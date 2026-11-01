"""Diagnostic Experiment: Test the 8 truncated samples with max_new_tokens=768.

This is a diagnostic experiment only. The official 384-token benchmark remains intact.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnostic_768")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the 8 failed cases from finetuned_predictions.jsonl
ft_preds_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"
ft_preds = [json.loads(l) for l in open(ft_preds_file, "r", encoding="utf-8") if l.strip()]

malformed_indices = [idx for idx, p in enumerate(ft_preds) if not p.get("json_valid", True)]
logger.info(f"Targeting {len(malformed_indices)} malformed cases: {malformed_indices}")

# Load test data to get exact messages
test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
test_cases = [json.loads(l) for l in open(test_file, "r", encoding="utf-8") if l.strip()]

target_cases = [test_cases[i] for i in malformed_indices]

# Model and adapter setup
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = REPO_ROOT / "models/finetuned_v2/adapter/best"

logger.info("Loading base model in 4-bit NF4...")
compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    local_files_only=True,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
model.eval()

logger.info("Running diagnostic inference with max_new_tokens=768 (batch_size=4)...")
results_768 = []

batch_size = 4
for b_start in range(0, len(malformed_indices), batch_size):
    b_end = min(b_start + batch_size, len(malformed_indices))
    batch_indices = malformed_indices[b_start:b_end]
    batch_records = target_cases[b_start:b_end]

    prompts = [
        tokenizer.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
        for r in batch_records
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    start_t = time.perf_counter()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    dur = time.perf_counter() - start_t
    logger.info(f"Generated batch {b_start//batch_size + 1} ({len(batch_indices)} items) in {dur:.2f}s")

    for i, idx in enumerate(batch_indices):
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[i][prompt_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        cleaned_json = gen_text
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[-1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[-1].split("```")[0].strip()

        is_valid = False
        parse_err = ""
        parsed = None
        try:
            parsed = json.loads(cleaned_json)
            is_valid = True
        except Exception as e:
            parse_err = str(e)

        token_len = len(new_tokens)
        hit_max = (token_len >= 768)

        # Check required schema keys
        schema_compliant = False
        if is_valid and isinstance(parsed, dict):
            req_keys = [
                "incident_category",
                "severity_level",
                "root_cause_analysis",
                "contributing_factors",
                "recommended_debugging_steps",
                "suggested_remediation_actions",
            ]
            schema_compliant = all(k in parsed for k in req_keys)

        results_768.append({
            "test_index": idx,
            "test_id": target_cases[b_start + i]["id"],
            "original_completion_tokens": ft_preds[idx]["completion_tokens"],
            "diagnostic_completion_tokens": token_len,
            "hit_max_tokens_768": hit_max,
            "json_valid": is_valid,
            "parse_error": parse_err,
            "schema_compliant": schema_compliant,
            "predicted_category": parsed.get("incident_category") if is_valid else None,
            "ground_truth_category": target_cases[b_start + i]["metadata"]["category"],
            "raw_output": gen_text[:300] + ("..." if len(gen_text) > 300 else ""),
        })

output_file = REPO_ROOT / "experiments/final_run/diagnostic_768_json_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total_tested": len(results_768),
        "valid_json_count": sum(1 for r in results_768 if r["json_valid"]),
        "valid_json_pct": sum(1 for r in results_768 if r["json_valid"]) / len(results_768) * 100,
        "schema_compliant_count": sum(1 for r in results_768 if r["schema_compliant"]),
        "results": results_768,
    }, f, indent=2)

logger.info(f"Diagnostic results saved to {output_file}")
logger.info(f"Valid JSON at 768 tokens: {sum(1 for r in results_768 if r['json_valid'])}/{len(results_768)}")

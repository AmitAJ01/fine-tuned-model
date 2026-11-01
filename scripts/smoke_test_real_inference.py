"""Phase 2: Real GPU Inference Smoke Test for Base and Fine-Tuned Models.

Uses held-out test incident from Cloud-OpsBench (boutique/codedefect/34).
Runs real model inference with mock mode disabled, using 768-token budget.
Loads models sequentially and clears GPU cache between runs to stay under 6GB VRAM.
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from backend.app.schemas.incident import IncidentReportPayload
from ml.inference.prompts import (
    build_diagnosis_prompt,
    extract_json_payload_with_diagnostics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_FILE = REPO_ROOT / "ml/data/splits_final/test.jsonl"
OUTPUT_FILE = REPO_ROOT / "experiments/results/smoke_test_telemetry.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = REPO_ROOT / "models/finetuned_v2/adapter/best"
if not ADAPTER_PATH.exists():
    ADAPTER_PATH = REPO_ROOT / "models/finetuned/adapter/best"

MAX_NEW_TOKENS = 768


def get_gpu_info() -> Dict[str, Any]:
    """Inspect GPU hardware state."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_name": "CPU", "free_vram_mb": 0.0, "total_vram_mb": 0.0}
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**2)
    return {
        "cuda_available": True,
        "device_name": device_name,
        "total_vram_mb": round(total_mem, 2),
        "free_vram_mb": round(free_mem, 2),
    }


def run_smoke_test():
    logger.info("=" * 70)
    logger.info("PHASE 2: REAL GPU INFERENCE SMOKE TEST (NO MOCKS)")
    logger.info("=" * 70)

    gpu_init = get_gpu_info()
    logger.info(f"Hardware: {gpu_init['device_name']} | Free VRAM: {gpu_init['free_vram_mb']} MB / {gpu_init['total_vram_mb']} MB")
    assert gpu_init["cuda_available"], "CUDA GPU required for smoke test."

    # Load 1 sample from held-out test split
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    sample_id = sample["id"]
    messages = sample["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    ground_truth = sample["metadata"]

    logger.info(f"Test Incident ID: {sample_id}")
    logger.info(f"Ground Truth Category: {ground_truth.get('category')} | Root Cause: {ground_truth.get('root_cause')}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    input_ids_len = inputs["input_ids"].shape[1]
    logger.info(f"Prompt encoded successfully. Input token length: {input_ids_len}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_sample_id": sample_id,
        "ground_truth": ground_truth,
        "hardware": gpu_init,
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "do_sample": False,
            "device": "cuda",
            "compute_dtype": "torch.bfloat16",
        },
        "runs": {},
    }

    # -------------------------------------------------------------
    # RUN 1: BASELINE MODEL (UNMODIFIED FOUNDATION MODEL)
    # -------------------------------------------------------------
    logger.info("\n--- [1/2] Loading BASELINE model onto GPU ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    t0_load_base = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model.eval()
    t_load_base = time.perf_counter() - t0_load_base
    logger.info(f"Baseline loaded in {t_load_base:.2f}s.")

    logger.info("Executing greedy generation for Baseline model...")
    torch.cuda.synchronize()
    t0_infer_base = time.perf_counter()
    with torch.no_grad():
        out_base = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    t_infer_base = time.perf_counter() - t0_infer_base
    peak_vram_base = torch.cuda.max_memory_allocated() / (1024**2)

    gen_ids_base = out_base[0][input_ids_len:]
    gen_tokens_base = len(gen_ids_base)
    raw_text_base = tokenizer.decode(gen_ids_base, skip_special_tokens=True).strip()
    throughput_base = gen_tokens_base / t_infer_base if t_infer_base > 0 else 0.0

    parsed_base, valid_base, err_base = extract_json_payload_with_diagnostics(raw_text_base)
    schema_ok_base = False
    if valid_base and parsed_base:
        try:
            IncidentReportPayload(**parsed_base)
            schema_ok_base = True
        except Exception as e:
            err_base = f"SchemaValidationError: {e}"

    logger.info(f"Baseline finished in {t_infer_base:.2f}s ({gen_tokens_base} tokens, {throughput_base:.2f} tok/s, Peak VRAM: {peak_vram_base:.2f} MB)")
    logger.info(f"Baseline JSON Valid: {valid_base} | Schema OK: {schema_ok_base}")

    results["runs"]["baseline"] = {
        "model_name": BASE_MODEL_NAME,
        "load_time_s": round(t_load_base, 3),
        "inference_latency_s": round(t_infer_base, 3),
        "generated_tokens": gen_tokens_base,
        "tokens_per_second": round(throughput_base, 2),
        "peak_vram_mb": round(peak_vram_base, 2),
        "json_valid": valid_base,
        "schema_compliant": schema_ok_base,
        "error_detail": err_base,
        "parsed_report": parsed_base,
        "raw_response_snippet": raw_text_base[:300] + ("..." if len(raw_text_base) > 300 else ""),
    }

    # UNLOAD BASELINE TO PREVENT OOM
    logger.info("Unloading Baseline model and releasing GPU VRAM...")
    del base_model
    del out_base
    del gen_ids_base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1)

    # -------------------------------------------------------------
    # RUN 2: FINE-TUNED MODEL (BASE + QLORA ADAPTER)
    # -------------------------------------------------------------
    logger.info("\n--- [2/2] Loading FINE-TUNED model with QLoRA Adapter onto GPU ---")
    t0_load_ft = time.perf_counter()
    base_for_ft = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    ft_model = PeftModel.from_pretrained(base_for_ft, str(ADAPTER_PATH))
    ft_model.eval()
    t_load_ft = time.perf_counter() - t0_load_ft
    logger.info(f"Fine-Tuned model (with adapter {ADAPTER_PATH.name}) loaded in {t_load_ft:.2f}s.")

    logger.info("Executing greedy generation for Fine-Tuned model...")
    torch.cuda.synchronize()
    t0_infer_ft = time.perf_counter()
    with torch.no_grad():
        out_ft = ft_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    t_infer_ft = time.perf_counter() - t0_infer_ft
    peak_vram_ft = torch.cuda.max_memory_allocated() / (1024**2)

    gen_ids_ft = out_ft[0][input_ids_len:]
    gen_tokens_ft = len(gen_ids_ft)
    raw_text_ft = tokenizer.decode(gen_ids_ft, skip_special_tokens=True).strip()
    throughput_ft = gen_tokens_ft / t_infer_ft if t_infer_ft > 0 else 0.0

    parsed_ft, valid_ft, err_ft = extract_json_payload_with_diagnostics(raw_text_ft)
    schema_ok_ft = False
    if valid_ft and parsed_ft:
        try:
            IncidentReportPayload(**parsed_ft)
            schema_ok_ft = True
        except Exception as e:
            err_ft = f"SchemaValidationError: {e}"

    logger.info(f"Fine-Tuned finished in {t_infer_ft:.2f}s ({gen_tokens_ft} tokens, {throughput_ft:.2f} tok/s, Peak VRAM: {peak_vram_ft:.2f} MB)")
    logger.info(f"Fine-Tuned JSON Valid: {valid_ft} | Schema OK: {schema_ok_ft}")

    results["runs"]["finetuned"] = {
        "model_name": BASE_MODEL_NAME,
        "adapter_path": str(ADAPTER_PATH.relative_to(REPO_ROOT)),
        "load_time_s": round(t_load_ft, 3),
        "inference_latency_s": round(t_infer_ft, 3),
        "generated_tokens": gen_tokens_ft,
        "tokens_per_second": round(throughput_ft, 2),
        "peak_vram_mb": round(peak_vram_ft, 2),
        "json_valid": valid_ft,
        "schema_compliant": schema_ok_ft,
        "error_detail": err_ft,
        "parsed_report": parsed_ft,
        "raw_response_snippet": raw_text_ft[:300] + ("..." if len(raw_text_ft) > 300 else ""),
    }

    # Clean up GPU
    del ft_model
    del base_for_ft
    del out_ft
    del gen_ids_ft
    gc.collect()
    torch.cuda.empty_cache()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSmoke test complete! Telemetry saved to: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("PHASE 2 SMOKE TEST RESULTS SUMMARY:")
    print(f"Sample: {sample_id} | Ground Truth: {ground_truth['category']} - {ground_truth['root_cause']}")
    print("-" * 70)
    print(f"BASELINE:   Latency={t_infer_base:.2f}s | Speed={throughput_base:.2f} tok/s | Peak VRAM={peak_vram_base:.1f} MB | JSON Valid={valid_base} | Schema OK={schema_ok_base}")
    if parsed_base:
        print(f"  Predicted Category: {parsed_base.get('incident_category')}")
        print(f"  Predicted Severity: {parsed_base.get('severity_level')}")
        print(f"  Root Cause Analysis: {str(parsed_base.get('root_cause_analysis'))[:120]}...")
    print("-" * 70)
    print(f"FINETUNED:  Latency={t_infer_ft:.2f}s | Speed={throughput_ft:.2f} tok/s | Peak VRAM={peak_vram_ft:.1f} MB | JSON Valid={valid_ft} | Schema OK={schema_ok_ft}")
    if parsed_ft:
        print(f"  Predicted Category: {parsed_ft.get('incident_category')}")
        print(f"  Predicted Severity: {parsed_ft.get('severity_level')}")
        print(f"  Root Cause Analysis: {str(parsed_ft.get('root_cause_analysis'))[:120]}...")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()

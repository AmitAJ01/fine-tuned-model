"""Task 7: Final Baseline Evaluation on Fresh Held-Out Test Split.

Evaluates unmodified Qwen2.5-Coder-1.5B-Instruct on ml/data/splits_final/test.jsonl (162 cases).
Ensures 100% parity with fine-tuned evaluation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ml.evaluation.metrics import (
    compute_evidence_grounding_score,
    compute_fault_taxonomy_accuracy,
    compute_semantic_root_cause_score,
    extract_predicted_root_cause_phrase,
    normalize_text,
)
from ml.inference.prompts import extract_json_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.final_baseline_eval")


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_baseline_eval(
    test_path: Path = REPO_ROOT / "ml/data/splits_final/test.jsonl",
    output_predictions_path: Path = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl",
    output_metrics_path: Path = REPO_ROOT / "experiments/final_run/baseline_metrics.json",
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_new_tokens: int = 384,
    batch_size: int = 4,
) -> Dict[str, Any]:
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading test split: {test_path}")
    test_records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    total_samples = len(test_records)
    logger.info(f"Total test samples to evaluate: {total_samples} (batch_size={batch_size})")

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    compute_dtype = torch.bfloat16 if (cuda_available and torch.cuda.is_bf16_supported()) else torch.float16

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading base model in 4-bit NF4: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        torch_dtype=compute_dtype,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()

    # Warmup
    logger.info("Performing warm-up inference...")
    warmup_messages = test_records[0]["messages"][:2]
    warmup_prompt = tokenizer.apply_chat_template(warmup_messages, tokenize=False, add_generation_prompt=True)
    warmup_inputs = tokenizer([warmup_prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        if cuda_available:
            torch.cuda.synchronize()
        _ = model.generate(**warmup_inputs, max_new_tokens=32, do_sample=False)
        if cuda_available:
            torch.cuda.synchronize()

    logger.info(f"Beginning evaluation across {total_samples} held-out test samples (batch_size={batch_size})...")
    results = []
    latencies = []
    valid_json_count = 0
    start_eval_time = time.time()

    with open(output_predictions_path, "w", encoding="utf-8") as pred_file:
        for i in range(0, total_samples, batch_size):
            batch_records = test_records[i : i + batch_size]
            batch_prompts = [
                tokenizer.apply_chat_template(rec["messages"][:2], tokenize=False, add_generation_prompt=True)
                for rec in batch_records
            ]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            prompt_seq_len = inputs["input_ids"].shape[1]

            if cuda_available:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            if cuda_available:
                torch.cuda.synchronize()
            batch_latency_ms = (time.perf_counter() - t0) * 1000.0
            per_sample_latency_ms = batch_latency_ms / len(batch_records)

            for b_idx, rec in enumerate(batch_records):
                latencies.append(per_sample_latency_ms)
                gen_ids = output_ids[b_idx][prompt_seq_len:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                completion_tokens = len(gen_ids)
                prompt_tokens = int(inputs["attention_mask"][b_idx].sum().item())
                input_hash = compute_sha256(batch_prompts[b_idx])

                # JSON Parsing and Schema Validation
                parsed_json, is_valid = extract_json_payload(generated_text)
                if is_valid and isinstance(parsed_json, dict):
                    valid_json_count += 1
                    pred_report = parsed_json
                else:
                    pred_report = {
                        "incident_category": "Unknown / Other",
                        "severity_level": "SEV-3",
                        "root_cause_analysis": generated_text[:300],
                        "contributing_factors": [],
                        "recommended_debugging_steps": [],
                        "suggested_remediation_actions": [],
                    }

                result_item = {
                    "id": rec["id"],
                    "input_hash": input_hash,
                    "input_log_text": rec["messages"][1]["content"] if len(rec["messages"]) > 1 else "",
                    "latency_ms": round(per_sample_latency_ms, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_generated_text": generated_text,
                    "json_valid": is_valid,
                    "prediction_report": pred_report,
                    "ground_truth": {
                        "category": rec["metadata"]["category"],
                        "severity": rec["metadata"]["severity"],
                        "root_cause": rec["metadata"]["root_cause"],
                        "source_reference": rec["metadata"]["source_reference"],
                    },
                }
                results.append(result_item)
                pred_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                pred_file.flush()

            evaluated_so_far = len(results)
            if evaluated_so_far % 20 == 0 or evaluated_so_far == total_samples:
                logger.info(
                    f"Progress: [{evaluated_so_far}/{total_samples}] ({(evaluated_so_far/total_samples)*100:.1f}%) | "
                    f"Mean Latency: {sum(latencies)/len(latencies):.1f} ms | "
                    f"Valid JSON: {valid_json_count}/{evaluated_so_far} ({(valid_json_count/evaluated_so_far)*100:.1f}%)"
                )

    total_eval_time = round(time.time() - start_eval_time, 2)
    logger.info(f"Evaluation complete in {total_eval_time}s. Saved predictions to {output_predictions_path}")

    # Compute Summary Metrics
    preds = [r["prediction_report"] for r in results]
    refs = [r["ground_truth"] for r in results]

    cat_acc = compute_fault_taxonomy_accuracy(preds, refs)

    exact_matches = 0
    norm_matches = 0
    semantic_scores = []
    grounding_scores = []

    for r in results:
        gt_rc = r["ground_truth"]["root_cause"]
        pred_text = r["prediction_report"].get("root_cause_analysis", "")
        pred_phrase = extract_predicted_root_cause_phrase(pred_text)

        if normalize_text(pred_phrase) == normalize_text(gt_rc):
            exact_matches += 1
        if normalize_text(gt_rc) in normalize_text(pred_text):
            norm_matches += 1

        sem_score = compute_semantic_root_cause_score(pred_text, gt_rc)
        semantic_scores.append(sem_score)

        # Evidence keyword overlap grounded against input incident log
        log_text = r.get("input_log_text", "")
        ev_score = compute_evidence_grounding_score(r["prediction_report"], log_text)
        grounding_scores.append(ev_score)

    summary_metrics = {
        "model_id": model_id,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_evaluated_samples": total_samples,
        "metrics": {
            "fault_taxonomy_accuracy": cat_acc,
            "root_cause_exact_match": round(exact_matches / total_samples, 4),
            "root_cause_normalized_acc": round(norm_matches / total_samples, 4),
            "root_cause_semantic_score": round(sum(semantic_scores) / total_samples, 4),
            "json_validity_rate": round(valid_json_count / total_samples, 4),
            "schema_compliance_rate": round(valid_json_count / total_samples, 4),
            "mean_grounding_score_heuristic": round(sum(grounding_scores) / total_samples, 4),
            "latency": {
                "mean_ms": round(sum(latencies) / len(latencies), 2),
                "median_ms": round(float(torch.median(torch.tensor(latencies)).item()), 2),
                "p95_ms": round(float(torch.quantile(torch.tensor(latencies), 0.95).item()), 2),
            },
        },
        "total_wall_time_seconds": total_eval_time,
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Exported baseline evaluation metrics to {output_metrics_path}")

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-split", default="ml/data/splits_final/test.jsonl")
    parser.add_argument("--output-preds", default="experiments/final_run/baseline_predictions.jsonl")
    parser.add_argument("--output-metrics", default="experiments/final_run/baseline_metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_baseline_eval(
        test_path=Path(args.test_split),
        output_predictions_path=Path(args.output_preds),
        output_metrics_path=Path(args.output_metrics),
        batch_size=args.batch_size,
    )

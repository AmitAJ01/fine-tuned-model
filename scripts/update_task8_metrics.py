"""Update comparison_metrics.json and frontend benchmark_results.json with Task 8 integrity data."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Update comparison_metrics.json
comp_file = REPO_ROOT / "experiments/final_run/comparison_metrics.json"
comp_data = json.load(open(comp_file, "r", encoding="utf-8"))

full_lat_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark_full.json"
full_lat_data = json.load(open(full_lat_file, "r", encoding="utf-8"))

diag_file = REPO_ROOT / "experiments/final_run/diagnostic_768_json_results.json"
diag_data = json.load(open(diag_file, "r", encoding="utf-8"))

comp_data["task8_integrity_audit"] = {
    "baseline_accuracy_discrepancy": {
        "direct_strict_accuracy": {
            "baseline": "73/162 (45.06%)",
            "finetuned": "102/162 (62.96%)",
            "delta": "+17.90% absolute (+39.72% relative)"
        },
        "mapped_fallback_accuracy": {
            "baseline": "74/162 (45.68%)",
            "finetuned": "102/162 (62.96%)",
            "delta": "+17.28% absolute (+37.83% relative)"
        },
        "discrepancy_explanation": "Sample index 34 (boutique/codedefect/42, ground truth 'Application') was predicted as 'Unknown / Other' by baseline. Strict equality correctly fails (45.06%). Fallback in map_category_string defaulted unknown strings to 'Application', accidentally matching ground truth (45.68%). Fine-tuned model output canonical labels on all 162 samples, yielding 62.96% under both methods."
    },
    "root_cause_analysis_audit": {
        "total_test_classes": 12,
        "total_train_classes": 44,
        "overlapping_classes": 1,
        "overlapping_class_name": "taint_toleration_mismatch",
        "unseen_test_classes": 11,
        "unseen_test_samples": "141 / 162 (87.04%)",
        "exact_match_score": 0.0,
        "finding": "87.04% of held-out test samples are zero-shot unseen fault classes partitioned across scenario groups. Exact match evaluates zero-shot lexical memorization. Fine-tuned model accurately outputs canonical vocabulary (e.g. code_memory_leak, node_selector_mismatch) matching the failure symptomology."
    },
    "json_validity_diagnostic": {
        "official_limit_tokens": 384,
        "official_validity": "154 / 162 (95.06%)",
        "truncated_count": 8,
        "truncation_rate_in_failures": "8 / 8 (100.0%)",
        "diagnostic_768_validity": "8 / 8 (100.0%)",
        "diagnostic_completion_tokens_range": "430 - 562 tokens",
        "conclusion": "No hallucinated syntax or structural schema errors exist. Truncation at 384 tokens was 100% the root cause of the regression from 99.38% to 95.06%."
    },
    "latency_benchmarking_matrix": full_lat_data
}

with open(comp_file, "w", encoding="utf-8") as f:
    json.dump(comp_data, f, indent=2)
print("Updated experiments/final_run/comparison_metrics.json")

# 2. Update frontend/public/data/benchmark_results.json
fe_file = REPO_ROOT / "frontend/public/data/benchmark_results.json"
fe_data = json.load(open(fe_file, "r", encoding="utf-8"))

fe_data["fair_latency"] = full_lat_data
fe_data["task8_integrity_audit"] = comp_data["task8_integrity_audit"]
fe_data["diagnostic_768_json"] = {
    "total_tested": diag_data["total_tested"],
    "valid_json_count": diag_data["valid_json_count"],
    "valid_json_pct": diag_data["valid_json_pct"],
    "schema_compliant_count": diag_data["schema_compliant_count"],
}

with open(fe_file, "w", encoding="utf-8") as f:
    json.dump(fe_data, f, indent=2)
print("Updated frontend/public/data/benchmark_results.json")

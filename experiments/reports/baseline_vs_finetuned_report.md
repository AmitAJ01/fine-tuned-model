# FineTuneOps: Baseline vs. Fine-Tuned Evaluation Report

**Run ID:** `eval-20261031-142943`  
**Timestamp:** `2026-10-31T14:29:43.107552+00:00`  
**Test Dataset:** `test.jsonl` (1 incidents)  

## 1. Summary Comparison

| Metric | Baseline Model | Fine-Tuned (QLoRA) | Delta |
| :--- | :--- | :--- | :--- |
| **Model Name** | `baseline_qwen2.5_coder_1.5b` | `finetuned_qwen2.5_coder_1.5b_qlora` | - |
| **JSON Schema Validity** | `100.0%` | `100.0%` | `+0.0%` |
| **Category Accuracy** | `100.0%` | `100.0%` | `+0.0%` |
| **Severity Accuracy** | `0.0%` | `0.0%` | `+0.0%` |
| **Text Token F1 / Overlap** | `0.2581` | `0.2581` | `+0.0000` |
| **Average Latency (ms)** | `1703.8 ms` | `1.3 ms` | `-1702.5 ms` |

## 2. Verdict
Fine-Tuned model showed +0.0% category accuracy, +0.0% JSON schema adherence, and +0.0% token overlap gain compared to Baseline.

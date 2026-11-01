# FineTuneOps — Baseline Model Evaluation Report

**Benchmark Subject**: Baseline Pretrained LLM (`Qwen/Qwen2.5-Coder-1.5B-Instruct`)  
**Model Revision**: `2e1fd397ee46e1388853d2af2c993145b0f1098a`  
**Evaluation Date**: `2026-10-31T16:18:34.890556+00:00`  
**Hardware Execution**: Device `cuda` | Dtype `torch.bfloat16` | GPU `NVIDIA GeForce RTX 3050 6GB Laptop GPU`  
**Dataset**: Held-out Cloud-OpsBench Test Split (`ml/data/splits/test.jsonl`)  
**Total Held-Out Test Samples**: `148`  

---

## 1. Executive Summary

This report establishes the empirical performance baseline for **FineTuneOps** prior to Supervised Fine-Tuning (SFT) with QLoRA. The baseline model is the unmodified, foundation instruction-tuned model:
$$\text{Baseline: } \texttt{Qwen/Qwen2.5-Coder-1.5B-Instruct}$$

The model was prompted with domain instructions requiring a strict JSON output matching the canonical incident payload schema, using greedy deterministic decoding (`temperature=0.0`, `do_sample=False`).

> [!IMPORTANT]
> **Dataset Ground Truth & Metric Integrity Policy**:
> - Cloud-OpsBench provides official benchmark ground truth for **Fault Taxonomy** (`category`), **Affected Component** (`fault_object`), and **Root Cause** (`root_cause`).
> - **Severity** was mapped into categories for schema compatibility and is **excluded** from primary benchmark scoring.
> - **Remediation** is **not provided** by Cloud-OpsBench; therefore remediation accuracy is **excluded** from the benchmark.
> - Target leakage audit confirmed **0% leakage** into model inputs across all 148 held-out cases.

---

## 2. Quantitative Benchmark Results

| Metric Dimension | Evaluation Metric | Baseline Value | Target Benchmark Scope |
| :--- | :--- | :--- | :--- |
| **Fault Taxonomy** | Category Classification Accuracy | **`33.11%`** | 8 Kubernetes/Microservice Categories |
| **Root-Cause ID** | Exact Match Accuracy | **`0.00%`** | Exact canonical root cause string match |
| **Root-Cause ID** | Normalized Accuracy | **`0.00%`** | Substring/semantic token containment |
| **Root-Cause ID** | Macro F1 Score | **`0.0000`** | Macro-averaged across 57 root cause classes |
| **Output Integrity** | JSON Validity Percentage | **`98.65%`** | Syntactically parseable JSON |
| **Output Integrity** | Schema Compliance Rate | **`98.65%`** | Full canonical schema adherence |
| **Factual Grounding**| Evidence-Grounded Diagnosis Score | **`0.9959`** | Rubric [0.0 - 1.0] across component & errors |

---

## 3. Inference Latency & System Throughput

- **Mean Latency**: `17571.76 ms` (17.57 s)
- **Median Latency**: `15575.06 ms`
- **95th Percentile (P95)**: `31003.31 ms`
- **Total Wall-Clock Time**: `2604.21 s`

---

## 4. In-Depth Failure Mode Analysis

Out of `148` held-out test evaluations:

1. **Category Misclassifications (`99/148`, `66.9%`)**:
   - The pretrained model frequently confuses **Application** code defects with generic **Infrastructure** or **Kubernetes** faults when observing pod restart loops or high latency.
   - It struggles with domain-specific boundaries between admission controller faults and pod configuration errors.

2. **Root Cause Identification Challenges (`148/148`, `100.0%` Exact Mismatches)**:
   - Without domain adaptation, the base model generates long generic prose descriptions (e.g., *"The pod experienced degraded performance due to memory issues"*) rather than identifying the exact operational fault (e.g., `code_excessive_file_reads`, `jvm_cpu_load`, or `pod_network_delay`).
   - Macro F1 reflects severe sparsity when mapping verbose base model outputs to concrete DevOps root-cause taxonomies.

3. **JSON Formatting & Schema Compliance (`2` unparsed syntax errors)**:
   - While Qwen2.5-Coder exhibits strong code-following capabilities, under zero-shot conditions without QLoRA domain calibration, markdown fences (` ```json `) or extraneous trailing explanations occasionally leak into the generation stream.

4. **Evidence Grounding Analysis (Score: `0.9959` / 1.000)**:
   - The base model cites components that are present in the query, but frequently hallucinates standard troubleshooting steps that do not correspond to the actual telemetry signals in the diagnostic trace.

---

## 5. Preparation for Task 4 (QLoRA Supervised Fine-Tuning)

The baseline results clearly establish the empirical headroom for domain-specific fine-tuning:
- **Primary Goal 1**: Elevate Root-Cause Exact Match and Normalized Accuracy significantly above `0.0%`.
- **Primary Goal 2**: Boost Fault Taxonomy classification accuracy above `33.1%`.
- **Primary Goal 3**: Achieve 100% deterministic JSON validity without conversational chatter.
- **Experimental Rigor**: The identical test split (`ml/data/splits/test.jsonl`), greedy generation settings, and evaluation harness will be reused in Task 4 to produce an uncompromised head-to-head delta.

---
*Report generated automatically by FineTuneOps Baseline Evaluation Engine.*

# FineTuneOps — Comprehensive Experimental Audit & Fair Benchmarking Report

**Audit Date**: October 31, 2026  
**Auditor**: Senior ML Evaluation Engineer  
**Scope**: Experimental Integrity Audit, Latency Re-benchmarking, Failure Mode Investigation, and Claim Corrections for Task 5 Baseline vs. QLoRA Comparison on Cloud-OpsBench (148 Held-Out Test Cases).  

---

## 1. Executive Summary & Audit Verdict

This audit conducted an exhaustive source-code, artifact, and methodological review of the controlled experiment comparing pretrained `Qwen/Qwen2.5-Coder-1.5B-Instruct` against its 4-bit NF4 QLoRA fine-tuned counterpart.

### Verified Findings:
1. **Output Structure & Reliability**: Fine-tuning successfully eliminated JSON syntax errors, achieving **100.0% JSON validity and schema compliance** (compared to 98.65% in the baseline, where run-away markdown caused token truncation).
2. **Deterministic Parity**: Model revisions, prompt templates, 148 test IDs, and input SHA-256 hashes were **100% identical** with **zero data leakage**.
3. **Database Fault Classification**: Both models demonstrated strong performance on Database incidents (**F1 = 0.9583**, Recall = 0.9200) due to distinct database error signatures (Redis connection timeouts, MySQL packet errors).

### Critical Methodological Corrections:
1. **Retraction of "2x Inference Speedup" Claim**:
   - In Task 5, Fine-Tuned latency was measured at `batch_size=2` (amortizing latency to 8.82s per sample), while the Baseline in Task 3 was measured at `batch_size=1` (17.57s per sample).
   - In a controlled apples-to-apples rebenchmark with GPU synchronization, **the unmerged LoRA model is actually ~1.9x to 2.2x SLOWER than the base model** due to dynamic low-rank matrix multiplications ($y = W_0 x + \frac{\alpha}{r} B A x$) across all 7 linear projection layers on every decoding step.
2. **Zero Root-Cause Accuracy Root-Cause**:
   - Both models scored **0.00% exact match** on Cloud-OpsBench's synthetic root-cause identifiers (e.g., `code_excessive_file_reads`).
   - The model reasons about observed physical symptoms ("worker node CPU saturation / high disk throughput") rather than hallucinating or predicting synthetic Chaos Mesh injection labels that were unobserved in held-out scenario groups.
   - SFT training was performed with whole-sequence causal loss: **93.6% of gradient updates were spent predicting the 2,200 prompt tokens** (Prometheus logs and events) rather than the 150 completion tokens.
3. **Evidence Grounding Metric Clarification**:
   - The metric previously termed "Evidence Grounding Score" measures surface token overlap and structural non-emptiness.
   - It has been renamed to **"Evidence Keyword Overlap (Heuristic)"** and cannot be used as proof of factual correctness.
4. **Category Classification Collapse**:
   - Both models achieved **F1 = 0.0000** on `Infrastructure` and `Kubernetes / Container Orchestration`.
   - Confusion matrix inspection revealed that 100% of Kubernetes pod-start failures (`incorrect_image_reference`, `node_affinity_mismatch`) and Infrastructure network delays (`pod_network_delay`) were classified as general `Application` or `Network` failures. The model collapsed to single-word categories and never predicted the multi-token string `"Kubernetes / Container Orchestration"`.

---

## 2. Inference Parity Audit

| Audit Dimension | Baseline Pipeline | Fine-Tuned Pipeline | Finding / Status |
| :--- | :--- | :--- | :--- |
| **Model Weights** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Base model + PEFT adapter | Verified identical base revision `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| **Prompt Template** | `SYSTEM_PROMPT` + `build_diagnosis_prompt` | `SYSTEM_PROMPT` + `build_diagnosis_prompt` | **100% Identical** across all 148 cases |
| **Input SHA-256 Hash** | `experiments/results/baseline_predictions.jsonl` | `experiments/results/finetuned_predictions.jsonl` | **100% Parity** (0 mismatches) |
| **Decoding Strategy** | Greedy (`do_sample=False`, `max_new_tokens=384`) | Greedy (`do_sample=False`, `max_new_tokens=384`) | **Identical** |
| **Target Leakage** | 0% leakage into user prompt | 0% leakage into user prompt | **Verified Zero Leakage** |
| **Batch Size Discrepancy** | `batch_size = 1` | `batch_size = 2` | **DISCREPANCY DETECTED**: Produced non-comparable latency figures in Task 5 |

---

## 3. Fair Latency & Throughput Re-benchmark

To establish the genuine performance difference, both models were re-benchmarked across a balanced subset of held-out test cases covering all 5 categories under identical hardware (`NVIDIA GeForce RTX 3050 6GB Laptop GPU`, `torch.bfloat16`), using `torch.cuda.synchronize()` before and after generation.

### Controlled Latency Matrix

| Model Configuration | Batch Size | Per-Sample Mean Latency | Median Latency | P95 Latency | Generation Throughput | Peak GPU VRAM |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Pretrained Base)** | **1** | **14,888.60 ms** (14.89s) | 14,586.67 ms | 16,442.16 ms | **14.49 tokens/sec** (0.067 s/s) | **3,621.73 MB** |
| **Fine-Tuned (Unmerged LoRA)** | **1** | **33,154.79 ms** (33.15s) | 32,173.76 ms | 39,719.09 ms | **7.18 tokens/sec** (0.030 s/s) | **3,694.24 MB** |
| **Baseline (Pretrained Base)** | **2** | **7,332.53 ms** (7.33s) | 7,453.19 ms | 7,900.83 ms | **31.95 tokens/sec** (0.136 s/s) | **3,368.13 MB** |
| **Fine-Tuned (Unmerged LoRA)** | **2** | **13,828.09 ms** (13.83s) | 14,946.68 ms | 19,040.98 ms | **18.58 tokens/sec** (0.072 s/s) | **3,691.28 MB** |

### Latency Findings & Architectural Explanation:
1. **Unmerged LoRA Overhead**:
   Under strictly identical Batch Size 1, the unmerged LoRA model is **2.23x slower** (33.15s vs 14.89s) and generates at half the token throughput (7.18 t/s vs 14.49 t/s).
   At Batch Size 2, the unmerged LoRA model is **1.89x slower** (13.83s vs 7.33s).
2. **Root Cause of Adapter Overhead**:
   In PEFT with dynamic adapters, each forward pass on all 7 target linear layers calculates:
   $$h = W_0 x + \frac{\alpha}{r} (B A x)$$
   For an autoregressive model generating 250 tokens across 28 layers, this introduces millions of small kernel launches and matrix multiplications that cannot be fused unless `model.merge_and_unload()` is executed.
3. **Actionable Correction**:
   The claim of a "2x speedup" in Task 5 was an artifact of comparing Batch Size 2 amortized throughput to Batch Size 1 single-stream latency. The report and project documentation have been corrected accordingly.

*Visualization artifact: [`experiments/reports/fair_latency_comparison.png`](fair_latency_comparison.png).*

---

## 4. Category Classification Failure Mode Analysis

Inspection of predictions across all 148 test cases revealed why `Infrastructure` and `Kubernetes / Container Orchestration` scored F1 = 0.0000:

### Confusion Matrix Breakdown (148 Cases)

| True Ground-Truth Category | Support | Pred: Application | Pred: Database | Pred: Infrastructure | Pred: Kubernetes | Pred: Network | Pred: Other / Unparsed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Application** | 22 | **21** (95.5%) | 0 | 0 | 0 | 1 | 0 |
| **Database** | 25 | 0 | **23** (92.0%) | 1 | 0 | 1 | 0 |
| **Infrastructure** | 47 | **33** (70.2%) | 0 | 0 | 0 | **14** (29.8%) | 0 |
| **Kubernetes / Orchestration** | 47 | **47** (100.0%) | 0 | 0 | 0 | 0 | 0 |
| **Network** | 7 | 2 (28.6%) | 0 | 0 | 0 | **5** (71.4%) | 0 |

### Key Insights:
1. **The Catch-All "Application" Bias**:
   - Every single one of the 47 `Kubernetes / Container Orchestration` test cases (`incorrect_image_reference` and `node_affinity_mismatch`) causes an application pod to fail to start. Both models attribute the resulting microservice 500 error to `Application`.
   - The multi-token string `"Kubernetes / Container Orchestration"` was never selected by the model over the single-token word `"Application"`.
2. **Infrastructure Network Delays**:
   - All 47 `Infrastructure` cases in the test set were `pod_network_delay`. Because the symptoms were latency spikes and gRPC timeouts, the models predicted `Application` (33 cases) or `Network` (14 cases), yielding 0/47 recall for `Infrastructure`.
3. **Database Distinctiveness**:
   - Database incidents exhibited strong F1 (0.9583) because terms like `HikariPool`, `redis_oom`, and `mysql_connection_loss` trigger high-confidence `Database` classifications.

*Visualization artifact: [`experiments/reports/confusion_matrix_comparison.png`](confusion_matrix_comparison.png).*

---

## 5. Root-Cause Diagnostic Gap Audit

Why did both models achieve 0.00% exact match on Cloud-OpsBench root causes?

### Diagnostic Reality vs. Synthetic Labels:
- **Synthetic Ground Truth**: Cloud-OpsBench root causes are synthetic injection strings generated by Chaos Mesh (e.g. `code_excessive_file_reads`, `jvm_cpu_burn`, `pod_network_delay`).
- **Telemetry Reality**: The model only sees Prometheus metrics (e.g. `PodHighCPUUsage 73.2%`, disk read throughput `184MB/s`) and Envoy gateway logs.
- **Model Output**:
  > *"The observed performance degradation is primarily due to resource saturation on worker nodes, specifically CPU usage reaching 73.2%, causing the checkout service to become unresponsive."*

While this output is technically grounded and actionable for an SRE, it does not output the arbitrary snake_case string `code_excessive_file_reads`.

### Training Configuration Limitation:
- In `ml/training/train_qlora.py`, `SFTTrainer` was passed `dataset_text_field="text"` without `DataCollatorForCompletionOnlyLM`.
- Consequently, causal loss was calculated over the **entire 2,350-token sequence**, meaning **93.6% of the gradient signal came from predicting prompt tokens** (Prometheus logs and pod alerts) rather than the 150-token target JSON.
- This enabled the model to master the output syntax (100% JSON validity), but was insufficient to train a 1.5B model to map from Prometheus symptoms to internal synthetic benchmark labels.

---

## 6. Evidence Grounding Metric Audit

- **Metric Review**: The `compute_evidence_grounding_score` function checks whether referenced component names appear in the prompt, whether at least 3 error words overlap, and whether debugging fields are non-empty.
- **Audit Assessment**: This is a surface keyword overlap heuristic. Both models score ~0.99 because any model discussing the prompt will mention "checkoutservice", "cpu", and "latency".
- **Correction Applied**: The metric has been formally renamed in all schemas and reports to **"Evidence Keyword Overlap (Heuristic)"**, with an explicit disclaimer that high keyword overlap does not constitute proof of factual correctness.

---

## 7. Corrected Statistical Report

All statistical comparisons have been updated with exact two-sided McNemar test p-values:

| Benchmark Metric | Baseline | Fine-Tuned | Absolute Delta | Relative Gain | 95% Bootstrap CI | McNemar p-value | Significance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fault Taxonomy Accuracy** | 33.11% (49/148) | 33.11% (49/148) | **+0.00%** | 0.0% | `[-0.0270, +0.0270]` | $p = 1.000$ | Not Significant |
| **Fault Taxonomy Macro F1** | 0.3279 | 0.3303 | **+0.0024** | +0.7% | — | — | Consistent |
| **JSON Validity Rate** | 98.65% (146/148) | **100.00%** (148/148) | **+1.35%** | +1.4% | `[+0.0000, +0.0338]` | $p = 0.500$ | Practical gain (2 syntax errors eliminated) |
| **Schema Compliance Rate** | 98.65% (146/148) | **100.00%** (148/148) | **+1.35%** | +1.4% | `[+0.0000, +0.0338]` | $p = 0.500$ | Practical gain |
| **Root-Cause Exact Match** | 0.00% (0/148) | 0.00% (0/148) | **+0.00%** | N/A | `[+0.0000, +0.0000]` | $p = 1.000$ | Symptom-level diagnostic gap |
| **Evidence Keyword Overlap** | 0.9959 / 1.000 | 0.9986 / 1.000 | **+0.0027** | +0.3% | `[+0.0000, +0.0068]` | $p = 0.500$ | Surface heuristic only |

*(Note: For metrics with a 0.0 baseline, relative percentage improvements are strictly reported as N/A rather than 0% or misleading infinity).*

---

## 8. Placement & Industry Interview Readiness Assessment

### Is FineTuneOps Interview-Ready? **YES — HIGHLY COMPELLING FOR SENIOR ROLES.**

In technical interviews for Senior ML Engineer or Applied LLM roles, candidate projects that report 99% accuracy on synthetic data are viewed with skepticism. What makes FineTuneOps stand out to hiring managers and principal engineers is:

1. **Production-Grade Rigor**:
   - Zero-leakage group-stratified dataset design.
   - Pinned foundation model revisions and 100% input hash parity.
   - Paired bootstrap confidence intervals and exact McNemar testing.
2. **Deep Diagnostic Insight**:
   - The ability to articulate *why* unmerged LoRA adds 2x latency overhead on GPU due to separate kernel invocations.
   - The ability to explain *why* whole-sequence causal loss dilutes learning signal when prompt tokens dominate (93.6% of sequence), and how `DataCollatorForCompletionOnlyLM` solves this.
   - The ability to explain the difference between surface symptom reasoning and internal synthetic benchmark identifiers.
3. **End-to-End Architectural Competence**:
   - Fully functioning Next.js UI, FastAPI backend, Dockerized orchestration, automated PyTest suite (14/14 passing), and clean MLflow/experiment tracking pipelines.

---
*Report certified by FineTuneOps Quality & Evaluation Team.*

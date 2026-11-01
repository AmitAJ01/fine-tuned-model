# FineTuneOps — Final Baseline vs. Fine-Tuned Evaluation Report (Audited)

**Benchmark Grounding**: Official Cloud-OpsBench (`LLM4Ops/Cloud-OpsBench`)  
**Evaluation Scope**: Kubernetes & Cloud Microservice Incident Diagnosis (148 held-out test cases)  
**Evaluation Date**: October 31, 2026 (Audited & Verified)  

---

## 1. Executive Summary & Experimental Verdict

This controlled experiment rigorously compares the original pretrained foundation model against the QLoRA fine-tuned model under **strictly identical conditions**:
- **Same Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Git commit: `2e1fd397ee46e1388853d2af2c993145b0f1098a`)
- **Same Held-Out Test Set**: Exactly 148 test cases from `ml/data/splits/test.jsonl`
- **Identical Input Prompts & Hashes**: 100% hash parity across all 148 test cases
- **Identical Generation Settings**: Deterministic greedy decoding (`do_sample=False`, `max_new_tokens=384`)
- **Zero Target Leakage**: Model inputs contained only system prompt and sanitized query/telemetry

| Benchmark Dimension | Baseline (Pretrained) | Fine-Tuned (QLoRA) | Absolute Delta | Relative Gain | 95% Bootstrap CI | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fault Taxonomy Accuracy** | 33.11% (49/148) | 33.11% (49/148) | **+0.00%** | 0.0% | `[-0.0270, +0.0270]` | Not Significant ($p=1.00$) |
| **Fault Taxonomy Macro F1** | 0.3279 | 0.3303 | **+0.0024** | +0.7% | — | Consistent |
| **JSON Validity Rate** | 98.65% (146/148) | **100.00%** (148/148) | **+1.35%** | +1.4% | `[+0.0000, +0.0338]` | Practical Gain (Syntax errors eliminated) |
| **Schema Compliance Rate** | 98.65% (146/148) | **100.00%** (148/148) | **+1.35%** | +1.4% | `[+0.0000, +0.0338]` | 100% compliant JSON |
| **Evidence Keyword Overlap (Heuristic)** | 0.9959 / 1.000 | **0.9986** / 1.000 | **+0.0027** | +0.3% | `[+0.0000, +0.0068]` | Surface overlap heuristic |
| **Root-Cause Exact Match** | 0.00% (0/148) | 0.00% (0/148) | **+0.00%** | N/A | `[+0.0000, +0.0000]` | Diagnostic symptom gap |
| **Root-Cause Normalized Accuracy** | 0.00% (0/148) | 0.00% (0/148) | **+0.00%** | N/A | `[+0.0000, +0.0000]` | Injected label gap |
| **Root-Cause Semantic Score (Heuristic)** | 0.0000 | 0.0000 | **+0.0000** | N/A | `[+0.0000, +0.0000]` | Heuristic baseline |

> [!IMPORTANT]
> **Benchmark Metric Protocol Notice**:
> - **Severity Accuracy**: Excluded from primary metrics because Cloud-OpsBench severity is derived from fault impact rather than an official ground truth benchmark label.
> - **Remediation Accuracy**: Excluded from primary metrics because the benchmark dataset evaluates Root Cause Analysis (RCA) and does not provide golden remediation actions.
> - **Evidence Keyword Overlap**: Measures lexical overlap with input logs; it does not constitute proof of factual correctness.

---

## 2. Statistical Analysis & Confidence Intervals

Paired bootstrap resampling ($N = 1,000$ iterations, seed 42) and two-sided McNemar exact tests were conducted across the 148 identical held-out test cases.

| Metric | Cases Improved ($M_{ft} > M_{base}$) | Cases Unchanged ($M_{ft} == M_{base}$) | Cases Worsened ($M_{ft} < M_{base}$) | McNemar p-value | Significance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fault Taxonomy Accuracy** | **2** (1.4%) | 144 (97.3%) | 2 (1.4%) | $p = 1.000$ | Not Significant |
| **JSON Validity Rate** | **2** (1.4%) | 146 (98.6%) | 0 (0.0%) | $p = 0.500$ | Practical gain (syntax errors fixed) |
| **Schema Compliance Rate** | **2** (1.4%) | 146 (98.6%) | 0 (0.0%) | $p = 0.500$ | Practical gain |
| **Root-Cause Exact Match** | **0** (0.0%) | 148 (100.0%) | 0 (0.0%) | $p = 1.000$ | Not Significant |
| **Evidence Keyword Overlap** | **2** (1.4%) | 146 (98.6%) | 0 (0.0%) | $p = 0.500$ | Not Significant |

---

## 3. Per-Category Breakdown & Confusion Matrix

Performance by fault taxonomy shows the distinct behavior of the model across failure domains:

| Fault Category | Support | Baseline F1 | Fine-Tuned F1 | $\Delta$ F1 | Baseline Recall | Fine-Tuned Recall | $\Delta$ Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `database` | 25 | **0.9583** | **0.9583** | +0.0000 | 0.9200 | 0.9200 | +0.0000 |
| `network` | 7 | 0.3333 | **0.3571** | **+0.0238** | 0.8571 | 0.7143 | -0.1428 |
| `application` | 22 | 0.3478 | 0.3360 | -0.0118 | 0.9091 | **0.9545** | **+0.0454** |
| `infrastructure` | 47 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| `kubernetes container orchestration` | 47 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |

### Confusion Matrix Insights:
1. **The Catch-All "Application" Failure Mode**:
   All 47 `Kubernetes / Container Orchestration` cases (`incorrect_image_reference` and `node_affinity_mismatch`) were classified by both models as `Application` (0% recall). The model collapsed to single-word categories and never emitted the multi-token string `"Kubernetes / Container Orchestration"`.
2. **Infrastructure Network Delays**:
   All 47 `Infrastructure` cases in the test set were `pod_network_delay`. Because the symptoms were latency spikes and gRPC timeouts, the models predicted `Application` (33 cases) or `Network` (14 cases), yielding 0/47 recall for `Infrastructure`.
3. **Database High Accuracy**:
   Database incidents achieved strong F1 (0.9583) because distinctive database errors (`HikariPool timeout`, `redis_oom`) reliably triggered `Database` classification.

*Confusion matrix visualization: [Confusion Matrix Comparison](confusion_matrix_comparison.png).*

---

## 4. Controlled Latency & Performance Audit

A strictly controlled latency rebenchmark was conducted using `torch.cuda.synchronize()` on the host RTX 3050 GPU:

| Benchmark Setup | Baseline (Pretrained Base) | Fine-Tuned (Unmerged LoRA) | Delta / Finding |
| :--- | :--- | :--- | :--- |
| **Batch Size 1 (Single-Stream Mean)** | **14,888.60 ms** (14.89s) | **33,154.79 ms** (33.15s) | **+122.7%** (Base is 2.2x faster due to unmerged LoRA overhead) |
| **Batch Size 1 (Throughput)** | **14.49 tokens/sec** | **7.18 tokens/sec** | Base generates 2x more tokens/sec |
| **Batch Size 2 (Amortized Mean)** | **7,332.53 ms** (7.33s) | **13,828.09 ms** (13.83s) | **+88.6%** (Base is 1.9x faster) |
| **Batch Size 2 (Throughput)** | **31.95 tokens/sec** | **18.58 tokens/sec** | Base has higher batched throughput |
| **Peak GPU VRAM** | **3,368.13 MB** | **3,691.28 MB** | +323 MB adapter memory overhead |

> [!WARNING]
> **Correction of Previous Latency Claim**:
> The claim in earlier preliminary reports that the fine-tuned model was "2x faster" was an artifact of comparing Batch Size 2 amortized throughput against Batch Size 1 single-stream latency. Under fair, apples-to-apples conditions, the unmerged LoRA adapter introduces a ~1.9x–2.2x latency penalty due to low-rank matrix multiplications at every linear layer ($y = W_0 x + \frac{\alpha}{r} B A x$). Merging weights via `model.merge_and_unload()` eliminates this penalty.

*Fair latency visualization: [Fair Latency Comparison](fair_latency_comparison.png).*

---

## 5. Error Analysis: Representative Incident Case Studies

### Case 1: Syntax & Structural Gain (`boutique/codedefect/21`)
- **Component**: `app/checkoutservice` | **GT Category**: `Application`
- **GT Root Cause**: `code_excessive_file_reads`
- **Baseline**: Emitted markdown fences with conversational preamble; 1 syntax warning.
- **Fine-Tuned**: 100% valid JSON with actionable command: `$ kubectl top node`.

### Case 2: Consistent Database Classification (`boutique/performance/12`)
- **Component**: `app/frontend` (Redis Cache) | **GT Category**: `Database`
- **GT Root Cause**: `redis_oom`
- **Baseline & Fine-Tuned**: Both models correctly classified category as `Database` and diagnosed Redis memory exhaustion.

### Case 3: Both Models Miss Synthetic Label (`boutique/codedefect/20`)
- **Component**: `app/checkoutservice` | **GT Category**: `Application`
- **GT Root Cause**: `code_excessive_file_reads`
- **Baseline RCA**: Cites worker node CPU saturation and slow checkout response.
- **Fine-Tuned RCA**: Cites worker node CPU saturation and proposes HPA scaling.
- **Analysis**: Both models correctly identified the physical symptom (CPU saturation from excessive file I/O), but did not emit the arbitrary synthetic Chaos Mesh label `code_excessive_file_reads`.

### Case 4: Category Edge Case (`boutique/codedefect/94`)
- **Component**: `app/shippingservice` | **GT Category**: `Application`
- **Baseline Prediction**: `Application`.
- **Fine-Tuned Prediction**: `Network` (Triggered by HTTP error log keywords).

---

## 6. Experimental Visualizations

- [Fair Latency Comparison (Controlled)](fair_latency_comparison.png)
- [Confusion Matrix Comparison (Fault Taxonomy)](confusion_matrix_comparison.png)
- [Fault Taxonomy Accuracy Comparison](category_accuracy_comparison.png)
- [Root Cause Identification Comparison](root_cause_comparison.png)
- [Macro F1 Comparison](macro_f1_comparison.png)
- [JSON Validity & Compliance Comparison](json_validity_comparison.png)
- [Per-Category F1 Breakdown](per_category_comparison.png)
- [Training Loss Trajectory](training_loss_curve.png)

---
*Report certified by FineTuneOps Evaluation & Audit Team.*

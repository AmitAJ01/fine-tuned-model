# FineTuneOps: Experiment Plan & Evaluation Methodology

**Document Version:** 2.0.0  
**Project:** FineTuneOps  
**Scope:** Baseline vs. Fine-Tuned Model Comparative Evaluation on Cloud-OpsBench  

---

## 1. Experimental Objective & Boundary Constraints

The core objective of FineTuneOps is to empirically validate whether domain-specialized **Supervised Fine-Tuning (SFT) with QLoRA** on **Cloud-OpsBench** produces statistically significant gains over an **unmodified pretrained instruction model** (`Qwen2.5-Coder-1.5B-Instruct`) for Kubernetes and cloud-native microservice Root Cause Analysis (RCA).

### Critical Experimental Rules
1. **Strictly Two Configurations:**
   - **Configuration A (Baseline Model):** Pretrained `Qwen/Qwen2.5-Coder-1.5B-Instruct` evaluated zero-shot with structured prompt.
   - **Configuration B (Fine-Tuned Model):** Same base model, fine-tuned using QLoRA and SFT on Cloud-OpsBench training data.
2. **Explicitly Prohibited Approaches:**
   - **No RAG (Retrieval-Augmented Generation):** No external vector store or search retriever.
   - **No DPO (Direct Preference Optimization):** No RLHF/DPO preference pairs.
   - **No Additional Third-Party Benchmarking:** Comparison is strictly confined to Configuration A vs Configuration B.
3. **Strict Parity Constraints:**
   - **Identical Test Dataset:** Both models must be evaluated on the exact same held-out test split (`ml/data/splits/test.jsonl`, 148 cases).
   - **Identical Prompt Formatting:** Both models receive identical system instructions and user message templates.
   - **Identical Generation Parameters:** Both models are invoked with identical decoding hyperparameters (temperature 0.1, max tokens 768, repetition penalty 1.05, greedy sampling).

---

## 2. Experimental Configurations

| Parameter / Dimension | Configuration A: Baseline | Configuration B: Fine-Tuned |
| :--- | :--- | :--- |
| **Base Architecture** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Domain Adaptation** | None (Pretrained Zero-Shot) | QLoRA SFT (4-bit NF4, $r=16, \alpha=32$) |
| **Trainable Parameters** | 0 (Frozen) | ~30M (< 1% of total parameters) |
| **Quantization Precision** | 4-bit NF4 / 16-bit float | 4-bit NF4 base + 16-bit LoRA adapter |
| **Input System Prompt** | Standard `SYSTEM_PROMPT` | Standard `SYSTEM_PROMPT` |
| **Temperature** | 0.1 | 0.1 |
| **Top-P Sampling** | 0.95 | 0.95 |
| **Max New Tokens** | 768 | 768 |
| **Sampling Mode** | Deterministic (`do_sample=False`) | Deterministic (`do_sample=False`) |

---

## 3. Dataset Splitting & Leakage Prevention

To ensure scientific validity and avoid data contamination:

```
Cloud-OpsBench Processed Dataset (754 Cases)
├── Training Split (68.2% / 514 cases)    -> Fed exclusively to SFTTrainer during training
├── Validation Split (12.2% / 92 cases)   -> Monitored during training for checkpoint selection & early stopping
└── Held-Out Test Split (19.6% / 148 cases) -> NEVER observed during training; used only for final evaluation
```

- **Group-Stratification:** Cases are grouped by `(system::category::fault_object::root_cause)`. Scenario groups are assigned atomically to a single split, ensuring **zero scenario overlap** between training, validation, and test partitions.
- **Isolation:** The test split is stored in `ml/data/splits/test.jsonl` and locked against modification during the training phase.

---

## 4. Evaluation Metrics & Success Criteria

### 4.1 Quantitative Benchmark Metrics

1. **Fault Taxonomy / Category Accuracy ($\text{Acc}_{\text{cat}}$):**
   $$\text{Acc}_{\text{cat}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{cat}}^{(i)} = y_{\text{cat}}^{(i)})$$
   Evaluates whether the model correctly assigns the incident to one of the 8 canonical Cloud-OpsBench fault categories (`Application_Code_Defect`, `Startup_Fault`, `Runtime_Fault`, `Scheduling_Fault`, `Service_Routing_Fault`, `Performance_Fault`, `Admission_Fault`, `Infrastructure_Fault`).

2. **Fault Taxonomy Macro F1:**
   Unweighted mean of F1 scores across all 8 fault categories, penalizing frequency bias toward dominant classes.

3. **Root-Cause Exact Match Accuracy ($\text{EM}_{\text{rc}}$):**
   $$\text{EM}_{\text{rc}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{normalize}(\hat{y}_{\text{rc}}^{(i)}) = \text{normalize}(y_{\text{rc}}^{(i)}))$$
   Strict equality between predicted root cause and ground truth (e.g. `code_excessive_file_reads`).

4. **Root-Cause Normalized Accuracy:**
   Evaluates whether the canonical ground-truth failure mechanism or token subset is accurately identified within the model's diagnostic analysis or achieves token F1 $\ge 0.60$.

5. **Semantic Root-Cause Scoring Rubric ($0.0 - 1.0$):**
   - **Level 4 ($1.00$):** Exact match.
   - **Level 3 ($0.80$):** Canonical token alignment without syntax discrepancies.
   - **Level 2 ($0.50$):** Component and core failure domain mechanism match.
   - **Level 1 ($0.25$):** Symptom-level diagnostic association.
   - **Level 0 ($0.00$):** Ungrounded or hallucinated failure.

6. **JSON Validity & Schema Compliance Rate:**
   $$\text{VR}_{\text{JSON}} = \frac{N_{\text{valid}}}{N_{\text{total}}}$$
   Evaluates whether model generations produce valid JSON and strictly contain required canonical fields (`incident_category`, `root_cause_analysis`).

7. **Evidence Grounding Score ($0.0 - 1.0$):**
   Evaluates whether cited services, pods, and error symptoms originate directly from the input telemetry:
   - Entity & Component Grounding ($0.40$)
   - Symptom & Error Grounding ($0.40$)
   - Structural Coherence ($0.20$)

8. **Inference Latency & Efficiency Profile:**
   Mean, Median, P95 latency (ms) and Peak GPU VRAM (MB).

### 4.2 Benchmark Metric Exclusion Protocol
- **Derived Severity Accuracy**: Excluded from primary metrics because Cloud-OpsBench severity is derived from fault impact rather than an official benchmark label.
- **Remediation Accuracy**: Excluded because the source dataset evaluates Root Cause Analysis (RCA) and does not provide golden remediation actions.

---

## 5. Statistical Significance & Confidence Intervals

Because both models are evaluated on the exact same 148 held-out test cases:
1. **Paired Bootstrap Resampling:**
   $N = 1,000$ bootstrap resamples (seed 42) are drawn with replacement to calculate the empirical 95% Confidence Interval for differences:
   $$\Delta = M_{\text{finetuned}} - M_{\text{baseline}}$$
2. **Significance Threshold:**
   An improvement is deemed statistically significant at $\alpha = 0.05$ if and only if the 95% bootstrap confidence interval strictly excludes zero:
   $$\text{CI}_{0.95}(\Delta) > 0$$
3. **Case-Level Outcome Tracking:**
   For every metric, report the exact tally of:
   - Cases improved ($M_{\text{ft}} > M_{\text{base}}$)
   - Cases unchanged ($M_{\text{ft}} == M_{\text{base}}$)
   - Cases worsened ($M_{\text{ft}} < M_{\text{base}}$)

---

## 5. Experiment Execution Workflow

1. **Generate Splits:**
   ```bash
   python scripts/prepare_dataset.py
   ```
2. **Execute QLoRA Training (Local GPU or Colab):**
   ```bash
   python ml/training/train_qlora.py --config ml/config/qlora_training.yaml
   ```
3. **Execute Benchmark Evaluation:**
   ```bash
   python scripts/run_evaluation.py --config ml/config/evaluation_config.yaml
   ```
4. **Inspect Generated Outputs:**
   - Machine-readable summary: `experiments/results/evaluation_summary.json`
   - Formatted markdown report: `experiments/reports/baseline_vs_finetuned_report.md`
   - MLflow trajectories: View via `mlflow ui` or local tracking directory.

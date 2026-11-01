# FineTuneOps — Task 13: Documentation Accuracy & Release Audit Report

**Date:** 2026-11-01  
**Auditor Role:** Senior ML Engineer & Technical Documentation Auditor  
**Repository:** [https://github.com/AmitAJ01/fine-tuned-model](https://github.com/AmitAJ01/fine-tuned-model)  
**Base Commit:** `cc94808`  
**Execution Status:** Verified & Clean  

---

## 1. Files Modified

The following files were inspected and updated in the working tree to ensure complete technical accuracy, empirical alignment with Task 7–10 results, and adherence to professional open-source release standards:

| File Path | Nature of Edits |
| :--- | :--- |
| `README.md` | Replaced placeholder clone URL with official GitHub repo; updated QLoRA training parameters (effective batch size: 16, optimizer steps: 33, duration: ~43m 26s, peak VRAM: 4,583 MB); removed unsupported enterprise/production claims; softened dashboard and serving labels; clarified JSON validity distinctions (standard 384 tokens vs 8-case diagnostic rerun); neutrally reported loss trajectory without claiming absence of overfitting; clarified merged BF16 execution parity vs unmerged LoRA latency overhead; updated reproduction guide. |
| `docs/ADAPTER_DISTRIBUTION.md` | Marked Hugging Face publication explicitly as **Pending Release**; replaced realistic-looking repository ID with clear placeholder notice; updated training reproduction duration to ~43m 26s on RTX 3050; added hardware floating-point non-deterministic variance notice; verified missing adapter error matches `ml/inference/engine.py`. |
| `.env.example` | Added detailed explanatory comment above `FINETUNED_ADAPTER_PATH` explaining that `adapter_model.safetensors` (~70.5 MB) is excluded from Git to prevent repository bloat, referencing `docs/ADAPTER_DISTRIBUTION.md`. |
| `models/finetuned_v2/adapter/README.md` | Replaced generic TRL template placeholder with a comprehensive model card detailing base model (`Qwen2.5-Coder-1.5B-Instruct`), dataset (`Cloud-OpsBench`), completion-only masking, training hyperparameters, verified evaluation metrics, and pending release notice. |
| `models/finetuned_v2/adapter/best/README.md` | Replaced generic PEFT stub (`[More Information Needed]`) with a complete, production-grade model card matching the best checkpoint specification and providing Python loading instructions. |
| `experiments/reports/task13_documentation_audit.md` | Created this deliverable report documenting all modifications, numerical sources, and audit checks. |

---

## 2. Claims Corrected

| Area | Previous Stated Claim | Audited & Corrected Truth | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Clone URL** | `https://github.com/username/FineTuneOps.git` | `https://github.com/AmitAJ01/fine-tuned-model.git` | Corrected to official published GitHub repository. |
| **System Scope & Labeling** | "Production Serving & Verification", "Next.js 14 Enterprise Dashboard" | "Inference Serving & Verification", "Next.js 14 Interactive Dashboard" | The system is a research prototype copilot for incident triage. "Production" and "Enterprise" overstate operational readiness. |
| **System Designation** | Implicit / scattered | Formally affirmed as: **"Human-Reviewed Incident Diagnostic Copilot / Research Prototype"** | Clarifies that the model assists human SREs and does not autonomously mutate production infrastructure. |
| **Training Duration** | "5 Minutes on GPU" / 311.59s | **2,606.0 seconds (~43 minutes 26 seconds)** | The 311s figure was from the obsolete unmasked run (effective batch size 8). The final completion-only masked training ran for 33 steps at effective batch size 16 with 3,072-token sequences, taking 43m 26s on an RTX 3050 Laptop GPU. |
| **Training Batch Size & Steps** | Effective $B=8$, steps=67 | Effective $B=16$ ($B=1 \times 16$ accum steps), **33 optimizer steps** | Aligned with `ml/config/qlora_training_final.yaml` and `experiments/final_run/training_metrics.json`. |
| **Peak Training VRAM** | 3,946 MB (3.85 GB) | **4,583 MB (~4.48 GB, 74.6% capacity)** | Verified against GPU telemetry recorded during final completion-only training. |
| **JSON Validity** | "100.0% (8/8)" presented without sufficient context | Primary test: **95.06% (154/162)** on 384 tokens; **100.0% (8/8)** on 8-case diagnostic rerun with 768 tokens | Clarified that the full 162-case benchmark had 8 mid-string truncations on long diagnostic responses. The 100% figure was strictly from an 8-case diagnostic re-run, not a 162-case re-evaluation. |
| **Overfitting Claims** | "Validation loss tracked below training loss, demonstrating healthy generalization without overfitting" | Neutral reporting: "Train loss decreased monotonically from 2.0710 to 0.0624; Val loss evaluated every 11 steps at 0.3301, 0.1120, 0.08941" | Lower validation loss does not prove lack of overfitting (validation sets may have different loss distributions or token lengths). Loss trajectory is now reported strictly as observed empirical data. |
| **Serving Latency & Parity** | Implication that fine-tuning accelerated or changed FLOPs | **Merged BF16 runs at physical hardware parity with base BF16 (within 1.34% latency)**; Unmerged LoRA incurs +241% latency overhead | Pre-merging weights ($W = W_0 + \Delta W$) yields identical compute shapes to the base model. Fine-tuning adjusts weights for domain knowledge, not computational speed. |
| **Hugging Face Release Status** | CLI download command shown as active | Explicitly marked **Pending Release** with placeholder notice | Prevents users from executing commands against non-existent Hugging Face repositories. |

---

## 3. Numerical Sources Used

All numerical claims in the corrected documentation are directly anchored to verified, immutable local experimental artifacts:

1. **Training Hyperparameters & Telemetry:**
   - Source File: `experiments/final_run/training_metrics.json`
   - Config File: `ml/config/qlora_training_final.yaml`
   - Key Values:
     - Base Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Git revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`)
     - Trainable Parameters: 18,464,768 (1.182% of 1,562,179,072)
     - Effective Batch Size: 16 (`per_device_train_batch_size: 1`, `gradient_accumulation_steps: 16`)
     - Optimizer Steps: 33 (evaluated at steps 11, 22, 33)
     - Sequence Length: 3,072 tokens
     - Training Duration: 2,606.0 seconds (43m 26s)
     - Loss: Train 2.0710 $\to$ 0.0624; Val 0.3301 $\to$ 0.1120 $\to$ 0.08941
     - Peak VRAM: 4,583.0 MB
     - Adapter File Size: 70.49 MB (73,912,488 bytes)

2. **Classification & Diagnostic Evaluation:**
   - Source Files: `experiments/final_run/baseline_predictions.jsonl`, `experiments/final_run/finetuned_predictions.jsonl`, `ml/data/splits_final/test.jsonl`
   - Key Values:
     - Test Sample Count: $N = 162$ incidents
     - Base Strict Accuracy: 45.06% (73/162)
     - Fine-Tuned Strict Accuracy: 62.96% (102/162)
     - Absolute Improvement: **+17.90 percentage points (pp)**
     - Relative Improvement: **+39.73%**
     - Discordant Pairs: 45 fine-tuned wins vs 16 baseline wins (net +29)
     - Exact Two-Sided McNemar Test: **$p = 0.000264$**
     - Asymptotic McNemar Test: $p = 0.000205$
     - 95% Paired Percentile Bootstrap CI ($B=10,000$): $[+8.64\text{ pp}, +26.54\text{ pp}]$
     - Kubernetes Recall: 4.35% (2/46) $\to$ 89.13% (41/46) (+84.78 pp)
     - Network Precision: 23.68% $\to$ 88.89% (+65.21 pp)
     - Application Precision: 48.19% $\to$ 74.51% (+26.32 pp)
     - Macro F1: 0.3578 $\to$ 0.5307 (+0.1729)

3. **Inference Latency & Throughput:**
   - Source File: `experiments/reports/task9_inference_benchmark.md`
   - Key Values (RTX 3050 Laptop GPU, CUDA synchronized, Batch Size = 2):
     - Base Pretrained BF16: 5,622.21 ms/sample, 41.62 tok/s, 3,399 MB VRAM
     - Fine-Tuned Merged BF16: 5,697.44 ms/sample, 43.70 tok/s, 4,707 MB VRAM (+1.34% latency delta)
     - Fine-Tuned Unmerged 4-bit LoRA: 19,178.03 ms/sample, 13.35 tok/s, 2,552 MB VRAM (+241.1% latency overhead)

---

## 4. Remaining Limitations & Operational Boundaries

The documentation explicitly preserves and highlights the four fundamental boundaries of the system:

1. **Research Prototype Boundary:**
   FineTuneOps is a diagnostic triage assistant designed for human engineers. It does not possess autonomous execution authority to modify cluster state, apply manifests, or delete resources.
2. **Root-Cause Analysis (RCA) Exact Match (0.0%):**
   Across the Cloud-OpsBench benchmark, 87.04% (141/162) of test incidents feature synthetic chaos injection labels never present in the training set (only 1 class overlaps). Furthermore, the evaluation prompt instructs the model to provide explanatory natural language descriptions rather than exact enum tokens. As a result, exact-match RCA accuracy is 0.0% for both baseline and fine-tuned models and is documented as an ungrounded/inconclusive metric.
3. **Absence of Ground-Truth Remediation:**
   The public Cloud-OpsBench benchmark evaluates diagnostic identification and does not provide golden remediation scripts. Remediation suggestions produced by the model are heuristic suggestions that require engineering validation.
4. **Token Headroom Sensitivity:**
   At a 384-token generation limit, 8 test incidents experienced mid-string truncations. While a diagnostic rerun at 768 tokens verified 100% syntax compliance, constrained deployments should configure appropriate token budgets or employ grammar-constrained decoding (e.g. Outlines, vLLM).

---

## 5. Automated Test Results

Verification of the complete regression test suite:
- **Command:** `pytest tests/ -q`
- **Total Tests:** 34
- **Passed:** 34
- **Failed:** 0
- **Duration:** 17.65s
- **Coverage Areas:** Environment config, schema validation (Pydantic v2 / RFC 8259), group-stratified split isolation, completion loss masking tensor verification, mock inference router, and real GPU inference compatibility.

---

## 6. Model & Experiment Artifact Integrity Confirmation

We formally certify that during Task 13:
- **Zero model weights** were retrained, modified, or deleted.
- **Zero dataset files** (`ml/data/splits_final/*.jsonl`) were modified or regenerated.
- **Zero prediction records** (`experiments/final_run/*_predictions.jsonl`) were altered.
- **Zero metric files** (`experiments/final_run/*_metrics.json`) were altered.
- **Zero code logic** in `backend/`, `ml/`, `frontend/`, or `tests/` was modified.
- All historical experimental records and figures remain 100% byte-for-byte identical to their Task 7–10 baseline.

---

## 7. Git & Release Status Confirmation

- **No automated Git commits** (`git commit`) were executed.
- **No Git pushes** (`git push`) were executed.
- **No remote repositories** were created or modified.
- **No Hugging Face uploads** or authentication routines were invoked.
- All modifications remain uncommitted in the local working directory, ready for manual review and approval by the user.

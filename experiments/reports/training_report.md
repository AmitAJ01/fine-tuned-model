# FineTuneOps — QLoRA Supervised Fine-Tuning Report

**Model Subject**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Base Model Revision**: `2e1fd397ee46e1388853d2af2c993145b0f1098a`  
**Training Date**: `2026-10-31T16:38:14.622319+00:00`  
**Hardware Used**: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (6.0 GB VRAM) | Compute Dtype: `torch.bfloat16`  
**Peak GPU Memory**: **`3946.18 MB`** (3.85 GB)  

---

## 1. Executive Summary

This report documents the Supervised Fine-Tuning (SFT) of **Qwen2.5-Coder-1.5B-Instruct** using **4-bit NormalFloat (NF4) QLoRA** on the official **Cloud-OpsBench** benchmark dataset.

The training objective is domain adaptation for Kubernetes microservice incident diagnosis: transforming raw Prometheus alerts and diagnostic telemetry into canonical structured JSON incident reports with exact root cause attribution.

> [!IMPORTANT]
> **Strict Benchmark Integrity Compliance**:
> - **Train Dataset**: `ml/data/splits/train.jsonl` (514 cases)
> - **Validation Dataset**: `ml/data/splits/val.jsonl` (92 cases)
> - **Untouched Held-Out Test Set**: `ml/data/splits/test.jsonl` (148 cases) — **STRICTLY ISOLATED (0% exposure during training)**.
> - The 148 test cases were never used for loss computation, early stopping, or checkpoint selection.

---

## 2. QLoRA Parameter & Architecture Profile

| Parameter Category | Configuration Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Quantization Format** | 4-bit NF4 (`bnb_4bit_use_double_quant=True`) | Compresses 1.54B base weights to ~1.1 GB VRAM |
| **Compute Dtype** | `torch.bfloat16` | Preserves dynamic range during forward/backward passes |
| **LoRA Rank ($r$)** | `16` | Expressive rank capacity for cross-entropy fine-tuning |
| **LoRA Alpha ($\alpha$)** | `32` | Scaling factor ($\alpha/r = 2.0$) for adapter weight updates |
| **LoRA Dropout** | `0.05` | Regularization against small-dataset memorization |
| **Target Projections** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full coverage of self-attention and MLP feed-forward blocks |
| **Trainable Parameters** | **`18,464,768`** | Lightweight parameter delta |
| **Total Base Parameters** | **`1,562,179,072`** | Complete foundation model |
| **Trainable Ratio** | **`1.1820%`** | Efficient parameter adaptation (< 1.2% modified) |

---

## 3. Training Dynamics & Loss Trajectory

- **Number of Epochs**: `3`
- **Effective Batch Size**: `16` (per-device batch size 1 $\times$ 16 gradient accumulation steps)
- **Learning Rate**: `2e-4` with Cosine decay schedule and `0.05` warmup ratio
- **Maximum Sequence Length**: `3072` tokens (preserves full diagnostic trace without truncation)
- **Initial Training Loss**: `N/A`
- **Final Training Loss**: **`1.5201`**
- **Final Validation Loss**: **`1.4673`**
- **Best Validation Loss**: **`1.4673`**
- **Total Training Duration**: **`311.59 s`** (5.19 minutes)

### Overfitting Assessment
- **Status**: `Healthy Generalization`
- **Diagnosis**: No obvious overfitting detected; validation loss decreased across epochs.

---

## 4. Model Artifacts & Checkpoint Verification

- **Final Adapter Directory**: [`models/finetuned/adapter`](../../models/finetuned/adapter)
- **Best Checkpoint Directory**: [`models/finetuned/adapter/best`](../../models/finetuned/adapter/best)
- **Loss Curve Visualization**: [`experiments/reports/training_loss_curve.png`](training_loss_curve.png)
- **Adapter Load Verification**: **`PASSED (Verified loadable into base model)`**

```
models/finetuned/adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── finetuneops_metadata.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

---
*Report generated automatically by FineTuneOps QLoRA Training Engine.*

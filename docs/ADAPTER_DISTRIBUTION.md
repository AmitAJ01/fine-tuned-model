# FineTuneOps — Adapter Weight Distribution & Setup Guide

This document describes how the fine-tuned QLoRA adapter weights for **FineTuneOps** are distributed, configured, and loaded.

---

## 1. Adapter Artifact Specification

- **Architecture:** PEFT LoRA adapter for `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Training Method:** 4-bit NormalFloat (NF4) Supervised Fine-Tuning with completion-only loss masking
- **LoRA Parameters:** Rank $r = 16$, Alpha $\alpha = 32$, Target modules: all 7 linear projections
- **File Name:** `adapter_model.safetensors`
- **File Size:** **70.49 MB** (73,912,488 bytes)
- **Local Target Directory:** `models/finetuned_v2/adapter/best/`
- **Accompanying Metadata Files (Tracked in Git):**
  - `adapter_config.json` (670 bytes) — PEFT configuration
  - `tokenizer.json` (10.89 MB) — Fast BPE tokenizer
  - `tokenizer_config.json` (9.1 KB) — Tokenizer parameters and special token definitions
  - `vocab.json` & `merges.txt` — Vocabulary dictionaries
  - `chat_template.jinja` — OpenAI ChatML Jinja template

---

## 2. Why Adapter Weights are Excluded from Git

Standard Git repositories are designed for source code and text artifacts. GitHub enforces:
- A **50 MB** soft recommendation limit per file (triggers warnings on push).
- A **100 MB** hard rejection limit per file (`GH001: Large files detected`).

Because `adapter_model.safetensors` is **70.49 MB**, committing it directly to Git history permanently inflates repository clone times and packfile sizes. Following open-source ML best practices, binary weights are excluded via `.gitignore` and distributed via dedicated model repositories.

---

## 3. Configuration & Environment Variables

The inference engine and FastAPI backend support configurable adapter locations:

| Setting | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Adapter Directory** | `FINETUNED_ADAPTER_PATH` | `models/finetuned_v2/adapter/best` | Path to directory containing `adapter_model.safetensors` |
| **Base Model** | `BASE_MODEL_NAME` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Upstream Hugging Face foundation model repository |
| **Device Target** | `DEVICE` | `cuda` (auto-detected) | Execution device (`cuda` or `cpu`) |
| **Force CPU** | `FORCE_CPU` | `0` | Set `FORCE_CPU=1` to disable GPU offloading |

To specify a custom adapter location:
```bash
# In your .env file or shell:
export FINETUNED_ADAPTER_PATH="/path/to/custom/adapter_checkpoint"
```

---

## 4. Setup Instructions for New Users

### Option A: Download Published Adapter (Pending Release)
Once published to Hugging Face Model Hub, download the adapter weights directly:
```bash
# Install Hugging Face Hub CLI
pip install huggingface_hub

# Download adapter weights into the expected directory:
huggingface-cli download <HF_ORG_OR_USER>/finetuneops-qwen2.5-1.5b-adapter \
  adapter_model.safetensors \
  --local-dir models/finetuned_v2/adapter/best/
```

### Option B: Local Training Reproduction (5 Minutes on GPU)
If you have an NVIDIA GPU ($\ge 6\text{GB}$ VRAM), you can re-train the adapter locally from the verified dataset:
```bash
# Run completion-only masked QLoRA fine-tuning:
python ml/training/train_qlora_final.py

# The trained adapter will automatically be saved to models/finetuned_v2/adapter/best/
```

### Option C: Run Without Local Adapter (Baseline or Simulation Mode)
You do not need the fine-tuned adapter to test the application:
1. **Baseline Pretrained Model:** Pass `model_target: "baseline"` to evaluate `Qwen/Qwen2.5-Coder-1.5B-Instruct` directly from Hugging Face.
2. **Mock Inference:** Pass `mock: true` to test the FastAPI endpoints and Next.js frontend UI without loading GPU tensors.

---

## 5. Missing Adapter Error Handling

If real inference (`mock=False`) is requested for the fine-tuned model and `adapter_model.safetensors` is not present on disk, the inference service will **never fabricate dummy outputs or silently fall back**. 

Instead, it returns a structured diagnostic error conforming to RFC 8259:
```json
{
  "report": null,
  "model_used": "Qwen/Qwen2.5-Coder-1.5B-Instruct + LoRA (best)",
  "is_fine_tuned": true,
  "latency_ms": 1.25,
  "json_valid": false,
  "error_detail": "Fine-tuned adapter weights (adapter_model.safetensors) not found at 'models/finetuned_v2/adapter/best'. For full inference, please download the adapter checkpoint into this directory, or set the FINETUNED_ADAPTER_PATH environment variable to a custom path. To run lightweight demo inference without local weights, set mock=True or use baseline mode."
}
```

---

## 6. Maintainer Step: Publishing the Adapter to Hugging Face Hub

> **Notice:** This step is to be executed by the repository maintainer upon public release. Do not execute this during local evaluation.

1. Install Hugging Face CLI and log in:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
2. Create model repository on Hugging Face:
   ```bash
   huggingface-cli repo create finetuneops-qwen2.5-1.5b-qlora --type model
   ```
3. Upload adapter directory:
   ```bash
   huggingface-cli upload <your-username>/finetuneops-qwen2.5-1.5b-qlora \
     models/finetuned_v2/adapter/best/ . \
     --type model
   ```
4. Update the repository placeholder in `README.md` and `.env.example` with the published model identifier.

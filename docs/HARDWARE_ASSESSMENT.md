# FineTuneOps: Hardware & Environment Assessment

**Date:** October 31, 2026  
**Auditor:** Senior ML Engineer & Full-Stack Architect  
**Project:** FineTuneOps (DevOps Incident Diagnosis LLM Platform)

---

## 1. Executive Summary

An empirical diagnostic assessment of the local host environment was performed using `nvidia-smi`, Python runtime introspection, and dedicated probing scripts.

The host machine is equipped with a dedicated **NVIDIA GeForce RTX 3050 6GB Laptop GPU** with **6,144 MB (6.00 GB) VRAM** and 15.31 GB of system RAM. However, the active Python environment currently runs **CPU-only PyTorch (`2.11.0+cpu`)**, and native Windows presents well-documented driver bindings and compilation constraints for `bitsandbytes` 4-bit CUDA kernels.

To balance realistic local developer workflows with high-performance model training, we establish a **Dual-Mode Execution Architecture**:
1. **Local Platform Mode:** Local API serving (FastAPI), Next.js dashboard, dataset preprocessing, prompt engineering, validation tests, and CPU/CUDA inference.
2. **Cloud/Colab Training Mode:** GPU-accelerated QLoRA training via a preconfigured Google Colab notebook (`ml/notebooks/finetune_ops_colab.ipynb`) or cloud Linux instance (T4/A100), exporting lightweight LoRA adapter weights (`~30 MB`) back to the local repository for inference and evaluation.

---

## 2. Empirical Environment Audit Data

| Component | Detected Specification | Operational Status & Constraints |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (Build 10.0.26300, AMD64) | Fully functional; requires POSIX-safe path and tool handling |
| **Python Runtime** | Python 3.10.11 (64-bit) | Compatible with PyTorch 2.x and Hugging Face ecosystem |
| **Physical System RAM** | 15.31 GB Total (~3.25 GB Available) | Sufficient for small model CPU inference; tight for multi-model loading |
| **Storage Volumes** | `C:\` 21.34 GB free<br>`E:\` 314.28 GB free<br>`F:\` 177.75 GB free | `C:\` is adequate for code and adapters; model cache (`HF_HOME`) can use `E:\` |
| **NVIDIA GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | Detected via `nvidia-smi` (Driver 610.60, CUDA 13.3 driver support) |
| **VRAM Capacity** | 6,144 MB (6.00 GB total, 5.86 GB free) | Hard VRAM ceiling: **6.00 GB** |
| **Active PyTorch** | `2.11.0+cpu` (CUDA Available: `False`) | Requires CUDA wheel (`pip install torch --index-url ... cu121`) for local GPU |
| **Container & Dev Tools** | Git 2.53.0, Docker 29.6.1, Compose v5.2.0 | Git & Docker available; Node.js/npm not in PATH |

---

## 3. Mathematical VRAM & Model Selection Analysis

Under Supervised Fine-Tuning (SFT) with QLoRA, peak GPU VRAM is governed by:

$$\text{VRAM}_{\text{peak}} = \text{VRAM}_{\text{model\_weights}} + \text{VRAM}_{\text{adapters\_and\_optimizer}} + \text{VRAM}_{\text{activations}} + \text{VRAM}_{\text{KV\_cache\_and\_context\_overhead}}$$

### Model Candidate Comparison Matrix

| Model Family & Size | 4-bit Base Weights | LoRA & 8-bit Paged AdamW | Activation Memory (Seq 1024, Grad Checkpoint) | Total SFT Training VRAM | Fits in Local 6 GB VRAM? | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-Coder-0.5B-Instruct** | ~0.4 GB | ~0.1 GB | ~0.6 GB | **~1.5 GB** | **Yes** (Generous headroom) | Viable, but lower incident reasoning depth |
| **Qwen/Qwen2.5-Coder-1.5B-Instruct** | **~1.1 GB** | **~0.25 GB** | **~1.2 GB** | **~3.3 – 3.8 GB** | **YES (Recommended)** | **Optimal balance of DevOps reasoning & VRAM efficiency** |
| **Qwen/Qwen2.5-Coder-3B-Instruct** | ~2.2 GB | ~0.4 GB | ~1.8 GB | **~4.9 – 5.5 GB** | **Marginal** (High risk of OOM on Windows WDDM) | Viable with batch size 1 and extreme sequence truncation |
| **Qwen/Qwen2.5-Coder-7B-Instruct** | ~4.6 GB | ~0.8 GB | ~2.5 GB | **~8.2 – 9.5 GB** | **NO (Exceeds 6 GB)** | **Recommended for Google Colab (T4 15 GB / A100)** |

### Why `Qwen/Qwen2.5-Coder-1.5B-Instruct` is Selected

1. **Hardware Fit:** In 4-bit NormalFloat (NF4), the base weights occupy only ~1.1 GB. With gradient accumulation ($N=8$), per-device batch size of 1, and activation checkpointing enabled, peak memory stays under 3.8 GB VRAM. This leaves ~2.2 GB safety margin under Windows WDDM desktop display overhead.
2. **Domain Suitability:** DevOps incident diagnosis requires parsing complex stack traces, kernel OOM notices, Kubernetes events, and producing syntactically valid JSON. The Qwen2.5-Coder family exhibits industry-leading code comprehension, regex/log parsing, and strict adherence to structured JSON output formatting.
3. **Local Inference Feasibility:** At 1.5B parameters, the model can execute zero-shot or LoRA inference locally on CPU (using ~3 GB RAM) or GPU (~1.5 GB VRAM), guaranteeing that the API backend and Next.js frontend remain fully interactive on any developer machine.

---

## 4. Execution Strategies: Local vs. Cloud

### Path A: Local GPU / CPU Workflow
- **Prerequisites for Local GPU Training:**
  - Install CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  - Install Windows bitsandbytes: `pip install bitsandbytes-windows`
- **Fallback Simulation / CPU Testing:**
  - The inference engine and evaluation harness feature automatic dry-run and mock fallbacks so full API and UI integration can be validated without downloading multi-gigabyte checkpoints.

### Path B: Cloud / Google Colab Workflow (Zero Local Friction)
- A dedicated notebook `ml/notebooks/finetune_ops_colab.ipynb` is provided.
- Train on free Google Colab (NVIDIA T4 with 15 GB VRAM) or Colab Pro (A100).
- After 3 epochs of SFT, download the tiny LoRA adapter directory (`~30 MB`) and place it in `FineTuneOps/models/finetuned/qwen2.5-coder-1.5b-incident-diagnosis`.
- Local FineTuneOps backend immediately loads the fine-tuned adapter on top of the base model for inference and evaluation.

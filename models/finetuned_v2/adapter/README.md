---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- sft
- devops
- incident-diagnosis
- cloud-opsbench
- kubernetes
license: mit
---

# Model Card: FineTuneOps Incident Diagnostic Adapter

This model repository contains the PEFT QLoRA adapter checkpoints fine-tuned on top of [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) for cloud microservices and Kubernetes incident diagnosis.

> **Distribution Notice:** The binary adapter weights (`adapter_model.safetensors`, 70.49 MB) are excluded from the Git repository to prevent repository bloat. Formal publication to the Hugging Face Model Hub is **pending release**. To reproduce or run inference with local weights, see [docs/ADAPTER_DISTRIBUTION.md](../../docs/ADAPTER_DISTRIBUTION.md) or train locally via `python ml/training/train_qlora_final.py`.

## Model Details

- **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Git revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`)
- **Adapter Type:** Low-Rank Adaptation (LoRA) via PEFT
- **Quantization:** 4-bit NormalFloat (NF4) with Double Quantization (`bitsandbytes`)
- **Compute Precision:** `torch.bfloat16`
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - Trainable Parameters: 18,464,768 (1.182% of 1,562,179,072 total parameters)
- **Dataset:** [Cloud-OpsBench v1.0.0](https://github.com/LLM4Ops/Cloud-OpsBench) (MIT License, 754 validated incident cases across Google Online Boutique and TrainTicket)
- **Training Strategy:** Completion-only loss masking (system prompt and user telemetry tokens assigned `label = -100`)
- **Effective Batch Size:** 16 ($B=1$ micro-batch $\times$ 16 gradient accumulation steps)
- **Training Steps:** 33 optimizer steps (1 epoch, evaluated every 11 steps)
- **Training Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM)
- **Training Duration:** 2,606.0 seconds (~43 minutes 26 seconds)
- **Loss Progression:** Train Loss: 2.0710 $\to$ 0.0624; Validation Loss: 0.3301 $\to$ 0.1120 $\to$ 0.08941
- **System Designation:** Human-Reviewed Incident Diagnostic Copilot / Research Prototype

## Intended Use

FineTuneOps assists Site Reliability Engineers (SREs) and DevOps engineers in diagnosing cloud infrastructure incidents by:
1. Classifying the failure domain into standard taxonomy categories (`Application`, `Kubernetes`, `Network`, `Database`, etc.).
2. Generating structured incident triage reports conforming to RFC 8259 JSON schema (`IncidentReportPayload`).
3. Recommending diagnostic commands (`kubectl get pods`, `kubectl describe`, error log filters) for human verification.

**Out-of-Scope:** This model is not an autonomous infrastructure remediation agent and must not execute mutative actions (`kubectl delete`, `kubectl apply`) without human engineer review.

## Evaluation Results (Controlled 162-Case Held-Out Test Set)

Evaluated head-to-head against the unmodified base model on the exact same 162 held-out test cases (`ml/data/splits_final/test.jsonl`) under greedy decoding (`temperature = 0.0`):

| Metric | Pretrained Base | Fine-Tuned (Completion QLoRA) | Delta | Statistical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Fault Category Accuracy (Strict)** | 45.06% (73/162) | **62.96%** (102/162) | **+17.90 pp** (+39.7%) | Exact McNemar $p = 0.000264$ (95% CI: $[+8.64\text{ pp}, +26.54\text{ pp}]$) |
| **Fault Category Macro F1** | 0.3578 | **0.5307** | **+0.1729** (+48.3%) | Broad gains across minority incident classes |
| **Kubernetes Recall** | 4.35% (2/46) | **89.13%** (41/46) | **+84.78 pp** | Resolves baseline failure mode collapsing K8s faults into Application |
| **Network Precision** | 23.68% | **88.89%** | **+65.21 pp** | F1 improved from 0.3214 to 0.5926 |
| **Application Precision** | 48.19% | **74.51%** | **+26.32 pp** | F1 improved from 0.6299 to 0.8000 |
| **JSON Validity (Standard 384 tokens)** | 99.38% (161/162) | 95.06% (154/162) | -4.32 pp | 8 mid-string truncations on long diagnostic responses |
| **JSON Validity (Diagnostic 768 tokens)** | N/A | **100.0%** (8/8) | N/A | Diagnostic rerun confirms zero syntax errors with token headroom |
| **Root-Cause Exact Match** | 0.00% (0/162) | 0.00% (0/162) | 0.00 pp | Inconclusive metric: prompt requests prose; 87% unseen chaos tags |

## How to Load and Use

When adapter weights are present in `models/finetuned_v2/adapter/best/`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "models/finetuned_v2/adapter/best"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2. Load Base Model in 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 3. Attach Fine-Tuned LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Citation & Attribution

- **Base Model:** Alibaba Cloud `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache 2.0)
- **Dataset:** LLM4Ops `Cloud-OpsBench` (MIT License)
- **Fine-Tuning Platform:** FineTuneOps (MIT License)

# FineTuneOps — DevOps Incident Diagnosis Platform

> **Supervised Fine-Tuning of Open-Source Code LLMs (`Qwen2.5-Coder-1.5B-Instruct`) using QLoRA with Completion-Only Loss Masking on Cloud-OpsBench.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)

FineTuneOps is an end-to-end Machine Learning Engineering platform designed to fine-tune and evaluate open-source Large Language Models on cloud infrastructure and Kubernetes incident diagnosis. Using the public benchmark **Cloud-OpsBench**, the platform conducts an empirically audited head-to-head evaluation between the unmodified foundation model (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) and its completion-masked QLoRA fine-tuned counterpart.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph DataEngine ["Data Engineering & Provenance"]
        A["Cloud-OpsBench v1.0.0\n(754 Raw Incidents)"] --> B["Group-Stratified Split\n(Key: system::category::root_cause)"]
        B --> C["Train (536 records, 71.1%)"]
        B --> D["Val (56 records, 7.4%)"]
        B --> E["Held-Out Test (162 records, 21.5%)\nSHA-256 Verified"]
    end

    subgraph TrainingPipeline ["QLoRA SFT Pipeline"]
        C --> F["Completion-Only Loss Masking\n(Prompt Tokens label = -100)"]
        F --> G["Qwen2.5-Coder-1.5B-Instruct\n(4-bit NF4 Quantization)"]
        G --> H["LoRA Adapter Training\n(r=16, alpha=32, 1.18% params)"]
        H --> I["Best Adapter Checkpoint\n(models/finetuned_v2/adapter/best)"]
    end

    subgraph InferenceServing ["Production Serving & Verification"]
        I --> J["FastAPI Backend\n(/api/v1/incident/diagnose)"]
        J --> K["Structured RFC 8259 JSON\n(IncidentReportPayload)"]
        K --> L["Next.js 14 Interactive Dashboard\n(CUDA Synchronized Telemetry)"]
    end

    subgraph EvaluationAudit ["Controlled Statistical Evaluation"]
        E --> M["Greedy Deterministic Decoding\n(T=0.0, 768-token budget)"]
        M --> N["Contingency Tables & McNemar Tests\n(Strict: +17.90 pp, p = 0.00026)"]
        M --> O["Fair Latency Benchmarking\n(Merged BF16 within 1.34% of Base)"]
    end

    I -.-> M
```

---

## 2. Key Features

- **Strict Controlled Experimentation:** Direct head-to-head comparison between base foundation weights and fine-tuned adapter weights under identical prompt hashes, 162 held-out test cases, and greedy decoding.
- **Completion-Only Loss Masking:** Tensors format prompt tokens with `label = -100`, eliminating gradient dilution and forcing backpropagation exclusively through the model's diagnostic response.
- **Hardware-Aware 4-Bit QLoRA:** Fits full parameter adaptation (18.46M trainable parameters across all 7 linear projections) within a consumer **6 GB GPU VRAM footprint** (peak VRAM: 3.85 GB during training).
- **FastAPI Diagnostic Microservice:** Implements robust endpoints (`/health`, `/hardware`, `/evaluation/comparison`, `/incident/diagnose`) with structured JSON parsing diagnostics and zero dummy fallback fabrication.
- **Next.js 14 Enterprise Dashboard:** Real-time triage console featuring incident scenario presets, side-by-side model comparison, and interactive benchmark visualizations.
- **Mathematical Significance Verification:** Exact two-sided McNemar binomial tests ($p < 0.0004$) and paired percentile bootstrap confidence intervals ($B = 10,000$ iterations) verifying reproducible domain gains.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Foundation Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.54B parameters, bfloat16) |
| **Fine-Tuning & Quantization**| Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit NF4), PyTorch 2.6.0+cu124 |
| **Dataset & Benchmarking** | Cloud-OpsBench v1.0.0 (MIT License), `datasets`, `huggingface_hub` |
| **Backend & Serving** | FastAPI, Pydantic v2, Uvicorn, Python 3.10 |
| **Frontend & Visualization** | Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide React |
| **Evaluation & Statistics** | `scipy.stats` (binomtest), `scikit-learn`, `matplotlib`, `seaborn` |
| **Testing & Quality** | `pytest`, `anyio` (34/34 automated regression tests passing) |

---

## 4. Dataset & Methodology

### 4.1 Benchmark Provenance & Licensing
FineTuneOps uses the official **Cloud-OpsBench** benchmark (`LLM4Ops/Cloud-OpsBench`), released under the **MIT License**. The dataset comprises 754 validated incident cases across two microservice architectures:
- **Google Online Boutique:** 550 incidents across 11 Go/gRPC microservices.
- **TrainTicket:** 204 incidents across 40+ Java/Spring Boot microservices.

### 4.2 Group-Stratified Zero-Leakage Split
To evaluate true out-of-distribution diagnostic generalization, records were partitioned using a group key: `f"{system}::{category}::{root_cause}"`:
- **Train Split:** **536 records** (71.09%, 55 scenario groups) in `ml/data/splits_final/train.jsonl`
- **Validation Split:** **56 records** (7.43%, 6 scenario groups) in `ml/data/splits_final/val.jsonl`
- **Held-Out Test Split:** **162 records** (21.49%, 13 scenario groups) in `ml/data/splits_final/test.jsonl`
- **Contamination Audit:** Verified **0% group overlap** ($\text{Train} \cap \text{Test} = \emptyset$) with cryptographic SHA-256 verification in `ml/data/splits_final/splits_manifest.json`.

### 4.3 Completion-Only Loss Masking
In early SFT iterations, concatenation of user and assistant tokens without masking caused loss dilution over user telemetry. FineTuneOps implements strict completion-only masking in `ml/training/train_qlora.py`:
$$\mathcal{L} = -\frac{1}{N_{\text{response}}} \sum_{i \in \text{response}} \log P(x_i \mid x_{<i})$$
Tokens in the system prompt and user telemetry have `labels[i] = -100`, ensuring cross-entropy loss is computed exclusively on the diagnostic output.

---

## 5. QLoRA Training Configuration

| Parameter | Value | Architectural Justification |
| :--- | :--- | :--- |
| **Base Architecture** | `Qwen2.5-Coder-1.5B-Instruct` | 28 transformer layers, 16 attention heads, 1.54B parameters |
| **Quantization** | 4-bit NF4 (`bitsandbytes`) | Quantizes base linear layers to 4-bit; double quantization enabled |
| **Compute Precision** | `torch.bfloat16` | Preserves dynamic range and numerical stability during backprop |
| **LoRA Rank ($r$)** | `16` | Low-rank projection dimension |
| **LoRA Alpha ($\alpha$)** | `32` | Scaling factor ($\alpha / r = 2.0$) |
| **Target Modules** | All 7 Linear Projections | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable Parameters** | **18,464,768** (1.18%) | Total model parameters: 1,562,179,072 |
| **Sequence Length** | 3,072 tokens | Captures multi-service pod logs and Prometheus metrics |
| **Batch Size & Accumulation**| $B=1$, Steps=8 (Effective $B=8$) | Fits within 6 GB VRAM with gradient checkpointing enabled |
| **Optimizer & Learning Rate**| `paged_adamw_8bit`, $2 \times 10^{-4}$ | Cosine schedule with 5% warmup |
| **Peak Training VRAM** | **3,946 MB** (3.85 GB) | Safe operation within 6.0 GB laptop GPU budget |

---

## 6. Verified Experimental Results (Task 10 Reconciled)

Both configurations were evaluated head-to-head across the **exact same 162 held-out test incidents** (`ml/data/splits_final/test.jsonl`) using greedy decoding (`temperature = 0.0`, `do_sample = False`) on an NVIDIA GeForce RTX 3050 Laptop GPU in `bfloat16`.

### 6.1 Diagnostic Accuracy & Classification Metrics

| Metric Dimension | Baseline Pretrained | Fine-Tuned (Completion QLoRA) | Absolute Gain ($\Delta$) | Relative Gain | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fault Taxonomy (Strict Direct)** | **45.06%** (73/162) | **62.96%** (102/162) | **+17.90 pp** | **+39.73%** | **Exact $p = 0.000264$**, Asymptotic $p = 0.000205$; 95% CI $[+8.64\text{ pp}, +26.54\text{ pp}]$ |
| **Fault Taxonomy (Mapped Fallback)** | **45.68%** (74/162) | **62.96%** (102/162) | **+17.28 pp** | **+37.84%** | **Exact $p = 0.000394$**, Asymptotic $p = 0.000301$; 95% CI $[+8.02\text{ pp}, +25.93\text{ pp}]$ |
| **Fault Taxonomy Macro F1** | **0.3578** | **0.5307** | **+0.1729** | **+48.32%** | Broad gains across minority incident classes |
| **Kubernetes Recall** | **4.35%** (2/46) | **89.13%** (41/46) | **+84.78 pp** | **+1,949%** | Cured baseline's collapse of K8s faults into Application |
| **Network Precision** | **23.68%** | **88.89%** | **+65.21 pp** | **+275%** | F1 increased from 0.3214 to 0.5926 |
| **Application Precision** | **48.19%** | **74.51%** | **+26.32 pp** | **+54.6%** | F1 increased from 0.6299 to 0.8000 |
| **JSON Validity Rate (384 tokens)** | **99.38%** (161/162) | **95.06%** (154/162) | **-4.32 pp** | -4.35% | 8 mid-string truncations due to rich technical responses |
| **JSON Validity Rate (768 tokens)** | **99.38%** | **100.0%** (8/8) | **+0.62 pp** | +0.62% | Diagnostic rerun confirms zero syntax errors with token headroom |
| **Root-Cause Exact Match** | **0.00%** (0/162) | **0.00%** (0/162) | **0.00 pp** | — | Inconclusive metric: prompt asks for prose; 87% unseen |

### 6.2 Paired 2×2 Contingency Table (Strict Matching)

$$\begin{array}{c|cc|c}
& Y_{\text{FT}} = 1 & Y_{\text{FT}} = 0 & \text{Total} \\
\hline
Y_{\text{Base}} = 1 & 57 & 16 & 73 \\
Y_{\text{Base}} = 0 & 45 & 44 & 89 \\
\hline
\text{Total} & 102 & 60 & 162
\end{array}$$

- **Total Discordant Pairs:** $45 + 16 = \mathbf{61}$
- **Net Improved Cases:** $45 - 16 = \mathbf{+29}$ cases ($+29 / 162 = \mathbf{+17.90\text{ pp}}$)
- **Exact Two-Sided Binomial McNemar Test:** $p = \mathbf{0.000264} \approx \mathbf{0.0003}$

### 6.3 Fair Serving Latency & Throughput Benchmark

Tested across 10 evaluation samples on NVIDIA GeForce RTX 3050 Laptop GPU with CUDA synchronization enabled:

| Serving Configuration | Batch Size | Precision | Mean Latency (ms/sample) | Throughput (tok/s) | Peak VRAM | Architectural Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base Foundation Model** | 1 | BF16 | 11,974.54 | 19.90 | 3,716 MB | Base Baseline |
| **Fine-Tuned Merged Weights** | 1 | BF16 | 13,016.10 | 19.13 | 5,025 MB | +8.7% (longer output length) |
| **Fine-Tuned Unmerged LoRA** | 1 | 4-bit NF4 | 26,496.37 | 9.44 | 2,596 MB | +121.3% (dynamic PEFT forward branches) |
| **Base Foundation Model** | 2 | BF16 | **5,622.21** | **41.62** | 3,399 MB | Base Baseline |
| **Fine-Tuned Merged Weights** | 2 | BF16 | **5,697.44** | **43.70** | 4,707 MB | **+1.34% (Zero Architectural Overhead)** |
| **Fine-Tuned Unmerged LoRA** | 2 | 4-bit NF4 | 19,178.03 | 13.35 | 2,552 MB | +241.1% (PEFT dynamic branch overhead) |

*Key Finding:* Pre-merging LoRA adapter weights (`model.merge_and_unload()`) completely eliminates PEFT runtime penalties, serving within **1.34% latency** of the base model while achieving **43.70 tokens/second**.

---

## 7. Visual Benchmarks & Telemetry

### 7.1 Confusion Matrix Comparison (Baseline vs. Fine-Tuned)
The baseline model collapsed 76.1% of Kubernetes cluster failures into the generic *Application* category. Supervised fine-tuning cured this failure mode, elevating Kubernetes recall from **4.35% to 89.13%**:

![Confusion Matrix Comparison](experiments/reports/confusion_matrix_comparison.png)

### 7.2 Fair Controlled Latency Benchmark
Controlled benchmarking with explicit CUDA synchronization demonstrates that merged BF16 weights eliminate unmerged LoRA adapter overhead:

![Fair Latency Comparison](experiments/reports/fair_latency_comparison.png)

### 7.3 Training Loss Trajectory
Validation loss consistently tracked below training cross-entropy loss, demonstrating healthy generalization without overfitting across 3,072-token sequences:

![Training Loss Curve](experiments/reports/training_loss_curve.png)

---

## 8. Evaluation Limitations & Operational Boundaries

1. **System Designation:** FineTuneOps is formally designated as a **Human-Reviewed Incident Diagnostic Copilot / Research Prototype**. It assists SREs by hypothesizing failure domains and recommending diagnostic commands; it is **not** an autonomous infrastructure mutator.
2. **Root-Cause Analysis (RCA) Exact Match (0.0%):** Cloud-OpsBench contains 44 root-cause classes in training and 12 in the test split, with only 1 overlapping class (`taint_toleration_mismatch`). 87.04% (141/162) of test cases feature unseen chaos injection tags, and the prompt schema requests natural language diagnostic text. Consequently, exact-match RCA is **inconclusive / ungrounded** under the current benchmark contract.
3. **Absence of Golden Remediation Actions:** The Cloud-OpsBench dataset does not provide ground-truth remediation scripts. Remediation suggestions generated by the model are heuristic guidance and must be validated by an engineer.
4. **Token Generation Headroom:** Constraining output to 384 tokens caused 8 mid-string JSON truncations. Setting the default budget to **768 tokens** resolved all truncations (100.0% validity).

---

## 9. Installation & Reproduction Guide

### Prerequisites
- **OS:** Windows 10/11 (64-bit) or Ubuntu 22.04+
- **GPU:** NVIDIA GPU with $\ge$ 6.0 GB VRAM (RTX 3050 Laptop GPU verified)
- **CUDA:** CUDA 12.1 or 12.4
- **Python:** Python 3.10.x
- **Node.js:** Node.js 20.x LTS with npm (required for frontend UI)

### Step 1: Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/username/FineTuneOps.git
cd FineTuneOps

# Create and activate Python virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### Step 2: Environment Configuration & Adapter Setup
Copy the environment template:
```bash
cp .env.example .env
```
*Note on Model Weights:* Foundation weights (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) are pulled on-demand from Hugging Face. The fine-tuned QLoRA adapter is configured via `FINETUNED_ADAPTER_PATH`. For details on downloading or re-training adapter weights, see [docs/ADAPTER_DISTRIBUTION.md](docs/ADAPTER_DISTRIBUTION.md).

### Step 3: Run Full Automated Regression Test Suite
Verify environment, schemas, split isolation, completion masking, and API routers:
```bash
pytest tests/ -v
# Output: 34 passed in ~18s
```

### Step 4: Run Real GPU Inference Smoke Test
Executes real forward passes for both base and fine-tuned models on held-out test case `boutique/codedefect/34`:
```bash
python scripts/smoke_test_real_inference.py
# Outputs telemetry to experiments/results/smoke_test_telemetry.json
```

### Step 5: Start FastAPI Backend Server
```bash
# Terminal 1:
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### Step 6: Verify API Integration
```bash
# Terminal 2:
python scripts/verify_api_integration.py
# Outputs telemetry to experiments/results/api_integration_results.json
```

### Step 7: Launch Next.js Frontend Dashboard
```bash
# Terminal 3 (Requires Node.js 20 LTS):
cd frontend
npm install
npm run dev
```
Open your browser to **`http://localhost:3000`** to access the diagnostic triage console and benchmark dashboard.

---

## 10. Future Improvements

1. **Constrained Grammar Decoding:** Integrate Outlines or vLLM regex/CFG constrained decoding to guarantee 100% syntactic JSON validity even under constrained token limits.
2. **Retrieval-Augmented Generation (RAG):** Index Kubernetes architectural manifests, Helm charts, and internal team runbooks to ground root-cause hypotheses on live cluster topologies.
3. **Direct Preference Optimization (DPO):** Align model remediation suggestions with SRE triage preferences using pairwise preference optimization.

---

## License & Citation

- **Code:** Distributed under the [MIT License](LICENSE).
- **Benchmark Dataset:** [Cloud-OpsBench](https://github.com/LLM4Ops/Cloud-OpsBench) is released under the MIT License by LLM4Ops.
- **Base Model:** [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) is released under the Apache 2.0 License by Alibaba Cloud.

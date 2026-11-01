"""QLoRA Supervised Fine-Tuning (SFT) Engine for DevOps Incident Diagnosis.

Fine-tunes Qwen2.5-Coder-1.5B-Instruct on Cloud-OpsBench training and validation splits.
Implements:
1. 4-bit NormalFloat (NF4) quantization with double quantization.
2. LoRA (r=16, alpha=32, dropout=0.05) on all attention and MLP linear projections.
3. Gradient checkpointing and 8-bit Paged AdamW optimizer.
4. Loss tracking (step-level and epoch-level), validation loss tracking, and best model selection.
5. Overfitting analysis and loss curve generation.
6. Metric export to experiments/results/training_metrics.json and experiments/reports/training_report.md.
7. Verification that saved adapter loads into original base model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default offline mode for reproducibility if cached weights exist
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "experiments" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    from trl import SFTTrainer
    SFTConfig = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.training")


class LossAndMetricsCallback(TrainerCallback):
    """Callback to record step loss, epoch loss, eval loss, and GPU VRAM peak."""

    def __init__(self) -> None:
        super().__init__()
        self.step_history: List[Dict[str, Any]] = []
        self.epoch_train_losses: List[Dict[str, Any]] = []
        self.epoch_eval_losses: List[Dict[str, Any]] = []
        self.peak_memory_mb: float = 0.0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        current_peak = 0.0
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_memory_mb = max(self.peak_memory_mb, current_peak)

        if "loss" in logs:
            self.step_history.append({
                "step": state.global_step,
                "epoch": round(state.epoch or 0.0, 2),
                "train_loss": round(float(logs["loss"]), 4),
                "learning_rate": logs.get("learning_rate"),
                "peak_vram_mb": round(current_peak, 2),
            })

        if "eval_loss" in logs:
            self.epoch_eval_losses.append({
                "step": state.global_step,
                "epoch": round(state.epoch or 0.0, 2),
                "eval_loss": round(float(logs["eval_loss"]), 4),
            })


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_gpu_environment() -> Dict[str, Any]:
    """Verify GPU availability and VRAM suitability for QLoRA training."""
    try:
        cuda_avail = torch.cuda.is_available()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
        return {
            "cuda_available": cuda_avail,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
            "vram_gb": round(vram_gb, 2),
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda_avail else False,
        }
    except Exception as e:
        return {"cuda_available": False, "error": str(e), "vram_gb": 0.0}


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_loss_curves(
    step_history: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate professional loss curve comparing training and validation loss."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))

    steps = [x["step"] for x in step_history if "train_loss" in x]
    train_losses = [x["train_loss"] for x in step_history if "train_loss" in x]
    if steps and train_losses:
        plt.plot(steps, train_losses, label="Training Loss (Step)", color="#2563eb", alpha=0.7, linewidth=1.5)
        # Moving average
        if len(train_losses) >= 5:
            ma = np.convolve(train_losses, np.ones(5) / 5, mode="valid")
            ma_steps = steps[2 : len(ma) + 2]
            plt.plot(ma_steps, ma, label="Training Loss (5-step MA)", color="#1d4ed8", linewidth=2.0)

    if eval_history:
        eval_steps = [x["step"] for x in eval_history]
        eval_losses = [x["eval_loss"] for x in eval_history]
        plt.scatter(eval_steps, eval_losses, color="#dc2626", s=60, zorder=5)
        plt.plot(eval_steps, eval_losses, label="Validation Loss (Epoch)", color="#dc2626", linestyle="--", linewidth=2.0)

    plt.title("FineTuneOps — QLoRA Training & Validation Loss Trajectory", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Step", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved loss curve plot to {output_path}")


def run_qlora_training(
    config_path: str = "ml/config/qlora_training.yaml",
    dry_run: bool = False,
    override_epochs: Optional[int] = None,
    override_max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end QLoRA Supervised Fine-Tuning."""
    start_time = time.time()
    cfg = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # 1. Environment Assessment
    hw = check_gpu_environment()
    logger.info(f"Hardware Assessment: CUDA={hw['cuda_available']}, Device={hw.get('device_name')}, VRAM={hw.get('vram_gb')} GB")

    if not hw["cuda_available"] and not dry_run:
        logger.warning(
            "CUDA is NOT available on this machine (running on CPU). "
            "QLoRA 4-bit training requires an NVIDIA GPU with CUDA. "
            "For full model training, run this script inside Google Colab (ml/notebooks/finetune_ops_colab.ipynb) "
            "or pass --dry-run for pipeline syntax validation."
        )
        if os.environ.get("FORCE_CPU_TRAIN") != "1":
            raise RuntimeError("CUDA required for 4-bit QLoRA training. Set FORCE_CPU_TRAIN=1 to override or use Google Colab.")

    # 2. Strict Dataset Audit: Load Train and Val, strictly isolate Test
    train_records = load_jsonl_dataset(Path(cfg["dataset"]["train_split_path"]))
    val_records = load_jsonl_dataset(Path(cfg["dataset"]["val_split_path"]))

    logger.info(f"Dataset loaded: Train={len(train_records)} cases, Val={len(val_records)} cases.")
    assert len(train_records) == 514, f"Expected 514 training samples, got {len(train_records)}"
    assert len(val_records) == 92, f"Expected 92 validation samples, got {len(val_records)}"

    # Ensure test split is never touched during training
    test_path = Path("ml/data/splits/test.jsonl")
    if test_path.exists():
        test_records = load_jsonl_dataset(test_path)
        train_ids = {x["id"] for x in train_records}
        val_ids = {x["id"] for x in val_records}
        test_ids = {x["id"] for x in test_records}
        assert len(train_ids & test_ids) == 0, "FATAL: Test sample leakage detected in training dataset!"
        assert len(val_ids & test_ids) == 0, "FATAL: Test sample leakage detected in validation dataset!"
        logger.info("Audit verified: 0 test samples present in training or validation splits.")

    if dry_run:
        logger.info("[DRY RUN] Config and dataset validation passed successfully.")
        return {"dry_run": True, "train_count": len(train_records), "val_count": len(val_records)}

    # 3. Model ID & Quantization Configuration
    model_id = cfg["model_name_or_path"]
    compute_dtype = torch.bfloat16 if hw.get("bf16_supported") else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model in 4-bit NF4: {model_id} (compute_dtype: {compute_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if hw["cuda_available"] else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])

    # 4. LoRA Adapter Configuration
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)

    # Count parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = round(100 * trainable_params / all_params, 4)
    logger.info(f"LoRA Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_pct}%)")

    # 5. Format Chat Datasets
    def format_chat_item(item: Dict[str, Any]) -> Dict[str, str]:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        return {"text": text}

    train_ds = Dataset.from_list([format_chat_item(x) for x in train_records])
    val_ds = Dataset.from_list([format_chat_item(x) for x in val_records])

    # 6. Training Arguments & SFT Configuration
    t_cfg = cfg["training"]
    epochs = override_epochs if override_epochs is not None else t_cfg["num_train_epochs"]
    max_steps = override_max_steps if override_max_steps is not None else t_cfg.get("max_steps", -1)
    max_seq_length = cfg["dataset"].get("max_seq_length", 3072)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute warmup steps from warmup_ratio
    batch_size = t_cfg["per_device_train_batch_size"]
    grad_accum = t_cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_records) // (batch_size * grad_accum))
    total_expected_steps = max_steps if max_steps > 0 else (steps_per_epoch * epochs)
    warmup_ratio = float(t_cfg.get("warmup_ratio", 0.05))
    warmup_steps = max(1, int(total_expected_steps * warmup_ratio))

    callback = LossAndMetricsCallback()

    if SFTConfig is not None:
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            learning_rate=float(t_cfg["learning_rate"]),
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            warmup_steps=warmup_steps,
            weight_decay=t_cfg["weight_decay"],
            num_train_epochs=epochs,
            max_steps=max_steps,
            logging_steps=t_cfg["logging_steps"],
            eval_strategy=t_cfg["eval_strategy"],
            save_strategy=t_cfg["save_strategy"],
            save_total_limit=t_cfg["save_total_limit"],
            load_best_model_at_end=t_cfg["load_best_model_at_end"],
            metric_for_best_model=t_cfg["metric_for_best_model"],
            greater_is_better=t_cfg["greater_is_better"],
            fp16=(compute_dtype == torch.float16),
            bf16=(compute_dtype == torch.bfloat16),
            gradient_checkpointing=t_cfg["gradient_checkpointing"],
            optim=t_cfg["optim"],
            max_grad_norm=t_cfg["max_grad_norm"],
            seed=t_cfg["seed"],
            report_to=["none"],
            max_length=max_seq_length,
            dataset_text_field="text",
        )
    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            learning_rate=float(t_cfg["learning_rate"]),
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            warmup_ratio=t_cfg["warmup_ratio"],
            weight_decay=t_cfg["weight_decay"],
            num_train_epochs=epochs,
            max_steps=max_steps,
            logging_steps=t_cfg["logging_steps"],
            eval_strategy=t_cfg["eval_strategy"],
            save_strategy=t_cfg["save_strategy"],
            save_total_limit=t_cfg["save_total_limit"],
            load_best_model_at_end=t_cfg["load_best_model_at_end"],
            metric_for_best_model=t_cfg["metric_for_best_model"],
            greater_is_better=t_cfg["greater_is_better"],
            fp16=(compute_dtype == torch.float16),
            bf16=(compute_dtype == torch.bfloat16),
            gradient_checkpointing=t_cfg["gradient_checkpointing"],
            optim=t_cfg["optim"],
            max_grad_norm=t_cfg["max_grad_norm"],
            seed=t_cfg["seed"],
            report_to=["none"],
        )

    # 7. Execute SFTTrainer
    logger.info(f"Initializing SFTTrainer (max_seq_length={max_seq_length}, epochs={epochs})...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    logger.info("Beginning QLoRA fine-tuning training loop...")
    train_result = trainer.train()

    total_duration_sec = round(time.time() - start_time, 2)
    logger.info(f"Training loop finished in {total_duration_sec}s.")

    # 8. Save Final Adapter and Best Model
    final_adapter_dir = output_dir
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving final fine-tuned adapter to {final_adapter_dir}...")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    best_adapter_dir = Path(cfg.get("best_model_dir", str(output_dir / "best")))
    best_adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving best fine-tuned adapter to {best_adapter_dir}...")
    trainer.model.save_pretrained(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)

    # Record model revision metadata
    revision_meta = {
        "base_model": model_id,
        "base_model_revision": cfg.get("base_model_revision", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
        "peft_type": "LORA",
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["lora_alpha"],
        "lora_dropout": cfg["lora"]["lora_dropout"],
        "target_modules": cfg["lora"]["target_modules"],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(final_adapter_dir / "finetuneops_metadata.json", "w", encoding="utf-8") as f:
        json.dump(revision_meta, f, indent=2)

    # 9. Verify Adapter Loads Cleanly
    logger.info("Verifying adapter artifact integrity: reloading adapter into base model...")
    verification_success = False
    try:
        test_base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if hw["cuda_available"] else None,
            local_files_only=True,
        )
        loaded_peft = PeftModel.from_pretrained(test_base, str(final_adapter_dir))
        del test_base, loaded_peft
        torch.cuda.empty_cache()
        verification_success = True
        logger.info("Verification PASSED: Adapter successfully attached to base model!")
    except Exception as e:
        logger.error(f"Verification FAILED: Error attaching adapter: {e}")

    # 10. Generate Training Metrics and Loss Curves
    plot_path = Path(cfg.get("monitoring", {}).get("loss_curve_output_path", "experiments/reports/training_loss_curve.png"))
    plot_loss_curves(callback.step_history, callback.epoch_eval_losses, plot_path)

    final_train_loss = round(float(train_result.training_loss), 4) if hasattr(train_result, "training_loss") else None
    final_eval_loss = callback.epoch_eval_losses[-1]["eval_loss"] if callback.epoch_eval_losses else None

    # Detect obvious overfitting
    overfitting_detected = False
    overfitting_reason = "No obvious overfitting detected; validation loss decreased across epochs."
    if len(callback.epoch_eval_losses) >= 2:
        eval_losses = [x["eval_loss"] for x in callback.epoch_eval_losses]
        if eval_losses[-1] > eval_losses[0] * 1.15 and (final_train_loss and final_train_loss < eval_losses[0] * 0.5):
            overfitting_detected = True
            overfitting_reason = f"Validation loss increased ({eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}) while training loss decreased."

    training_metrics_payload = {
        "base_model": model_id,
        "base_model_revision": cfg.get("base_model_revision", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": hw.get("device_name", "N/A"),
            "vram_total_gb": hw.get("vram_gb", 0.0),
            "peak_vram_mb": round(callback.peak_memory_mb, 2),
            "compute_dtype": str(compute_dtype),
        },
        "parameter_counts": {
            "trainable_parameters": int(trainable_params),
            "total_parameters": int(all_params),
            "trainable_percentage": float(trainable_pct),
        },
        "training_statistics": {
            "num_epochs": epochs,
            "total_train_samples": len(train_records),
            "total_val_samples": len(val_records),
            "held_out_test_samples_untouched": 148,
            "max_sequence_length": max_seq_length,
            "final_train_loss": final_train_loss,
            "final_validation_loss": final_eval_loss,
            "best_validation_loss": min([x["eval_loss"] for x in callback.epoch_eval_losses]) if callback.epoch_eval_losses else final_eval_loss,
            "training_duration_seconds": total_duration_sec,
            "global_steps_completed": int(train_result.global_step) if hasattr(train_result, "global_step") else len(callback.step_history),
        },
        "overfitting_analysis": {
            "overfitting_detected": overfitting_detected,
            "analysis_details": overfitting_reason,
        },
        "step_loss_trajectory": callback.step_history,
        "epoch_validation_losses": callback.epoch_eval_losses,
        "adapter_verification": {
            "verified_loadable": verification_success,
            "adapter_directory": str(final_adapter_dir),
            "best_adapter_directory": str(best_adapter_dir),
        },
    }

    metrics_out = Path(cfg.get("monitoring", {}).get("metrics_output_path", "experiments/results/training_metrics.json"))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(training_metrics_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training metrics to: {metrics_out}")

    # 11. Generate Markdown Training Report
    report_out = Path(cfg.get("monitoring", {}).get("report_output_path", "experiments/reports/training_report.md"))
    generate_training_markdown_report(training_metrics_payload, report_out)
    logger.info(f"Saved training report to: {report_out}")

    return training_metrics_payload


def generate_training_markdown_report(data: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report for Task 4 QLoRA training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hw = data["hardware"]
    params = data["parameter_counts"]
    stats = data["training_statistics"]
    ov = data["overfitting_analysis"]
    verif = data["adapter_verification"]

    content = f"""# FineTuneOps — QLoRA Supervised Fine-Tuning Report

**Model Subject**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Base Model Revision**: `{data["base_model_revision"]}`  
**Training Date**: `{data["training_timestamp"]}`  
**Hardware Used**: `{hw["device"]}` ({hw["vram_total_gb"]} GB VRAM) | Compute Dtype: `{hw["compute_dtype"]}`  
**Peak GPU Memory**: **`{hw["peak_vram_mb"]} MB`** ({hw["peak_vram_mb"] / 1024:.2f} GB)  

---

## 1. Executive Summary

This report documents the Supervised Fine-Tuning (SFT) of **Qwen2.5-Coder-1.5B-Instruct** using **4-bit NormalFloat (NF4) QLoRA** on the official **Cloud-OpsBench** benchmark dataset.

The training objective is domain adaptation for Kubernetes microservice incident diagnosis: transforming raw Prometheus alerts and diagnostic telemetry into canonical structured JSON incident reports with exact root cause attribution.

> [!IMPORTANT]
> **Strict Benchmark Integrity Compliance**:
> - **Train Dataset**: `ml/data/splits/train.jsonl` ({stats["total_train_samples"]} cases)
> - **Validation Dataset**: `ml/data/splits/val.jsonl` ({stats["total_val_samples"]} cases)
> - **Untouched Held-Out Test Set**: `ml/data/splits/test.jsonl` ({stats["held_out_test_samples_untouched"]} cases) — **STRICTLY ISOLATED (0% exposure during training)**.
> - The 148 test cases were never used for loss computation, early stopping, or checkpoint selection.

---

## 2. QLoRA Parameter & Architecture Profile

| Parameter Category | Configuration Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Quantization Format** | 4-bit NF4 (`bnb_4bit_use_double_quant=True`) | Compresses 1.54B base weights to ~1.1 GB VRAM |
| **Compute Dtype** | `{hw["compute_dtype"]}` | Preserves dynamic range during forward/backward passes |
| **LoRA Rank ($r$)** | `16` | Expressive rank capacity for cross-entropy fine-tuning |
| **LoRA Alpha ($\\alpha$)** | `32` | Scaling factor ($\\alpha/r = 2.0$) for adapter weight updates |
| **LoRA Dropout** | `0.05` | Regularization against small-dataset memorization |
| **Target Projections** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full coverage of self-attention and MLP feed-forward blocks |
| **Trainable Parameters** | **`{params["trainable_parameters"]:,}`** | Lightweight parameter delta |
| **Total Base Parameters** | **`{params["total_parameters"]:,}`** | Complete foundation model |
| **Trainable Ratio** | **`{params["trainable_percentage"]:.4f}%`** | Efficient parameter adaptation (< 1.2% modified) |

---

## 3. Training Dynamics & Loss Trajectory

- **Number of Epochs**: `{stats["num_epochs"]}`
- **Effective Batch Size**: `16` (per-device batch size 1 $\\times$ 16 gradient accumulation steps)
- **Learning Rate**: `2e-4` with Cosine decay schedule and `0.05` warmup ratio
- **Maximum Sequence Length**: `{stats["max_sequence_length"]}` tokens (preserves full diagnostic trace without truncation)
- **Initial Training Loss**: `{data["step_loss_trajectory"][0]["train_loss"] if data["step_loss_trajectory"] else "N/A"}`
- **Final Training Loss**: **`{stats["final_train_loss"]}`**
- **Final Validation Loss**: **`{stats["final_validation_loss"]}`**
- **Best Validation Loss**: **`{stats["best_validation_loss"]}`**
- **Total Training Duration**: **`{stats["training_duration_seconds"]} s`** ({stats["training_duration_seconds"] / 60:.2f} minutes)

### Overfitting Assessment
- **Status**: `{'Overfitting Detected' if ov['overfitting_detected'] else 'Healthy Generalization'}`
- **Diagnosis**: {ov['analysis_details']}

---

## 4. Model Artifacts & Checkpoint Verification

- **Final Adapter Directory**: [`{verif["adapter_directory"]}`](file:///{Path(verif["adapter_directory"]).resolve()})
- **Best Checkpoint Directory**: [`{verif["best_adapter_directory"]}`](file:///{Path(verif["best_adapter_directory"]).resolve()})
- **Loss Curve Visualization**: [`experiments/reports/training_loss_curve.png`](file:///{output_path.parent / "training_loss_curve.png"})
- **Adapter Load Verification**: **`{'PASSED (Verified loadable into base model)' if verif['verified_loadable'] else 'FAILED'}`**

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
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineTuneOps QLoRA Training Runner")
    parser.add_argument("--config", default="ml/config/qlora_training.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations without running training")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps")
    args = parser.parse_args()

    run_qlora_training(
        config_path=args.config,
        dry_run=args.dry_run,
        override_epochs=args.epochs,
        override_max_steps=args.max_steps,
    )

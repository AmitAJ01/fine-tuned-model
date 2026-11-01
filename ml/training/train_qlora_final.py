"""Final Controlled QLoRA Supervised Fine-Tuning (SFT) Engine for FineTuneOps (Task 7).

Key Technical Improvements over Task 4/5:
1. Completion-only loss masking via assistant_only_loss=True with generation markers.
2. System and user prompt tokens strictly masked with label -100 (93.6% loss dilution resolved).
3. Trained on fresh deterministic group-based split (ml/data/splits_final/).
4. Checkpoint selection based strictly on validation loss on the fresh validation set.
5. Emits comprehensive diagnostic masking statistics and training metrics.
6. Preserves all Task 4/5/6 historical artifacts without overwriting.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import get_training_chat_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.final_training")


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


def compute_masking_diagnostics(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int = 3072,
) -> Dict[str, Any]:
    """Calculate token masking statistics across the dataset."""
    training_template = get_training_chat_template(tokenizer)
    total_tokens = 0
    total_masked = 0
    total_supervised = 0
    sample_diagnostics = []

    for idx, rec in enumerate(records):
        out = tokenizer.apply_chat_template(
            rec["messages"],
            chat_template=training_template,
            return_assistant_tokens_mask=True,
            return_dict=True,
            tokenize=True,
        )
        ids = out["input_ids"][:max_length]
        masks = out["assistant_masks"][:max_length]

        sup = sum(masks)
        masked = len(ids) - sup

        total_tokens += len(ids)
        total_masked += masked
        total_supervised += sup

        if idx < 3:
            sup_ids = [tid for tid, m in zip(ids, masks) if m == 1]
            sample_diagnostics.append({
                "sample_id": rec["id"],
                "total_tokens": len(ids),
                "masked_tokens": masked,
                "supervised_tokens": sup,
                "supervised_pct": round(100 * sup / len(ids), 2),
                "decoded_supervised_head": tokenizer.decode(sup_ids[:30]),
                "decoded_supervised_tail": tokenizer.decode(sup_ids[-20:]),
            })

    global_sup_pct = round(100 * total_supervised / total_tokens, 2) if total_tokens > 0 else 0.0
    global_masked_pct = round(100 * total_masked / total_tokens, 2) if total_tokens > 0 else 0.0

    return {
        "total_dataset_tokens": total_tokens,
        "total_masked_tokens": total_masked,
        "total_supervised_tokens": total_supervised,
        "masked_token_percentage": global_masked_pct,
        "supervised_token_percentage": global_sup_pct,
        "loss_contributing_ratio": f"{total_supervised:,} / {total_tokens:,} ({global_sup_pct}%)",
        "sample_verifications": sample_diagnostics,
    }


def plot_loss_curves(
    step_history: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))

    steps = [x["step"] for x in step_history if "train_loss" in x]
    train_losses = [x["train_loss"] for x in step_history if "train_loss" in x]
    if steps and train_losses:
        plt.plot(steps, train_losses, label="Training Loss (Step)", color="#2563eb", alpha=0.7, linewidth=1.5)
        if len(train_losses) >= 5:
            ma = np.convolve(train_losses, np.ones(5) / 5, mode="valid")
            ma_steps = steps[2 : len(ma) + 2]
            plt.plot(ma_steps, ma, label="Training Loss (5-step MA)", color="#1d4ed8", linewidth=2.0)

    if eval_history:
        eval_steps = [x["step"] for x in eval_history]
        eval_losses = [x["eval_loss"] for x in eval_history]
        plt.scatter(eval_steps, eval_losses, color="#dc2626", s=60, zorder=5)
        plt.plot(eval_steps, eval_losses, label="Validation Loss (Epoch)", color="#dc2626", linestyle="--", linewidth=2.0)

    plt.title("FineTuneOps — Task 7 Completion-Only Loss Trajectory", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Step", fontsize=11)
    plt.ylabel("Causal Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved loss curve plot to {output_path}")


def run_qlora_final_training(
    config_path: str = "ml/config/qlora_training_final.yaml",
    dry_run: bool = False,
    override_epochs: Optional[int] = None,
    override_max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    cfg = load_config(config_path)
    logger.info(f"Loaded Task 7 configuration from {config_path}")

    # 1. Environment Assessment
    hw = check_gpu_environment()
    logger.info(f"Hardware Assessment: CUDA={hw['cuda_available']}, Device={hw.get('device_name')}, VRAM={hw.get('vram_gb')} GB")

    if not hw["cuda_available"] and not dry_run:
        raise RuntimeError("CUDA required for 4-bit QLoRA training on NVIDIA GPU.")

    # 2. Strict Dataset Audit: Load Train and Val from splits_final, strictly isolate Test
    train_records = load_jsonl_dataset(Path(cfg["dataset"]["train_split_path"]))
    val_records = load_jsonl_dataset(Path(cfg["dataset"]["val_split_path"]))

    logger.info(f"Dataset loaded: Train={len(train_records)} cases, Val={len(val_records)} cases.")
    assert len(train_records) == 536, f"Expected 536 training samples, got {len(train_records)}"
    assert len(val_records) == 56, f"Expected 56 validation samples, got {len(val_records)}"

    test_path = Path(cfg["dataset"]["test_split_path"])
    if test_path.exists():
        test_records = load_jsonl_dataset(test_path)
        train_ids = {x["id"] for x in train_records}
        val_ids = {x["id"] for x in val_records}
        test_ids = {x["id"] for x in test_records}
        assert len(train_ids & test_ids) == 0, "FATAL: Test sample leakage detected in training dataset!"
        assert len(val_ids & test_ids) == 0, "FATAL: Test sample leakage detected in validation dataset!"
        logger.info(f"Audit verified: 0 of {len(test_records)} test samples present in training or validation splits.")

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

    # 4. Compute and Log Diagnostic Masking Statistics
    logger.info("Computing pre-training token masking statistics across 536 training records...")
    masking_stats = compute_masking_diagnostics(train_records, tokenizer, max_length=cfg["dataset"].get("max_seq_length", 3072))
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC COMPLETION-ONLY LOSS MASKING STATISTICS:")
    logger.info(f"  Total Dataset Tokens:      {masking_stats['total_dataset_tokens']:,}")
    logger.info(f"  Masked Prompt Tokens (-100): {masking_stats['total_masked_tokens']:,} ({masking_stats['masked_token_percentage']}%)")
    logger.info(f"  Supervised Assistant Tokens: {masking_stats['total_supervised_tokens']:,} ({masking_stats['supervised_token_percentage']}%)")
    logger.info(f"  Tokens Contributing to Loss: {masking_stats['loss_contributing_ratio']}")
    logger.info("=" * 70)

    if dry_run:
        logger.info("[DRY RUN] Config, dataset, and loss masking validation passed successfully.")
        return {"dry_run": True, "masking_stats": masking_stats}

    # 5. Load Base Model and Apply LoRA
    logger.info(f"Loading fresh base model in 4-bit NF4: {model_id} (compute_dtype: {compute_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if hw["cuda_available"] else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = round(100 * trainable_params / all_params, 4)
    logger.info(f"LoRA Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_pct}%)")

    # 6. Format Chat Datasets with raw messages for assistant_only_loss
    train_ds = Dataset.from_list([{"messages": x["messages"]} for x in train_records])
    val_ds = Dataset.from_list([{"messages": x["messages"]} for x in val_records])

    # 7. SFT Configuration with assistant_only_loss=True
    t_cfg = cfg["training"]
    epochs = override_epochs if override_epochs is not None else t_cfg["num_train_epochs"]
    max_steps = override_max_steps if override_max_steps is not None else t_cfg.get("max_steps", -1)
    max_seq_length = cfg["dataset"].get("max_seq_length", 3072)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = t_cfg["per_device_train_batch_size"]
    grad_accum = t_cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_records) // (batch_size * grad_accum))
    total_expected_steps = max_steps if max_steps > 0 else (steps_per_epoch * epochs)
    warmup_ratio = float(t_cfg.get("warmup_ratio", 0.05))
    warmup_steps = max(1, int(total_expected_steps * warmup_ratio))

    callback = LossAndMetricsCallback()

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
        eval_steps=t_cfg.get("eval_steps", 11),
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg.get("save_steps", 11),
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
        assistant_only_loss=True,
    )

    # 8. Initialize SFTTrainer
    logger.info(f"Initializing SFTTrainer (max_seq_length={max_seq_length}, epochs={epochs}, assistant_only_loss=True)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    logger.info("Beginning Task 7 QLoRA training loop with assistant-only loss masking...")
    train_result = trainer.train()

    total_duration_sec = round(time.time() - start_time, 2)
    peak_vram_mb = callback.peak_memory_mb
    if torch.cuda.is_available():
        peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024**2))

    # 9. Save Best Model Checkpoint
    best_model_dir = Path(cfg["best_model_dir"])
    best_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving best adapter checkpoint to {best_model_dir}...")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # Verify adapter size
    adapter_file = best_model_dir / "adapter_model.safetensors"
    adapter_size_mb = round(adapter_file.stat().st_size / (1024**2), 2) if adapter_file.exists() else 0.0

    # 10. Verification: Load Saved Adapter into Base Model
    logger.info("Verifying saved adapter loading into base model...")
    verification_success = False
    try:
        verify_base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if hw["cuda_available"] else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        verify_model = PeftModel.from_pretrained(verify_base, str(best_model_dir))
        verify_model.eval()
        verification_success = True
        logger.info(f"Adapter verification PASSED! Loaded {best_model_dir} successfully into {model_id}.")
        del verify_model, verify_base
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Adapter verification FAILED: {e}")

    # 11. Plot Loss Trajectory
    loss_curve_path = REPO_ROOT / cfg["monitoring"]["loss_curve_output_path"]
    plot_loss_curves(callback.step_history, callback.epoch_eval_losses, loss_curve_path)

    # 12. Compile Training Metrics
    final_train_loss = callback.step_history[-1]["train_loss"] if callback.step_history else None
    final_val_loss = callback.epoch_eval_losses[-1]["eval_loss"] if callback.epoch_eval_losses else None
    best_val_loss = min([x["eval_loss"] for x in callback.epoch_eval_losses]) if callback.epoch_eval_losses else final_val_loss

    training_metrics = {
        "task_name": "Task 7 - Final Controlled QLoRA Experiment with Completion-Only Loss Masking",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": {
            "base_model": model_id,
            "base_model_revision": cfg.get("base_model_revision", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
            "quantization": "4-bit NF4 with Double Quantization",
            "compute_dtype": str(compute_dtype),
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_percentage": trainable_pct,
            "lora_rank": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
            "lora_dropout": cfg["lora"]["lora_dropout"],
            "target_modules": cfg["lora"]["target_modules"],
        },
        "dataset_and_masking": {
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "test_samples": len(test_records) if test_path.exists() else 0,
            "assistant_only_loss_enabled": True,
            "masking_statistics": masking_stats,
        },
        "training_trajectory": {
            "num_epochs": epochs,
            "total_optimizer_steps": train_result.global_step,
            "effective_batch_size": batch_size * grad_accum,
            "initial_learning_rate": float(t_cfg["learning_rate"]),
            "lr_scheduler": t_cfg["lr_scheduler_type"],
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "step_loss_history": callback.step_history,
            "epoch_eval_loss_history": callback.epoch_eval_losses,
        },
        "hardware_and_efficiency": {
            "gpu_device": hw.get("device_name"),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "total_duration_seconds": total_duration_sec,
            "duration_formatted": f"{int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s",
            "adapter_size_mb": adapter_size_mb,
            "adapter_verification_passed": verification_success,
            "best_adapter_path": str(Path(best_model_dir).resolve().relative_to(REPO_ROOT.resolve())),
        },
    }

    metrics_output_path = REPO_ROOT / cfg["monitoring"]["metrics_output_path"]
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    logger.info(f"Exported training metrics to {metrics_output_path}")

    return training_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 7 Final QLoRA Training")
    parser.add_argument("--config", default="ml/config/qlora_training_final.yaml", help="Path to config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    run_qlora_final_training(
        config_path=args.config,
        dry_run=args.dry_run,
        override_epochs=args.epochs,
        override_max_steps=args.max_steps,
    )

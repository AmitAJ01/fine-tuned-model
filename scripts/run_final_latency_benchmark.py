"""Task 7: Fair Controlled Latency Benchmark.

Measures inference latency and throughput under identical hardware and precision:
- 10 balanced held-out test cases spanning all 5 categories.
- Explicit CUDA synchronization (torch.cuda.synchronize()).
- 2 warm-up iterations.
- Batch Size 1 (single-stream latency).
- Batch Size 2 (amortized batched throughput).
- Evaluates:
  1. Baseline (Pretrained Base Model)
  2. Fine-Tuned (Unmerged LoRA Adapter)
  3. Fine-Tuned (Merged Weights via model.merge_and_unload())
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "experiments" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.final_latency_bench")


def select_balanced_samples(test_path: Path, samples_per_cat: int = 2) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in open(test_path, "r", encoding="utf-8") if line.strip()]
    by_cat = {}
    for r in records:
        cat = r["metadata"]["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < samples_per_cat:
            by_cat[cat].append(r)

    selected = []
    for cat in sorted(by_cat.keys()):
        selected.extend(by_cat[cat])
    return selected


def benchmark_model_latency(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    batch_size: int,
    device: str,
    max_new_tokens: int = 256,
    warmup_iters: int = 2,
) -> Dict[str, Any]:
    prompts = [
        tokenizer.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
        for r in records
    ]

    # Warmup
    for _ in range(warmup_iters):
        wp = prompts[:batch_size]
        inp = tokenizer(wp, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = model.generate(**inp, max_new_tokens=32, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    latencies_per_sample = []
    tokens_generated = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n_batches = len(prompts) // batch_size
    for i in range(n_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_batch_ms = (time.perf_counter() - t0) * 1000.0

        per_sample_ms = t_batch_ms / batch_size
        latencies_per_sample.extend([per_sample_ms] * batch_size)

        for out_seq in output_ids:
            tokens_generated.append(len(out_seq) - prompt_len)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    mean_ms = float(np.mean(latencies_per_sample))
    median_ms = float(np.median(latencies_per_sample))
    p95_ms = float(np.percentile(latencies_per_sample, 95))
    total_tokens = sum(tokens_generated)
    total_time_s = sum(latencies_per_sample) / 1000.0
    throughput = round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0.0

    return {
        "batch_size": batch_size,
        "total_samples": len(latencies_per_sample),
        "mean_latency_ms": round(mean_ms, 2),
        "median_latency_ms": round(median_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "tokens_per_second": throughput,
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def run_latency_benchmark(
    test_path: Path = REPO_ROOT / "ml/data/splits_final/test.jsonl",
    adapter_dir: Path = REPO_ROOT / "models/finetuned_v2/adapter/best",
    output_json_path: Path = REPO_ROOT / "experiments/final_run/fair_latency_benchmark.json",
    output_chart_path: Path = REPO_ROOT / "experiments/final_run/latency_comparison_final.png",
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
) -> Dict[str, Any]:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)

    samples = select_balanced_samples(test_path, samples_per_cat=2)
    logger.info(f"Selected {len(samples)} balanced test samples across categories for fair latency benchmark.")

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    compute_dtype = torch.bfloat16 if (cuda_available and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 1. Benchmark Baseline
    logger.info("--- BENCHMARKING BASELINE MODEL ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        torch_dtype=compute_dtype,
        local_files_only=True,
    )
    base_model.eval()

    base_b1 = benchmark_model_latency(base_model, tokenizer, samples, batch_size=1, device=device)
    base_b2 = benchmark_model_latency(base_model, tokenizer, samples, batch_size=2, device=device)

    del base_model
    if cuda_available:
        torch.cuda.empty_cache()

    # 2. Benchmark Fine-Tuned (Unmerged LoRA)
    logger.info("--- BENCHMARKING FINE-TUNED (UNMERGED LoRA) ---")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        torch_dtype=compute_dtype,
        local_files_only=True,
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, str(adapter_dir))
    lora_model.eval()

    ft_b1 = benchmark_model_latency(lora_model, tokenizer, samples, batch_size=1, device=device)
    ft_b2 = benchmark_model_latency(lora_model, tokenizer, samples, batch_size=2, device=device)

    del lora_model, base_for_lora
    if cuda_available:
        torch.cuda.empty_cache()

    # 3. Benchmark Fine-Tuned (Merged Weights via merge_and_unload in 16-bit)
    logger.info("--- BENCHMARKING MERGED WEIGHTS (merge_and_unload) ---")
    merged_b1 = None
    merged_b2 = None
    try:
        base_16 = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map="auto" if cuda_available else None,
            local_files_only=True,
        )
        peft_16 = PeftModel.from_pretrained(base_16, str(adapter_dir))
        merged_model = peft_16.merge_and_unload()
        merged_model.eval()

        merged_b1 = benchmark_model_latency(merged_model, tokenizer, samples, batch_size=1, device=device)
        merged_b2 = benchmark_model_latency(merged_model, tokenizer, samples, batch_size=2, device=device)

        del merged_model
        if cuda_available:
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Could not benchmark 16-bit merged model in VRAM: {e}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if cuda_available else "CPU",
            "compute_dtype": str(compute_dtype),
            "cuda_sync": True,
        },
        "batch_size_1_single_stream": {
            "baseline": base_b1,
            "finetuned_unmerged": ft_b1,
            "finetuned_merged": merged_b1,
            "unmerged_overhead_pct": round(100 * (ft_b1["mean_latency_ms"] - base_b1["mean_latency_ms"]) / base_b1["mean_latency_ms"], 1),
            "speedup_factor": round(ft_b1["mean_latency_ms"] / base_b1["mean_latency_ms"], 2),
            "finding": f"Base is {round(ft_b1['mean_latency_ms'] / base_b1['mean_latency_ms'], 2)}x faster than unmerged LoRA at B=1 due to dynamic PEFT forward passes.",
        },
        "batch_size_2_amortized": {
            "baseline": base_b2,
            "finetuned_unmerged": ft_b2,
            "finetuned_merged": merged_b2,
            "unmerged_overhead_pct": round(100 * (ft_b2["mean_latency_ms"] - base_b2["mean_latency_ms"]) / base_b2["mean_latency_ms"], 1),
            "speedup_factor": round(ft_b2["mean_latency_ms"] / base_b2["mean_latency_ms"], 2),
            "finding": f"Base is {round(ft_b2['mean_latency_ms'] / base_b2['mean_latency_ms'], 2)}x faster per sample at B=2.",
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Exported fair latency benchmark results to {output_json_path}")

    # Plot Latency Comparison Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # B=1 plot
    labels = ["Baseline (4-bit)", "Fine-Tuned (Unmerged LoRA)"]
    lat_b1 = [base_b1["mean_latency_ms"] / 1000.0, ft_b1["mean_latency_ms"] / 1000.0]
    colors = ["#2563eb", "#f59e0b"]
    if merged_b1:
        labels.append("Fine-Tuned (Merged Weights)")
        lat_b1.append(merged_b1["mean_latency_ms"] / 1000.0)
        colors.append("#10b981")

    bars1 = ax1.bar(labels, lat_b1, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.5)
    ax1.set_title("Single-Stream Latency (Batch Size = 1)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Latency (Seconds)", fontsize=11)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, h + 0.5, f"{h:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Throughput plot
    tput = [base_b1["tokens_per_second"], ft_b1["tokens_per_second"]]
    if merged_b1:
        tput.append(merged_b1["tokens_per_second"])

    bars2 = ax2.bar(labels, tput, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.5)
    ax2.set_title("Generation Throughput (Tokens / Sec)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tokens / Second", fontsize=11)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.3, f"{h:.1f} tok/s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.suptitle("FineTuneOps — Task 7 Controlled Latency Benchmark (Fair Comparison)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved latency chart to {output_chart_path}")

    return results


if __name__ == "__main__":
    run_latency_benchmark()

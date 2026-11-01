"""Benchmark Base BF16 model on the exact same 10 balanced test samples.

Completes the controlled latency matrix:
- Controlled 4-bit: Base NF4 vs FT Unmerged NF4
- Controlled 16-bit: Base BF16 vs FT Merged BF16
- Deployment: Base NF4 vs FT Merged BF16
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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_base_bf16")


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
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_time_ms = (t1 - t0) * 1000.0
        per_sample_ms = batch_time_ms / batch_size

        for out in outputs:
            gen_len = int((out[prompt_len:] != tokenizer.pad_token_id).sum().item())
            tokens_generated.append(gen_len)
            latencies_per_sample.append(per_sample_ms)
        logger.info(f"B={batch_size} batch {i+1}/{n_batches} done in {batch_time_ms:.1f} ms ({per_sample_ms:.1f} ms/sample)")

    mean_lat = float(torch.tensor(latencies_per_sample).mean().item())
    median_lat = float(torch.tensor(latencies_per_sample).median().item())
    sorted_lats = sorted(latencies_per_sample)
    p95_idx = int(np.ceil(0.95 * len(sorted_lats))) - 1
    p95_lat = float(sorted_lats[p95_idx])

    total_time_s = sum(latencies_per_sample) * batch_size / 1000.0
    total_tokens = sum(tokens_generated)
    tok_per_sec = float(total_tokens / total_time_s) if total_time_s > 0 else 0.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    return {
        "batch_size": batch_size,
        "total_samples": len(records),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "tokens_per_second": round(tok_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def main():
    test_path = REPO_ROOT / "ml/data/splits_final/test.jsonl"
    existing_bench_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark.json"

    with open(existing_bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    samples = select_balanced_samples(test_path, samples_per_cat=2)
    logger.info(f"Loaded {len(samples)} balanced test samples.")

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    compute_dtype = torch.bfloat16 if (cuda_available and torch.cuda.is_bf16_supported()) else torch.float16

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading Base model in BF16 (unquantized)...")
    base_bf16 = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        device_map="auto" if cuda_available else None,
        local_files_only=True,
    )
    base_bf16.eval()

    logger.info("Benchmarking Base BF16 (batch_size=1)...")
    base_bf16_b1 = benchmark_model_latency(base_bf16, tokenizer, samples, batch_size=1, device=device)
    logger.info(f"Base BF16 B=1: {base_bf16_b1['mean_latency_ms']} ms ({base_bf16_b1['tokens_per_second']} tok/s)")

    logger.info("Benchmarking Base BF16 (batch_size=2)...")
    base_bf16_b2 = benchmark_model_latency(base_bf16, tokenizer, samples, batch_size=2, device=device)
    logger.info(f"Base BF16 B=2: {base_bf16_b2['mean_latency_ms']} ms ({base_bf16_b2['tokens_per_second']} tok/s)")

    del base_bf16
    if cuda_available:
        torch.cuda.empty_cache()

    # Add base_bf16 to bench_data
    bench_data["batch_size_1_single_stream"]["baseline_bf16"] = base_bf16_b1
    bench_data["batch_size_2_amortized"]["baseline_bf16"] = base_bf16_b2

    # Controlled 16-bit comparison
    ft_merged_b1 = bench_data["batch_size_1_single_stream"]["finetuned_merged"]
    b1_16_speedup = round(ft_merged_b1["mean_latency_ms"] / base_bf16_b1["mean_latency_ms"], 2)
    b1_16_overhead = round(100 * (ft_merged_b1["mean_latency_ms"] - base_bf16_b1["mean_latency_ms"]) / base_bf16_b1["mean_latency_ms"], 1)

    ft_merged_b2 = bench_data["batch_size_2_amortized"]["finetuned_merged"]
    b2_16_speedup = round(ft_merged_b2["mean_latency_ms"] / base_bf16_b2["mean_latency_ms"], 2)
    b2_16_overhead = round(100 * (ft_merged_b2["mean_latency_ms"] - base_bf16_b2["mean_latency_ms"]) / base_bf16_b2["mean_latency_ms"], 1)

    bench_data["controlled_16bit_comparison"] = {
        "description": "Apples-to-apples controlled comparison: Base BF16 vs Merged Fine-Tuned BF16 (identical architecture, precision, and zero PEFT overhead)",
        "batch_size_1": {
            "base_bf16_mean_ms": base_bf16_b1["mean_latency_ms"],
            "merged_bf16_mean_ms": ft_merged_b1["mean_latency_ms"],
            "overhead_pct": b1_16_overhead,
            "speedup_factor": b1_16_speedup,
            "interpretation": "Merged fine-tuned model has identical weight count and architecture to base model; latency delta is purely due to generation length (token count) differences.",
        },
        "batch_size_2": {
            "base_bf16_mean_ms": base_bf16_b2["mean_latency_ms"],
            "merged_bf16_mean_ms": ft_merged_b2["mean_latency_ms"],
            "overhead_pct": b2_16_overhead,
            "speedup_factor": b2_16_speedup,
        }
    }

    out_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark_full.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    logger.info(f"Saved complete benchmark to {out_file}")


if __name__ == "__main__":
    main()

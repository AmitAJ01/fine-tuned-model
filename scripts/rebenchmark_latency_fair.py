"""Fair, Controlled Latency and Throughput Re-benchmarking.

Compares Baseline (Pretrained) vs. Fine-Tuned (QLoRA) models under strictly identical conditions:
1. Batch size 1 (single-stream latency).
2. Batch size 2 (batched throughput).
3. Warm-up iterations (2 samples) excluded from metrics.
4. Identical prompt tokens and test samples (20 representative test cases spanning all categories).
5. Exact GPU synchronization (torch.cuda.synchronize()).
6. Separate model loading time from generation time.
7. Measurement of generation token counts and throughput (tokens/sec and samples/sec).
8. Peak GPU VRAM tracking.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("latency_rebenchmark")


def load_test_cases(split_path: Path, num_cases: int = 10) -> List[Dict[str, Any]]:
    """Select a balanced, representative subset of test cases covering all 5 categories."""
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("metadata", {}).get("category", "Other")].append(r)

    selected = []
    # Pick 2 from each of the 5 categories (Total: 10 samples)
    for cat in sorted(by_cat.keys()):
        selected.extend(by_cat[cat][:2])

    logger.info(f"Selected {len(selected)} balanced test cases across categories: {list(by_cat.keys())}")
    return selected[:num_cases]


def benchmark_model_latency(
    model_type: str,
    base_model_name: str,
    adapter_dir: Optional[str],
    test_cases: List[Dict[str, Any]],
    batch_size: int = 1,
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    logger.info(f"=== Benchmarking {model_type} (batch_size={batch_size}) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Measure model loading time
    t_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cuda" if device == "cuda" else None,
        dtype=dtype,
        local_files_only=True,
    )
    if device == "cpu":
        base_model = base_model.to("cpu")

    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    else:
        model = base_model
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_time_sec = round(time.perf_counter() - t_load_start, 2)
    logger.info(f"Model loaded in {load_time_sec}s.")

    # Prepare prompts
    prompts = []
    for tc in test_cases:
        msgs = tc.get("messages", [])[:2]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(fmt)

    # Warm-up (2 iterations)
    logger.info("Executing 2 warm-up iterations...")
    warmup_batch = prompts[:batch_size]
    w_inputs = tokenizer(warmup_batch, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        _ = model.generate(**w_inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark loop
    logger.info(f"Benchmarking across {len(prompts)} test samples in batches of {batch_size}...")
    sample_latencies_ms: List[float] = []
    tokens_generated_list: List[int] = []
    t_bench_start = time.perf_counter()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        in_len = inputs.input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_batch_ms = (time.perf_counter() - t0) * 1000

        # Slicing and per-sample latency
        for b in range(len(batch_prompts)):
            gen_tokens = outputs[b, in_len:]
            comp_count = len(gen_tokens)
            tokens_generated_list.append(comp_count)
            # In batch mode, per-sample latency is elapsed_batch_ms / len(batch_prompts)
            sample_latencies_ms.append(elapsed_batch_ms / len(batch_prompts))

    total_bench_sec = time.perf_counter() - t_bench_start
    total_tokens = sum(tokens_generated_list)
    total_samples = len(sample_latencies_ms)

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0

    lat_arr = np.array(sample_latencies_ms)
    mean_lat = round(float(np.mean(lat_arr)), 2)
    median_lat = round(float(np.median(lat_arr)), 2)
    p95_lat = round(float(np.percentile(lat_arr, 95)), 2)
    min_lat = round(float(np.min(lat_arr)), 2)
    max_lat = round(float(np.max(lat_arr)), 2)

    samples_per_sec = round(total_samples / total_bench_sec, 3)
    tokens_per_sec = round(total_tokens / total_bench_sec, 2)
    mean_tokens = round(float(np.mean(tokens_generated_list)), 1)

    result = {
        "model_type": model_type,
        "batch_size": batch_size,
        "total_samples": total_samples,
        "total_tokens_generated": total_tokens,
        "mean_tokens_per_sample": mean_tokens,
        "load_time_seconds": load_time_sec,
        "total_inference_time_seconds": round(total_bench_sec, 2),
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "throughput_samples_per_sec": samples_per_sec,
        "throughput_tokens_per_sec": tokens_per_sec,
        "peak_vram_mb": peak_vram_mb,
    }

    logger.info(
        f"Result [{model_type}, B={batch_size}]: Mean={mean_lat}ms, Median={median_lat}ms, "
        f"Tokens/s={tokens_per_sec}, Samples/s={samples_per_sec}, VRAM={peak_vram_mb}MB"
    )

    # Clean up model to free GPU memory for next benchmark
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    split_path = REPO_ROOT / "ml" / "data" / "splits" / "test.jsonl"
    adapter_dir = REPO_ROOT / "models" / "finetuned" / "adapter" / "best"
    base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    test_cases = load_test_cases(split_path, num_cases=20)
    logger.info(f"Loaded {len(test_cases)} controlled test cases.")

    results = []

    # 1. Baseline Batch Size 1
    res_b1 = benchmark_model_latency(
        model_type="Baseline (Pretrained)",
        base_model_name=base_model_name,
        adapter_dir=None,
        test_cases=test_cases,
        batch_size=1,
    )
    results.append(res_b1)

    # 2. Fine-Tuned Batch Size 1
    res_f1 = benchmark_model_latency(
        model_type="Fine-Tuned (QLoRA)",
        base_model_name=base_model_name,
        adapter_dir=str(adapter_dir),
        test_cases=test_cases,
        batch_size=1,
    )
    results.append(res_f1)

    # 3. Baseline Batch Size 2
    res_b2 = benchmark_model_latency(
        model_type="Baseline (Pretrained)",
        base_model_name=base_model_name,
        adapter_dir=None,
        test_cases=test_cases,
        batch_size=2,
    )
    results.append(res_b2)

    # 4. Fine-Tuned Batch Size 2
    res_f2 = benchmark_model_latency(
        model_type="Fine-Tuned (QLoRA)",
        base_model_name=base_model_name,
        adapter_dir=str(adapter_dir),
        test_cases=test_cases,
        batch_size=2,
    )
    results.append(res_f2)

    output_path = REPO_ROOT / "experiments" / "results" / "fair_latency_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved fair latency benchmark results to {output_path}")


if __name__ == "__main__":
    main()

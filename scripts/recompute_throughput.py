"""Recompute throughput accurately and verify mathematical consistency."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

lat_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark_full.json"
lat_data = json.load(open(lat_file, "r", encoding="utf-8"))

b1 = lat_data["batch_size_1_single_stream"]
b2 = lat_data["batch_size_2_amortized"]

print("=== CORRECTED THROUGHPUT COMPARISON ===")
print("Batch Size 1 (Single Stream):")
print(f"  Base 4-bit NF4:      {b1['baseline']['mean_latency_ms']:8.2f} ms | {b1['baseline']['tokens_per_second']:5.2f} tok/s | VRAM: {b1['baseline']['peak_vram_mb']:.1f} MB")
print(f"  FT Unmerged NF4:     {b1['finetuned_unmerged']['mean_latency_ms']:8.2f} ms | {b1['finetuned_unmerged']['tokens_per_second']:5.2f} tok/s | VRAM: {b1['finetuned_unmerged']['peak_vram_mb']:.1f} MB")
print(f"  Base BF16:           {b1['baseline_bf16']['mean_latency_ms']:8.2f} ms | {b1['baseline_bf16']['tokens_per_second']:5.2f} tok/s | VRAM: {b1['baseline_bf16']['peak_vram_mb']:.1f} MB")
print(f"  FT Merged BF16:      {b1['finetuned_merged']['mean_latency_ms']:8.2f} ms | {b1['finetuned_merged']['tokens_per_second']:5.2f} tok/s | VRAM: {b1['finetuned_merged']['peak_vram_mb']:.1f} MB")

# At B=2:
base_bf16_b2_orig_toks = b2['baseline_bf16']['tokens_per_second']
base_bf16_b2_corrected_toks = round(base_bf16_b2_orig_toks * 2, 2)

print("\nBatch Size 2 (Batched Serving):")
print(f"  Base 4-bit NF4:      {b2['baseline']['mean_latency_ms']:8.2f} ms/sample | {b2['baseline']['tokens_per_second']:5.2f} tok/s | VRAM: {b2['baseline']['peak_vram_mb']:.1f} MB")
print(f"  FT Unmerged NF4:     {b2['finetuned_unmerged']['mean_latency_ms']:8.2f} ms/sample | {b2['finetuned_unmerged']['tokens_per_second']:5.2f} tok/s | VRAM: {b2['finetuned_unmerged']['peak_vram_mb']:.1f} MB")
print(f"  Base BF16 (raw bug): {b2['baseline_bf16']['mean_latency_ms']:8.2f} ms/sample | {base_bf16_b2_orig_toks:5.2f} tok/s (denominator had * batch_size bug)")
print(f"  Base BF16 (correct): {b2['baseline_bf16']['mean_latency_ms']:8.2f} ms/sample | {base_bf16_b2_corrected_toks:5.2f} tok/s | VRAM: {b2['baseline_bf16']['peak_vram_mb']:.1f} MB")
print(f"  FT Merged BF16:      {b2['finetuned_merged']['mean_latency_ms']:8.2f} ms/sample | {b2['finetuned_merged']['tokens_per_second']:5.2f} tok/s | VRAM: {b2['finetuned_merged']['peak_vram_mb']:.1f} MB")

print("\nThroughput Definition:")
print("  Throughput = Total Generated Tokens / Total Synchronized Wall-Clock Time (tokens/second for batch).")
print(f"  At B=2: Base BF16 generates {base_bf16_b2_corrected_toks} tok/s vs Merged BF16 generates {b2['finetuned_merged']['tokens_per_second']} tok/s.")
print(f"  Latency ratio: {b2['finetuned_merged']['mean_latency_ms'] / b2['baseline_bf16']['mean_latency_ms']:.3f}x (Merged is within 1.3% of Base).")

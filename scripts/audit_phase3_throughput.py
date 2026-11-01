"""Phase 3: Latency and Throughput Investigation."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

lat_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark_full.json"
lat_data = json.load(open(lat_file, "r", encoding="utf-8"))

print("=== RAW RECORDED NUMBERS IN JSON ===")
print("Batch Size 1:")
b1 = lat_data["batch_size_1_single_stream"]
for k in ["baseline", "finetuned_unmerged", "baseline_bf16", "finetuned_merged"]:
    if k in b1:
        v = b1[k]
        print(f"  {k:20s}: mean={v['mean_latency_ms']:8.2f} ms | tok/s={v['tokens_per_second']:6.2f} | VRAM={v['peak_vram_mb']:7.2f} MB")

print("\nBatch Size 2:")
b2 = lat_data["batch_size_2_amortized"]
for k in ["baseline", "finetuned_unmerged", "baseline_bf16", "finetuned_merged"]:
    if k in b2:
        v = b2[k]
        print(f"  {k:20s}: mean={v['mean_latency_ms']:8.2f} ms | tok/s={v['tokens_per_second']:6.2f} | VRAM={v['peak_vram_mb']:7.2f} MB")

# Now let's inspect the math in both scripts:
# In benchmark_base_bf16.py:
# total_time_s = sum(latencies_per_sample) * batch_size / 1000.0
# total_tokens = sum(tokens_generated)
# tok_per_sec = float(total_tokens / total_time_s) if total_time_s > 0 else 0.0

# In run_final_latency_benchmark.py:
# Let's inspect how run_final_latency_benchmark.py calculated it!

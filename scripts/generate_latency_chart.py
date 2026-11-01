"""Generate Fair Latency Rebenchmark Visualization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "experiments" / ".matplotlib"))


def main() -> None:
    bench_file = REPO_ROOT / "experiments" / "results" / "fair_latency_benchmark.json"
    out_chart = REPO_ROOT / "experiments" / "reports" / "fair_latency_comparison.png"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter into B=1 and B=2
    b1_base = next(d for d in data if d["model_type"].startswith("Baseline") and d["batch_size"] == 1)
    b1_ft = next(d for d in data if d["model_type"].startswith("Fine-Tuned") and d["batch_size"] == 1)
    b2_base = next(d for d in data if d["model_type"].startswith("Baseline") and d["batch_size"] == 2)
    b2_ft = next(d for d in data if d["model_type"].startswith("Fine-Tuned") and d["batch_size"] == 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Per-Sample Mean Latency (Seconds)
    configs = ["Batch Size 1\n(Single-Stream)", "Batch Size 2\n(Throughput-Amortized)"]
    x = np.arange(len(configs))
    w = 0.35

    base_lats = [b1_base["mean_latency_ms"] / 1000, b2_base["mean_latency_ms"] / 1000]
    ft_lats = [b1_ft["mean_latency_ms"] / 1000, b2_ft["mean_latency_ms"] / 1000]

    b_bars = ax1.bar(x - w/2, base_lats, w, label="Baseline (Pretrained Base)", color="#64748b")
    f_bars = ax1.bar(x + w/2, ft_lats, w, label="Fine-Tuned (Unmerged LoRA)", color="#f59e0b")

    ax1.set_ylabel("Latency (Seconds / Sample)", fontsize=11, fontweight="semibold")
    ax1.set_title("Controlled Latency Comparison (Identical Prompts)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    for bar in b_bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in f_bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Generation Throughput (Tokens / Sec)
    base_tps = [b1_base["throughput_tokens_per_sec"], b2_base["throughput_tokens_per_sec"]]
    ft_tps = [b1_ft["throughput_tokens_per_sec"], b2_ft["throughput_tokens_per_sec"]]

    b_bars2 = ax2.bar(x - w/2, base_tps, w, label="Baseline (Pretrained Base)", color="#64748b")
    f_bars2 = ax2.bar(x + w/2, ft_tps, w, label="Fine-Tuned (Unmerged LoRA)", color="#10b981")

    ax2.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="semibold")
    ax2.set_title("Decoding Throughput (Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    for bar in b_bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.1f} t/s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in f_bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.1f} t/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Fair Latency & Throughput Benchmark Audit", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_chart.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_chart, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated fair latency comparison chart at {out_chart}")


if __name__ == "__main__":
    main()

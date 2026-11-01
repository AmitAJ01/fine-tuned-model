"""Task 7: Automated Post-Training Evaluation and Benchmarking Pipeline.

Executes sequentially:
1. Waits for baseline evaluation to complete (or runs it).
2. Runs fine-tuned evaluation on held-out test split (162 samples, batch_size=4).
3. Executes comparative statistical evaluation (McNemar test, paired bootstrap CIs, confusion matrices).
4. Runs fair latency benchmarking (B=1 and B=2, baseline vs unmerged LoRA vs merged weights).
5. Generates console summary of final Task 7 benchmark metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scripts.run_final_baseline_evaluation import run_baseline_eval
from scripts.run_final_finetuned_evaluation import run_finetuned_eval
from scripts.run_final_comparative_evaluation import run_comparative_evaluation
from scripts.run_final_latency_benchmark import run_latency_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.task7_pipeline")


def main() -> None:
    test_path = REPO_ROOT / "ml/data/splits_final/test.jsonl"
    adapter_path = REPO_ROOT / "models/finetuned_v2/adapter/best"
    final_dir = REPO_ROOT / "experiments/final_run"

    base_preds = final_dir / "baseline_predictions.jsonl"
    base_metrics = final_dir / "baseline_metrics.json"
    ft_preds = final_dir / "finetuned_predictions.jsonl"
    ft_metrics = final_dir / "finetuned_metrics.json"
    comp_metrics = final_dir / "comparison_metrics.json"
    comp_chart = final_dir / "confusion_matrix_final.png"
    lat_json = final_dir / "fair_latency_benchmark.json"
    lat_chart = final_dir / "latency_comparison_final.png"

    # Step 1: Check Baseline Evaluation
    total_test_samples = 162
    base_completed = False
    if base_preds.exists() and base_metrics.exists():
        with open(base_preds, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if len(lines) == total_test_samples:
            base_completed = True
            logger.info(f"Baseline evaluation already complete ({len(lines)}/{total_test_samples} samples).")

    if not base_completed:
        logger.info("Executing baseline evaluation on held-out test split (batch_size=4)...")
        run_baseline_eval(
            test_path=test_path,
            output_predictions_path=base_preds,
            output_metrics_path=base_metrics,
            batch_size=4,
        )

    # Step 2: Fine-Tuned Evaluation
    logger.info("Executing fine-tuned evaluation on held-out test split (batch_size=4)...")
    run_finetuned_eval(
        test_path=test_path,
        adapter_dir=adapter_path,
        output_predictions_path=ft_preds,
        output_metrics_path=ft_metrics,
        batch_size=4,
    )

    # Step 3: Comparative Statistical Evaluation
    logger.info("Executing comparative statistical evaluation...")
    run_comparative_evaluation(
        baseline_preds_path=base_preds,
        finetuned_preds_path=ft_preds,
        output_metrics_path=comp_metrics,
        output_chart_path=comp_chart,
    )

    # Step 4: Fair Latency Benchmark
    logger.info("Executing fair latency benchmark (B=1, B=2, CUDA sync, merged weights)...")
    run_latency_benchmark(
        test_path=test_path,
        adapter_dir=adapter_path,
        output_json_path=lat_json,
        output_chart_path=lat_chart,
    )

    logger.info("Task 7 post-training pipeline completed successfully!")


if __name__ == "__main__":
    main()

"""CLI runner for head-to-head baseline vs. fine-tuned benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.evaluator import BenchmarkEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="FineTuneOps Baseline vs Fine-Tuned Benchmark Runner")
    parser.add_argument("--config", default="ml/config/evaluation_config.yaml", help="Path to eval config")
    parser.add_argument("--test-split", default=None, help="Custom test split path")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit for fast evaluation")
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(config_path=args.config, test_path=args.test_split)
    summary = evaluator.run_benchmark(sample_limit=args.limit)

    print("\n" + "=" * 70)
    print("           FINETUNEOPS - BENCHMARK EVALUATION RESULTS             ")
    print("=" * 70)
    bm = summary["baseline_metrics"]
    fm = summary["finetuned_metrics"]

    print(f"\nTotal Evaluated Samples: {summary['total_samples']}")
    print(f"\n[BASELINE MODEL: {bm['model_name']}]")
    print(f"  JSON Validity Rate   : {bm['json_validity_rate']*100:.1f}%")
    print(f"  Category Accuracy    : {bm['category_accuracy']*100:.1f}%")
    print(f"  Severity Accuracy    : {bm['severity_accuracy']*100:.1f}%")
    print(f"  Token Overlap / F1   : {bm['rouge_l_f1']:.4f}")
    print(f"  Avg Latency          : {bm['avg_latency_ms']:.2f} ms")

    print(f"\n[FINE-TUNED MODEL: {fm['model_name']}]")
    print(f"  JSON Validity Rate   : {fm['json_validity_rate']*100:.1f}%")
    print(f"  Category Accuracy    : {fm['category_accuracy']*100:.1f}%")
    print(f"  Severity Accuracy    : {fm['severity_accuracy']*100:.1f}%")
    print(f"  Token Overlap / F1   : {fm['rouge_l_f1']:.4f}")
    print(f"  Avg Latency          : {fm['avg_latency_ms']:.2f} ms")

    print(f"\nVerdict: {summary['verdict_summary']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

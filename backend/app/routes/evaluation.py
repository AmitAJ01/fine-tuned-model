"""Evaluation and baseline-vs-finetuned benchmarking router."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from backend.app.core.config import settings
from backend.app.schemas.evaluation import (
    EvaluationComparisonResponse,
    EvaluationRunRequest,
    MetricSummary,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarks"])


@router.get(
    "/latest",
    response_model=EvaluationComparisonResponse,
    summary="Get Latest Evaluation Benchmark Results",
)
async def get_latest_evaluation() -> EvaluationComparisonResponse:
    """Retrieve the most recent baseline vs. fine-tuned evaluation summary."""
    summary_file = settings.PROJECT_ROOT / "experiments" / "results" / "evaluation_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationComparisonResponse(**data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read evaluation summary: {e}",
            )

    # Return empty / initial benchmark placeholder if no run has occurred yet
    return EvaluationComparisonResponse(
        run_id="pending-first-run",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name="held-out-test-split",
        total_samples=0,
        baseline_metrics=None,
        finetuned_metrics=None,
        sample_comparisons=[],
        verdict_summary="No evaluation run recorded yet. Execute scripts/run_evaluation.py to produce benchmark report.",
    )


@router.get(
    "/comparison",
    summary="Get Comprehensive Head-to-Head Benchmark Metrics",
)
async def get_full_comparison_metrics(version: str = "final") -> dict:
    """Retrieve full comparative metrics, baseline, fine-tuned, and training details."""
    final_dir = settings.PROJECT_ROOT / "experiments" / "final_run"
    legacy_dir = settings.PROJECT_ROOT / "experiments" / "results"

    use_final = (version == "final") and (final_dir / "comparison_metrics.json").exists()
    target_dir = final_dir if use_final else legacy_dir

    comp_file = target_dir / "comparison_metrics.json"
    base_file = target_dir / "baseline_metrics.json"
    ft_file = target_dir / "finetuned_metrics.json"
    train_file = target_dir / "training_metrics.json"
    latency_file = target_dir / "fair_latency_benchmark.json"

    res = {
        "status": "ready" if comp_file.exists() else "pending",
        "benchmark_version": "final_run_task7" if use_final else "historical_task6",
        "comparison": None,
        "baseline": None,
        "finetuned": None,
        "training": None,
        "fair_latency": None,
    }

    try:
        if comp_file.exists():
            with open(comp_file, "r", encoding="utf-8") as f:
                res["comparison"] = json.load(f)
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                res["baseline"] = json.load(f)
        if ft_file.exists():
            with open(ft_file, "r", encoding="utf-8") as f:
                res["finetuned"] = json.load(f)
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                res["training"] = json.load(f)
        if latency_file.exists():
            with open(latency_file, "r", encoding="utf-8") as f:
                res["fair_latency"] = json.load(f)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading benchmark metrics: {e}",
        )


@router.get(
    "/final-run",
    summary="Get Task 7 Final Controlled Experiment Benchmark Metrics",
)
async def get_final_run_metrics() -> dict:
    """Retrieve Task 7 final run metrics with completion-only loss masking."""
    final_dir = settings.PROJECT_ROOT / "experiments" / "final_run"
    comp_file = final_dir / "comparison_metrics.json"
    base_file = final_dir / "baseline_metrics.json"
    ft_file = final_dir / "finetuned_metrics.json"
    train_file = final_dir / "training_metrics.json"
    latency_file = final_dir / "fair_latency_benchmark.json"

    res = {
        "status": "ready" if comp_file.exists() else "pending",
        "comparison": None,
        "baseline": None,
        "finetuned": None,
        "training": None,
        "fair_latency": None,
    }

    try:
        if comp_file.exists():
            with open(comp_file, "r", encoding="utf-8") as f:
                res["comparison"] = json.load(f)
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                res["baseline"] = json.load(f)
        if ft_file.exists():
            with open(ft_file, "r", encoding="utf-8") as f:
                res["finetuned"] = json.load(f)
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                res["training"] = json.load(f)
        if latency_file.exists():
            with open(latency_file, "r", encoding="utf-8") as f:
                res["fair_latency"] = json.load(f)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading final run metrics: {e}",
        )


@router.post(
    "/run",
    response_model=EvaluationComparisonResponse,
    summary="Trigger Benchmark Evaluation Run",
)
async def trigger_evaluation(request: EvaluationRunRequest) -> EvaluationComparisonResponse:
    """Trigger evaluation comparing baseline and fine-tuned models on test dataset."""
    from ml.evaluation.evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(test_path=request.test_split_path)
    res = evaluator.run_benchmark(sample_limit=request.sample_limit)
    return EvaluationComparisonResponse(**res)

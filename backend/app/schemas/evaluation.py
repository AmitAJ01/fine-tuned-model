"""Evaluation schemas for Baseline vs. Fine-Tuned Model Benchmarks."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationRunRequest(BaseModel):
    """Trigger an evaluation run across the held-out test split."""
    test_split_path: Optional[str] = Field(
        default="ml/data/splits/test.jsonl",
        description="Path to evaluation test split",
    )
    sample_limit: Optional[int] = Field(
        default=None,
        description="Limit the number of samples for rapid benchmarking (null for all)",
    )
    compare_baseline: bool = Field(default=True)
    compare_finetuned: bool = Field(default=True)


class MetricSummary(BaseModel):
    """Summary metrics for a single model configuration."""
    model_name: str
    is_fine_tuned: bool
    total_evaluated: int
    json_validity_rate: float
    category_accuracy: float
    severity_accuracy: float
    rouge_l_f1: float
    avg_latency_ms: float
    tokens_per_second: float


class IncidentComparisonItem(BaseModel):
    """Item-level comparison for a single incident in the test set."""
    incident_id: str
    input_log: str
    ground_truth: Dict
    baseline_output: Optional[Dict] = None
    finetuned_output: Optional[Dict] = None
    baseline_latency_ms: Optional[float] = None
    finetuned_latency_ms: Optional[float] = None
    baseline_json_valid: bool = False
    finetuned_json_valid: bool = False


class EvaluationComparisonResponse(BaseModel):
    """Aggregate benchmark comparison between Baseline and Fine-Tuned models."""
    run_id: str
    timestamp: str
    dataset_name: str
    total_samples: int
    baseline_metrics: Optional[MetricSummary] = None
    finetuned_metrics: Optional[MetricSummary] = None
    sample_comparisons: List[IncidentComparisonItem] = Field(default_factory=list)
    verdict_summary: str

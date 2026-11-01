"""Phase 5: Automated API Verification Tests for Backend Routers and Schemas."""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.routes.health import health_check, hardware_status
from backend.app.routes.evaluation import get_full_comparison_metrics
from backend.app.routes.incident import diagnose
from backend.app.schemas.incident import (
    IncidentCategory,
    IncidentDiagnosisResponse,
    IncidentReportPayload,
    IncidentRequest,
    ModelTarget,
    SeverityLevel,
)
from backend.app.services.inference_service import InferenceService


def run_coroutine(coro):
    """Execute coroutines directly without needing background asyncio event loop runner."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_health_check_endpoint():
    """Verify GET /api/v1/health returns healthy status and model configuration."""
    resp = run_coroutine(health_check())
    assert resp["status"] == "healthy"
    assert resp["service"] == "FineTuneOps"
    assert "base_model" in resp
    assert "finetuned_adapter_path" in resp


def test_evaluation_comparison_endpoint():
    """Verify GET /api/v1/evaluation/comparison returns full comparative benchmark metrics."""
    resp = run_coroutine(get_full_comparison_metrics(version="final"))
    assert isinstance(resp, dict)
    assert resp["status"] == "ready"
    assert resp["benchmark_version"] == "final_run_task7"
    assert "comparison" in resp
    comp = resp["comparison"]
    assert "metrics_comparison" in comp
    assert "fault_taxonomy_accuracy" in comp["metrics_comparison"]
    assert "task8_integrity_audit" in comp
    assert comp["total_evaluated_samples"] == 162


def test_incident_diagnosis_baseline_routing():
    """Verify POST /api/v1/incident/diagnose with baseline model target in simulation/mock mode."""
    service = InferenceService()
    req = IncidentRequest(
        log_text="kernel: Out of memory: Killed process 4123 (node) total-vm:4194304kB, anon-rss:2097152kB",
        system_context="E-commerce boutique microservice on Kubernetes",
        model_target=ModelTarget.BASELINE,
        max_tokens=768,
        mock=True,
    )
    resp = run_coroutine(diagnose(req, service=service))
    assert isinstance(resp, IncidentDiagnosisResponse)
    assert resp.is_fine_tuned is False
    assert resp.json_valid is True
    assert resp.report is not None
    assert resp.report.incident_category in [
        IncidentCategory.INFRASTRUCTURE,
        IncidentCategory.APPLICATION,
        IncidentCategory.KUBERNETES,
    ]
    assert resp.error_detail is None


def test_incident_diagnosis_finetuned_routing():
    """Verify POST /api/v1/incident/diagnose with fine-tuned model target in simulation/mock mode."""
    service = InferenceService()
    req = IncidentRequest(
        log_text="postgres: FATAL: remaining connection slots are reserved for non-replication superuser connections",
        system_context="Order database PostgreSQL",
        model_target=ModelTarget.FINETUNED,
        max_tokens=768,
        mock=True,
    )
    resp = run_coroutine(diagnose(req, service=service))
    assert isinstance(resp, IncidentDiagnosisResponse)
    assert resp.is_fine_tuned is True
    assert resp.json_valid is True
    assert resp.report is not None
    assert resp.report.incident_category == IncidentCategory.DATABASE
    assert resp.error_detail is None


def test_incident_diagnosis_error_handling_and_no_silent_fabrication():
    """Verify that when an engine produces malformed/truncated output, no fake report is fabricated."""
    from ml.inference.engine import IncidentInferenceEngine

    engine = IncidentInferenceEngine()
    broken_output = "Unterminated model generation that was cut off..."

    resp = IncidentDiagnosisResponse(
        report=None,
        model_used="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        is_fine_tuned=False,
        latency_ms=10.0,
        raw_response=broken_output,
        json_valid=False,
        error_detail="JSONDecodeError: Unterminated string (generation truncated at token limit)",
    )
    assert resp.json_valid is False
    assert resp.report is None
    assert resp.error_detail is not None

"""Phase 4 Tests: Robust JSON Generation, Extraction, and Schema Contract Tests."""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schemas.incident import (
    IncidentCategory,
    IncidentDiagnosisResponse,
    IncidentReportPayload,
    SeverityLevel,
)
from ml.inference.prompts import extract_json_payload, extract_json_payload_with_diagnostics


def test_valid_json_extraction():
    """Verify extraction and schema validation of completely valid JSON."""
    raw = """Here is the incident diagnosis:
```json
{
  "incident_category": "Kubernetes / Container Orchestration",
  "severity_level": "Critical",
  "root_cause_analysis": "OOM killer killed container app/cartservice.",
  "contributing_factors": ["High memory spike"],
  "recommended_debugging_steps": ["kubectl top pod"],
  "suggested_remediation_actions": ["Increase memory limit"]
}
```
Diagnostic complete."""
    parsed, is_valid, err = extract_json_payload_with_diagnostics(raw)
    assert is_valid is True
    assert err is None
    assert parsed["incident_category"] == "Kubernetes / Container Orchestration"

    # Validate Pydantic schema
    report = IncidentReportPayload(**parsed)
    assert report.incident_category == IncidentCategory.KUBERNETES
    assert report.severity_level == SeverityLevel.CRITICAL


def test_truncated_json_detection():
    """Verify that mid-generation truncated JSON returns is_valid=False with clear error."""
    # Simulate an output truncated at 384 tokens mid-string
    truncated_raw = """{
  "incident_category": "Kubernetes / Container Orchestration",
  "severity_level": "Critical",
  "root_cause_analysis": "Root cause identified as 'node_selector_mismatch' in component 'app/recommendationservice'.",
  "contributing_factors": ["Pod scheduled on worker-01"],
  "recommended_debugging_steps": ["kubectl describe pod recommendationservice", "kubectl get events -n boutique --sort-by='.metadata.c"""

    parsed, is_valid, err = extract_json_payload_with_diagnostics(truncated_raw)
    assert is_valid is False
    assert parsed is None
    assert "truncated" in err.lower() or "unterminated string" in err.lower()

    # Ensure IncidentDiagnosisResponse handles it without fabricating fields
    resp = IncidentDiagnosisResponse(
        report=None,
        model_used="Qwen/Qwen2.5-Coder-1.5B-Instruct + LoRA",
        is_fine_tuned=True,
        latency_ms=1200.0,
        raw_response=truncated_raw,
        json_valid=False,
        error_detail=err,
    )
    assert resp.report is None
    assert resp.json_valid is False
    assert resp.error_detail is not None
    assert "truncated" in resp.error_detail.lower() or "unterminated string" in resp.error_detail.lower()


def test_invalid_syntax_json():
    """Verify that completely broken non-JSON output returns is_valid=False."""
    bad_raw = "This is a purely conversational error without any JSON braces or format."
    parsed, is_valid, err = extract_json_payload_with_diagnostics(bad_raw)
    assert is_valid is False
    assert parsed is None
    assert err is not None


def test_missing_required_fields_schema_validation():
    """Verify that JSON missing mandatory fields fails schema validation."""
    incomplete_json = {
        "incident_category": "Application",
        # "severity_level" is missing
        "root_cause_analysis": "Database lock contention",
    }
    with pytest.raises(Exception):
        IncidentReportPayload(**incomplete_json)


def test_diagnostic_768_cases_integrity():
    """Verify that diagnostic 768 results contain 8/8 valid schema-compliant cases."""
    import json
    from pathlib import Path

    diag_file = Path(__file__).resolve().parent.parent / "experiments/final_run/diagnostic_768_json_results.json"
    assert diag_file.exists(), "Diagnostic 768 file must exist"

    data = json.load(open(diag_file, "r", encoding="utf-8"))
    assert data["total_tested"] == 8
    assert data["valid_json_count"] == 8
    assert data["valid_json_pct"] == 100.0
    assert data["schema_compliant_count"] == 8

    for r in data["results"]:
        assert r["json_valid"] is True
        assert r["schema_compliant"] is True
        assert 384 < r["diagnostic_completion_tokens"] < 768

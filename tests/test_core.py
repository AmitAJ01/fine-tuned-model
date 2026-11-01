"""Unit tests for FineTuneOps core components."""

import sys
from pathlib import Path
import pytest

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.incident import (
    IncidentCategory,
    IncidentDiagnosisResponse,
    IncidentReportPayload,
    IncidentRequest,
    ModelTarget,
    SeverityLevel,
)
from ml.evaluation.metrics import compute_incident_metrics, compute_token_f1
from ml.inference.prompts import build_diagnosis_prompt, extract_json_payload
from ml.training.dataset import DevOpsIncidentDataset


def test_incident_schemas():
    """Verify incident schema validation and enum parsing."""
    sample = {
        "incident_category": "Infrastructure",
        "severity_level": "SEV-1",
        "root_cause_analysis": "OOM-killer invoked on billing pod.",
        "contributing_factors": ["High traffic spike"],
        "recommended_debugging_steps": ["kubectl top pod"],
        "suggested_remediation_actions": ["Increase memory limit to 8Gi"],
    }
    payload = IncidentReportPayload(**sample)
    assert payload.incident_category == IncidentCategory.INFRASTRUCTURE
    assert payload.severity_level == SeverityLevel.SEV1
    assert len(payload.recommended_debugging_steps) == 1


def test_prompt_builder_and_json_extraction():
    """Verify prompt formatting and robust JSON extraction."""
    log = "kernel: Out of memory: Killed process 1234 (node)"
    prompt = build_diagnosis_prompt(log, system_context="AWS ECS cluster")
    assert "Incident Logs / Description:" in prompt
    assert "AWS ECS cluster" in prompt

    raw_output = """Here is the analysis:
    ```json
    {
      "incident_category": "Infrastructure",
      "severity_level": "SEV-1",
      "root_cause_analysis": "OOM killer killed node container.",
      "contributing_factors": [],
      "recommended_debugging_steps": [],
      "suggested_remediation_actions": []
    }
    ```
    Hope this helps!"""

    extracted, valid = extract_json_payload(raw_output)
    assert valid is True
    assert extracted["incident_category"] == "Infrastructure"


def test_metric_computations():
    """Verify token F1 and aggregate metrics calculation."""
    f1 = compute_token_f1("Database connection timeout", "Database connection pool timeout occurred")
    assert f1 > 0.6

    preds = [
        {
            "json_valid": True,
            "report": {
                "incident_category": "Infrastructure",
                "severity_level": "SEV-1",
                "root_cause_analysis": "Pod OOMKilled",
            },
        }
    ]
    refs = [
        {
            "incident_category": "Infrastructure",
            "severity_level": "SEV-1",
            "root_cause_analysis": "Pod memory limit exceeded triggering OOMKilled",
        }
    ]
    metrics = compute_incident_metrics(preds, refs, [120.0])
    assert metrics["json_validity_rate"] == 1.0
    assert metrics["category_accuracy"] == 1.0
    assert metrics["severity_accuracy"] == 1.0


def test_dataset_chat_formatting():
    """Verify dataset converter properly wraps messages for SFT chat template."""
    log = "504 Gateway Timeout"
    target = {
        "incident_category": "Network",
        "severity_level": "SEV-2",
        "root_cause_analysis": "CoreDNS timeout",
        "contributing_factors": ["Policy dropped UDP 53"],
        "recommended_debugging_steps": ["dig @10.96.0.10"],
        "suggested_remediation_actions": ["Scale CoreDNS"],
    }
    sample = DevOpsIncidentDataset.format_sample(log, target, system_context="K8s cluster")
    messages = sample["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_cloud_ops_bench_dataset_splits_exist_and_valid():
    """Verify Cloud-OpsBench splits exist, match 754 total, and adhere to chat format."""
    import json
    root = Path(__file__).resolve().parent.parent
    train_path = root / "ml" / "data" / "splits" / "train.jsonl"
    val_path = root / "ml" / "data" / "splits" / "val.jsonl"
    test_path = root / "ml" / "data" / "splits" / "test.jsonl"

    assert train_path.exists(), "train.jsonl missing"
    assert val_path.exists(), "val.jsonl missing"
    assert test_path.exists(), "test.jsonl missing"

    counts = {}
    for p, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        with open(p, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
            counts[name] = len(lines)
            assert len(lines) > 0, f"{name}.jsonl is empty"
            sample = lines[0]
            assert "messages" in sample
            assert len(sample["messages"]) == 3
            assert sample["messages"][0]["role"] == "system"
            assert sample["messages"][1]["role"] == "user"
            assert sample["messages"][2]["role"] == "assistant"

    total = counts["train"] + counts["val"] + counts["test"]
    assert total == 754, f"Expected 754 records, got {total}"


def test_zero_scenario_leakage_between_splits():
    """Verify strictly zero scenario leakage across train, val, and test splits."""
    import json
    root = Path(__file__).resolve().parent.parent
    splits = {}
    for name in ["train", "val", "test"]:
        path = root / "ml" / "data" / "splits" / f"{name}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            splits[name] = [json.loads(l) for l in f if l.strip()]

    def extract_group(item):
        meta = item.get("metadata", {})
        ref = meta.get("source_reference", "")
        sys_name = ref.split("(")[-1].split(")")[0].split("/")[0] if "(" in ref else "unknown"
        return f"{sys_name}::{meta.get('category')}::{meta.get('root_cause')}"

    train_groups = set(extract_group(x) for x in splits["train"])
    val_groups = set(extract_group(x) for x in splits["val"])
    test_groups = set(extract_group(x) for x in splits["test"])

    assert len(train_groups & val_groups) == 0, "Leakage between train and val"
    assert len(train_groups & test_groups) == 0, "Leakage between train and test"
    assert len(val_groups & test_groups) == 0, "Leakage between val and test"


def test_baseline_ground_truth_isolation():
    """Verify strictly zero target leakage into input prompt across held-out test split."""
    import json
    root = Path(__file__).resolve().parent.parent
    test_path = root / "ml" / "data" / "splits" / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]

    assert len(cases) == 148, f"Expected 148 test cases, found {len(cases)}"
    for case in cases:
        user_prompt = case["messages"][1]["content"].lower()
        root_cause = case["metadata"]["root_cause"].lower()
        # Verify exact root cause label does not appear in user input prompt
        assert root_cause not in user_prompt, f"Target leakage of '{root_cause}' in {case['id']}"


def test_baseline_json_parsing_and_schema_validation():
    """Verify JSON extractor handles clean JSON, markdown fences, and bad syntax."""
    clean = '{"incident_category": "Database", "severity_level": "SEV-2", "root_cause_analysis": "Connection pool timeout"}'
    parsed, ok = extract_json_payload(clean)
    assert ok is True
    assert parsed["incident_category"] == "Database"

    fenced = '```json\n{"incident_category": "Application", "root_cause_analysis": "Null pointer in checkout"}\n```'
    parsed, ok = extract_json_payload(fenced)
    assert ok is True
    assert parsed["incident_category"] == "Application"

    bad = "This is not json at all."
    parsed, ok = extract_json_payload(bad)
    assert ok is False
    assert parsed is None


def test_root_cause_metric_calculations():
    """Verify exact match, normalized accuracy, and macro F1 metric calculations."""
    from ml.evaluation.metrics import compute_root_cause_metrics, compute_fault_taxonomy_accuracy

    predictions = [
        {"incident_category": "Application", "root_cause_analysis": "Root cause identified as 'code_excessive_file_reads' in checkout"},
        {"incident_category": "Kubernetes", "root_cause_analysis": "Missing service account in adservice"},
        {"incident_category": "Network", "root_cause_analysis": "High CPU throttle and memory exhaustion"},
    ]
    references = [
        {"incident_category": "Application", "root_cause": "code_excessive_file_reads"},
        {"incident_category": "Kubernetes", "root_cause": "missing_service_account"},
        {"incident_category": "Network", "root_cause": "pod_network_delay"},
    ]

    cat_acc = compute_fault_taxonomy_accuracy(predictions, references)
    assert cat_acc == 1.0

    rc_metrics = compute_root_cause_metrics(predictions, references)
    assert rc_metrics["exact_match_accuracy"] >= 0.33  # First matches exact quoted string
    assert rc_metrics["normalized_accuracy"] >= 0.66  # First two match normalized
    assert "macro_f1" in rc_metrics
    assert 0.0 <= rc_metrics["macro_f1"] <= 1.0


def test_evidence_grounding_score_rubric():
    """Verify transparent evidence grounding scoring rubric."""
    from ml.evaluation.metrics import compute_evidence_grounding_score

    input_log = "Error in app/cartservice: Redis connection timed out on port 6379 with latency spike."
    grounded_report = {
        "incident_category": "Database",
        "root_cause_analysis": "cartservice failed to connect to Redis port 6379 due to connection timeout",
        "contributing_factors": ["High redis latency"],
        "recommended_debugging_steps": ["Check redis pod"],
    }
    score = compute_evidence_grounding_score(grounded_report, input_log)
    assert score >= 0.60, f"Expected high grounding score, got {score}"

    ungrounded_report = {
        "incident_category": "Security",
        "root_cause_analysis": "Unauthorized SSH brute force on external bastion host",
        "contributing_factors": [],
        "recommended_debugging_steps": [],
    }
    low_score = compute_evidence_grounding_score(ungrounded_report, input_log)
    assert low_score < score, "Ungrounded report should score lower"


def test_semantic_root_cause_score_rubric():
    """Verify semantic root cause evaluation method and tiered scoring rubric."""
    from ml.evaluation.metrics import compute_semantic_root_cause_score

    # Level 4: Exact match = 1.0
    exact_score = compute_semantic_root_cause_score(
        "Root cause identified as 'code_excessive_file_reads' in checkout",
        "code_excessive_file_reads",
        "app/checkoutservice",
    )
    assert exact_score == 1.0

    # Level 3: Canonical tokens containment = 0.8
    token_score = compute_semantic_root_cause_score(
        "The checkout service suffered from excessive file reads in the source code.",
        "code_excessive_file_reads",
        "app/checkoutservice",
    )
    assert token_score == 0.80

    # Level 2: Component + core failure domain = 0.50
    component_score = compute_semantic_root_cause_score(
        "checkoutservice suffered from file I/O operations",
        "code_excessive_file_reads",
        "app/checkoutservice",
    )
    assert component_score == 0.50

    # Level 0: Completely unrelated = 0.0
    unrelated_score = compute_semantic_root_cause_score(
        "Postgres connection timeout on payment pod",
        "code_excessive_file_reads",
        "app/checkoutservice",
    )
    assert unrelated_score == 0.0


def test_comparative_eval_bootstrap_confidence_interval():
    """Verify bootstrap confidence interval computation properties."""
    from scripts.run_finetuned_evaluation import compute_bootstrap_confidence_interval

    # If both models have identical scores, mean difference is 0 and CI is [0, 0]
    identical_a = [1.0] * 50 + [0.0] * 50
    identical_b = [1.0] * 50 + [0.0] * 50
    ci_identical = compute_bootstrap_confidence_interval(identical_a, identical_b, n_bootstraps=500, seed=42)
    assert ci_identical["mean_difference"] == 0.0
    assert ci_identical["ci_lower"] <= 0.0 <= ci_identical["ci_upper"]

    # When model B strictly dominates model A, CI is strictly positive
    scores_a = [0.0] * 100
    scores_b = [1.0] * 50 + [0.0] * 50
    ci_res = compute_bootstrap_confidence_interval(scores_a, scores_b, n_bootstraps=500, seed=42)
    assert ci_res["mean_difference"] > 0.45
    assert ci_res["ci_lower"] > 0.35
    assert ci_res["ci_upper"] < 0.65


def test_baseline_predictions_count_and_hashes():
    """Verify exactly 148 baseline predictions exist and match test split input hashes."""
    import hashlib
    import json
    root = Path(__file__).resolve().parent.parent
    test_path = root / "ml" / "data" / "splits" / "test.jsonl"
    base_preds_path = root / "experiments" / "results" / "baseline_predictions.jsonl"

    assert test_path.exists(), "test.jsonl missing"
    assert base_preds_path.exists(), "baseline_predictions.jsonl missing"

    with open(test_path, "r", encoding="utf-8") as f:
        tests = [json.loads(l) for l in f if l.strip()]
    with open(base_preds_path, "r", encoding="utf-8") as f:
        preds = [json.loads(l) for l in f if l.strip()]

    assert len(tests) == 148
    assert len(preds) == 148

    for t, p in zip(tests, preds):
        assert t["id"] == p["case_id"]
        messages = t.get("messages", [])
        sys_p = messages[0]["content"] if len(messages) > 0 else ""
        usr_p = messages[1]["content"] if len(messages) > 1 else ""
        expected_hash = hashlib.sha256(f"{sys_p}\n---\n{usr_p}".encode("utf-8")).hexdigest()
        assert p["input_hash"] == expected_hash


def test_finetuned_predictions_parity_and_count():
    """Verify exactly 148 fine-tuned predictions exist and match baseline case IDs and hashes."""
    import json
    root = Path(__file__).resolve().parent.parent
    base_preds_path = root / "experiments" / "results" / "baseline_predictions.jsonl"
    ft_preds_path = root / "experiments" / "results" / "finetuned_predictions.jsonl"

    assert base_preds_path.exists(), "baseline_predictions.jsonl missing"
    assert ft_preds_path.exists(), "finetuned_predictions.jsonl missing"

    with open(base_preds_path, "r", encoding="utf-8") as f:
        base_preds = [json.loads(l) for l in f if l.strip()]
    with open(ft_preds_path, "r", encoding="utf-8") as f:
        ft_preds = [json.loads(l) for l in f if l.strip()]

    assert len(base_preds) == 148
    assert len(ft_preds) == 148

    for bp, fp in zip(base_preds, ft_preds):
        assert bp["case_id"] == fp["case_id"]
        assert bp["input_hash"] == fp["input_hash"]
        assert fp["json_valid"] is True
        assert fp["parsing_status"] == "schema_compliant"






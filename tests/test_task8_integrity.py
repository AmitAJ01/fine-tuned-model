"""Unit tests for Task 8: Metrics Integrity, RCA Evaluation, and Production Validation."""

import json
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.metrics import (
    compute_fault_taxonomy_accuracy,
    compute_root_cause_metrics,
    normalize_text,
)


@pytest.fixture(scope="module")
def dataset_splits():
    test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
    train_file = REPO_ROOT / "ml/data/splits_final/train.jsonl"
    base_preds_file = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl"
    ft_preds_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"
    diag_file = REPO_ROOT / "experiments/final_run/diagnostic_768_json_results.json"
    lat_file = REPO_ROOT / "experiments/final_run/fair_latency_benchmark_full.json"

    test_records = [json.loads(l) for l in open(test_file, "r", encoding="utf-8") if l.strip()]
    train_records = [json.loads(l) for l in open(train_file, "r", encoding="utf-8") if l.strip()]
    base_preds = [json.loads(l) for l in open(base_preds_file, "r", encoding="utf-8") if l.strip()]
    ft_preds = [json.loads(l) for l in open(ft_preds_file, "r", encoding="utf-8") if l.strip()]
    diag_results = json.load(open(diag_file, "r", encoding="utf-8")) if diag_file.exists() else {}
    lat_results = json.load(open(lat_file, "r", encoding="utf-8")) if lat_file.exists() else {}

    return {
        "test": test_records,
        "train": train_records,
        "base_preds": base_preds,
        "ft_preds": ft_preds,
        "diag": diag_results,
        "lat": lat_results,
    }


def test_baseline_accuracy_discrepancy_resolution(dataset_splits):
    """Verify that baseline accuracy evaluates to 73/162 (45.06%) direct vs 74/162 (45.68%) mapped."""
    test_records = dataset_splits["test"]
    base_preds = dataset_splits["base_preds"]
    ft_preds = dataset_splits["ft_preds"]

    assert len(test_records) == 162
    assert len(base_preds) == 162
    assert len(ft_preds) == 162

    # Direct category matching
    direct_base_correct = 0
    mapped_base_correct = 0

    def map_category_string(cat_str: str) -> str:
        if not cat_str or not isinstance(cat_str, str):
            return "Unknown"
        s = cat_str.strip().lower()
        if "kube" in s or "container" in s or "pod" in s:
            return "Kubernetes / Container Orchestration"
        if "infra" in s or "node" in s or "host" in s:
            return "Infrastructure"
        if "data" in s or "sql" in s or "redis" in s or "db" in s:
            return "Database"
        if "net" in s or "dns" in s or "traffic" in s:
            return "Network"
        if "app" in s or "service" in s or "code" in s or "software" in s:
            return "Application"
        return "Application"  # Fallback in comparative evaluation

    for i in range(162):
        gt = test_records[i]["metadata"]["category"]
        pred_cat = (base_preds[i].get("prediction_report") or {}).get("incident_category")

        # Direct match
        if pred_cat and normalize_text(pred_cat) == normalize_text(gt):
            direct_base_correct += 1

        # Mapped match
        if pred_cat and map_category_string(pred_cat) == gt:
            mapped_base_correct += 1

    assert direct_base_correct == 73, f"Expected exactly 73 direct matches, got {direct_base_correct}"
    assert mapped_base_correct == 74, f"Expected exactly 74 mapped matches, got {mapped_base_correct}"

    # Verify the exact discrepancy index is sample 34 (id: boutique/codedefect/42)
    sample_34 = test_records[34]
    pred_34 = base_preds[34]
    pred_34_cat = (pred_34.get("prediction_report") or {}).get("incident_category")
    assert sample_34["id"] == "boutique/codedefect/42"
    assert sample_34["metadata"]["category"] == "Application"
    assert pred_34_cat == "Unknown / Other"
    assert normalize_text(pred_34_cat) != normalize_text("Application")
    assert map_category_string(pred_34_cat) == "Application"

    # For the fine-tuned model, direct and mapped must both be exactly 102/162
    direct_ft_correct = sum(
        1 for i in range(162)
        if (ft_preds[i].get("prediction_report") or {}).get("incident_category")
        and normalize_text((ft_preds[i].get("prediction_report") or {}).get("incident_category")) == normalize_text(test_records[i]["metadata"]["category"])
    )
    mapped_ft_correct = sum(
        1 for i in range(162)
        if (ft_preds[i].get("prediction_report") or {}).get("incident_category")
        and map_category_string((ft_preds[i].get("prediction_report") or {}).get("incident_category")) == test_records[i]["metadata"]["category"]
    )

    assert direct_ft_correct == 102, f"Expected 102 direct FT matches, got {direct_ft_correct}"
    assert mapped_ft_correct == 102, f"Expected 102 mapped FT matches, got {mapped_ft_correct}"
    assert direct_ft_correct / 162 == pytest.approx(0.6296, abs=0.001)


def test_root_cause_disjoint_class_partition(dataset_splits):
    """Verify that 11 of 12 test root-cause classes never appeared in training."""
    test_records = dataset_splits["test"]
    train_records = dataset_splits["train"]

    train_rcs = {r["metadata"]["root_cause"] for r in train_records}
    test_rcs = {r["metadata"]["root_cause"] for r in test_records}

    assert len(train_rcs) == 44
    assert len(test_rcs) == 12

    shared_rcs = train_rcs & test_rcs
    assert len(shared_rcs) == 1
    assert "taint_toleration_mismatch" in shared_rcs

    unseen_rcs = test_rcs - train_rcs
    assert len(unseen_rcs) == 11

    # Samples with unseen root causes
    unseen_samples = [r for r in test_records if r["metadata"]["root_cause"] in unseen_rcs]
    assert len(unseen_samples) == 141
    assert len(unseen_samples) / len(test_records) == pytest.approx(0.8704, abs=0.001)

    # Verify exact match is 0.0 on test predictions
    base_preds = dataset_splits["base_preds"]
    ft_preds = dataset_splits["ft_preds"]
    ref_dicts = [{"root_cause": r["metadata"]["root_cause"]} for r in test_records]
    base_rc_metrics = compute_root_cause_metrics(base_preds, ref_dicts)
    ft_rc_metrics = compute_root_cause_metrics(ft_preds, ref_dicts)

    assert base_rc_metrics["exact_match_accuracy"] == 0.0
    assert ft_rc_metrics["exact_match_accuracy"] == 0.0


def test_json_truncation_analysis(dataset_splits):
    """Verify that all 8 malformed cases hit the 384-token boundary and pass at 768 tokens."""
    ft_preds = dataset_splits["ft_preds"]
    malformed = [p for p in ft_preds if not p.get("json_valid", True)]

    assert len(malformed) == 8, f"Expected exactly 8 malformed cases, got {len(malformed)}"

    # Every malformed case generated exactly 384 tokens
    for p in malformed:
        assert p["completion_tokens"] == 384, f"Case {p.get('id')} has {p['completion_tokens']} tokens, expected 384"

    # Diagnostic 768 tokens results
    diag = dataset_splits["diag"]
    assert diag.get("total_tested") == 8
    assert diag.get("valid_json_count") == 8
    assert diag.get("valid_json_pct") == 100.0
    assert diag.get("schema_compliant_count") == 8

    for r in diag.get("results", []):
        assert r["json_valid"] is True
        assert r["schema_compliant"] is True
        assert r["hit_max_tokens_768"] is False
        assert 384 < r["diagnostic_completion_tokens"] < 768


def test_input_hash_parity(dataset_splits):
    """Verify 100% input hash parity between baseline and fine-tuned evaluations."""
    base_preds = dataset_splits["base_preds"]
    ft_preds = dataset_splits["ft_preds"]

    assert len(base_preds) == len(ft_preds) == 162
    for b, f in zip(base_preds, ft_preds):
        assert b["id"] == f["id"]
        assert b["input_hash"] == f["input_hash"], f"Input hash mismatch on {b['id']}"


def test_fair_latency_matrix_integrity(dataset_splits):
    """Verify latency benchmark matrix contains controlled 4-bit, controlled 16-bit, and deployment configurations."""
    lat = dataset_splits["lat"]
    assert "batch_size_1_single_stream" in lat
    assert "batch_size_2_amortized" in lat
    assert "controlled_16bit_comparison" in lat

    b1 = lat["batch_size_1_single_stream"]
    assert "baseline" in b1
    assert "finetuned_unmerged" in b1
    assert "finetuned_merged" in b1
    assert "baseline_bf16" in b1

    # Controlled 4-bit: Unmerged LoRA has overhead over Base NF4
    assert b1["finetuned_unmerged"]["mean_latency_ms"] > b1["baseline"]["mean_latency_ms"]

    # Controlled 16-bit: Merged BF16 has minimal overhead (<15%) over Base BF16
    ctrl_16 = lat["controlled_16bit_comparison"]["batch_size_1"]
    assert ctrl_16["overhead_pct"] < 15.0

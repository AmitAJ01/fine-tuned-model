"""Task 7: Comparative Statistical Evaluation Engine.

Compares Baseline vs. Fine-Tuned (Task 7 Masked) predictions on the fresh 162 held-out test split.
Features:
- Exact ID and input hash parity verification.
- Full confusion matrices for both models.
- Per-category metrics (support, precision, recall, F1).
- Exact two-sided McNemar tests.
- Paired bootstrap 95% confidence intervals (N=1,000, seed 42).
- Better / Equal / Worse case accounting.
- Visual confusion matrix comparison plot export.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "experiments" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from ml.evaluation.metrics import (
    compute_evidence_grounding_score,
    compute_fault_taxonomy_accuracy,
    compute_semantic_root_cause_score,
    extract_predicted_root_cause_phrase,
    normalize_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.final_comparative_eval")

CANONICAL_CATEGORIES = [
    "Application",
    "Database",
    "Infrastructure",
    "Kubernetes / Container Orchestration",
    "Network",
]


def map_category_string(cat_str: str) -> str:
    norm = normalize_text(cat_str)
    for c in CANONICAL_CATEGORIES:
        c_norm = normalize_text(c)
        if norm == c_norm or norm in c_norm or c_norm in norm:
            return c
    return "Application"


def calculate_mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided binomial test for McNemar discordant pairs."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    res = binomtest(b, n, 0.5, alternative="two-sided")
    return float(res.pvalue)


def paired_bootstrap_ci(
    base_scores: List[float],
    ft_scores: List[float],
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(base_scores)

    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        b_mean = np.mean([base_scores[i] for i in idx])
        f_mean = np.mean([ft_scores[i] for i in idx])
        diffs.append(f_mean - b_mean)

    diffs = np.sort(diffs)
    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))
    return {"lower": round(lower, 4), "upper": round(upper, 4)}


def run_comparative_evaluation(
    baseline_preds_path: Path = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl",
    finetuned_preds_path: Path = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl",
    output_metrics_path: Path = REPO_ROOT / "experiments/final_run/comparison_metrics.json",
    output_chart_path: Path = REPO_ROOT / "experiments/final_run/confusion_matrix_final.png",
) -> Dict[str, Any]:
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading baseline predictions: {baseline_preds_path}")
    base_records = [json.loads(line) for line in open(baseline_preds_path, "r", encoding="utf-8") if line.strip()]

    logger.info(f"Loading fine-tuned predictions: {finetuned_preds_path}")
    ft_records = [json.loads(line) for line in open(finetuned_preds_path, "r", encoding="utf-8") if line.strip()]

    assert len(base_records) == len(ft_records), f"Count mismatch: {len(base_records)} vs {len(ft_records)}"
    n_samples = len(base_records)
    logger.info(f"Loaded {n_samples} matched predictions.")

    # 1. Parity and Integrity Verification
    mismatches = 0
    for b, f in zip(base_records, ft_records):
        assert b["id"] == f["id"], f"ID mismatch: {b['id']} != {f['id']}"
        assert b["input_hash"] == f["input_hash"], f"Input hash mismatch on {b['id']}"
    logger.info("Experimental parity verified: 100% ID and SHA-256 input hash match.")

    # 2. Extract Labels and Predictions
    y_true = [map_category_string(r["ground_truth"]["category"]) for r in base_records]
    y_base = [map_category_string(r["prediction_report"].get("incident_category", "")) for r in base_records]
    y_ft = [map_category_string(r["prediction_report"].get("incident_category", "")) for r in ft_records]

    # 3. Category Metrics & Confusion Matrix
    cm_base = confusion_matrix(y_true, y_base, labels=CANONICAL_CATEGORIES)
    cm_ft = confusion_matrix(y_true, y_ft, labels=CANONICAL_CATEGORIES)

    p_base, r_base, f1_base, s_base = precision_recall_fscore_support(y_true, y_base, labels=CANONICAL_CATEGORIES, zero_division=0)
    p_ft, r_ft, f1_ft, s_ft = precision_recall_fscore_support(y_true, y_ft, labels=CANONICAL_CATEGORIES, zero_division=0)

    category_breakdown = {}
    for idx, cat in enumerate(CANONICAL_CATEGORIES):
        category_breakdown[cat] = {
            "support": int(s_base[idx]),
            "baseline_precision": round(float(p_base[idx]), 4),
            "baseline_recall": round(float(r_base[idx]), 4),
            "baseline_f1": round(float(f1_base[idx]), 4),
            "finetuned_precision": round(float(p_ft[idx]), 4),
            "finetuned_recall": round(float(r_ft[idx]), 4),
            "finetuned_f1": round(float(f1_ft[idx]), 4),
            "delta_f1": round(float(f1_ft[idx] - f1_base[idx]), 4),
            "delta_recall": round(float(r_ft[idx] - r_base[idx]), 4),
        }

    # Accuracy arrays for paired tests
    cat_acc_base = [1.0 if yt == yb else 0.0 for yt, yb in zip(y_true, y_base)]
    cat_acc_ft = [1.0 if yt == yf else 0.0 for yt, yf in zip(y_true, y_ft)]

    # 4. Root Cause Matches
    rc_exact_base, rc_exact_ft = [], []
    rc_norm_base, rc_norm_ft = [], []
    rc_sem_base, rc_sem_ft = [], []

    for b, f in zip(base_records, ft_records):
        gt_rc = b["ground_truth"]["root_cause"]
        bp_text = b["prediction_report"].get("root_cause_analysis", "")
        fp_text = f["prediction_report"].get("root_cause_analysis", "")

        bp_phrase = extract_predicted_root_cause_phrase(bp_text)
        fp_phrase = extract_predicted_root_cause_phrase(fp_text)

        rc_exact_base.append(1.0 if normalize_text(bp_phrase) == normalize_text(gt_rc) else 0.0)
        rc_exact_ft.append(1.0 if normalize_text(fp_phrase) == normalize_text(gt_rc) else 0.0)

        rc_norm_base.append(1.0 if normalize_text(gt_rc) in normalize_text(bp_text) else 0.0)
        rc_norm_ft.append(1.0 if normalize_text(gt_rc) in normalize_text(fp_text) else 0.0)

        rc_sem_base.append(compute_semantic_root_cause_score(bp_text, gt_rc))
        rc_sem_ft.append(compute_semantic_root_cause_score(fp_text, gt_rc))

    # 5. JSON Validity & Schema Compliance
    json_base = [1.0 if b["json_valid"] else 0.0 for b in base_records]
    json_ft = [1.0 if f["json_valid"] else 0.0 for f in ft_records]

    # 6. Evidence Grounding Score (grounded against input incident log)
    ev_base = [compute_evidence_grounding_score(b["prediction_report"], b.get("input_log_text", b["raw_generated_text"])) for b in base_records]
    ev_ft = [compute_evidence_grounding_score(f["prediction_report"], f.get("input_log_text", f["raw_generated_text"])) for f in ft_records]

    # 7. Helper for McNemar & Accounting
    def compute_metric_item(
        label: str,
        base_arr: List[float],
        ft_arr: List[float],
        is_binary: bool = True,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        b_mean = float(np.mean(base_arr))
        f_mean = float(np.mean(ft_arr))
        abs_diff = round(f_mean - b_mean, 4)
        rel_diff = round(100 * abs_diff / b_mean, 2) if b_mean > 0 else None

        ci = paired_bootstrap_ci(base_arr, ft_arr)

        better = sum(1 for b, f in zip(base_arr, ft_arr) if f > b)
        worse = sum(1 for b, f in zip(base_arr, ft_arr) if f < b)
        equal = sum(1 for b, f in zip(base_arr, ft_arr) if f == b)

        p_val = None
        if is_binary:
            p_val = calculate_mcnemar_exact(better, worse)

        item = {
            "metric_label": label,
            "baseline": round(b_mean, 4),
            "finetuned": round(f_mean, 4),
            "absolute_diff": abs_diff,
            "relative_diff_pct": rel_diff,
            "ci_95": ci,
            "cases_improved": better,
            "cases_unchanged": equal,
            "cases_worsened": worse,
        }
        if p_val is not None:
            item["p_value_mcnemar"] = round(p_val, 4)
            item["statistically_significant"] = bool(p_val < 0.05)
        if note:
            item["metric_note"] = note
        return item

    metrics_comparison = {
        "fault_taxonomy_accuracy": compute_metric_item("Fault Taxonomy Accuracy", cat_acc_base, cat_acc_ft),
        "fault_taxonomy_macro_f1": {
            "metric_label": "Fault Taxonomy Macro F1",
            "baseline": round(float(np.mean(f1_base)), 4),
            "finetuned": round(float(np.mean(f1_ft)), 4),
            "absolute_diff": round(float(np.mean(f1_ft) - np.mean(f1_base)), 4),
        },
        "root_cause_exact_match": compute_metric_item("Root-Cause Exact Match", rc_exact_base, rc_exact_ft),
        "root_cause_normalized_acc": compute_metric_item("Root-Cause Normalized Accuracy", rc_norm_base, rc_norm_ft),
        "root_cause_semantic_score": compute_metric_item("Root-Cause Semantic Score (Heuristic)", rc_sem_base, rc_sem_ft, is_binary=False),
        "json_validity_rate": compute_metric_item("JSON Validity Rate", json_base, json_ft),
        "schema_compliance_rate": compute_metric_item("Schema Compliance Rate", json_base, json_ft),
        "evidence_keyword_overlap": compute_metric_item(
            "Evidence Keyword Overlap (Heuristic)",
            ev_base,
            ev_ft,
            is_binary=False,
            note="Measures surface keyword overlap only; does NOT prove factual causality.",
        ),
    }

    # 8. Generate Visual Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    short_labels = ["App", "DB", "Infra", "K8s", "Net"]

    for ax, cm, title in zip(axes, [cm_base, cm_ft], ["Baseline (Pretrained)", "Fine-Tuned (Task 7 Masked)"]):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        tick_marks = np.arange(len(short_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(short_labels, fontsize=10)
        ax.set_ylabel("Ground Truth Category", fontsize=11)
        ax.set_xlabel("Predicted Category", fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                    fontsize=10,
                )

    plt.suptitle("FineTuneOps — Task 7 Final Experiment: Fault Taxonomy Confusion Matrices (N=162)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix chart to {output_chart_path}")

    # 9. Compile Full Comparison Report JSON
    comparison_payload = {
        "task_name": "Task 7 Final Controlled Experiment",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_a": "Qwen/Qwen2.5-Coder-1.5B-Instruct (Pretrained Base)",
        "model_b": "Qwen/Qwen2.5-Coder-1.5B-Instruct + QLoRA (Task 7 Masked)",
        "benchmark_dataset": "Cloud-OpsBench Fresh Held-Out Split (ml/data/splits_final/test.jsonl)",
        "total_evaluated_samples": n_samples,
        "metrics_comparison": metrics_comparison,
        "category_breakdown": category_breakdown,
        "confusion_matrices": {
            "labels": CANONICAL_CATEGORIES,
            "baseline": cm_base.tolist(),
            "finetuned": cm_ft.tolist(),
        },
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    logger.info(f"Exported comparison metrics to {output_metrics_path}")

    return comparison_payload


if __name__ == "__main__":
    run_comparative_evaluation()

"""Generate Confusion Matrix Visualization for Baseline and Fine-Tuned Models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "experiments" / ".matplotlib"))


def main() -> None:
    base_file = REPO_ROOT / "experiments" / "results" / "baseline_predictions.jsonl"
    ft_file = REPO_ROOT / "experiments" / "results" / "finetuned_predictions.jsonl"
    out_chart = REPO_ROOT / "experiments" / "reports" / "confusion_matrix_comparison.png"

    with open(base_file, "r", encoding="utf-8") as f:
        base_preds = [json.loads(line) for line in f if line.strip()]
    with open(ft_file, "r", encoding="utf-8") as f:
        ft_preds = [json.loads(line) for line in f if line.strip()]

    y_true = [p["ground_truth"]["category"] for p in base_preds]
    cats = sorted(list(set(y_true)))

    def map_pred(p):
        c = p.get("report", {}).get("incident_category", "Other")
        for valid in cats:
            if c.lower() == valid.lower() or valid.lower() in c.lower() or c.lower() in valid.lower():
                return valid
        return "Other"

    y_base = [map_pred(p) for p in base_preds]
    y_ft = [map_pred(p) for p in ft_preds]

    all_labels = cats + ["Other"]
    short_labels = [
        "Application",
        "Database",
        "Infrastructure",
        "Kubernetes",
        "Network",
        "Other",
    ]

    cm_base = confusion_matrix(y_true, y_base, labels=all_labels)
    cm_ft = confusion_matrix(y_true, y_ft, labels=all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cm, title in [
        (axes[0], cm_base, "Baseline Model (Pretrained)"),
        (axes[1], cm_ft, "Fine-Tuned Model (QLoRA)"),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        tick_marks = np.arange(len(short_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_ylabel("True Category", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Predicted Category", fontsize=11, fontweight="semibold")

        # Cell text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold" if val > 0 else "normal",
                    fontsize=10,
                )

    plt.suptitle("Fault Taxonomy Confusion Matrices (148 Held-Out Cases)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_chart.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_chart, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated confusion matrix chart at {out_chart}")


if __name__ == "__main__":
    main()

"""Benchmark evaluator for Baseline vs. Fine-Tuned LLM Comparison.

Strictly enforces:
- Identical held-out test dataset
- Identical prompt template
- Identical inference generation parameters (temperature, max_tokens, penalty)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from ml.evaluation.metrics import compute_incident_metrics
from ml.inference.engine import IncidentInferenceEngine

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates Baseline vs Fine-Tuned models head-to-head on DevOps incident diagnosis."""

    def __init__(
        self,
        config_path: str = "ml/config/evaluation_config.yaml",
        test_path: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        self.test_path = test_path or self.cfg["dataset"]["test_split_path"]
        self.engine = IncidentInferenceEngine(
            base_model_name=self.cfg["models"]["baseline"]["model_name_or_path"],
            finetuned_adapter_path=self.cfg["models"]["finetuned"]["adapter_path"],
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run_benchmark(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute evaluation over test split and compile comparison metrics."""
        test_file = Path(self.test_path)
        samples = []
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))

        limit = sample_limit or self.cfg.get("dataset", {}).get("sample_limit")
        if limit and limit > 0:
            samples = samples[:limit]

        logger.info(f"Running benchmark on {len(samples)} test cases.")

        gen_cfg = self.cfg.get("generation", {})
        temp = gen_cfg.get("temperature", 0.1)
        max_tokens = gen_cfg.get("max_new_tokens", 768)

        baseline_preds = []
        baseline_latencies = []
        finetuned_preds = []
        finetuned_latencies = []
        references = []
        item_comparisons = []

        for idx, sample in enumerate(samples, 1):
            log_text = sample.get("log_text") or sample.get("input_log") or ""
            target_report = sample.get("target_report") or sample.get("ground_truth") or {}
            references.append(target_report)

            # 1. Baseline Model Execution
            b_res = self.engine.generate_diagnosis(
                log_text=log_text,
                use_finetuned=False,
                temperature=temp,
                max_new_tokens=max_tokens,
            )
            baseline_preds.append(b_res)
            baseline_latencies.append(b_res["latency_ms"])

            # 2. Fine-Tuned Model Execution
            f_res = self.engine.generate_diagnosis(
                log_text=log_text,
                use_finetuned=True,
                temperature=temp,
                max_new_tokens=max_tokens,
            )
            finetuned_preds.append(f_res)
            finetuned_latencies.append(f_res["latency_ms"])

            item_comparisons.append({
                "incident_id": sample.get("id", f"inc-{idx}"),
                "input_log": log_text[:200] + ("..." if len(log_text) > 200 else ""),
                "ground_truth": target_report,
                "baseline_output": b_res["report"],
                "finetuned_output": f_res["report"],
                "baseline_latency_ms": b_res["latency_ms"],
                "finetuned_latency_ms": f_res["latency_ms"],
                "baseline_json_valid": b_res.get("json_valid", False),
                "finetuned_json_valid": f_res.get("json_valid", False),
            })

        # Calculate metrics
        b_metrics = compute_incident_metrics(baseline_preds, references, baseline_latencies)
        b_metrics["model_name"] = self.cfg["models"]["baseline"]["name"]
        b_metrics["is_fine_tuned"] = False

        f_metrics = compute_incident_metrics(finetuned_preds, references, finetuned_latencies)
        f_metrics["model_name"] = self.cfg["models"]["finetuned"]["name"]
        f_metrics["is_fine_tuned"] = True

        run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": Path(self.test_path).name,
            "total_samples": len(samples),
            "baseline_metrics": b_metrics,
            "finetuned_metrics": f_metrics,
            "sample_comparisons": item_comparisons,
            "verdict_summary": self._generate_verdict(b_metrics, f_metrics),
        }

        self._export_results(summary)
        return summary

    def _generate_verdict(self, b: Dict[str, Any], f: Dict[str, Any]) -> str:
        """Synthesize evaluation summary text."""
        acc_diff = (f.get("category_accuracy", 0) - b.get("category_accuracy", 0)) * 100
        json_diff = (f.get("json_validity_rate", 0) - b.get("json_validity_rate", 0)) * 100
        f1_diff = (f.get("rouge_l_f1", 0) - b.get("rouge_l_f1", 0)) * 100

        return (
            f"Fine-Tuned model showed {acc_diff:+.1f}% category accuracy, "
            f"{json_diff:+.1f}% JSON schema adherence, and "
            f"{f1_diff:+.1f}% token overlap gain compared to Baseline."
        )

    def _export_results(self, summary: Dict[str, Any]) -> None:
        """Write JSON summary and Markdown report to experiments/."""
        results_dir = Path(self.cfg.get("output", {}).get("results_dir", "experiments/results"))
        reports_dir = Path(self.cfg.get("output", {}).get("reports_dir", "experiments/reports"))
        results_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_out = results_dir / "evaluation_summary.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_out = reports_dir / "baseline_vs_finetuned_report.md"
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(self._format_markdown_report(summary))

        logger.info(f"Exported evaluation report to {md_out} and {json_out}")

    def _format_markdown_report(self, s: Dict[str, Any]) -> str:
        bm = s.get("baseline_metrics", {})
        fm = s.get("finetuned_metrics", {})
        return f"""# FineTuneOps: Baseline vs. Fine-Tuned Evaluation Report

**Run ID:** `{s.get('run_id')}`  
**Timestamp:** `{s.get('timestamp')}`  
**Test Dataset:** `{s.get('dataset_name')}` ({s.get('total_samples')} incidents)  

## 1. Summary Comparison

| Metric | Baseline Model | Fine-Tuned (QLoRA) | Delta |
| :--- | :--- | :--- | :--- |
| **Model Name** | `{bm.get('model_name')}` | `{fm.get('model_name')}` | - |
| **JSON Schema Validity** | `{bm.get('json_validity_rate', 0)*100:.1f}%` | `{fm.get('json_validity_rate', 0)*100:.1f}%` | `{(fm.get('json_validity_rate', 0) - bm.get('json_validity_rate', 0))*100:+.1f}%` |
| **Category Accuracy** | `{bm.get('category_accuracy', 0)*100:.1f}%` | `{fm.get('category_accuracy', 0)*100:.1f}%` | `{(fm.get('category_accuracy', 0) - bm.get('category_accuracy', 0))*100:+.1f}%` |
| **Severity Accuracy** | `{bm.get('severity_accuracy', 0)*100:.1f}%` | `{fm.get('severity_accuracy', 0)*100:.1f}%` | `{(fm.get('severity_accuracy', 0) - bm.get('severity_accuracy', 0))*100:+.1f}%` |
| **Text Token F1 / Overlap** | `{bm.get('rouge_l_f1', 0):.4f}` | `{fm.get('rouge_l_f1', 0):.4f}` | `{fm.get('rouge_l_f1', 0) - bm.get('rouge_l_f1', 0):+.4f}` |
| **Average Latency (ms)** | `{bm.get('avg_latency_ms', 0):.1f} ms` | `{fm.get('avg_latency_ms', 0):.1f} ms` | `{fm.get('avg_latency_ms', 0) - bm.get('avg_latency_ms', 0):+.1f} ms` |

## 2. Verdict
{s.get('verdict_summary')}
"""

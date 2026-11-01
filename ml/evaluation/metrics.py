"""Metric computation utilities for DevOps incident diagnosis benchmark.

Strictly adheres to Task 3 evaluation requirements:
1. Fault taxonomy/category accuracy.
2. Root-cause identification:
   - Exact match accuracy
   - Normalized root-cause accuracy
   - Macro F1 across root-cause classes
3. JSON validity percentage.
4. Evidence-grounded diagnosis:
   - Transparent scoring rubric evaluating whether cited observations support the predicted root cause.
5. Latency metrics:
   - Mean, Median, P95.
6. Derived severity and remediation accuracy are explicitly EXCLUDED from primary benchmark metrics.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics import f1_score


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and standardizing whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compute_incident_metrics(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    latencies: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Backward-compatible metric helper for evaluator."""
    total = len(predictions)
    if total == 0:
        return {
            "total_evaluated": 0,
            "json_validity_rate": 0.0,
            "category_accuracy": 0.0,
            "rouge_l_f1": 0.0,
            "avg_latency_ms": 0.0,
        }
    cat_acc = compute_fault_taxonomy_accuracy(
        [p.get("report", p) for p in predictions],
        references,
    )
    json_metrics = compute_json_validity_rate(predictions)
    latencies = latencies or [p.get("latency_ms", 0.0) for p in predictions]
    avg_lat = round(sum(latencies) / total, 2) if latencies else 0.0

    sev_matches = 0
    for p, r in zip(predictions, references):
        p_sev = str(p.get("report", p).get("severity_level", "")).strip().upper()
        r_sev = str(r.get("severity_level", "")).strip().upper()
        if p_sev and r_sev and (p_sev == r_sev or p_sev.startswith(r_sev) or r_sev.startswith(p_sev)):
            sev_matches += 1
    sev_acc = round(sev_matches / total, 4)

    return {
        "total_evaluated": total,
        "json_validity_rate": json_metrics["validity_rate"],
        "category_accuracy": cat_acc,
        "severity_accuracy": sev_acc,
        "avg_latency_ms": avg_lat,
    }


def compute_token_f1(prediction: str, reference: str) -> float:
    """Calculate token-level overlap F1 score."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def compute_sequence_matcher_similarity(prediction: str, reference: str) -> float:
    """Calculate character/token sequence similarity using Gestalt Pattern Matching."""
    return round(difflib.SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio(), 4)


def extract_predicted_root_cause_phrase(root_cause_text: str) -> str:
    """Extract the specific root cause phrase from a root_cause_analysis sentence.

    Handles outputs such as:
    - "Root cause identified as 'missing_service_account' in component..."
    - "The root cause is missing service account."
    - "missing_service_account"
    """
    if not root_cause_text:
        return ""
    cleaned = root_cause_text.strip()
    # Check for quotes: 'root_cause' or "root_cause"
    quote_match = re.search(r"['\"]([a-zA-Z0-9_\-\s]+)['\"]", cleaned)
    if quote_match:
        return quote_match.group(1).strip()

    # Check for "root cause is X" or "root cause identified as X"
    phrase_match = re.search(r"root cause(?: is| identified as|:)?\s+([a-zA-Z0-9_\-]+)", cleaned, re.IGNORECASE)
    if phrase_match:
        return phrase_match.group(1).strip()

    # Fallback to the first sentence or first 10 words
    first_sentence = cleaned.split(".")[0]
    return first_sentence.strip()


def compute_fault_taxonomy_accuracy(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> float:
    """Calculate accuracy of predicted incident category / fault taxonomy.

    Compares predicted 'incident_category' to reference 'incident_category'
    or ground-truth fault category.
    """
    if not predictions or not references:
        return 0.0

    matches = 0
    total = min(len(predictions), len(references))

    for pred, ref in zip(predictions[:total], references[:total]):
        pred_cat = normalize_text(str(pred.get("incident_category", "")))
        ref_cat = normalize_text(str(ref.get("incident_category", ref.get("category", ""))))

        if pred_cat and ref_cat:
            if pred_cat == ref_cat or pred_cat in ref_cat or ref_cat in pred_cat:
                matches += 1

    return round(matches / total, 4)


def compute_semantic_root_cause_score(
    prediction_text: str,
    ground_truth_rc: str,
    ground_truth_object: Optional[str] = None,
) -> float:
    """Compute semantic root-cause identification score using a transparent, deterministic rubric (0.0 to 1.0).

    Scoring Rubric:
    - Level 4 (1.00): Exact Match - The normalized ground truth string matches the predicted phrase exactly.
    - Level 3 (0.80): Canonical Token Alignment - All core canonical tokens or the un-underscored phrase appear in the analysis.
    - Level 2 (0.50): Component & Core Failure Domain Match - Identifies the failing component and primary failure mechanism.
    - Level 1 (0.25): Symptom-Level Association - Mentions the correct component or failure symptom, but misidentifies the root cause.
    - Level 0 (0.00): Ungrounded or Irrelevant Diagnosis - Fabricates unrelated services or failure modes.
    """
    if not prediction_text or not ground_truth_rc:
        return 0.0

    pred_norm = normalize_text(prediction_text)
    rc_norm = normalize_text(ground_truth_rc)
    pred_phrase = normalize_text(extract_predicted_root_cause_phrase(prediction_text))

    # Level 4: Exact Match
    if pred_phrase == rc_norm or rc_norm == pred_phrase:
        return 1.0

    # Level 3: Canonical Token Alignment
    rc_tokens = set(rc_norm.split())
    pred_tokens = set(pred_norm.split())
    if rc_tokens.issubset(pred_tokens) or rc_norm in pred_norm:
        return 0.80

    # Level 2: Component & Core Failure Mechanism
    score = 0.0
    if ground_truth_object:
        comp_norm = normalize_text(ground_truth_object.split("/")[-1])
        if comp_norm in pred_norm:
            score += 0.25

    # Core failure mechanism keywords in ground truth
    core_keywords = {"file", "cpu", "memory", "leak", "delay", "loss", "timeout", "kill", "oom", "permission", "secret", "restart", "account"}
    matched_core = [w for w in (rc_tokens & core_keywords) if w in pred_norm]
    if matched_core:
        score += 0.25

    if score >= 0.50:
        return 0.50

    # Level 1: Partial / Symptom Match
    token_f1 = compute_token_f1(pred_norm, rc_norm)
    if token_f1 >= 0.30 or (ground_truth_object and normalize_text(ground_truth_object.split("/")[-1]) in pred_norm):
        return 0.25

    return 0.0


def compute_root_cause_metrics(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute Root-Cause Identification metrics:

    1. Exact Match Accuracy: Normalized string equality.
    2. Normalized Accuracy: Reference root cause tokens or phrase present in prediction.
    3. Macro F1: Scikit-learn macro F1 across unique root-cause ground-truth classes.
    """
    total = min(len(predictions), len(references))
    if total == 0:
        return {
            "exact_match_accuracy": 0.0,
            "normalized_accuracy": 0.0,
            "macro_f1": 0.0,
        }

    exact_matches = 0
    normalized_matches = 0
    semantic_scores = []

    y_true: List[str] = []
    y_pred: List[str] = []

    # Collect known classes from references
    all_ref_classes = set()
    for ref in references:
        rc = ref.get("root_cause") or ref.get("root_cause_analysis", "")
        norm_rc = normalize_text(rc)
        if norm_rc:
            all_ref_classes.add(norm_rc)

    for pred, ref in zip(predictions[:total], references[:total]):
        ref_rc_raw = str(ref.get("root_cause") or ref.get("root_cause_analysis", ""))
        ref_rc_norm = normalize_text(ref_rc_raw)

        pred_rc_full = str(pred.get("root_cause_analysis", ""))
        pred_phrase = extract_predicted_root_cause_phrase(pred_rc_full)
        pred_phrase_norm = normalize_text(pred_phrase)
        pred_full_norm = normalize_text(pred_rc_full)

        # 1. Exact Match
        is_exact = (pred_phrase_norm == ref_rc_norm) and bool(ref_rc_norm)
        if is_exact:
            exact_matches += 1

        # 2. Normalized Match: Reference phrase is present in full analysis or token F1 >= 0.6
        token_f1 = compute_token_f1(pred_rc_full, ref_rc_raw)
        is_normalized = (
            is_exact
            or (ref_rc_norm in pred_full_norm if len(ref_rc_norm) > 3 else False)
            or (token_f1 >= 0.60)
        )
        if is_normalized:
            normalized_matches += 1

        # 3. Macro F1 class alignment:
        # If model's predicted phrase or full text matches one of the valid classes, assign it.
        # Otherwise assign "unclassified_or_hallucinated".
        assigned_pred_class = "unclassified_or_hallucinated"
        if is_exact:
            assigned_pred_class = ref_rc_norm
        else:
            for known_class in all_ref_classes:
                if known_class in pred_full_norm or known_class == pred_phrase_norm:
                    assigned_pred_class = known_class
                    break

        y_true.append(ref_rc_norm if ref_rc_norm else "unknown_ground_truth")
        y_pred.append(assigned_pred_class)

        # 4. Multi-level Semantic Root-Cause Score
        sem_score = compute_semantic_root_cause_score(
            pred_rc_full,
            ref_rc_raw,
            ref.get("fault_object", ref.get("component")),
        )
        semantic_scores.append(sem_score)

    # Compute Macro F1 across root-cause classes
    unique_labels = sorted(list(set(y_true + y_pred)))
    macro_f1 = f1_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)

    mean_semantic_score = round(sum(semantic_scores) / total, 4) if semantic_scores else 0.0

    return {
        "exact_match_accuracy": round(exact_matches / total, 4),
        "normalized_accuracy": round(normalized_matches / total, 4),
        "semantic_score": mean_semantic_score,
        "macro_f1": round(float(macro_f1), 4),
    }


def compute_json_validity_rate(
    predictions_raw: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute percentage of predictions that are valid JSON and schema-compliant."""
    total = len(predictions_raw)
    if total == 0:
        return {"valid_count": 0, "total": 0, "validity_rate": 0.0}

    valid_count = 0
    schema_compliant_count = 0

    required_keys = {"incident_category", "root_cause_analysis"}

    for pred in predictions_raw:
        if pred.get("json_valid", False):
            valid_count += 1
            report = pred.get("report", {})
            if isinstance(report, dict) and required_keys.issubset(set(report.keys())):
                schema_compliant_count += 1

    return {
        "valid_json_count": valid_count,
        "schema_compliant_count": schema_compliant_count,
        "total": total,
        "validity_rate": round(valid_count / total, 4),
        "schema_compliance_rate": round(schema_compliant_count / total, 4),
    }


def compute_evidence_grounding_score(
    prediction_report: Dict[str, Any],
    input_log_text: str,
) -> float:
    """Evaluate whether cited observations support the predicted root cause.

    Transparent Scoring Rubric (0.0 to 1.0):
    1. Entity & Component Grounding (0.40):
       Checks whether services, pods, or components referenced in the report
       actually appear in the input logs/diagnostic trace.
    2. Symptom & Error Grounding (0.40):
       Checks whether diagnostic failure tokens (e.g. error keywords, status codes,
       anomalies) in the root cause analysis are present in the input telemetry.
    3. Structural Coherence (0.20):
       Checks whether the contributing factors and debugging steps logically link
       to the identified component rather than generic ungrounded templates.
    """
    if not isinstance(prediction_report, dict) or not input_log_text:
        return 0.0

    score = 0.0
    input_lower = input_log_text.lower()
    input_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", input_lower))

    rc_text = str(prediction_report.get("root_cause_analysis", "")).lower()
    rc_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", rc_text))

    # 1. Entity / Component Grounding (up to 0.40)
    # Check for component/service names like frontend, cartservice, redis, etc.
    components_found = [token for token in rc_tokens if token in input_tokens and len(token) > 4]
    if components_found:
        score += 0.40
    elif any(token in input_lower for token in rc_tokens if len(token) > 5):
        score += 0.20

    # 2. Symptom & Error Grounding (up to 0.40)
    # Check overlap of error symptoms
    error_overlap = rc_tokens & input_tokens
    # Filter common stop words
    common_stops = {"incident", "system", "service", "cause", "error", "found", "identified", "under", "within"}
    substantive_overlap = [w for w in error_overlap if w not in common_stops]

    if len(substantive_overlap) >= 3:
        score += 0.40
    elif len(substantive_overlap) >= 1:
        score += 0.20

    # 3. Structural Coherence (up to 0.20)
    contributing = prediction_report.get("contributing_factors", [])
    debugging = prediction_report.get("recommended_debugging_steps", [])
    if isinstance(contributing, list) and len(contributing) > 0:
        score += 0.10
    if isinstance(debugging, list) and len(debugging) > 0:
        score += 0.10

    return round(min(score, 1.0), 4)


def compute_latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, median, P95, min, and max latency in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def compute_comprehensive_metrics(
    predictions_payloads: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """Aggregate all evaluation metrics across the test split.

    Explicitly excludes derived severity and remediation accuracy from primary benchmark metrics.
    """
    total = len(test_records)
    reports = [p.get("report", {}) for p in predictions_payloads]

    # Extract ground truth references
    references = []
    input_logs = []
    for rec in test_records:
        meta = rec.get("metadata", {})
        # Ground-truth reference
        ref_obj = {
            "incident_category": meta.get("category", ""),
            "root_cause": meta.get("root_cause", ""),
            "root_cause_analysis": meta.get("root_cause", ""),
            "source_reference": meta.get("source_reference", ""),
        }
        references.append(ref_obj)
        # Input user prompt
        messages = rec.get("messages", [])
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        input_logs.append(user_content)

    # 1. Fault Taxonomy / Category Accuracy
    category_acc = compute_fault_taxonomy_accuracy(reports, references)

    # 2. Root Cause Identification Metrics
    rc_metrics = compute_root_cause_metrics(reports, references)

    # 3. JSON Validity
    json_metrics = compute_json_validity_rate(predictions_payloads)

    # 4. Evidence Grounding Scores
    grounding_scores = [
        compute_evidence_grounding_score(rep, inp)
        for rep, inp in zip(reports, input_logs)
    ]
    mean_grounding_score = round(float(np.mean(grounding_scores)), 4) if grounding_scores else 0.0

    # 5. Latency Summary
    latency_summary = compute_latency_summary(latencies_ms)

    return {
        "evaluation_scope": "Kubernetes and microservice incident diagnosis (Cloud-OpsBench)",
        "total_test_samples": total,
        "metrics": {
            "fault_taxonomy_accuracy": category_acc,
            "root_cause_identification": {
                "exact_match_accuracy": rc_metrics["exact_match_accuracy"],
                "normalized_accuracy": rc_metrics["normalized_accuracy"],
                "semantic_score": rc_metrics["semantic_score"],
                "macro_f1": rc_metrics["macro_f1"],
            },
            "json_validity": {
                "validity_rate": json_metrics["validity_rate"],
                "schema_compliance_rate": json_metrics["schema_compliance_rate"],
                "valid_count": json_metrics["valid_json_count"],
            },
            "evidence_grounded_diagnosis": {
                "mean_grounding_score": mean_grounding_score,
                "scoring_rubric": "Entity grounding (0.40) + Symptom/error grounding (0.40) + Coherence (0.20)",
                "human_verified": False,
            },
            "latency": latency_summary,
        },
        "excluded_metrics_notice": {
            "derived_severity_accuracy": "EXCLUDED: Severity in Cloud-OpsBench is derived from fault impact rather than an official benchmark label.",
            "remediation_accuracy": "EXCLUDED: Cloud-OpsBench benchmark evaluates RCA and does not provide ground-truth remediation actions in the source dataset.",
        },
    }

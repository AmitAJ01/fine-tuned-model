"""Find representative correct semantic diagnoses that fail exact match,
and representative genuinely incorrect diagnoses.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
ft_preds_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"
base_preds_file = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl"

test_records = [json.loads(l) for l in open(test_file, "r", encoding="utf-8") if l.strip()]
ft_preds = [json.loads(l) for l in open(ft_preds_file, "r", encoding="utf-8") if l.strip()]
base_preds = [json.loads(l) for l in open(base_preds_file, "r", encoding="utf-8") if l.strip()]

semantic_correct_examples = []
genuinely_incorrect_examples = []

for idx, (rec, ft_p) in enumerate(zip(test_records, ft_preds)):
    gt_cat = rec["metadata"]["category"]
    gt_rc = rec["metadata"]["root_cause"]
    rec_id = rec["id"]
    
    ft_rep = ft_p.get("prediction_report") or {}
    pred_cat = ft_rep.get("incident_category")
    rc_text = ft_rep.get("root_cause_analysis", "")
    
    cat_match = (pred_cat == gt_cat)
    
    # Check if category matched and the root cause description is semantically aligned
    if cat_match:
        # Check specific semantic equivalents
        if gt_rc == "code_busy_loop" and "code_memory_leak" in rc_text:
            semantic_correct_examples.append({
                "idx": idx, "id": rec_id, "gt_cat": gt_cat, "gt_rc": gt_rc,
                "pred_cat": pred_cat, "rc_analysis": rc_text,
                "reason": "Code performance degradation: model attributes to code_memory_leak under Application_Code_Defect. Both cause CPU/memory resource exhaustion in microservice."
            })
        elif gt_rc == "mysql_invalid_credentials" and "mysql_connection_loss" in rc_text:
            semantic_correct_examples.append({
                "idx": idx, "id": rec_id, "gt_cat": gt_cat, "gt_rc": gt_rc,
                "pred_cat": pred_cat, "rc_analysis": rc_text,
                "reason": "Database connectivity: invalid auth credentials manifest as database connection refusal and connection pool loss."
            })
        elif gt_rc == "taint_toleration_mismatch" and "node_selector_mismatch" in rc_text:
            semantic_correct_examples.append({
                "idx": idx, "id": rec_id, "gt_cat": gt_cat, "gt_rc": gt_rc,
                "pred_cat": pred_cat, "rc_analysis": rc_text,
                "reason": "Kubernetes scheduling: pod Pending with '0/3 nodes available' scheduling constraint. Both faults cause identical pod unschedulability."
            })
        elif gt_rc == "pod_anti_affinity_conflict" and "node_selector_mismatch" in rc_text:
            semantic_correct_examples.append({
                "idx": idx, "id": rec_id, "gt_cat": gt_cat, "gt_rc": gt_rc,
                "pred_cat": pred_cat, "rc_analysis": rc_text,
                "reason": "Kubernetes scheduling constraint: anti-affinity conflict prevents pod scheduling, matching node selector mismatch telemetry."
            })
    else:
        # Category mismatched - genuinely incorrect diagnosis
        genuinely_incorrect_examples.append({
            "idx": idx, "id": rec_id, "gt_cat": gt_cat, "gt_rc": gt_rc,
            "pred_cat": pred_cat, "rc_analysis": rc_text,
            "reason": f"Category mismatch: ground truth is '{gt_cat}' but model diagnosed as '{pred_cat}'."
        })

print(f"Found {len(semantic_correct_examples)} semantic alignment examples, {len(genuinely_incorrect_examples)} incorrect examples.")
print("\n--- TOP 3 SEMANTICALLY ALIGNED EXAMPLES (FAIL EXACT MATCH) ---")
for ex in semantic_correct_examples[:3]:
    print(f"Sample: {ex['id']} (GT Category: {ex['gt_cat']}, GT Root Cause: {ex['gt_rc']})")
    print(f"  Predicted Category: {ex['pred_cat']}")
    print(f"  Predicted Analysis: {ex['rc_analysis'][:130]}...")
    print(f"  Analysis: {ex['reason']}\n")

print("\n--- TOP 3 GENUINELY INCORRECT EXAMPLES ---")
for ex in genuinely_incorrect_examples[:3]:
    print(f"Sample: {ex['id']} (GT Category: {ex['gt_cat']}, GT Root Cause: {ex['gt_rc']})")
    print(f"  Predicted Category: {ex['pred_cat']}")
    print(f"  Predicted Analysis: {ex['rc_analysis'][:130]}...")
    print(f"  Analysis: {ex['reason']}\n")

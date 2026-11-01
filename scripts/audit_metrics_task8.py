"""Task 8: Metrics Integrity Audit & RCA Deep-Dive Script.

This script independently recomputes all metrics from:
- ml/data/splits_final/test.jsonl
- experiments/final_run/baseline_predictions.jsonl
- experiments/final_run/finetuned_predictions.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.metrics import (
    normalize_text,
    compute_token_f1,
    extract_predicted_root_cause_phrase,
    compute_fault_taxonomy_accuracy,
    compute_semantic_root_cause_score,
    compute_evidence_grounding_score,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
baseline_file = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl"
finetuned_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"

test_data = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]
baseline_preds = [json.loads(line) for line in open(baseline_file, "r", encoding="utf-8")]
finetuned_preds = [json.loads(line) for line in open(finetuned_file, "r", encoding="utf-8")]

print(f"Total test cases: {len(test_data)}")
print(f"Baseline predictions: {len(baseline_preds)}")
print(f"Fine-tuned predictions: {len(finetuned_preds)}")

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

print("\n" + "="*60)
print("SECTION 1: BASELINE CATEGORY ACCURACY AUDIT (73 vs 74)")
print("="*60)

matches_compute = 0
matches_map = 0

for i, r in enumerate(baseline_preds):
    gt_cat = r["ground_truth"]["category"]
    pred_cat = r["prediction_report"].get("incident_category", "")
    
    # 1. compute_fault_taxonomy_accuracy method
    pred_norm = normalize_text(str(pred_cat))
    ref_norm = normalize_text(str(gt_cat))
    match_1 = bool(pred_norm and ref_norm and (pred_norm == ref_norm or pred_norm in ref_norm or ref_norm in pred_norm))
    if match_1:
        matches_compute += 1
        
    # 2. map_category_string method
    yt = map_category_string(gt_cat)
    yp = map_category_string(pred_cat)
    match_2 = (yt == yp)
    if match_2:
        matches_map += 1
        
    if match_1 != match_2:
        rec_id = r["id"]
        print(f"Discrepancy at index {i} (id={rec_id}):")
        print(f"  gt_cat: {repr(gt_cat)} -> norm: {repr(ref_norm)} -> mapped: {repr(yt)}")
        print(f"  pred_cat: {repr(pred_cat)} -> norm: {repr(pred_norm)} -> mapped: {repr(yp)}")
        print(f"  match_1 (compute_fault_taxonomy_accuracy): {match_1}")
        print(f"  match_2 (map_category_string): {match_2}")

print(f"\ncompute_fault_taxonomy_accuracy matches: {matches_compute} / 162 = {matches_compute/162:.4f} ({matches_compute/162*100:.2f}%)")
print(f"map_category_string matches: {matches_map} / 162 = {matches_map/162:.4f} ({matches_map/162*100:.2f}%)")

# Check fine-tuned for the same:
ft_matches_compute = 0
ft_matches_map = 0
for i, r in enumerate(finetuned_preds):
    gt_cat = r["ground_truth"]["category"]
    pred_cat = r["prediction_report"].get("incident_category", "")
    
    pred_norm = normalize_text(str(pred_cat))
    ref_norm = normalize_text(str(gt_cat))
    match_1 = bool(pred_norm and ref_norm and (pred_norm == ref_norm or pred_norm in ref_norm or ref_norm in pred_norm))
    if match_1:
        ft_matches_compute += 1
        
    yt = map_category_string(gt_cat)
    yp = map_category_string(pred_cat)
    match_2 = (yt == yp)
    if match_2:
        ft_matches_map += 1

print("\n" + "="*60)
print("SECTION 2: ROOT-CAUSE PREDICTION AUDIT ACROSS ALL 162 CASES")
print("="*60)

train_cases = [json.loads(l) for l in open(REPO_ROOT / "ml/data/splits_final/train.jsonl", "r", encoding="utf-8")]
val_cases = [json.loads(l) for l in open(REPO_ROOT / "ml/data/splits_final/val.jsonl", "r", encoding="utf-8")]

train_rcs = set(c["metadata"]["root_cause"] for c in train_cases)
val_rcs = set(c["metadata"]["root_cause"] for c in val_cases)
test_rcs = set(c["metadata"]["root_cause"] for c in test_data)

print(f"Unique root causes in train: {len(train_rcs)}")
print(f"Unique root causes in val: {len(val_rcs)}")
print(f"Unique root causes in test: {len(test_rcs)}")
print(f"Train & Test RC overlap: {len(train_rcs & test_rcs)} -> {sorted(list(train_rcs & test_rcs))}")
print(f"Test RCs NOT in train: {len(test_rcs - train_rcs)} -> {sorted(list(test_rcs - train_rcs))}")

# Check extracted root cause phrases for all 162 predictions
audit_table = []
exact_matches_count = 0
in_text_matches_count = 0

for idx, (p, t) in enumerate(zip(finetuned_preds, test_data)):
    gt_rc = t["metadata"]["root_cause"]
    rep = p.get("prediction_report", {})
    rc_text = rep.get("root_cause_analysis", "") if isinstance(rep, dict) else ""
    extracted_phrase = extract_predicted_root_cause_phrase(rc_text)
    
    norm_gt = normalize_text(gt_rc)
    norm_extracted = normalize_text(extracted_phrase)
    norm_text = normalize_text(rc_text)
    
    is_exact = bool(norm_gt and norm_extracted == norm_gt)
    is_in_text = bool(norm_gt and norm_gt in norm_text)
    
    if is_exact:
        exact_matches_count += 1
    if is_in_text:
        in_text_matches_count += 1
        
    reason = []
    if gt_rc not in train_rcs:
        reason.append("Unseen RC class (not in train)")
    if is_exact:
        reason.append("Exact match!")
    elif is_in_text:
        reason.append("Mentioned in text, but extracted phrase differed")
    else:
        reason.append(f"Predicted '{extracted_phrase}' instead of '{gt_rc}'")
        
    audit_table.append({
        "idx": idx,
        "id": p["id"],
        "gt_rc": gt_rc,
        "extracted_phrase": extracted_phrase,
        "is_exact": is_exact,
        "is_in_text": is_in_text,
        "explanation": "; ".join(reason),
    })

print(f"\nTotal Exact Matches found: {exact_matches_count} / 162")
print(f"Total In-Text Matches found: {in_text_matches_count} / 162")

# Check specifically the 21 cases where gt_rc IS in train (taint_toleration_mismatch)
print("\n--- Detailed Inspection of Cases Where GT Root Cause was Seen in Training (taint_toleration_mismatch) ---")
seen_count = 0
for item in audit_table:
    if item["gt_rc"] in train_rcs:
        seen_count += 1
        p = finetuned_preds[item["idx"]]
        rc_text = p.get("prediction_report", {}).get("root_cause_analysis", "")
        print(f"ID: {item['id']}")
        print(f"  GT: {item['gt_rc']}")
        print(f"  Extracted: {item['extracted_phrase']}")
        print(f"  Full text snippet: {rc_text[:120]}...")
        print(f"  Exact match: {item['is_exact']}")
        print()
print(f"Total seen cases: {seen_count}")

print("\n" + "="*60)
print("SECTION 3: SPLIT GENERALIZATION AUDIT")
print("="*60)

splits_manifest_path = REPO_ROOT / "experiments/final_run/splits_manifest.json"
manifest = json.loads(open(splits_manifest_path, "r", encoding="utf-8").read())

def get_system(c):
    return c["id"].split("/")[0]

def get_scenario_key(c):
    m = c["metadata"]
    sys_name = get_system(c)
    return f"{sys_name}::{m['category']}::{m['root_cause']}"

train_scenarios = set(get_scenario_key(c) for c in train_cases)
val_scenarios = set(get_scenario_key(c) for c in val_cases)
test_scenarios = set(get_scenario_key(c) for c in test_data)

train_categories = set(c["metadata"]["category"] for c in train_cases)
test_categories = set(c["metadata"]["category"] for c in test_data)

train_systems = set(get_system(c) for c in train_cases)
test_systems = set(get_system(c) for c in test_data)

def get_fault_object(rec_id):
    meta_path = REPO_ROOT / f"ml/data/raw/cloud_ops_bench/benchmark/{rec_id}/metadata.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data.get("result", {}).get("fault_object")
    return None

train_objects = set(filter(None, [get_fault_object(c["id"]) for c in train_cases]))
test_objects = set(filter(None, [get_fault_object(c["id"]) for c in test_data]))

print(f"Scenario-group overlap (train vs test): {len(train_scenarios & test_scenarios)}")
print(f"Root-cause label overlap (train vs test): {len(train_rcs & test_rcs)} / {len(test_rcs)} test classes ({len(train_rcs & test_rcs)/len(test_rcs)*100:.1f}%)")
print(f"Category overlap (train vs test): {len(train_categories & test_categories)} / {len(test_categories)} test categories ({len(train_categories & test_categories)/len(test_categories)*100:.1f}%)")
print(f"System overlap (train vs test): {len(train_systems & test_systems)} / {len(test_systems)} test systems ({len(train_systems & test_systems)/len(test_systems)*100:.1f}%)")
print(f"Fault-object overlap (train vs test): {len(train_objects & test_objects)} / {len(test_objects)} test objects ({len(train_objects & test_objects)/len(test_objects)*100:.1f}%)")

print("\nWhat the split tests:")
print("1. Scenario Groups: ZERO overlap. All 13 test scenario groups are unseen.")
print(f"2. Root Causes: Disjoint across 11 of 12 classes ({len(test_rcs - train_rcs)}/12 = {(len(test_rcs - train_rcs))/12*100:.1f}% unseen classes, covering {sum(1 for t in test_data if t['metadata']['root_cause'] not in train_rcs)}/162 = {sum(1 for t in test_data if t['metadata']['root_cause'] not in train_rcs)/162*100:.1f}% of test samples).")
print("   -> For Root Cause Analysis, this is predominantly an UNSEEN-CLASS zero-shot generalization test!")
print(f"3. Incident Categories: All 5 canonical categories are shared ({sorted(list(train_categories & test_categories))}).")
print("   -> For Fault Taxonomy / Category classification, this is a generalization test to UNSEEN SCENARIOS OF KNOWN CATEGORIES!")
print(f"4. Systems: Both systems ({sorted(list(train_systems & test_systems))}) are shared.")
print(f"5. Fault Objects: {len(train_objects & test_objects)} / {len(test_objects)} ({len(train_objects & test_objects)/len(test_objects)*100:.1f}%) microservices/components are shared.")

print("\n" + "="*60)
print("SECTION 4: INVESTIGATION OF THE 8 MALFORMED JSON OUTPUTS")
print("="*60)

malformed_cases = []
for idx, p in enumerate(finetuned_preds):
    if not p.get("json_valid", True):
        malformed_cases.append((idx, p))

print(f"Found {len(malformed_cases)} malformed cases:")
for idx, p in malformed_cases:
    raw = p["raw_generated_text"]
    tokens = p["completion_tokens"]
    last_chars = raw[-60:] if len(raw) >= 60 else raw
    # Try parsing JSON to get exact exception:
    json_err = ""
    try:
        json.loads(raw)
    except Exception as e:
        json_err = str(e)
    print(f"Test ID: {p['id']} (idx={idx})")
    print(f"  Completion Tokens: {tokens} (max limit is 384)")
    print(f"  Truncated at 384 tokens: {tokens >= 384}")
    print(f"  JSON Parsing Error: {json_err}")
    print(f"  Last 60 characters: {repr(last_chars)}")
    print()



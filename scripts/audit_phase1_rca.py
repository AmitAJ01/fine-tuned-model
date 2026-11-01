"""Phase 1: In-depth Root-Cause Analysis (RCA) Audit Script.

Investigates:
1. Prompt and schema requirements (does prompt ask for canonical ID or descriptive analysis?).
2. Training target format across samples.
3. Raw predictions on seen vs unseen root-cause classes in test set.
4. Detailed inspection of all 21 seen cases (taint_toleration_mismatch).
5. Representative correct vs incorrect semantic diagnoses.
"""

import json
from pathlib import Path
from collections import Counter
import re

REPO_ROOT = Path(__file__).resolve().parent.parent

test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
train_file = REPO_ROOT / "ml/data/splits_final/train.jsonl"
base_preds_file = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl"
ft_preds_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"

test_records = [json.loads(l) for l in open(test_file, "r", encoding="utf-8") if l.strip()]
train_records = [json.loads(l) for l in open(train_file, "r", encoding="utf-8") if l.strip()]
base_preds = [json.loads(l) for l in open(base_preds_file, "r", encoding="utf-8") if l.strip()]
ft_preds = [json.loads(l) for l in open(ft_preds_file, "r", encoding="utf-8") if l.strip()]

train_rcs = {r["metadata"]["root_cause"] for r in train_records}
test_rcs = {r["metadata"]["root_cause"] for r in test_records}

seen_rc_classes = test_rcs & train_rcs
unseen_rc_classes = test_rcs - train_rcs

print("=== RCA CLASS PARTITIONING ===")
print(f"Total Unique Train Classes: {len(train_rcs)}")
print(f"Total Unique Test Classes: {len(test_rcs)}")
print(f"Shared Classes: {seen_rc_classes}")
print(f"Unseen Classes ({len(unseen_rc_classes)}): {sorted(list(unseen_rc_classes))}")

# Check prompt instruction in test_records[0]
sample_user_msg = test_records[0]["messages"][1]["content"]
print("\n=== SAMPLE PROMPT INSTRUCTION (USER PROMPT) ===")
# Print the JSON schema portion of the user prompt
if "```json" in sample_user_msg:
    print(sample_user_msg.split("```json")[1].split("```")[0])
elif "{" in sample_user_msg:
    lines = [l for l in sample_user_msg.split("\n") if "root_cause" in l or "incident_category" in l or "schema" in l.lower()]
    print("\n".join(lines[:10]))

# Check training target format in train_records[0..2]
print("\n=== SAMPLE TRAINING TARGETS (ASSISTANT MESSAGE) ===")
for i in range(3):
    ast_msg = train_records[i]["messages"][2]["content"]
    print(f"Train sample {i} (id={train_records[i]['id']}, GT={train_records[i]['metadata']['root_cause']}):")
    try:
        parsed = json.loads(ast_msg)
        print(f"  root_cause_analysis: {parsed.get('root_cause_analysis')[:120]}...")
    except Exception as e:
        print(f"  Raw: {ast_msg[:120]}...")

# Calculate exact-match on seen vs unseen
def check_rc_exact(pred_obj, gt_rc):
    rep = pred_obj.get("prediction_report") or {}
    rc_text = rep.get("root_cause_analysis", "")
    if not rc_text:
        return False
    norm_gt = gt_rc.lower().replace("_", " ").strip()
    norm_text = rc_text.lower().replace("_", " ").strip()
    # Check exact match or token containment
    return norm_gt in norm_text

print("\n=== EXACT MATCH / CONTAINMENT ON SEEN VS UNSEEN CLASSES ===")
seen_indices = [i for i, r in enumerate(test_records) if r["metadata"]["root_cause"] in seen_rc_classes]
unseen_indices = [i for i, r in enumerate(test_records) if r["metadata"]["root_cause"] in unseen_rc_classes]

print(f"Seen sample count: {len(seen_indices)} (all taint_toleration_mismatch)")
print(f"Unseen sample count: {len(unseen_indices)}")

# Baseline
base_seen_exact = 0
base_seen_contain = 0
for idx in seen_indices:
    gt_rc = test_records[idx]["metadata"]["root_cause"]
    rep = base_preds[idx].get("prediction_report") or {}
    rc_text = rep.get("root_cause_analysis", "")
    if gt_rc.lower() in rc_text.lower():
        base_seen_exact += 1
    if gt_rc.lower().replace("_", " ") in rc_text.lower().replace("_", " "):
        base_seen_contain += 1

print(f"Baseline on Seen (N={len(seen_indices)}): exact string={base_seen_exact}, normalized contain={base_seen_contain}")

# Fine-Tuned
ft_seen_exact = 0
ft_seen_contain = 0
for idx in seen_indices:
    gt_rc = test_records[idx]["metadata"]["root_cause"]
    rep = ft_preds[idx].get("prediction_report") or {}
    rc_text = rep.get("root_cause_analysis", "")
    if gt_rc.lower() in rc_text.lower():
        ft_seen_exact += 1
    if gt_rc.lower().replace("_", " ") in rc_text.lower().replace("_", " "):
        ft_seen_contain += 1

print(f"Fine-Tuned on Seen (N={len(seen_indices)}): exact string={ft_seen_exact}, normalized contain={ft_seen_contain}")

# Inspect all 21 seen cases predictions
print("\n=== ALL 21 SEEN CASES (taint_toleration_mismatch) INSPECTION ===")
for idx in seen_indices:
    rec = test_records[idx]
    ft_rep = ft_preds[idx].get("prediction_report") or {}
    base_rep = base_preds[idx].get("prediction_report") or {}
    ft_rc = ft_rep.get("root_cause_analysis", "")[:100]
    base_rc = base_rep.get("root_cause_analysis", "")[:100]
    print(f"ID={rec['id']}:")
    print(f"  FT:   {ft_rc}")
    print(f"  Base: {base_rc}")

# Inspect representative unseen cases
print("\n=== SAMPLE UNSEEN CASES INSPECTION ===")
for idx in unseen_indices[:5]:
    rec = test_records[idx]
    gt_rc = rec["metadata"]["root_cause"]
    ft_rep = ft_preds[idx].get("prediction_report") or {}
    base_rep = base_preds[idx].get("prediction_report") or {}
    print(f"ID={rec['id']} (GT Root Cause: {gt_rc}):")
    print(f"  FT:   {ft_rep.get('root_cause_analysis', '')[:120]}")
    print(f"  Base: {base_rep.get('root_cause_analysis', '')[:120]}")

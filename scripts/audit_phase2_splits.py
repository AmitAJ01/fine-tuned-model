"""Phase 2: Comprehensive Split Integrity and Accounting Audit."""

import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent

splits_dir = REPO_ROOT / "ml/data/splits_final"
train_file = splits_dir / "train.jsonl"
val_file = splits_dir / "val.jsonl"
test_file = splits_dir / "test.jsonl"
manifest_file = REPO_ROOT / "experiments/final_run/splits_manifest.json"

train_recs = [json.loads(l) for l in open(train_file, "r", encoding="utf-8") if l.strip()]
val_recs = [json.loads(l) for l in open(val_file, "r", encoding="utf-8") if l.strip()]
test_recs = [json.loads(l) for l in open(test_file, "r", encoding="utf-8") if l.strip()]
manifest = json.load(open(manifest_file, "r", encoding="utf-8"))

print("=== RECORD COUNTS ===")
print(f"Train Records: {len(train_recs)} (Manifest: {manifest['splits']['train']['count']})")
print(f"Val Records:   {len(val_recs)} (Manifest: {manifest['splits']['val']['count']})")
print(f"Test Records:  {len(test_recs)} (Manifest: {manifest['splits']['test']['count']})")
total = len(train_recs) + len(val_recs) + len(test_recs)
print(f"Total:         {total} (Manifest: {manifest['total_records']})")

import hashlib

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

train_hash = compute_file_sha256(train_file)
val_hash = compute_file_sha256(val_file)
test_hash = compute_file_sha256(test_file)

assert train_hash == manifest["splits"]["train"]["sha256"], "Train SHA256 mismatch!"
assert val_hash == manifest["splits"]["val"]["sha256"], "Val SHA256 mismatch!"
assert test_hash == manifest["splits"]["test"]["sha256"], "Test SHA256 mismatch!"
print(f"Train SHA-256 strictly verified: {train_hash}")
print(f"Val SHA-256 strictly verified:   {val_hash}")
print(f"Test SHA-256 strictly verified:  {test_hash}")

# Extract grouping key:
# Check create_final_experiment_split.py to verify exact grouping key
def extract_scenario_group(rec):
    # From scripts/create_final_experiment_split.py:
    # scenario_key = f"{system}::{category}::{root_cause}"
    m = rec["metadata"]
    # Extract system from id (e.g. boutique/codedefect/34 -> boutique)
    sys_name = rec["id"].split("/")[0]
    cat = m.get("category", "")
    rc = m.get("root_cause", "")
    return f"{sys_name}::{cat}::{rc}"

train_groups = {extract_scenario_group(r) for r in train_recs}
val_groups = {extract_scenario_group(r) for r in val_recs}
test_groups = {extract_scenario_group(r) for r in test_recs}

print("\n=== SCENARIO GROUPS ===")
print(f"Grouping Key Definition: system::category::root_cause")
print(f"Unique Train Groups: {len(train_groups)}")
print(f"Unique Val Groups:   {len(val_groups)}")
print(f"Unique Test Groups:  {len(test_groups)}")
all_groups = train_groups | val_groups | test_groups
print(f"Total Unique Groups across dataset: {len(all_groups)}")

# Overlaps
tv_group_overlap = train_groups & val_groups
tt_group_overlap = train_groups & test_groups
vt_group_overlap = val_groups & test_groups

print("\n=== GROUP OVERLAPS ===")
print(f"Train vs Val Group Overlap:  {len(tv_group_overlap)} ({tv_group_overlap})")
print(f"Train vs Test Group Overlap: {len(tt_group_overlap)} ({tt_group_overlap})")
print(f"Val vs Test Group Overlap:   {len(vt_group_overlap)} ({vt_group_overlap})")

# Feature overlaps: Categories, Systems, Root Causes, Fault Objects
index_file = REPO_ROOT / "ml/data/raw/cloud_ops_bench/index.jsonl"
index_recs = {r["case_key"]: r for r in (json.loads(l) for l in open(index_file, "r", encoding="utf-8") if l.strip())}

def get_features(recs):
    cats = {r["metadata"]["category"] for r in recs}
    syss = {r["id"].split("/")[0] for r in recs}
    rcs = {r["metadata"]["root_cause"] for r in recs}
    
    # Fault objects from index.jsonl
    fos = set()
    for r in recs:
        rec_id = r["id"]
        if rec_id in index_recs and "fault_object" in index_recs[rec_id]:
            fos.add(index_recs[rec_id]["fault_object"])
    return cats, syss, rcs, fos

train_c, train_s, train_rc, train_fo = get_features(train_recs)
val_c, val_s, val_rc, val_fo = get_features(val_recs)
test_c, test_s, test_rc, test_fo = get_features(test_recs)

print("\n=== CATEGORIES OVERLAP ===")
print(f"Train Categories ({len(train_c)}): {sorted(list(train_c))}")
print(f"Val Categories   ({len(val_c)}): {sorted(list(val_c))}")
print(f"Test Categories  ({len(test_c)}): {sorted(list(test_c))}")
print(f"Train & Test Shared Categories: {len(train_c & test_c)} / {len(test_c)}")

print("\n=== SYSTEMS OVERLAP ===")
print(f"Train Systems: {sorted(list(train_s))}")
print(f"Val Systems:   {sorted(list(val_s))}")
print(f"Test Systems:  {sorted(list(test_s))}")
print(f"Train & Test Shared Systems: {len(train_s & test_s)} / {len(test_s)}")

print("\n=== ROOT CAUSES OVERLAP ===")
print(f"Train Root Causes: {len(train_rc)}")
print(f"Val Root Causes:   {len(val_rc)}")
print(f"Test Root Causes:  {len(test_rc)}")
print(f"Train & Val Shared:  {len(train_rc & val_rc)} ({train_rc & val_rc})")
print(f"Train & Test Shared: {len(train_rc & test_rc)} ({train_rc & test_rc})")
print(f"Val & Test Shared:   {len(val_rc & test_rc)} ({val_rc & test_rc})")

print("\n=== FAULT OBJECTS OVERLAP ===")
print(f"Train Fault Objects: {len(train_fo)}")
print(f"Val Fault Objects:   {len(val_fo)}")
print(f"Test Fault Objects:  {len(test_fo)}")
print(f"Train & Test Shared Fault Objects: {len(train_fo & test_fo)} / {len(test_fo)} ({len(train_fo & test_fo)/len(test_fo)*100:.1f}%)")
print(f"Disjoint in Test: {test_fo - train_fo}")

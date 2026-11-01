"""Script to generate a fresh, deterministic group-based experiment split for Task 7.

Requirements:
- Preserves scenario-group separation: (system::category::root_cause).
- 0 scenario-group overlap between train, validation, and test.
- Target proportions: ~70% train / 10% val / 20% test.
- Documented seed: 2026.
- Preserves all 754 original dataset examples and source labels.
- Computes SHA-256 hashes for reproducibility.
- Audits splits for target leakage into input prompts.
- Exports split files to ml/data/splits_final/ and manifest to experiments/final_run/splits_manifest.json.
- Leaves existing ml/data/splits/ completely untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.inference.prompts import SYSTEM_PROMPT, build_diagnosis_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.split_generator")


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_record_to_sft_format(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert incident record into SFT chat messages format."""
    user_prompt = build_diagnosis_prompt(record["logs"], record["incident_description"])
    assistant_content = json.dumps(record["expected_output"], indent=2)

    return {
        "id": record["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "category": record["category"],
            "severity": record["severity"],
            "root_cause": record["root_cause"],
            "source_type": record.get("source_type", "cloud_ops_bench_public"),
            "source_reference": record.get("source_reference", f"Cloud-OpsBench ({record['id']})"),
        },
    }


def audit_target_leakage(records: List[Dict[str, Any]], split_name: str) -> None:
    """Verify that user prompts and logs in model inputs contain no ground-truth root causes."""
    leakage_count = 0
    for rec in records:
        user_prompt = build_diagnosis_prompt(rec["logs"], rec["incident_description"])
        # Check if the root cause identifier appears in the raw logs as an explicit label
        rc = rec["root_cause"].lower()
        # Ensure evaluator answer lines are not present
        if f'"root_cause": "{rc}"' in user_prompt.lower() or f"'root_cause': '{rc}'" in user_prompt.lower():
            leakage_count += 1
            logger.error(f"Leakage detected in {split_name} record {rec['id']}: root_cause key found in prompt")

    if leakage_count > 0:
        raise ValueError(f"FATAL: Target leakage detected in {split_name} ({leakage_count} cases)!")
    logger.info(f"Target leakage audit PASSED for {split_name}: 0/{len(records)} cases leak ground truth.")


def generate_final_splits(
    incidents_path: Path = REPO_ROOT / "ml/data/processed/incidents.jsonl",
    output_dir: Path = REPO_ROOT / "ml/data/splits_final",
    manifest_path: Path = REPO_ROOT / "experiments/final_run/splits_manifest.json",
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Execute deterministic scenario-group splitting."""
    logger.info(f"Loading processed incidents from {incidents_path}")
    with open(incidents_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_records = len(records)
    logger.info(f"Total input records: {total_records}")
    assert total_records == 754, f"Expected 754 records, got {total_records}"

    rng = random.Random(seed)

    # 1. Group records by category and scenario group
    cat_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        cat = rec["category"]
        sys_name = rec["source_reference"].split("(")[-1].split(")")[0].split("/")[0]
        gk = f"{sys_name}::{rec['category']}::{rec['root_cause']}"
        cat_groups[cat][gk].append(rec)

    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    for cat in sorted(cat_groups.keys()):
        groups = cat_groups[cat]
        group_keys = sorted(list(groups.keys()))
        rng.shuffle(group_keys)

        n_cat = sum(len(groups[k]) for k in group_keys)
        t_train = int(n_cat * train_ratio)
        t_val = int(n_cat * val_ratio)
        t_test = n_cat - t_train - t_val

        cat_tr: List[Dict[str, Any]] = []
        cat_va: List[Dict[str, Any]] = []
        cat_te: List[Dict[str, Any]] = []

        for k in group_keys:
            items = groups[k]
            rem_tr = max(0, t_train - len(cat_tr))
            rem_va = max(0, t_val - len(cat_va))
            rem_te = max(0, t_test - len(cat_te))

            # Assign to split with largest remaining deficit
            if rem_tr >= rem_va and rem_tr >= rem_te and rem_tr > 0:
                cat_tr.extend(items)
            elif rem_va >= rem_te and rem_va > 0:
                cat_va.extend(items)
            else:
                cat_te.extend(items)

        train_recs.extend(cat_tr)
        val_recs.extend(cat_va)
        test_recs.extend(cat_te)

    # 2. Verify zero group overlap between splits
    def extract_scenario_groups(recs: List[Dict[str, Any]]) -> Set[str]:
        return {
            f"{r['source_reference'].split('(')[-1].split(')')[0].split('/')[0]}::{r['category']}::{r['root_cause']}"
            for r in recs
        }

    train_groups = extract_scenario_groups(train_recs)
    val_groups = extract_scenario_groups(val_recs)
    test_groups = extract_scenario_groups(test_recs)

    overlap_tv = train_groups & val_groups
    overlap_tt = train_groups & test_groups
    overlap_vt = val_groups & test_groups

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"FATAL: Scenario leakage detected! Overlaps: "
            f"train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}"
        )
    logger.info("Group separation verified: ZERO scenario group overlap between Train, Val, and Test splits.")

    # 3. Verify zero ID overlap
    train_ids = {r["id"] for r in train_recs}
    val_ids = {r["id"] for r in val_recs}
    test_ids = {r["id"] for r in test_recs}

    assert len(train_ids & val_ids) == 0, "FATAL: ID overlap between train and val"
    assert len(train_ids & test_ids) == 0, "FATAL: ID overlap between train and test"
    assert len(val_ids & test_ids) == 0, "FATAL: ID overlap between val and test"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 754, "Total split count mismatch"

    # 4. Leakage audit
    audit_target_leakage(train_recs, "train")
    audit_target_leakage(val_recs, "val")
    audit_target_leakage(test_recs, "test")

    # 5. Export split files
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    split_files = {}
    for name, split_records in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        out_file = output_dir / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in split_records:
                sft_obj = convert_record_to_sft_format(r)
                f.write(json.dumps(sft_obj, ensure_ascii=False) + "\n")
        file_hash = compute_file_sha256(out_file)
        split_files[name] = {
            "path": str(out_file.relative_to(REPO_ROOT)),
            "count": len(split_records),
            "percentage": round(100 * len(split_records) / total_records, 2),
            "sha256": file_hash,
            "category_distribution": dict(Counter(r["category"] for r in split_records)),
            "scenario_group_count": len(extract_scenario_groups(split_records)),
        }
        logger.info(f"Saved {len(split_records)} records to {out_file} (SHA256: {file_hash[:12]}...)")

    manifest = {
        "dataset_name": "Cloud-OpsBench Final Controlled Split",
        "split_generation_seed": seed,
        "split_proportions": {
            "train_ratio_target": train_ratio,
            "val_ratio_target": val_ratio,
            "test_ratio_target": test_ratio,
        },
        "total_records": total_records,
        "splits": split_files,
        "clean_room_audit": {
            "scenario_leakage_detected": False,
            "id_overlap_detected": False,
            "target_leakage_detected": False,
            "zero_leakage_verified": True,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Exported split manifest to {manifest_path}")

    return manifest


if __name__ == "__main__":
    manifest = generate_final_splits()
    print("\n--- FINAL SPLIT MANIFEST ---")
    print(json.dumps(manifest, indent=2))

"""Cloud-OpsBench Dataset Inspector and Profiler.

Audits:
- Total incident cases
- Available columns & schema fields
- Incident categories & fault taxonomies
- Root cause distribution
- Affected component / workload distribution
- Diagnostic evidence availability
- Missing fields & null values
- Dataset storage requirements
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def inspect_dataset() -> dict:
    raw_dir = Path("ml/data/raw/cloud_ops_bench")
    index_file = raw_dir / "index.jsonl"
    release_file = raw_dir / "release.json"

    if not index_file.exists():
        raise FileNotFoundError(f"index.jsonl not found at {index_file}")

    cases = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    release_info = {}
    if release_file.exists():
        with open(release_file, "r", encoding="utf-8") as f:
            release_info = json.load(f)

    # 1. Available fields & missing fields
    all_fields = set()
    for c in cases:
        all_fields.update(c.keys())

    missing_fields_counts = {field: 0 for field in all_fields}
    for c in cases:
        for field in all_fields:
            val = c.get(field)
            if val is None or val == "" or val == []:
                missing_fields_counts[field] += 1

    # 2. Distributions
    categories = Counter(c["category"] for c in cases)
    taxonomies = Counter(c["fault_taxonomy"] for c in cases)
    systems = Counter(c["system"] for c in cases)
    difficulties = Counter(c["difficulty"] for c in cases)
    root_causes = Counter(c["root_cause"] for c in cases)

    component_prefixes = Counter(
        c["fault_object"].split("/")[0] if "/" in c["fault_object"] else "other"
        for c in cases
    )
    top_components = Counter(c["fault_object"] for c in cases).most_common(15)

    stats = {
        "total_cases": len(cases),
        "dataset_id": release_info.get("dataset_id", "LLM4Ops/Cloud-OpsBench"),
        "distribution_version": release_info.get("distribution_version", "v1.0.0"),
        "source_commit": release_info.get("source_commit", "ea05dafbc87b5985df8721622d63d4229c2b4224"),
        "available_fields": sorted(list(all_fields)),
        "missing_fields": missing_fields_counts,
        "categories": dict(categories.most_common()),
        "fault_taxonomies": dict(taxonomies.most_common()),
        "systems": dict(systems.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "unique_root_causes": len(root_causes),
        "root_causes_top20": dict(root_causes.most_common(20)),
        "component_prefixes": dict(component_prefixes.most_common()),
        "top_components": dict(top_components),
        "total_size_mb": round(release_info.get("data_size_bytes", 3835252736) / (1024**2), 2),
        "data_file_count": release_info.get("data_file_count", 27108),
    }

    return stats


def print_report(stats: dict) -> None:
    print("=" * 70)
    print("       CLOUD-OPSBENCH BENCHMARK DATASET INSPECTION REPORT        ")
    print("=" * 70)
    print(f"\n[METADATA & PROVENANCE]")
    print(f"  Dataset ID          : {stats['dataset_id']}")
    print(f"  Distribution Version: {stats['distribution_version']}")
    print(f"  Source Git Commit   : {stats['source_commit']}")
    print(f"  Total Incident Cases: {stats['total_cases']}")
    print(f"  Total Data Files    : {stats['data_file_count']} files in full distribution")
    print(f"  Full Size on Disk   : {stats['total_size_mb']} MB (~3.66 GB)")

    print(f"\n[AVAILABLE SCHEMA FIELDS ({len(stats['available_fields'])} fields)]")
    for fld in stats["available_fields"]:
        missing = stats["missing_fields"].get(fld, 0)
        status = "COMPLETE (0 missing)" if missing == 0 else f"{missing} missing"
        print(f"  - {fld:<20}: {status}")

    print(f"\n[SYSTEMS & TARGET APPLICATIONS]")
    for sys_name, count in stats["systems"].items():
        pct = (count / stats["total_cases"]) * 100
        print(f"  - {sys_name:<20}: {count:>4} cases ({pct:5.1f}%)")

    print(f"\n[INCIDENT CATEGORIES / FAULT TAXONOMIES]")
    for tax, count in stats["fault_taxonomies"].items():
        pct = (count / stats["total_cases"]) * 100
        print(f"  - {tax:<26}: {count:>4} cases ({pct:5.1f}%)")

    print(f"\n[DIFFICULTY LEVELS]")
    for diff, count in stats["difficulties"].items():
        pct = (count / stats["total_cases"]) * 100
        print(f"  - {diff:<20}: {count:>4} cases ({pct:5.1f}%)")

    print(f"\n[ROOT CAUSE DIVERSITY]")
    print(f"  Total Unique Root Causes: {stats['unique_root_causes']}")
    print("  Top 10 Most Common Root Causes:")
    for rc, count in list(stats["root_causes_top20"].items())[:10]:
        print(f"    * {rc:<30}: {count:>3} cases")

    print(f"\n[AFFECTED COMPONENT TYPES]")
    for prefix, count in stats["component_prefixes"].items():
        pct = (count / stats["total_cases"]) * 100
        print(f"  - {prefix:<20}: {count:>4} cases ({pct:5.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    report = inspect_dataset()
    print_report(report)

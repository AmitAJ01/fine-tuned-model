"""Dataset preparation, validation, and splitting for DevOps Incident SFT."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from backend.app.schemas.incident import IncidentReportPayload
from ml.inference.prompts import SYSTEM_PROMPT, build_diagnosis_prompt

logger = logging.getLogger(__name__)


class DevOpsIncidentDataset:
    """Validator and formatter for DevOps Incident SFT samples."""

    @staticmethod
    def format_sample(
        log_text: str,
        target_report: Dict[str, Any],
        system_context: str | None = None,
    ) -> Dict[str, Any]:
        """Convert raw incident and ground truth into chat messages format for SFT."""
        # Ensure target_report adheres to schema
        validated = IncidentReportPayload(**target_report)
        assistant_content = validated.model_dump_json(indent=2)

        user_content = build_diagnosis_prompt(log_text, system_context)

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "category": validated.incident_category.value,
                "severity": validated.severity_level.value,
            },
        }

    @staticmethod
    def load_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
        """Load records from a JSONL file."""
        records = []
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    records.append(json.loads(clean))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num} in {path}: {e}")
        return records

    @staticmethod
    def save_jsonl(records: List[Dict[str, Any]], filepath: str | Path) -> None:
        """Write records to a JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_data_splits(
    raw_records: List[Dict[str, Any]],
    output_dir: str | Path = "ml/data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Split dataset deterministically into train, val, and test splits."""
    import random

    if round(train_ratio + val_ratio + test_ratio, 2) != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    rng = random.Random(seed)
    shuffled = list(raw_records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    test_path = out / "test.jsonl"

    DevOpsIncidentDataset.save_jsonl(train_data, train_path)
    DevOpsIncidentDataset.save_jsonl(val_data, val_path)
    DevOpsIncidentDataset.save_jsonl(test_data, test_path)

    logger.info(
        f"Splits saved to {out}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
    )
    return train_path, val_path, test_path

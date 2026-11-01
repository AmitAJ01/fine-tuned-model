"""Unit tests for Task 7: Completion-only loss masking and final experiment integrity."""

import json
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer
from trl.trainer.sft_trainer import get_training_chat_template
from backend.app.schemas.incident import IncidentCategory
from ml.evaluation.metrics import (
    compute_evidence_grounding_score,
    compute_fault_taxonomy_accuracy,
    compute_semantic_root_cause_score,
)


def test_completion_only_loss_masking():
    """Verify that prompt tokens are masked with -100 and assistant tokens are supervised."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    training_template = get_training_chat_template(tok)
    assert training_template is not None, "Failed to retrieve training chat template with generation markers"

    sample_messages = [
        {"role": "system", "content": "You are a DevOps assistant."},
        {"role": "user", "content": "Analyze these logs: OOM error in pod checkout."},
        {
            "role": "assistant",
            "content": json.dumps({
                "incident_category": "Infrastructure",
                "severity_level": "SEV-1",
                "root_cause_analysis": "OOM killer killed container",
                "contributing_factors": ["High memory spike"],
                "recommended_debugging_steps": ["kubectl top pod"],
                "suggested_remediation_actions": ["Increase memory limit"],
            }),
        },
    ]

    out = tok.apply_chat_template(
        sample_messages,
        chat_template=training_template,
        return_assistant_tokens_mask=True,
        return_dict=True,
        tokenize=True,
    )

    input_ids = out["input_ids"]
    assistant_masks = out["assistant_masks"]
    labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, assistant_masks)]

    total_tokens = len(input_ids)
    supervised_tokens = sum(assistant_masks)
    masked_tokens = total_tokens - supervised_tokens

    assert total_tokens > 0
    assert masked_tokens > 0, "Prompt tokens must be masked!"
    assert supervised_tokens > 0, "Assistant tokens must be supervised!"

    # Verify that the decoded supervised tokens contain the JSON content
    supervised_ids = [tid for tid, l in zip(input_ids, labels) if l != -100]
    decoded_supervised = tok.decode(supervised_ids)
    assert '"incident_category": "Infrastructure"' in decoded_supervised
    assert '"root_cause_analysis"' in decoded_supervised

    # Verify that prompt content is NOT in the supervised tokens
    assert "Analyze these logs" not in decoded_supervised
    assert "You are a DevOps assistant." not in decoded_supervised

    # Verify that <|im_end|> is included in the assistant mask
    im_end_id = tok.encode("<|im_end|>", add_special_tokens=False)[0]
    assert im_end_id in supervised_ids, "End-of-turn token must be supervised so model learns to stop"


def test_final_split_manifest_and_group_isolation():
    """Verify group-based split integrity and zero scenario leakage across final splits."""
    manifest_path = REPO_ROOT / "experiments/final_run/splits_manifest.json"
    assert manifest_path.exists(), "Splits manifest must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_records"] == 754
    assert manifest["splits"]["train"]["count"] == 536
    assert manifest["splits"]["val"]["count"] == 56
    assert manifest["splits"]["test"]["count"] == 162

    # Load records directly from jsonl files
    train_file = REPO_ROOT / "ml/data/splits_final/train.jsonl"
    val_file = REPO_ROOT / "ml/data/splits_final/val.jsonl"
    test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"

    assert train_file.exists() and val_file.exists() and test_file.exists()

    def get_records(path):
        return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

    train_recs = get_records(train_file)
    val_recs = get_records(val_file)
    test_recs = get_records(test_file)

    train_ids = {r["id"] for r in train_recs}
    val_ids = {r["id"] for r in val_recs}
    test_ids = {r["id"] for r in test_recs}

    assert len(train_ids & val_ids) == 0, "No ID overlap between train and val"
    assert len(train_ids & test_ids) == 0, "No ID overlap between train and test"
    assert len(val_ids & test_ids) == 0, "No ID overlap between val and test"

    def get_groups(recs):
        return {
            f"{r['metadata']['source_reference'].split('(')[-1].split(')')[0].split('/')[0]}::{r['metadata']['category']}::{r['metadata']['root_cause']}"
            for r in recs
        }

    tg = get_groups(train_recs)
    vg = get_groups(val_recs)
    teg = get_groups(test_recs)

    assert len(tg & vg) == 0, "Scenario leakage detected between train and val"
    assert len(tg & teg) == 0, "Scenario leakage detected between train and test"
    assert len(vg & teg) == 0, "Scenario leakage detected between val and test"


def test_no_target_leakage_in_test_prompts():
    """Verify that test user prompts do not leak evaluator ground truth."""
    test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
    recs = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]

    for r in recs:
        user_msg = r["messages"][1]["content"]
        rc = r["metadata"]["root_cause"]
        # Evaluator answer format must not be in user input
        assert f'"root_cause": "{rc}"' not in user_msg
        assert f"'root_cause': '{rc}'" not in user_msg


def test_canonical_categories_match_schema():
    """Verify that all categories in final splits match the canonical IncidentCategory enum."""
    test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
    recs = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]

    valid_cats = {c.value for c in IncidentCategory}
    for r in recs:
        cat = r["metadata"]["category"]
        assert cat in valid_cats, f"Category '{cat}' is not in IncidentCategory enum"


def test_metric_reproducibility_helpers():
    """Verify metric computation reproducibility on deterministic inputs."""
    preds = [
        {"incident_category": "Database", "root_cause_analysis": "Connection timeout in HikariPool"},
        {"incident_category": "Application", "root_cause_analysis": "NullPointerException in service"},
    ]
    refs = [
        {"category": "Database", "root_cause": "mysql_connection_loss"},
        {"category": "Application", "root_cause": "code_defect"},
    ]

    acc = compute_fault_taxonomy_accuracy(preds, refs)
    assert acc == 1.0

    # Semantic root cause rubric check
    score = compute_semantic_root_cause_score("Connection timeout in HikariPool", "mysql_connection_loss", "app/ts-order-service")
    assert 0.0 <= score <= 1.0

    # Evidence keyword overlap check
    ev_score = compute_evidence_grounding_score(
        {"root_cause_analysis": "Connection timeout in order service", "recommended_debugging_steps": ["kubectl logs"]},
        "order service reported Connection timeout with HikariPool error in app/ts-order-service",
    )
    assert ev_score >= 0.5

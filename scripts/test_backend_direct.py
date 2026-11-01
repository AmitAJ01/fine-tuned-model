"""Test backend routers and services directly."""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.routes.health import health_check, hardware_status
from backend.app.routes.evaluation import get_full_comparison_metrics
from backend.app.routes.incident import diagnose
from backend.app.schemas.incident import IncidentRequest, ModelTarget
from backend.app.services.inference_service import get_inference_service

async def main():
    print("Testing /health endpoint...")
    h = await health_check()
    print("Health response:", h)
    assert h["status"] == "healthy"
    print("✓ Health endpoint OK")

    print("\nTesting /hardware endpoint...")
    hw = await hardware_status()
    print(f"Hardware response: RAM={hw['ram']['total_gb']}GB, GPU detected={hw['gpu']['gpu_detected']}")
    print("✓ Hardware endpoint OK")

    print("\nTesting /evaluation/comparison endpoint...")
    eval_comp = await get_full_comparison_metrics(version="final")
    print(f"Comparison response keys: {list(eval_comp.keys())}")
    assert "metrics_comparison" in eval_comp
    print("✓ Evaluation comparison endpoint OK")

    print("\nTesting /incident/diagnose with simulation/mock...")
    service = get_inference_service()

    # Test baseline routing
    req_base = IncidentRequest(
        log_text="kernel: Out of memory: Killed process 4123 (node) total-vm:4194304kB, anon-rss:2097152kB",
        system_context="E-commerce boutique microservice on Kubernetes",
        model_target=ModelTarget.BASELINE,
        max_tokens=768,
        mock=True,
    )
    resp_base = await diagnose(req_base, service=service)
    print("Baseline diagnosis response:")
    print(f"  Model used: {resp_base.model_used}")
    print(f"  Category:   {resp_base.report.incident_category if resp_base.report else 'N/A'}")
    print(f"  Severity:   {resp_base.report.severity_level if resp_base.report else 'N/A'}")
    print(f"  JSON valid: {resp_base.json_valid}")
    assert resp_base.json_valid is True
    assert resp_base.is_fine_tuned is False

    # Test fine-tuned routing
    req_ft = IncidentRequest(
        log_text="postgres: FATAL: remaining connection slots are reserved for non-replication superuser connections",
        system_context="Order database PostgreSQL",
        model_target=ModelTarget.FINETUNED,
        max_tokens=768,
        mock=True,
    )
    resp_ft = await diagnose(req_ft, service=service)
    print("\nFine-tuned diagnosis response:")
    print(f"  Model used: {resp_ft.model_used}")
    print(f"  Category:   {resp_ft.report.incident_category if resp_ft.report else 'N/A'}")
    print(f"  Severity:   {resp_ft.report.severity_level if resp_ft.report else 'N/A'}")
    print(f"  JSON valid: {resp_ft.json_valid}")
    assert resp_ft.json_valid is True
    assert resp_ft.is_fine_tuned is True

    print("\n✓ ALL BACKEND ENDPOINTS AND ROUTING VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())

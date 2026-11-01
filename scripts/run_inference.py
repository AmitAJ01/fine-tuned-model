"""CLI runner for testing incident diagnosis inference on baseline vs. fine-tuned model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.incident import IncidentRequest, ModelTarget
from backend.app.services.inference_service import get_inference_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="FineTuneOps Incident Inference Runner")
    parser.add_argument(
        "--model",
        choices=["baseline", "finetuned"],
        default="finetuned",
        help="Model target (baseline vs finetuned)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="2026-10-31T14:10:04Z kubelet: OOMKilled: Pod billing-service exceeded memory limit (16Gi)",
        help="Log excerpt or incident text",
    )
    parser.add_argument("--context", type=str, default=None, help="System architecture context")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature")
    args = parser.parse_args()

    service = get_inference_service()
    req = IncidentRequest(
        log_text=args.log,
        system_context=args.context,
        model_target=ModelTarget.BASELINE if args.model == "baseline" else ModelTarget.FINETUNED,
        temperature=args.temp,
    )

    print(f"\n[Running Inference: Model Target = {args.model.upper()}]")
    res = service.diagnose_incident(req)

    print(f"Model ID   : {res.model_used}")
    print(f"Latency    : {res.latency_ms} ms")
    print(f"JSON Valid : {res.json_valid}")
    print("\n--- Structured Incident Report ---")
    print(json.dumps(res.report.model_dump(), indent=2))


if __name__ == "__main__":
    main()

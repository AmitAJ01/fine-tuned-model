"""Service layer handling model inference and business validation."""

from __future__ import annotations

import logging
from typing import Optional
from backend.app.core.config import settings
from backend.app.schemas.incident import (
    IncidentDiagnosisResponse,
    IncidentReportPayload,
    IncidentRequest,
    ModelTarget,
)
from ml.inference.engine import IncidentInferenceEngine

logger = logging.getLogger(__name__)


class InferenceService:
    """Manages LLM inference execution for baseline and fine-tuned configurations."""

    def __init__(self) -> None:
        self.engine = IncidentInferenceEngine(
            base_model_name=settings.BASE_MODEL_NAME,
            finetuned_adapter_path=str(settings.PROJECT_ROOT / settings.FINETUNED_ADAPTER_PATH),
            device_map=settings.DEVICE,
        )

    def diagnose_incident(self, request: IncidentRequest) -> IncidentDiagnosisResponse:
        """Route incident request to requested model and validate structured output."""
        use_finetuned = request.model_target == ModelTarget.FINETUNED

        raw_result = self.engine.generate_diagnosis(
            log_text=request.log_text,
            system_context=request.system_context,
            use_finetuned=use_finetuned,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            mock=request.mock,
        )

        report = None
        is_valid = raw_result.get("json_valid", True)
        error_detail = raw_result.get("error_detail")

        if is_valid and raw_result.get("report"):
            try:
                report = IncidentReportPayload(**raw_result["report"])
            except Exception as e:
                is_valid = False
                error_detail = f"Pydantic schema validation error: {str(e)}"
                report = None

        return IncidentDiagnosisResponse(
            report=report,
            model_used=raw_result["model_used"],
            is_fine_tuned=raw_result["is_fine_tuned"],
            latency_ms=raw_result["latency_ms"],
            raw_response=raw_result["raw_response"],
            tokens_generated=raw_result.get("tokens_generated"),
            json_valid=is_valid,
            error_detail=error_detail,
        )


_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Singleton getter for InferenceService."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service

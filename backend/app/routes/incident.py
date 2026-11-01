"""Incident diagnosis API router."""

from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.schemas.incident import IncidentDiagnosisResponse, IncidentRequest
from backend.app.services.inference_service import InferenceService, get_inference_service

router = APIRouter(prefix="/incident", tags=["Incident Diagnosis"])


@router.post(
    "/diagnose",
    response_model=IncidentDiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Diagnose DevOps Incident",
    description="Accepts DevOps error logs or incident description and produces a structured diagnosis.",
)
async def diagnose(
    request: IncidentRequest,
    service: InferenceService = Depends(get_inference_service),
) -> IncidentDiagnosisResponse:
    try:
        return service.diagnose_incident(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )

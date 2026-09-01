"""FastAPI routes for patient clinical analysis."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from models.patient import PatientRequest
from models.treatment import AnalysisResponse
from services.clinical_analysis import ClinicalAnalysisService

router = APIRouter(prefix="/api/patient", tags=["patient"])


@lru_cache(maxsize=1)
def get_clinical_analysis_service() -> ClinicalAnalysisService:
    """Create and fit the application models once per process."""
    return ClinicalAnalysisService(random_state=42)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_patient(patient: PatientRequest) -> AnalysisResponse:
    """Analyze one validated synthetic patient through the existing Epic pipeline."""
    try:
        result = get_clinical_analysis_service().analyze(patient)
        return result.to_api_response(request_id=_next_request_id())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_request_number = 0


def _next_request_id() -> str:
    global _request_number
    _request_number += 1
    return f"REQUEST-{_request_number:06d}"

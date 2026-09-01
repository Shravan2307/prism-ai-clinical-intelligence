"""Structured API response models for clinical analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RiskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    score: float
    factors: list[dict[str, Any]]


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    doctor_review_required: bool
    message: str


class TreatmentOptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    estimated_effect: float
    rank: int
    explanations: list[dict[str, Any]]


class TreatmentAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_options: list[TreatmentOptionResponse]
    label: str = "estimated treatment-effect ranking for clinician review"


class DoctorCaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: str
    priority: str


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: str
    risk: RiskResponse
    triage: TriageResponse
    treatment_analysis: TreatmentAnalysisResponse | None = None
    doctor_case: DoctorCaseResponse | None = None
    health_report: dict[str, Any]


class PatientAnalysisResponse(AnalysisResponse):
    """Named response type for the patient analysis endpoint."""

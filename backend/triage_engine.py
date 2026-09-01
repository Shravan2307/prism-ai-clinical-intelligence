"""Deterministic care-navigation triage for the application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.risk_engine import RiskAssessment


@dataclass(frozen=True)
class TriageDecision:
    category: str
    doctor_review_required: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "doctor_review_required": self.doctor_review_required,
            "message": self.message,
        }


class TriageEngine:
    """Route patients to care-navigation workflows without diagnosing them."""

    def triage(self, risk_assessment: RiskAssessment) -> TriageDecision:
        if not isinstance(risk_assessment, RiskAssessment):
            raise ValueError("risk_assessment must be a RiskAssessment")
        if risk_assessment.emergency or risk_assessment.risk_level == "EMERGENCY":
            return TriageDecision("emergency", True, "Immediate professional medical evaluation is recommended.")
        if risk_assessment.risk_level == "HIGH":
            return TriageDecision("urgent_doctor_review", True, "Priority clinical review is recommended.")
        if risk_assessment.risk_level == "MODERATE":
            return TriageDecision("doctor_review", bool(risk_assessment.clinician_review_required), "Clinical evaluation may be appropriate.")
        if risk_assessment.risk_level == "LOW":
            return TriageDecision("routine_follow_up", bool(risk_assessment.clinician_review_required), "Routine monitoring may be appropriate.")
        raise ValueError("Unsupported risk level")

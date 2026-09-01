"""Hard safety boundary for the Epic 7 application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.risk_engine import RiskAssessment


@dataclass(frozen=True)
class SafetyDecision:
    emergency: bool
    doctor_review_required: bool
    safe_candidates: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "emergency": self.emergency,
            "doctor_review_required": self.doctor_review_required,
            "safe_candidates": list(self.safe_candidates),
            "message": self.message,
        }


class SafetyGate:
    """Apply the existing Epic 4 safety result before treatment analysis."""

    SUPPORTED_TREATMENTS = ("Metformin", "SGLT2i", "GLP-1")

    def evaluate(self, risk_assessment: RiskAssessment) -> SafetyDecision:
        if not isinstance(risk_assessment, RiskAssessment):
            raise ValueError("risk_assessment must be a RiskAssessment")
        if risk_assessment.emergency or risk_assessment.risk_level == "EMERGENCY":
            return SafetyDecision(
                emergency=True,
                doctor_review_required=True,
                safe_candidates=(),
                message="Immediate professional medical evaluation is recommended. Do not continue treatment analysis through this emergency path.",
            )
        return SafetyDecision(
            emergency=False,
            doctor_review_required=bool(risk_assessment.clinician_review_required),
            safe_candidates=self.SUPPORTED_TREATMENTS,
            message="No emergency override was triggered; treatment effects remain analytical outputs for clinician review only.",
        )

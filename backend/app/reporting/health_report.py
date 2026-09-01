"""Deterministic explainable health reports for the Epic 6 prototype.

This module is a presentation and care-navigation layer. It does not diagnose,
prescribe, or make autonomous treatment decisions. All source data is expected
to come from Epics 3, 4, and 5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from app.clinical.doctor_workflow import ClinicalCase
from app.ml.risk_engine import RiskAssessment
from app.ml.treatment_ranker import FeatureContribution, RankedTreatment


@dataclass(frozen=True)
class FeatureExplanation:
    feature: str
    contribution: float
    direction: str
    explanation: str


@dataclass(frozen=True)
class AnalyticalFinding:
    treatment: str
    expected_hba1c_reduction: float
    rank: int


@dataclass(frozen=True)
class HealthReport:
    report_id: str
    case_id: str
    patient_id: str
    risk_level: str
    risk_score: float
    summary: str
    key_risk_indicators: list[str]
    explainability: list[FeatureExplanation]
    analytical_findings: list[AnalyticalFinding]
    recommended_next_step: str
    clinician_review_required: bool
    emergency: bool
    emergency_guidance: str | None
    limitations: list[str]
    disclaimer: str
    generated_at: str
    case_status: str | None = None
    case_priority: str | None = None
    ai_recommendation: str | None = None
    doctor_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary containing no enum or NumPy values."""
        return _json_safe(asdict(self))


class HealthReportGenerator:
    """Combine already-computed clinical, analytical, and workflow results."""

    _next_report_number = 1
    _CARE_NAVIGATION = {
        "ROUTINE_SELF_CARE": "ROUTINE_MONITORING",
        "ROUTINE_CLINICAL_REVIEW": "CLINICAL_EVALUATION",
        "PRIORITY_CLINICAL_REVIEW": "PRIORITY_CLINICAL_REVIEW",
        "URGENT_EMERGENCY_CARE": "URGENT_EMERGENCY_CARE",
    }
    _LIMITATIONS = [
        "This report is generated from the information available to the system.",
        "Risk estimates are not a diagnosis.",
        "Model outputs may contain uncertainty or error.",
        "Treatment-effect estimates are analytical and are not prescriptions.",
        "A qualified healthcare professional should make clinical decisions.",
        "If symptoms are severe or rapidly worsening, seek appropriate medical care.",
    ]
    _DISCLAIMER = (
        "This report is intended for informational and clinical decision-support purposes only. "
        "It does not provide a medical diagnosis, prescription, or substitute for evaluation "
        "by a qualified healthcare professional."
    )

    def generate(
        self,
        clinical_case: ClinicalCase,
        risk_assessment: RiskAssessment,
        ranked_treatments: Sequence[RankedTreatment] | None = None,
        explanations: Mapping[str, Sequence[FeatureContribution]] | None = None,
    ) -> HealthReport:
        self._validate_dependencies(clinical_case, risk_assessment)
        ranked = self._validate_ranked_treatments(ranked_treatments)
        explanation_values = self._validate_explanations(explanations, ranked)
        report_id = f"REPORT-{self._next_report_number:06d}"
        type(self)._next_report_number += 1

        # Emergency cases remain on the Epic 4 safety path; analytical treatment
        # information is not surfaced as though ordinary workflow continued.
        if risk_assessment.emergency:
            ranked = []
            explanation_values = []
        return HealthReport(
            report_id=report_id,
            case_id=clinical_case.case_id,
            patient_id=clinical_case.patient_id,
            risk_level=risk_assessment.risk_level,
            risk_score=float(risk_assessment.risk_score),
            summary=self._summary_for(risk_assessment.risk_level),
            key_risk_indicators=[factor.name for factor in risk_assessment.risk_factors],
            explainability=explanation_values,
            analytical_findings=[
                AnalyticalFinding(item.treatment, float(item.expected_hba1c_reduction), int(item.rank))
                for item in ranked
            ],
            recommended_next_step=self._CARE_NAVIGATION[clinical_case.care_pathway],
            clinician_review_required=bool(risk_assessment.clinician_review_required),
            emergency=bool(risk_assessment.emergency),
            emergency_guidance=(
                "URGENT MEDICAL ATTENTION REQUIRED. Seek immediate professional medical assistance "
                "or contact local emergency services if you believe you are experiencing a medical emergency."
                if risk_assessment.emergency
                else None
            ),
            limitations=list(self._LIMITATIONS),
            disclaimer=self._DISCLAIMER,
            generated_at=datetime.now(timezone.utc).isoformat(),
            case_status=clinical_case.status,
            case_priority=clinical_case.priority,
            ai_recommendation=clinical_case.ai_recommendation,
            doctor_decision=(clinical_case.doctor_review.decision if clinical_case.doctor_review else None),
        )

    @staticmethod
    def _summary_for(risk_level: str) -> str:
        summaries = {
            "LOW": "The current information shows relatively low-risk indicators. Continue appropriate routine health monitoring and discuss concerns with a healthcare professional if symptoms persist or change.",
            "MODERATE": "The current information shows some health indicators that may benefit from further clinical evaluation.",
            "HIGH": "The current information shows elevated health-risk indicators. Clinical evaluation is recommended to better understand these findings.",
            "EMERGENCY": "The available information contains indicators requiring urgent professional medical attention.",
        }
        return summaries[risk_level]

    @classmethod
    def _validate_dependencies(cls, clinical_case: ClinicalCase, risk_assessment: RiskAssessment) -> None:
        if not isinstance(clinical_case, ClinicalCase):
            raise ValueError("clinical_case must be a ClinicalCase")
        if not isinstance(risk_assessment, RiskAssessment):
            raise ValueError("risk_assessment must be a RiskAssessment")
        if not clinical_case.case_id or not clinical_case.patient_id:
            raise ValueError("clinical_case must contain non-empty case_id and patient_id")
        if clinical_case.patient_id != clinical_case.patient_id.strip():
            raise ValueError("clinical_case patient_id is invalid")
        if risk_assessment.risk_level not in {"LOW", "MODERATE", "HIGH", "EMERGENCY"}:
            raise ValueError("Unsupported risk level")
        if not isinstance(risk_assessment.risk_score, (int, float)) or not math.isfinite(float(risk_assessment.risk_score)) or not 0 <= float(risk_assessment.risk_score) <= 100:
            raise ValueError("risk_score must be a finite value from 0 to 100")
        if clinical_case.risk_level != risk_assessment.risk_level or float(clinical_case.risk_score) != float(risk_assessment.risk_score):
            raise ValueError("ClinicalCase and RiskAssessment risk information does not match")
        if clinical_case.emergency != risk_assessment.emergency:
            raise ValueError("ClinicalCase and RiskAssessment emergency status does not match")
        if clinical_case.emergency and clinical_case.care_pathway != "URGENT_EMERGENCY_CARE":
            raise ValueError("Emergency case must preserve URGENT_EMERGENCY_CARE")
        if clinical_case.care_pathway not in cls._CARE_NAVIGATION:
            raise ValueError("Unsupported care pathway")

    @staticmethod
    def _validate_ranked_treatments(
        ranked_treatments: Sequence[RankedTreatment] | None,
    ) -> list[RankedTreatment]:
        if ranked_treatments is None:
            return []
        if not isinstance(ranked_treatments, (list, tuple)):
            raise ValueError("ranked_treatments must be a list of RankedTreatment")
        result = list(ranked_treatments)
        seen: set[str] = set()
        for item in result:
            if not isinstance(item, RankedTreatment):
                raise ValueError("ranked_treatments contains an invalid item")
            if item.treatment not in {"Metformin", "SGLT2i", "GLP-1"}:
                raise ValueError(f"Unsupported treatment: {item.treatment}")
            if item.treatment in seen:
                raise ValueError(f"Duplicate treatment: {item.treatment}")
            seen.add(item.treatment)
            if not isinstance(item.rank, int) or item.rank <= 0 or not isinstance(item.expected_hba1c_reduction, (int, float)) or not math.isfinite(float(item.expected_hba1c_reduction)):
                raise ValueError("Treatment effects and ranks must be finite valid values")
        return result

    @staticmethod
    def _validate_explanations(
        explanations: Mapping[str, Sequence[FeatureContribution]] | None,
        ranked: list[RankedTreatment],
    ) -> list[FeatureExplanation]:
        if explanations is None:
            return []
        if not isinstance(explanations, Mapping):
            raise ValueError("explanations must be a mapping")
        allowed_treatments = {item.treatment for item in ranked}
        invalid_treatments = set(explanations) - allowed_treatments
        if invalid_treatments:
            raise ValueError(f"Explanations contain treatments not in ranked_treatments: {sorted(invalid_treatments)}")
        output: list[FeatureExplanation] = []
        for treatment in [item.treatment for item in ranked]:
            for contribution in explanations.get(treatment, []):
                if not isinstance(contribution, FeatureContribution):
                    raise ValueError("explanations contains an invalid FeatureContribution")
                value = float(contribution.contribution)
                if not contribution.feature or not math.isfinite(value):
                    raise ValueError("SHAP contributions must have valid names and finite values")
                positive = value >= 0
                output.append(
                    FeatureExplanation(
                        feature=contribution.feature,
                        contribution=value,
                        direction="POSITIVE_MODEL_CONTRIBUTION" if positive else "NEGATIVE_MODEL_CONTRIBUTION",
                        explanation=(
                            f"{contribution.feature} contributed {'positively' if positive else 'negatively'} "
                            "to the model's estimated treatment effect; this is not clinical causation."
                        ),
                    )
                )
        return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value

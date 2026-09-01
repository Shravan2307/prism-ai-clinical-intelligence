"""Application-level orchestration for the complete Epic 1–7 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from app.clinical.case_manager import CaseManager
from app.clinical.doctor_workflow import ClinicalCase
from app.ml.causal_data import generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine
from app.ml.risk_engine import ClinicalRiskEngine, RiskAssessment
from app.ml.treatment_ranker import FeatureContribution, RankedTreatment, TreatmentRankingEngine
from models.patient import PatientRequest
from models.treatment import (
    AnalysisResponse,
    DoctorCaseResponse,
    RiskResponse,
    TreatmentAnalysisResponse,
    TreatmentOptionResponse,
    TriageResponse,
)
from app.reporting.health_report import HealthReportGenerator
from safety_gate import SafetyDecision, SafetyGate
from triage_engine import TriageDecision, TriageEngine


@dataclass(frozen=True)
class ClinicalAnalysisResult:
    status: str
    risk: RiskAssessment
    triage: TriageDecision
    safety: SafetyDecision
    ranked_treatments: list[RankedTreatment]
    explanations: dict[str, list[FeatureContribution]]
    clinical_case: ClinicalCase | None
    health_report: Any

    def to_api_response(self, request_id: str) -> AnalysisResponse:
        treatment_analysis = None
        if not self.safety.emergency:
            treatment_analysis = TreatmentAnalysisResponse(
                ranked_options=[
                    TreatmentOptionResponse(
                        treatment=item.treatment,
                        estimated_effect=float(item.expected_hba1c_reduction),
                        rank=int(item.rank),
                        explanations=[asdict(value) for value in self.explanations.get(item.treatment, [])],
                    )
                    for item in self.ranked_treatments
                ]
            )
        doctor_case = None
        if self.clinical_case is not None:
            doctor_case = DoctorCaseResponse(
                case_id=self.clinical_case.case_id,
                status=self.clinical_case.status,
                priority=self.clinical_case.priority,
            )
        return AnalysisResponse(
            request_id=request_id,
            status=self.status,
            risk=RiskResponse(
                level=self.risk.risk_level,
                score=float(self.risk.risk_score),
                factors=[asdict(factor) for factor in self.risk.risk_factors],
            ),
            triage=TriageResponse(**self.triage.to_dict()),
            treatment_analysis=treatment_analysis,
            doctor_case=doctor_case,
            health_report=self.health_report.to_dict(),
        )


class ClinicalAnalysisService:
    """Reuse and sequence the existing Epic 1–6 engines for API requests."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.risk_engine = ClinicalRiskEngine()
        self.triage_engine = TriageEngine()
        self.safety_gate = SafetyGate()
        self.case_manager = CaseManager()
        self.report_generator = HealthReportGenerator()
        training_data = generate_synthetic_causal_data(random_state=random_state)
        self.causal_engine = CausalTreatmentEffectEngine(random_state=random_state).fit(training_data)
        self.ranking_engine = TreatmentRankingEngine(self.causal_engine)

    def analyze(self, patient: PatientRequest | dict[str, Any], patient_id: str = "API-PATIENT") -> ClinicalAnalysisResult:
        request = patient if isinstance(patient, PatientRequest) else PatientRequest.model_validate(patient)
        self._validate_patient_id(patient_id)
        patient_data = request.to_patient_dict()
        risk = self.risk_engine.assess(pd.Series(patient_data))
        triage = self.triage_engine.triage(risk)
        safety = self.safety_gate.evaluate(risk)
        if safety.emergency:
            ranked: list[RankedTreatment] = []
            explanations: dict[str, list[FeatureContribution]] = {}
        else:
            analysis = self.ranking_engine.analyze(patient_data, list(safety.safe_candidates))
            ranked = analysis["ranked_treatments"]
            explanations = analysis["explanations"]

        clinical_case = None
        if triage.doctor_review_required:
            clinical_case = self.case_manager.create_case(patient_id, risk)
        report_case = clinical_case or self._report_case(patient_id, risk)
        report = self.report_generator.generate(report_case, risk, ranked, explanations)
        return ClinicalAnalysisResult(
            status="emergency" if safety.emergency else "completed",
            risk=risk,
            triage=triage,
            safety=safety,
            ranked_treatments=ranked,
            explanations=explanations,
            clinical_case=clinical_case,
            health_report=report,
        )

    @staticmethod
    def _validate_patient_id(patient_id: str) -> None:
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("patient_id must be a non-empty string")

    @staticmethod
    def _report_case(patient_id: str, risk: RiskAssessment) -> ClinicalCase:
        priority = {
            "EMERGENCY": "CRITICAL",
            "HIGH": "HIGH",
            "MODERATE": "NORMAL",
            "LOW": "ROUTINE",
        }[risk.risk_level]
        return ClinicalCase(
            case_id="NO_DOCTOR_CASE",
            patient_id=patient_id,
            risk_score=float(risk.risk_score),
            risk_level=risk.risk_level,
            emergency=bool(risk.emergency),
            clinician_review_required=bool(risk.clinician_review_required),
            risk_factors=list(risk.risk_factors),
            care_pathway=risk.care_pathway,
            ai_recommendation=risk.care_pathway,
            status="NOT_CREATED",
            priority=priority,
            created_at="NOT_AVAILABLE",
        )

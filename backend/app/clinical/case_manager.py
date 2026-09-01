"""In-memory clinical case management for the Epic 5 doctor workflow."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any

from app.ml.risk_engine import RiskAssessment

from .doctor_workflow import (
    AuditEvent,
    CasePriority,
    CaseStatus,
    ClinicalCase,
    DoctorDecision,
    DoctorReview,
)


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PRIORITY_ORDER = {
    CasePriority.CRITICAL.value: 0,
    CasePriority.HIGH.value: 1,
    CasePriority.NORMAL.value: 2,
    CasePriority.ROUTINE.value: 3,
}
_ALLOWED_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "EMERGENCY"}
_ALLOWED_PATHWAYS = {
    "ROUTINE_SELF_CARE",
    "ROUTINE_CLINICAL_REVIEW",
    "PRIORITY_CLINICAL_REVIEW",
    "URGENT_EMERGENCY_CARE",
}
_TRANSITIONS = {
    CaseStatus.PENDING_REVIEW.value: {CaseStatus.IN_REVIEW.value},
    CaseStatus.IN_REVIEW.value: {
        CaseStatus.REVIEWED.value,
        CaseStatus.ESCALATED.value,
    },
    CaseStatus.REVIEWED.value: {CaseStatus.CLOSED.value},
    CaseStatus.ESCALATED.value: set(),
    CaseStatus.CLOSED.value: set(),
}


class InMemoryCaseRepository:
    """Small replaceable repository abstraction for the hackathon prototype."""

    def __init__(self) -> None:
        self._cases: dict[str, ClinicalCase] = {}
        self._audit: dict[str, list[AuditEvent]] = {}

    def add(self, case: ClinicalCase) -> None:
        if case.case_id in self._cases:
            raise ValueError(f"Case already exists: {case.case_id}")
        self._cases[case.case_id] = deepcopy(case)
        self._audit[case.case_id] = []

    def get(self, case_id: str) -> ClinicalCase:
        if case_id not in self._cases:
            raise KeyError(f"Case not found: {case_id}")
        return deepcopy(self._cases[case_id])

    def save(self, case: ClinicalCase) -> None:
        if case.case_id not in self._cases:
            raise KeyError(f"Case not found: {case.case_id}")
        self._cases[case.case_id] = deepcopy(case)

    def all(self) -> list[ClinicalCase]:
        return [deepcopy(case) for case in self._cases.values()]

    def add_audit(self, event: AuditEvent) -> None:
        if event.case_id not in self._audit:
            raise KeyError(f"Case not found: {event.case_id}")
        self._audit[event.case_id].append(event)

    def audit(self, case_id: str) -> list[AuditEvent]:
        if case_id not in self._audit:
            raise KeyError(f"Case not found: {case_id}")
        return list(self._audit[case_id])


class CaseManager:
    """Create, queue, review, escalate, close, and audit clinical cases."""

    def __init__(self, repository: InMemoryCaseRepository | None = None) -> None:
        self.repository = repository or InMemoryCaseRepository()
        self._next_case_number = 1

    def create_case(self, patient_id: str, risk_assessment: RiskAssessment) -> ClinicalCase:
        self._validate_id(patient_id, "patient_id")
        self._validate_assessment(risk_assessment)
        case_id = self._new_case_id()
        priority = self._priority_for(risk_assessment.risk_level)
        if risk_assessment.emergency:
            priority = CasePriority.CRITICAL.value
        case = ClinicalCase(
            case_id=case_id,
            patient_id=patient_id,
            risk_score=float(risk_assessment.risk_score),
            risk_level=risk_assessment.risk_level,
            emergency=bool(risk_assessment.emergency),
            clinician_review_required=bool(risk_assessment.clinician_review_required),
            risk_factors=deepcopy(risk_assessment.risk_factors),
            care_pathway=risk_assessment.care_pathway,
            ai_recommendation=risk_assessment.care_pathway,
            status=CaseStatus.PENDING_REVIEW.value,
            priority=priority,
            created_at=self._now(),
        )
        self.repository.add(case)
        self._audit(case_id, "SYSTEM", "SYSTEM", "CASE_CREATED", "Clinical case created from Epic 4 RiskAssessment.")
        return self.repository.get(case_id)

    def get_doctor_queue(self) -> list[dict[str, Any]]:
        """Return only fields required for prioritization and review triage."""
        active = [case for case in self.repository.all() if case.status != CaseStatus.CLOSED.value]
        active.sort(key=lambda case: (_PRIORITY_ORDER[case.priority], case.created_at, case.case_id))
        return [
            {
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "risk_level": case.risk_level,
                "priority": case.priority,
                "status": case.status,
                "clinician_review_required": case.clinician_review_required,
                "created_at": case.created_at,
            }
            for case in active
        ]

    def get_case(self, case_id: str) -> ClinicalCase:
        self._validate_case_id(case_id)
        return self.repository.get(case_id)

    def start_review(self, case_id: str, doctor_id: str) -> ClinicalCase:
        self._validate_case_id(case_id)
        self._validate_id(doctor_id, "doctor_id")
        case = self.repository.get(case_id)
        self._require_transition(case, CaseStatus.IN_REVIEW.value)
        case.status = CaseStatus.IN_REVIEW.value
        case._reviewing_doctor_id = doctor_id
        self.repository.save(case)
        self._audit(case_id, "DOCTOR", doctor_id, "REVIEW_STARTED", "Doctor started case review.")
        return self.repository.get(case_id)

    def submit_review(self, case_id: str, doctor_id: str, decision: str, clinical_notes: str) -> ClinicalCase:
        self._validate_case_id(case_id)
        self._validate_id(doctor_id, "doctor_id")
        if not isinstance(decision, str) or decision not in {item.value for item in DoctorDecision}:
            raise ValueError(f"Unsupported doctor decision: {decision}")
        if not isinstance(clinical_notes, str) or not clinical_notes.strip():
            raise ValueError("clinical_notes must be a non-empty string")
        case = self.repository.get(case_id)
        self._require_transition(case, CaseStatus.REVIEWED.value if decision != DoctorDecision.ESCALATE.value else CaseStatus.ESCALATED.value)
        reviewing_doctor = getattr(case, "_reviewing_doctor_id", doctor_id)
        if reviewing_doctor != doctor_id:
            raise ValueError("Only the doctor who started the review may submit it")
        case.doctor_review = DoctorReview(case_id, doctor_id, decision, clinical_notes, self._now())
        case.status = CaseStatus.ESCALATED.value if decision == DoctorDecision.ESCALATE.value else CaseStatus.REVIEWED.value
        self.repository.save(case)
        self._audit(case_id, "DOCTOR", doctor_id, "DECISION_SUBMITTED", f"Doctor decision submitted: {decision}.")
        if decision == DoctorDecision.ESCALATE.value:
            self._audit(case_id, "DOCTOR", doctor_id, "CASE_ESCALATED", "Doctor escalated the case for further clinical attention.")
        return self.repository.get(case_id)

    def close_case(self, case_id: str, doctor_id: str) -> ClinicalCase:
        self._validate_case_id(case_id)
        self._validate_id(doctor_id, "doctor_id")
        case = self.repository.get(case_id)
        self._require_transition(case, CaseStatus.CLOSED.value)
        if case.doctor_review is None or case.doctor_review.doctor_id != doctor_id:
            raise ValueError("Only the reviewing doctor may close a reviewed case")
        case.status = CaseStatus.CLOSED.value
        self.repository.save(case)
        self._audit(case_id, "DOCTOR", doctor_id, "CASE_CLOSED", "Reviewed case closed by doctor.")
        return self.repository.get(case_id)

    def get_audit_history(self, case_id: str) -> list[AuditEvent]:
        self._validate_case_id(case_id)
        return list(self.repository.audit(case_id))

    def _new_case_id(self) -> str:
        while True:
            case_id = f"CASE-{self._next_case_number:06d}"
            self._next_case_number += 1
            try:
                self.repository.get(case_id)
            except KeyError:
                return case_id

    def _audit(self, case_id: str, actor_type: str, actor_id: str, action: str, details: str) -> None:
        self.repository.add_audit(AuditEvent(case_id, actor_type, actor_id, action, self._now(), details))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_id(value: str, field: str) -> None:
        if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"Invalid {field}")

    @classmethod
    def _validate_case_id(cls, case_id: str) -> None:
        cls._validate_id(case_id, "case_id")
        if not case_id.startswith("CASE-"):
            raise ValueError("Invalid case_id")

    @staticmethod
    def _priority_for(risk_level: str) -> str:
        return {
            "EMERGENCY": CasePriority.CRITICAL.value,
            "HIGH": CasePriority.HIGH.value,
            "MODERATE": CasePriority.NORMAL.value,
            "LOW": CasePriority.ROUTINE.value,
        }[risk_level]

    @staticmethod
    def _validate_assessment(assessment: RiskAssessment) -> None:
        if not isinstance(assessment, RiskAssessment):
            raise ValueError("risk_assessment must be a RiskAssessment")
        if assessment.risk_level not in _ALLOWED_RISK_LEVELS:
            raise ValueError("Invalid risk level in RiskAssessment")
        if not isinstance(assessment.risk_score, (int, float)) or not 0 <= float(assessment.risk_score) <= 100:
            raise ValueError("Invalid risk score in RiskAssessment")
        if not isinstance(assessment.emergency, bool) or not isinstance(assessment.clinician_review_required, bool):
            raise ValueError("Invalid emergency or clinician-review flags")
        if assessment.emergency and (
            assessment.risk_level != "EMERGENCY"
            or not assessment.clinician_review_required
            or assessment.care_pathway != "URGENT_EMERGENCY_CARE"
        ):
            raise ValueError("Emergency RiskAssessment must preserve emergency safety fields")
        if assessment.care_pathway not in _ALLOWED_PATHWAYS:
            raise ValueError("Invalid care pathway in RiskAssessment")
        if not isinstance(assessment.risk_factors, list):
            raise ValueError("risk_factors must be a list")

    @staticmethod
    def _require_transition(case: ClinicalCase, target: str) -> None:
        if target not in _TRANSITIONS.get(case.status, set()):
            raise ValueError(f"Invalid status transition: {case.status} -> {target}")

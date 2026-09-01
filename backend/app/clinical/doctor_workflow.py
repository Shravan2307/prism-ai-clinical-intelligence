"""Models and controlled vocabularies for the Epic 5 doctor workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class CaseStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class CasePriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    ROUTINE = "ROUTINE"


class DoctorDecision(str, Enum):
    CONFIRM = "CONFIRM"
    MODIFY = "MODIFY"
    ESCALATE = "ESCALATE"
    DISMISS = "DISMISS"


@dataclass
class DoctorReview:
    case_id: str
    doctor_id: str
    decision: str
    clinical_notes: str
    reviewed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditEvent:
    case_id: str
    actor_type: str
    actor_id: str
    action: str
    timestamp: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalCase:
    case_id: str
    patient_id: str
    risk_score: float
    risk_level: str
    emergency: bool
    clinician_review_required: bool
    risk_factors: list
    care_pathway: str
    ai_recommendation: str
    status: str
    priority: str
    created_at: str
    doctor_review: DoctorReview | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = str(self.status)
        result["priority"] = str(self.priority)
        return result

import pytest

from app.clinical.case_manager import CaseManager
from app.ml.risk_engine import ClinicalRiskEngine


def assessment(**updates):
    value = {
        "age": 40,
        "bmi": 22.0,
        "hba1c": 5.8,
        "fasting_glucose": 90.0,
        "egfr": 100.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 75.0,
        "current_meds": [],
    }
    value.update(updates)
    return ClinicalRiskEngine().assess(value)


def make_case(manager=None, **updates):
    manager = manager or CaseManager()
    return manager, manager.create_case("PATIENT-001", assessment(**updates))


def test_doctor_can_start_review():
    manager, case = make_case()
    updated = manager.start_review(case.case_id, "DOCTOR-001")
    assert updated.status == "IN_REVIEW"


def test_doctor_can_submit_review():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    updated = manager.submit_review(case.case_id, "DOCTOR-001", "CONFIRM", "Reviewed indicators.")
    assert updated.status == "REVIEWED"
    assert updated.doctor_review.decision == "CONFIRM"


def test_doctor_decision_is_separate_from_ai_recommendation():
    manager, case = make_case(hba1c=7.0, fasting_glucose=140.0, bmi=32.0)
    manager.start_review(case.case_id, "DOCTOR-001")
    updated = manager.submit_review(case.case_id, "DOCTOR-001", "ESCALATE", "Further clinical evaluation required.")
    assert updated.ai_recommendation == "ROUTINE_CLINICAL_REVIEW"
    assert updated.doctor_review.decision == "ESCALATE"
    assert updated.ai_recommendation != updated.doctor_review.decision


def test_doctor_can_escalate_case():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    updated = manager.submit_review(case.case_id, "DOCTOR-001", "ESCALATE", "Further clinical evaluation required.")
    assert updated.status == "ESCALATED"


def test_reviewed_case_can_be_closed():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    manager.submit_review(case.case_id, "DOCTOR-001", "CONFIRM", "Reviewed.")
    updated = manager.close_case(case.case_id, "DOCTOR-001")
    assert updated.status == "CLOSED"


def test_unreviewed_case_cannot_be_closed():
    manager, case = make_case()
    with pytest.raises(ValueError, match="Invalid status transition"):
        manager.close_case(case.case_id, "DOCTOR-001")


def test_closed_case_cannot_be_reviewed():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    manager.submit_review(case.case_id, "DOCTOR-001", "CONFIRM", "Reviewed.")
    manager.close_case(case.case_id, "DOCTOR-001")
    with pytest.raises(ValueError, match="Invalid status transition"):
        manager.start_review(case.case_id, "DOCTOR-002")


def test_audit_events_are_created_for_workflow():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    manager.submit_review(case.case_id, "DOCTOR-001", "CONFIRM", "Reviewed.")
    manager.close_case(case.case_id, "DOCTOR-001")
    assert [event.action for event in manager.get_audit_history(case.case_id)] == [
        "CASE_CREATED", "REVIEW_STARTED", "DECISION_SUBMITTED", "CASE_CLOSED"
    ]


def test_emergency_case_requires_review_and_preserves_priority_pathway():
    manager, case = make_case(fasting_glucose=300.0)
    assert case.emergency is True
    assert case.clinician_review_required is True
    assert case.priority == "CRITICAL"
    assert case.care_pathway == "URGENT_EMERGENCY_CARE"


def test_invalid_patient_id_is_rejected():
    with pytest.raises(ValueError, match="Invalid patient_id"):
        CaseManager().create_case("", assessment())


def test_invalid_doctor_id_is_rejected():
    manager, case = make_case()
    with pytest.raises(ValueError, match="Invalid doctor_id"):
        manager.start_review(case.case_id, "")


def test_invalid_case_id_is_rejected():
    with pytest.raises(ValueError, match="Invalid case_id"):
        CaseManager().get_case("not-a-case")


def test_invalid_decision_is_rejected():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    with pytest.raises(ValueError, match="Unsupported doctor decision"):
        manager.submit_review(case.case_id, "DOCTOR-001", "PRESCRIBE", "Invalid.")


def test_invalid_risk_assessment_is_rejected():
    with pytest.raises(ValueError, match="risk_assessment must be a RiskAssessment"):
        CaseManager().create_case("PATIENT-001", object())


def test_nonexistent_case_is_rejected():
    with pytest.raises(KeyError, match="Case not found"):
        CaseManager().get_case("CASE-999999")


def test_two_doctors_cannot_review_same_case():
    manager, case = make_case()
    manager.start_review(case.case_id, "DOCTOR-001")
    with pytest.raises(ValueError, match="Invalid status transition"):
        manager.start_review(case.case_id, "DOCTOR-002")


def test_audit_history_is_not_exposed_as_mutable_internal_state():
    manager, case = make_case()
    history = manager.get_audit_history(case.case_id)
    history.clear()
    assert len(manager.get_audit_history(case.case_id)) == 1

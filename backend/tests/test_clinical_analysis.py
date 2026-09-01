from models.patient import PatientRequest
from services.clinical_analysis import ClinicalAnalysisService


def payload(**updates):
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
    return value


def test_full_clinical_pipeline():
    result = ClinicalAnalysisService(random_state=42).analyze(
        PatientRequest.model_validate(payload(age=70, bmi=35.0, hba1c=9.5, fasting_glucose=260.0, systolic_bp=160.0)),
        patient_id="PATIENT-001",
    )
    assert result.status == "completed"
    assert result.risk.risk_level == "HIGH"
    assert result.triage.category == "urgent_doctor_review"
    assert result.safety.emergency is False
    assert result.ranked_treatments
    assert result.explanations
    assert result.clinical_case is not None
    assert result.health_report.risk_level == result.risk.risk_level


def test_emergency_workflow_stops_treatment_analysis():
    result = ClinicalAnalysisService(random_state=42).analyze(
        PatientRequest.model_validate(payload(fasting_glucose=300.0)),
        patient_id="PATIENT-002",
    )
    assert result.status == "emergency"
    assert result.triage.category == "emergency"
    assert result.safety.safe_candidates == ()
    assert result.ranked_treatments == []
    assert result.explanations == {}
    assert result.clinical_case is not None
    assert result.health_report.emergency is True


def test_routine_patient_does_not_create_doctor_case():
    result = ClinicalAnalysisService(random_state=42).analyze(
        PatientRequest.model_validate(payload()),
        patient_id="PATIENT-003",
    )
    assert result.triage.category == "routine_follow_up"
    assert result.clinical_case is None
    assert result.ranked_treatments


def test_invalid_patient_id_is_rejected():
    service = ClinicalAnalysisService(random_state=42)
    try:
        service.analyze(PatientRequest.model_validate(payload()), patient_id="")
    except ValueError as exc:
        assert "patient_id" in str(exc)
    else:
        raise AssertionError("Expected invalid patient_id to be rejected")

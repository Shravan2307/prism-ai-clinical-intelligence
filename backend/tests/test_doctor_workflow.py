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


def test_case_created_from_risk_assessment():
    case = CaseManager().create_case("PATIENT-001", assessment(hba1c=7.0, fasting_glucose=140.0, bmi=32.0))
    assert case.case_id == "CASE-000001"
    assert case.risk_level == "MODERATE"
    assert case.risk_score == 32.0
    assert case.ai_recommendation == "ROUTINE_CLINICAL_REVIEW"
    assert case.status == "PENDING_REVIEW"


def test_emergency_case_gets_critical_priority():
    case = CaseManager().create_case("PATIENT-001", assessment(fasting_glucose=300.0))
    assert case.priority == "CRITICAL"
    assert case.emergency is True
    assert case.clinician_review_required is True
    assert case.care_pathway == "URGENT_EMERGENCY_CARE"


def test_high_risk_case_gets_high_priority():
    case = CaseManager().create_case("PATIENT-001", assessment(age=70, bmi=35.0, hba1c=9.5, fasting_glucose=260.0, systolic_bp=160.0))
    assert case.priority == "HIGH"


def test_moderate_case_gets_normal_priority():
    case = CaseManager().create_case("PATIENT-001", assessment(hba1c=7.0, fasting_glucose=140.0, bmi=32.0))
    assert case.priority == "NORMAL"


def test_low_risk_case_gets_routine_priority():
    case = CaseManager().create_case("PATIENT-001", assessment())
    assert case.priority == "ROUTINE"


def test_doctor_queue_prioritizes_emergency_cases():
    manager = CaseManager()
    manager.create_case("PATIENT-LOW", assessment())
    manager.create_case("PATIENT-EMERGENCY", assessment(fasting_glucose=300.0))
    manager.create_case("PATIENT-HIGH", assessment(age=70, bmi=35.0, hba1c=9.5, fasting_glucose=260.0, systolic_bp=160.0))
    queue = manager.get_doctor_queue()
    assert [item["priority"] for item in queue] == ["CRITICAL", "HIGH", "ROUTINE"]


def test_doctor_queue_order_is_deterministic():
    manager = CaseManager()
    manager.create_case("PATIENT-B", assessment())
    manager.create_case("PATIENT-A", assessment())
    assert manager.get_doctor_queue() == manager.get_doctor_queue()


def test_queue_does_not_expose_detailed_clinical_information():
    manager = CaseManager()
    manager.create_case("PATIENT-001", assessment(hba1c=7.0))
    item = manager.get_doctor_queue()[0]
    assert set(item) == {"case_id", "patient_id", "risk_level", "priority", "status", "clinician_review_required", "created_at"}
    assert "risk_factors" not in item
    assert "hba1c" not in item

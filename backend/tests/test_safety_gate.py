from app.ml.risk_engine import ClinicalRiskEngine
from safety_gate import SafetyGate


def make_patient(**updates):
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


def test_safety_gate_allows_non_emergency_analytical_candidates():
    risk = ClinicalRiskEngine().assess(make_patient())
    decision = SafetyGate().evaluate(risk)
    assert decision.emergency is False
    assert decision.safe_candidates == ("Metformin", "SGLT2i", "GLP-1")


def test_safety_gate_stops_emergency_treatment_analysis():
    risk = ClinicalRiskEngine().assess(make_patient(fasting_glucose=300.0))
    decision = SafetyGate().evaluate(risk)
    assert decision.emergency is True
    assert decision.doctor_review_required is True
    assert decision.safe_candidates == ()
    assert "professional medical evaluation" in decision.message

from app.ml.risk_engine import ClinicalRiskEngine
from triage_engine import TriageEngine


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


def risk(**updates):
    return ClinicalRiskEngine().assess(make_patient(**updates))


def test_triage_categories():
    engine = TriageEngine()
    assert engine.triage(risk(fasting_glucose=300.0)).category == "emergency"
    assert engine.triage(risk(age=70, bmi=35.0, hba1c=9.5, fasting_glucose=260.0, systolic_bp=160.0)).category == "urgent_doctor_review"
    assert engine.triage(risk(hba1c=7.0, fasting_glucose=140.0, bmi=32.0)).category == "doctor_review"
    assert engine.triage(risk()).category == "routine_follow_up"


def test_triage_does_not_diagnose():
    decision = TriageEngine().triage(risk(hba1c=7.0))
    assert "diagnos" not in decision.message.lower()

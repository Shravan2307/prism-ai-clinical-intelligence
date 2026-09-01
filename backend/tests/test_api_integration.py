from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def payload(**updates):
    value = {
        "age": 70,
        "bmi": 35.0,
        "hba1c": 9.5,
        "fasting_glucose": 260.0,
        "egfr": 100.0,
        "systolic_bp": 160.0,
        "diastolic_bp": 85.0,
        "current_meds": [],
    }
    value.update(updates)
    return value


def test_api_docs_are_available():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Prism AI Clinical Intelligence" in response.text


def test_non_emergency_response_contains_complete_pipeline_sections():
    response = client.post("/api/patient/analyze", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "status", "risk", "triage", "treatment_analysis", "doctor_case", "health_report"}
    assert body["risk"]["level"] == "HIGH"
    assert body["triage"]["doctor_review_required"] is True
    assert body["treatment_analysis"]["ranked_options"]
    assert body["treatment_analysis"]["label"] == "estimated treatment-effect ranking for clinician review"
    assert all(option["explanations"] for option in body["treatment_analysis"]["ranked_options"])
    assert body["doctor_case"]["status"] == "PENDING_REVIEW"
    assert body["health_report"]["risk_level"] == "HIGH"


def test_emergency_never_returns_treatment_analysis():
    response = client.post("/api/patient/analyze", json=payload(fasting_glucose=300.0))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert body["treatment_analysis"] is None
    assert body["health_report"]["analytical_findings"] == []
    assert "URGENT MEDICAL ATTENTION REQUIRED" in body["health_report"]["emergency_guidance"]

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def valid_payload(**updates):
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


def test_valid_patient_request():
    response = client.post("/api/patient/analyze", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["risk"]["level"]
    assert body["health_report"]["disclaimer"]


def test_invalid_patient_request_returns_422():
    response = client.post("/api/patient/analyze", json={"age": 10})
    assert response.status_code == 422


def test_emergency_request_returns_valid_emergency_workflow():
    response = client.post("/api/patient/analyze", json=valid_payload(fasting_glucose=300.0))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert body["triage"]["category"] == "emergency"
    assert body["treatment_analysis"] is None
    assert body["doctor_case"]["priority"] == "CRITICAL"
    assert body["health_report"]["emergency"] is True


def test_missing_and_extra_fields_are_rejected():
    missing = client.post("/api/patient/analyze", json=valid_payload(age=None))
    extra = client.post("/api/patient/analyze", json=valid_payload(unexpected="value"))
    assert missing.status_code == 422
    assert extra.status_code == 422

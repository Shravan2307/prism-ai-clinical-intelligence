import json
import math

import pytest

from app.clinical.case_manager import CaseManager
from app.ml.risk_engine import ClinicalRiskEngine
from app.ml.treatment_ranker import TreatmentRankingEngine
from app.reporting.health_report import HealthReportGenerator


def build_outputs(emergency=False):
    patient = {
        "age": 40,
        "bmi": 22.0,
        "hba1c": 5.8 if not emergency else 10.0,
        "fasting_glucose": 90.0 if not emergency else 300.0,
        "egfr": 100.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 75.0,
        "current_meds": [],
    }
    risk = ClinicalRiskEngine().assess(patient)
    case = CaseManager().create_case("PATIENT-001", risk)
    if emergency:
        return patient, risk, case, None, None
    causal_engine = __import__("app.ml.causal_engine", fromlist=["CausalTreatmentEffectEngine"]).CausalTreatmentEffectEngine(random_state=42)
    causal_data = __import__("app.ml.causal_data", fromlist=["generate_synthetic_causal_data"]).generate_synthetic_causal_data(random_state=42)
    causal_engine.fit(causal_data)
    ranker = TreatmentRankingEngine(causal_engine)
    effects = causal_engine.estimate_effects(patient, ["Metformin", "SGLT2i"])
    ranked = ranker.rank(effects, ["Metformin", "SGLT2i"])
    explanations = {item.treatment: ranker.top_contributors(patient, item.treatment, top_k=3) for item in ranked}
    return patient, risk, case, ranked, explanations


def test_report_generation():
    _, risk, case, ranked, explanations = build_outputs()
    report = HealthReportGenerator().generate(case, risk, ranked, explanations)
    assert report.report_id.startswith("REPORT-")
    assert report.case_id == case.case_id
    assert report.patient_id == case.patient_id
    assert report.risk_level == risk.risk_level
    assert report.summary
    assert report.recommended_next_step
    assert report.disclaimer


def test_report_preserves_epic4_risk_information():
    _, risk, case, _, _ = build_outputs()
    report = HealthReportGenerator().generate(case, risk)
    assert report.risk_level == risk.risk_level
    assert report.risk_score == risk.risk_score
    assert report.key_risk_indicators == [factor.name for factor in risk.risk_factors]


def test_emergency_report_contains_urgent_guidance_and_preserves_status():
    _, risk, case, _, _ = build_outputs(emergency=True)
    report = HealthReportGenerator().generate(case, risk)
    assert report.emergency is True
    assert report.clinician_review_required is True
    assert "URGENT MEDICAL ATTENTION REQUIRED" in report.emergency_guidance
    assert report.analytical_findings == []


def test_report_contains_epic4_risk_factors():
    _, risk, case, _, _ = build_outputs()
    report = HealthReportGenerator().generate(case, risk)
    assert report.key_risk_indicators == [factor.name for factor in risk.risk_factors]


def test_report_contains_shap_explanations_with_actual_names_and_finite_values():
    _, risk, case, ranked, explanations = build_outputs()
    report = HealthReportGenerator().generate(case, risk, ranked, explanations)
    assert report.explainability
    assert any(item.feature == "hba1c" for item in report.explainability)
    assert all(math.isfinite(item.contribution) for item in report.explainability)


def test_report_contains_safe_treatment_effects_only():
    _, risk, case, ranked, explanations = build_outputs()
    report = HealthReportGenerator().generate(case, risk, ranked, explanations)
    assert {item.treatment for item in report.analytical_findings} == {"Metformin", "SGLT2i"}
    assert "GLP-1" not in {item.treatment for item in report.analytical_findings}


def test_treatment_effects_are_not_presented_as_prescriptions():
    _, risk, case, ranked, explanations = build_outputs()
    report = HealthReportGenerator().generate(case, risk, ranked, explanations)
    text = json.dumps(report.to_dict()).lower()
    assert "recommended_medication" not in text
    assert "prescription" in text
    assert "not prescriptions" in text


def test_report_shows_clinician_review_and_preserves_doctor_decision():
    _, risk, case, _, _ = build_outputs()
    manager = CaseManager()
    case = manager.create_case("PATIENT-002", risk)
    manager.start_review(case.case_id, "DOCTOR-001")
    reviewed_case = manager.submit_review(case.case_id, "DOCTOR-001", "MODIFY", "Clinical review completed.")
    report = HealthReportGenerator().generate(reviewed_case, risk)
    assert report.clinician_review_required == risk.clinician_review_required
    assert report.case_status == "REVIEWED"
    assert report.doctor_decision == "MODIFY"


def test_report_always_contains_disclaimer_and_no_diagnosis():
    _, risk, case, _, _ = build_outputs()
    report = HealthReportGenerator().generate(case, risk)
    text = json.dumps(report.to_dict()).lower()
    assert report.disclaimer
    assert "does not provide a medical diagnosis" in report.disclaimer
    assert "you have diabetes" not in text
    assert "start medication" not in text


def test_same_inputs_produce_same_report_content_except_id_and_time():
    _, risk, case, ranked, explanations = build_outputs()
    first = HealthReportGenerator().generate(case, risk, ranked, explanations).to_dict()
    second = HealthReportGenerator().generate(case, risk, ranked, explanations).to_dict()
    first.pop("report_id")
    first.pop("generated_at")
    second.pop("report_id")
    second.pop("generated_at")
    assert first == second


def test_report_serializes_to_json_safe_dictionary():
    _, risk, case, ranked, explanations = build_outputs()
    report = HealthReportGenerator().generate(case, risk, ranked, explanations)
    payload = report.to_dict()
    json.dumps(payload)
    assert all(value not in {"nan", "inf", "-inf"} for value in json.dumps(payload).lower().split())


def test_invalid_dependencies_are_rejected():
    _, risk, case, _, _ = build_outputs()
    generator = HealthReportGenerator()
    with pytest.raises(ValueError, match="clinical_case"):
        generator.generate(object(), risk)
    with pytest.raises(ValueError, match="risk_assessment"):
        generator.generate(case, object())


def test_invalid_treatment_effect_is_rejected():
    _, risk, case, _, _ = build_outputs()
    from app.ml.treatment_ranker import RankedTreatment
    with pytest.raises(ValueError, match="finite"):
        HealthReportGenerator().generate(case, risk, [RankedTreatment("Metformin", float("nan"), 1)])


def test_invalid_shap_contribution_is_rejected():
    _, risk, case, ranked, _ = build_outputs()
    from app.ml.treatment_ranker import FeatureContribution
    with pytest.raises(ValueError, match="finite"):
        HealthReportGenerator().generate(case, risk, ranked, {"Metformin": [FeatureContribution("hba1c", float("inf"))]})

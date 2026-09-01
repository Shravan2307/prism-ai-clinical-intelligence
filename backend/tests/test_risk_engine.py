import json

import pytest

from app.ml.risk_engine import ClinicalRiskEngine, RiskThresholds
from app.ml.synthetic_data import SyntheticPatientGenerator


def patient(**updates):
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


def test_low_risk_patient():
    assessment = ClinicalRiskEngine().assess(patient())
    assert assessment.risk_level == "LOW"
    assert assessment.emergency is False


def test_moderate_risk_patient():
    assessment = ClinicalRiskEngine().assess(patient(hba1c=7.0, fasting_glucose=140.0, bmi=32.0))
    assert assessment.risk_level == "MODERATE"


def test_high_risk_patient_requires_clinician_review():
    assessment = ClinicalRiskEngine().assess(
        patient(age=70, bmi=35.0, hba1c=9.5, fasting_glucose=260.0, systolic_bp=160.0)
    )
    assert assessment.risk_level == "HIGH"
    assert assessment.clinician_review_required is True


def test_emergency_patient_overrides_ordinary_scoring():
    assessment = ClinicalRiskEngine().assess(patient(fasting_glucose=300.0))
    assert assessment.risk_level == "EMERGENCY"
    assert assessment.emergency is True
    assert assessment.clinician_review_required is True
    assert assessment.care_pathway == "URGENT_EMERGENCY_CARE"


def test_hba1c_risk_factor():
    assessment = ClinicalRiskEngine().assess(patient(hba1c=7.0))
    assert any(factor.name == "Elevated HbA1c" for factor in assessment.risk_factors)


def test_egfr_risk_factor():
    assessment = ClinicalRiskEngine().assess(patient(egfr=50.0))
    assert any(factor.name == "Reduced eGFR" for factor in assessment.risk_factors)


def test_blood_pressure_risk_factors():
    assessment = ClinicalRiskEngine().assess(patient(systolic_bp=150.0, diastolic_bp=95.0))
    names = {factor.name for factor in assessment.risk_factors}
    assert "Elevated systolic blood pressure" in names
    assert "Elevated diastolic blood pressure" in names


def test_bmi_risk_factor():
    assessment = ClinicalRiskEngine().assess(patient(bmi=31.0))
    assert any(factor.name == "Elevated BMI" for factor in assessment.risk_factors)


def test_risk_score_bounds():
    for values in [patient(), patient(hba1c=14.0, fasting_glucose=350.0, egfr=15.0, systolic_bp=200.0, diastolic_bp=125.0, bmi=45.0, age=85)]:
        score = ClinicalRiskEngine().assess(values).risk_score
        assert 0 <= score <= 100


def test_determinism():
    engine = ClinicalRiskEngine()
    assert engine.assess(patient(hba1c=8.0)) == engine.assess(patient(hba1c=8.0))


def test_invalid_patient_is_rejected():
    with pytest.raises(ValueError, match="Missing required patient fields"):
        ClinicalRiskEngine().assess({"age": 40})


def test_invalid_medication_reuses_epic1_validation():
    with pytest.raises(ValueError, match="Invalid patient input"):
        ClinicalRiskEngine().assess(patient(current_meds=["UnknownDrug"]))


def test_output_has_no_treatment_prescription_fields_and_is_json_serializable():
    output = ClinicalRiskEngine().assess(patient()).to_dict()
    assert "recommended_medication" not in output
    assert "dose" not in output
    assert "drug_to_start" not in output
    json.dumps(output)


def test_emergency_pathway_is_not_ordinary_pathway():
    assessment = ClinicalRiskEngine().assess(patient(egfr=30.0))
    assert assessment.emergency is True
    assert assessment.care_pathway == "URGENT_EMERGENCY_CARE"


def test_explanation_exists_and_is_non_diagnostic():
    assessment = ClinicalRiskEngine().assess(patient(hba1c=7.0))
    assert assessment.explanation
    assert "diagnosis" in assessment.explanation
    assert "prescription" in assessment.explanation


@pytest.mark.parametrize(
    "field,threshold,below,at,above,factor_name",
    [
        ("hba1c", 6.5, 6.49, 6.5, 6.51, "Elevated HbA1c"),
        ("bmi", 30.0, 29.99, 30.0, 30.01, "Elevated BMI"),
        ("egfr", 60.0, 60.01, 60.0, 59.99, "Reduced eGFR"),
        ("systolic_bp", 140.0, 139.99, 140.0, 140.01, "Elevated systolic blood pressure"),
        ("diastolic_bp", 90.0, 89.99, 90.0, 90.01, "Elevated diastolic blood pressure"),
    ],
)
def test_threshold_boundaries(field, threshold, below, at, above, factor_name):
    kwargs = {field: below}
    if field == "egfr":
        kwargs = {field: below}
    assert not any(f.name == factor_name for f in ClinicalRiskEngine().assess(patient(**kwargs)).risk_factors)
    assert any(f.name == factor_name for f in ClinicalRiskEngine().assess(patient(**{field: at})).risk_factors)
    assert any(f.name == factor_name for f in ClinicalRiskEngine().assess(patient(**{field: above})).risk_factors)


def test_custom_threshold_configuration_is_used():
    thresholds = RiskThresholds(bmi_high=25.0)
    assessment = ClinicalRiskEngine(thresholds).assess(patient(bmi=25.0))
    assert any(f.name == "Elevated BMI" for f in assessment.risk_factors)

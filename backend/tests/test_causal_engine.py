import numpy as np
import pandas as pd
import pytest

from app.ml.causal_data import CONTROL_TREATMENT, TREATMENTS, generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine
from app.ml.synthetic_data import SyntheticPatientGenerator


def test_synthetic_causal_dataset_creation_and_required_columns():
    causal_data = generate_synthetic_causal_data()
    assert isinstance(causal_data.data, pd.DataFrame)
    assert {"treatment", "outcome_hba1c"}.issubset(causal_data.data.columns)
    assert not causal_data.X.empty
    assert len(causal_data.T) == len(causal_data.Y) == 500


def test_synthetic_causal_data_is_reproducible():
    left = generate_synthetic_causal_data(random_state=42).data
    right = generate_synthetic_causal_data(random_state=42).data
    pd.testing.assert_frame_equal(left, right)


def test_engine_fits_and_reports_fitted_state():
    engine = CausalTreatmentEffectEngine(random_state=42)
    engine.fit(generate_synthetic_causal_data())
    assert engine.is_fitted is True


def test_predict_before_fit_raises_clear_exception():
    patient = SyntheticPatientGenerator(1, 42).generate().iloc[0]
    with pytest.raises(RuntimeError, match="not fitted"):
        CausalTreatmentEffectEngine().estimate_effects(patient, ["Metformin"])


def test_candidate_filtering_is_strict():
    engine = CausalTreatmentEffectEngine(random_state=42).fit(generate_synthetic_causal_data())
    patient = SyntheticPatientGenerator(1, 42).generate().iloc[0]
    effects = engine.estimate_effects(patient, ["Metformin", "SGLT2i"])
    assert set(effects) == {"Metformin", "SGLT2i"}


def test_unknown_treatment_is_rejected():
    engine = CausalTreatmentEffectEngine(random_state=42).fit(generate_synthetic_causal_data())
    patient = SyntheticPatientGenerator(1, 42).generate().iloc[0]
    with pytest.raises(ValueError, match="Unsupported treatments"):
        engine.estimate_effects(patient, ["UnknownDrug"])


def test_one_finite_numeric_effect_per_candidate():
    engine = CausalTreatmentEffectEngine(random_state=42).fit(generate_synthetic_causal_data())
    patient = SyntheticPatientGenerator(1, 42).generate().iloc[0]
    effects = engine.estimate_effects(patient, list(TREATMENTS))
    assert set(effects) == set(TREATMENTS)
    assert all(isinstance(value, float) and np.isfinite(value) for value in effects.values())


def test_effects_can_vary_for_different_patient_profiles():
    engine = CausalTreatmentEffectEngine(random_state=42).fit(generate_synthetic_causal_data())
    patients = SyntheticPatientGenerator(2, 42).generate()
    first = engine.estimate_effects(patients.iloc[0], ["GLP-1"])["GLP-1"]
    second = engine.estimate_effects(patients.iloc[1], ["GLP-1"])["GLP-1"]
    assert first != second


def test_prediction_does_not_require_hidden_true_effect_field():
    engine = CausalTreatmentEffectEngine(random_state=42).fit(generate_synthetic_causal_data())
    patient_series = SyntheticPatientGenerator(1, 42).generate().iloc[0]
    patient = dict(patient_series)
    patient["true_treatment_effect"] = 9999.0
    effects = engine.estimate_effects(patient, ["Metformin"])
    assert np.isfinite(effects["Metformin"])


def test_training_validation_rejects_unknown_treatment():
    causal_data = generate_synthetic_causal_data().data.copy()
    causal_data.loc[0, "treatment"] = "UnknownDrug"
    with pytest.raises(ValueError, match="Unsupported treatment values"):
        CausalTreatmentEffectEngine().fit(causal_data)


def test_control_category_is_present_in_training_data():
    data = generate_synthetic_causal_data().data
    assert CONTROL_TREATMENT in set(data["treatment"])

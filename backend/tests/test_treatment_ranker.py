import numpy as np
import pytest

from app.ml.causal_data import generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine
from app.ml.treatment_ranker import FeatureContribution, RankedTreatment, TreatmentRankingEngine
from app.ml.synthetic_data import SyntheticPatientGenerator


@pytest.fixture
def ranker_and_patient():
    causal_data = generate_synthetic_causal_data(random_state=42)
    causal_engine = CausalTreatmentEffectEngine(random_state=42).fit(causal_data)
    return TreatmentRankingEngine(causal_engine), causal_data.X.iloc[0]


def test_ranking_order(ranker_and_patient):
    ranker, _ = ranker_and_patient
    results = ranker.rank({"Metformin": 0.8, "SGLT2i": 1.2, "GLP-1": 1.6}, ["Metformin", "SGLT2i", "GLP-1"])
    assert [item.treatment for item in results] == ["GLP-1", "SGLT2i", "Metformin"]


def test_safe_candidate_filtering(ranker_and_patient):
    ranker, _ = ranker_and_patient
    results = ranker.rank({"Metformin": 0.8, "SGLT2i": 1.2, "GLP-1": 1.6}, ["Metformin", "SGLT2i"])
    assert [item.treatment for item in results] == ["SGLT2i", "Metformin"]


def test_missing_effect_is_rejected(ranker_and_patient):
    ranker, _ = ranker_and_patient
    with pytest.raises(ValueError, match="Missing effects"):
        ranker.rank({"Metformin": 0.8}, ["Metformin", "SGLT2i"])


def test_unknown_treatment_is_rejected(ranker_and_patient):
    ranker, _ = ranker_and_patient
    with pytest.raises(ValueError, match="Unsupported treatments"):
        ranker.rank({"UnknownDrug": 1.0}, ["UnknownDrug"])


def test_duplicate_candidates_are_rejected(ranker_and_patient):
    ranker, _ = ranker_and_patient
    with pytest.raises(ValueError, match="duplicates"):
        ranker.rank({"Metformin": 0.8, "SGLT2i": 1.2}, ["Metformin", "Metformin"])


def test_empty_candidates_are_rejected(ranker_and_patient):
    ranker, _ = ranker_and_patient
    with pytest.raises(ValueError, match="non-empty"):
        ranker.rank({}, [])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_effects_are_rejected(value, ranker_and_patient):
    ranker, _ = ranker_and_patient
    with pytest.raises(ValueError, match="finite"):
        ranker.rank({"Metformin": value}, ["Metformin"])


def test_ties_use_deterministic_alphabetical_order(ranker_and_patient):
    ranker, _ = ranker_and_patient
    results = ranker.rank({"Metformin": 1.0, "SGLT2i": 1.0}, ["SGLT2i", "Metformin"])
    assert [item.treatment for item in results] == ["Metformin", "SGLT2i"]


def test_rank_numbers_are_sequential(ranker_and_patient):
    ranker, _ = ranker_and_patient
    results = ranker.rank({"Metformin": 0.8, "SGLT2i": 1.2, "GLP-1": 1.6}, ["Metformin", "SGLT2i", "GLP-1"])
    assert [item.rank for item in results] == [1, 2, 3]
    assert all(isinstance(item, RankedTreatment) for item in results)


def test_shap_output_contains_finite_feature_contributions(ranker_and_patient):
    ranker, patient = ranker_and_patient
    results = ranker.explain(patient, "GLP-1")
    assert results
    assert all(isinstance(item, FeatureContribution) for item in results)
    assert all(np.isfinite(item.contribution) for item in results)


def test_shap_uses_actual_encoded_feature_names(ranker_and_patient):
    ranker, patient = ranker_and_patient
    results = ranker.explain(patient, "GLP-1")
    names = {item.feature for item in results}
    assert "hba1c" in names
    assert "current_med_metformin" in names
    assert not any(name.startswith("feature_") for name in names)


def test_top_k_returns_at_most_requested_number(ranker_and_patient):
    ranker, patient = ranker_and_patient
    assert len(ranker.top_contributors(patient, "GLP-1", top_k=5)) <= 5


def test_shap_explanations_are_deterministic(ranker_and_patient):
    ranker, patient = ranker_and_patient
    first = ranker.explain(patient, "GLP-1")
    second = ranker.explain(patient, "GLP-1")
    assert first == second


def test_safe_candidate_integration_only_ranks_and_explains_safe_candidates(ranker_and_patient):
    ranker, patient = ranker_and_patient
    result = ranker.analyze(patient, ["Metformin", "SGLT2i"])
    assert {item.treatment for item in result["ranked_treatments"]} == {"Metformin", "SGLT2i"}
    assert set(result["explanations"]) == {"Metformin", "SGLT2i"}
    assert "GLP-1" not in result["explanations"]

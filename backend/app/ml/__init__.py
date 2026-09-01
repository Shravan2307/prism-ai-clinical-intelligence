"""Machine-learning data utilities for the hackathon backend."""

from .synthetic_data import (
    FeatureRanges,
    SyntheticPatientGenerator,
    get_training_data,
    inspect_training_data,
)

__all__ = [
    "FeatureRanges",
    "SyntheticPatientGenerator",
    "get_training_data",
    "inspect_training_data",
]
from .causal_data import CONTROL_TREATMENT, TREATMENTS, SyntheticCausalData, encode_patient_features, generate_synthetic_causal_data
from .causal_engine import CausalTreatmentEffectEngine, TreatmentEffect

__all__ += [
    "CONTROL_TREATMENT",
    "TREATMENTS",
    "SyntheticCausalData",
    "encode_patient_features",
    "generate_synthetic_causal_data",
    "CausalTreatmentEffectEngine",
    "TreatmentEffect",
]
from .treatment_ranker import FeatureContribution, RankedTreatment, TreatmentRankingEngine

__all__ += [
    "FeatureContribution",
    "RankedTreatment",
    "TreatmentRankingEngine",
]
from .risk_engine import ClinicalRiskEngine, RiskAssessment, RiskFactor, RiskThresholds

__all__ += [
    "ClinicalRiskEngine",
    "RiskAssessment",
    "RiskFactor",
    "RiskThresholds",
]

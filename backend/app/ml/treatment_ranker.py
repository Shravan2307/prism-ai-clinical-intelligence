"""Treatment ranking and SHAP explainability for the Epic 3 analytical prototype.

Ranking is performed only over treatments already approved by the upstream safety
layer. SHAP values describe model-feature contributions to an estimated treatment
effect; they are not diagnoses, prescriptions, or eligibility decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from .causal_data import TREATMENTS, encode_patient_features
from .causal_engine import CausalTreatmentEffectEngine
from .synthetic_data import SyntheticPatientGenerator


@dataclass(frozen=True)
class RankedTreatment:
    treatment: str
    expected_hba1c_reduction: float
    rank: int


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    contribution: float


class _LinearEffectModel:
    """Sklearn-compatible signed linear view of EconML's final-stage effect model."""

    def __init__(self, model: Any) -> None:
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        # EconML's fit_intercept=False final model stores its intercept as the
        # first coefficient; the remaining coefficients align with X columns.
        self.intercept_ = float(-coefficients[0])
        self.coef_ = -coefficients[1:][np.newaxis, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_.reshape(-1)


class TreatmentRankingEngine:
    """Rank Epic 2 estimated effects and explain them with SHAP."""

    SUPPORTED_TREATMENTS: tuple[str, ...] = TREATMENTS

    def __init__(self, causal_engine: CausalTreatmentEffectEngine) -> None:
        if not isinstance(causal_engine, CausalTreatmentEffectEngine):
            raise ValueError("causal_engine must be a fitted CausalTreatmentEffectEngine")
        if not causal_engine.is_fitted:
            raise RuntimeError("CausalTreatmentEffectEngine is not fitted")
        self.causal_engine = causal_engine

    def rank(self, effects: dict[str, float], safe_candidates: list[str]) -> list[RankedTreatment]:
        """Return deterministic descending effect ranks for safe candidates only."""
        self._validate_inputs(effects, safe_candidates)
        ordered = sorted(
            safe_candidates,
            key=lambda treatment: (-float(effects[treatment]), treatment),
        )
        return [
            RankedTreatment(treatment, float(effects[treatment]), index)
            for index, treatment in enumerate(ordered, start=1)
        ]

    def explain(
        self, patient: pd.Series | dict[str, Any], treatment: str, top_k: int | None = None
    ) -> list[FeatureContribution]:
        """Explain one Epic 2 treatment-effect prediction with named SHAP features."""
        if treatment not in self.SUPPORTED_TREATMENTS:
            raise ValueError(f"Unsupported treatment: {treatment}")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        patient_df = pd.DataFrame([dict(patient) if isinstance(patient, pd.Series) else patient])
        SyntheticPatientGenerator(1).validate(patient_df)
        X_patient = encode_patient_features(patient_df).loc[:, self.causal_engine._feature_columns]
        estimator = self.causal_engine._estimators[treatment]
        model = _LinearEffectModel(estimator.model_final_)
        background = getattr(self.causal_engine, "_training_data", None)
        if background is None or background.empty:
            background = self.causal_engine._training_features
        explainer = shap.LinearExplainer(model, background)
        shap_values = np.asarray(explainer(X_patient).values, dtype=float).reshape(-1)
        contributions = [
            FeatureContribution(feature, float(value))
            for feature, value in zip(self.causal_engine._feature_columns, shap_values)
        ]
        contributions.sort(key=lambda item: (-abs(item.contribution), item.feature))
        return contributions[:top_k] if top_k is not None else contributions

    def top_contributors(
        self, patient: pd.Series | dict[str, Any], treatment: str, top_k: int = 5
    ) -> list[FeatureContribution]:
        return self.explain(patient, treatment, top_k=top_k)

    def analyze(
        self, patient: pd.Series | dict[str, Any], safe_candidates: list[str]
    ) -> dict[str, Any]:
        """Run the Epic 2 effects → ranking → SHAP flow for safe candidates."""
        effects = self.causal_engine.estimate_effects(patient, safe_candidates)
        ranked = self.rank(effects, safe_candidates)
        explanations = {
            item.treatment: self.explain(patient, item.treatment)
            for item in ranked
        }
        return {"ranked_treatments": ranked, "explanations": explanations}

    def _validate_inputs(self, effects: dict[str, float], safe_candidates: list[str]) -> None:
        if not isinstance(effects, dict) or not effects:
            raise ValueError("effects must be a non-empty dict[str, float]")
        for treatment, value in effects.items():
            if not isinstance(treatment, str) or not treatment:
                raise ValueError("effects must map treatment names to numeric values")
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Effect for {treatment} must be numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"Effect for {treatment} must be finite")
        if not isinstance(safe_candidates, list) or not safe_candidates:
            raise ValueError("safe_candidates must be a non-empty list[str]")
        if any(not isinstance(item, str) for item in safe_candidates):
            raise ValueError("safe_candidates must be a list[str]")
        if len(set(safe_candidates)) != len(safe_candidates):
            raise ValueError("safe_candidates must not contain duplicates")
        unknown = [item for item in safe_candidates if item not in self.SUPPORTED_TREATMENTS]
        if unknown:
            raise ValueError(f"Unsupported treatments: {unknown}")
        missing = [item for item in safe_candidates if item not in effects]
        if missing:
            raise ValueError(f"Missing effects for safe candidates: {missing}")

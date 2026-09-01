"""EconML causal treatment-effect engine for the Epic 2 demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .causal_data import CONTROL_TREATMENT, TREATMENTS, encode_patient_features
from .synthetic_data import SyntheticPatientGenerator


@dataclass(frozen=True)
class TreatmentEffect:
    """Estimated HbA1c reduction for one requested treatment.

    ``expected_hba1c_reduction`` is positive when treatment is expected to reduce
    HbA1c relative to the synthetic no-additional-therapy control.
    """

    treatment: str
    expected_hba1c_reduction: float


class CausalTreatmentEffectEngine:
    """Estimate heterogeneous, synthetic treatment effects using EconML LinearDML."""

    SUPPORTED_TREATMENTS: tuple[str, ...] = TREATMENTS
    TRAINING_COLUMNS: tuple[str, ...] = ("treatment", "outcome_hba1c")

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state
        self._estimators: dict[str, LinearDML] = {}
        self._feature_columns: list[str] = []
        self._training_features = pd.DataFrame()
        self._training_data = pd.DataFrame()
        self.is_fitted = False

    def fit(self, training_data: pd.DataFrame | Any) -> "CausalTreatmentEffectEngine":
        """Fit one-vs-control EconML est.loc[:, SyntheticPatientGenerator.REQUIRED_COLUMNS]imator for each supported treatment."""
        data = training_data.data if hasattr(training_data, "data") else training_data
        self._validate_training_data(data)
        X = encode_patient_features(data)
        Y = data["outcome_hba1c"].to_numpy(dtype=float)
        self._feature_columns = list(X.columns)
        self._training_features = X.copy()
        # Backward-compatible alias used by Epic 3 SHAP integrations.
        self._training_data = X.copy()
        estimators: dict[str, LinearDML] = {}
        for treatment_name in self.SUPPORTED_TREATMENTS:
            binary_treatment = (data["treatment"] == treatment_name).astype(int).to_numpy()
            estimator = LinearDML(
                model_y=RandomForestRegressor(
                    n_estimators=80, min_samples_leaf=8, random_state=self.random_state, n_jobs=1
                ),
                model_t=make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                ),
                discrete_treatment=True,
                cv=3,
                random_state=self.random_state,
            )
            estimator.fit(Y, binary_treatment, X=X)
            estimators[treatment_name] = estimator
        self._estimators = estimators
        self.is_fitted = True
        return self

    def estimate_effects(
        self, patient: pd.Series | dict[str, Any], safe_candidates: list[str]
    ) -> dict[str, float]:
        """Estimate only the requested candidate effects for one patient."""
        if not self.is_fitted:
            raise RuntimeError("CausalTreatmentEffectEngine is not fitted")
        if not isinstance(safe_candidates, list) or not safe_candidates:
            raise ValueError("safe_candidates must be a non-empty list")
        unknown = [item for item in safe_candidates if item not in self.SUPPORTED_TREATMENTS]
        if unknown:
            raise ValueError(f"Unsupported treatments: {unknown}")
        patient_df = pd.DataFrame([dict(patient) if isinstance(patient, pd.Series) else patient])
        SyntheticPatientGenerator(1).validate(patient_df)
        X = encode_patient_features(patient_df).loc[:, self._feature_columns]
        effects: dict[str, float] = {}
        for treatment_name in safe_candidates:
            # EconML returns Y(treatment)-Y(control); negate for reduction convention.
            effect = -float(np.asarray(self._estimators[treatment_name].effect(X))[0])
            if not np.isfinite(effect):
                raise ValueError(f"Non-finite effect returned for {treatment_name}")
            effects[treatment_name] = effect
        return effects

    @classmethod
    def _validate_training_data(cls, data: Any) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("training_data must be a pandas DataFrame or SyntheticCausalData")
        if data.empty:
            raise ValueError("training_data must not be empty")
        missing = [column for column in cls.TRAINING_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required training columns: {missing}")
        patient_columns = [column for column in SyntheticPatientGenerator.REQUIRED_COLUMNS if column in data.columns]
        if len(patient_columns) != len(SyntheticPatientGenerator.REQUIRED_COLUMNS):
            missing_patients = [column for column in SyntheticPatientGenerator.REQUIRED_COLUMNS if column not in data.columns]
            raise ValueError(f"Missing required patient columns: {missing_patients}")
        SyntheticPatientGenerator(len(data)).validate(data.loc[:, SyntheticPatientGenerator.REQUIRED_COLUMNS])
        if data["outcome_hba1c"].isna().any() or not pd.api.types.is_numeric_dtype(data["outcome_hba1c"]):
            raise ValueError("outcome_hba1c must be numeric and contain no NaN values")
        allowed = set(cls.SUPPORTED_TREATMENTS) | {CONTROL_TREATMENT}
        unknown = sorted(set(data["treatment"]) - allowed)
        if unknown:
            raise ValueError(f"Unsupported treatment values: {unknown}")
        if data["treatment"].nunique() < 2:
            raise ValueError("Sufficient treatment variation is required")
        for treatment_name in cls.SUPPORTED_TREATMENTS:
            if (data["treatment"] == treatment_name).sum() < 2:
                raise ValueError(f"Insufficient observations for treatment: {treatment_name}")
        if not np.isfinite(data["outcome_hba1c"].to_numpy(dtype=float)).all():
            raise ValueError("outcome_hba1c must contain finite numeric values")

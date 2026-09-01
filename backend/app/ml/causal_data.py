"""Synthetic treatment/outcome generation for the Epic 2 demo.

All treatment assignments and outcomes in this module are synthetic and exist only
for controlled model development. They are not clinical evidence or recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic_data import SyntheticPatientGenerator


CONTROL_TREATMENT = "No additional therapy"
TREATMENTS: tuple[str, ...] = ("Metformin", "SGLT2i", "GLP-1")
CAUSAL_DATA_COLUMNS: tuple[str, ...] = (
    *SyntheticPatientGenerator.REQUIRED_COLUMNS,
    "treatment",
    "outcome_hba1c",
)


@dataclass(frozen=True)
class SyntheticCausalData:
    """Container for the generated causal-training DataFrame."""

    data: pd.DataFrame

    @property
    def X(self) -> pd.DataFrame:
        return self.data.drop(columns=["treatment", "outcome_hba1c"])

    @property
    def T(self) -> pd.Series:
        return self.data["treatment"]

    @property
    def Y(self) -> pd.Series:
        return self.data["outcome_hba1c"]


def encode_patient_features(patients: pd.DataFrame) -> pd.DataFrame:
    """Encode Epic 1 list-valued medications as deterministic binary indicators."""
    generator = SyntheticPatientGenerator(len(patients))
    required = list(generator.REQUIRED_COLUMNS)
    missing = [column for column in required if column not in patients.columns]
    if missing:
        raise ValueError(f"Missing required patient columns: {missing}")
    patient_features = patients.loc[:, required]
    generator.validate(patient_features)
    encoded = patient_features.drop(columns=["current_meds"]).copy()
    for medication in sorted(generator.APPROVED_MEDICATIONS):
        key = medication.lower().replace("-", "").replace("i", "i")
        encoded[f"current_med_{key}"] = patients["current_meds"].map(
            lambda values, item=medication: int(item in values)
        )
    return encoded


def _assignment_probabilities(patients: pd.DataFrame) -> np.ndarray:
    """Return softmax treatment probabilities driven by patient characteristics."""
    hba1c = patients["hba1c"].to_numpy()
    bmi = patients["bmi"].to_numpy()
    meds = patients["current_meds"]
    existing_diabetes_meds = np.array(
        [sum(item in values for item in ("Metformin", "SGLT2i", "GLP-1")) for values in meds]
    )
    baseline = 0.25 + 0.16 * (hba1c - 6.5) + 0.025 * (bmi - 28) - 0.10 * existing_diabetes_meds
    logits = np.column_stack(
        [
            baseline + 0.20,
            baseline + 0.10 + 0.025 * (bmi - 28) - 0.004 * np.maximum(0, 60 - patients["egfr"]),
            baseline + 0.04 + 0.06 * (bmi - 30) + 0.10 * (hba1c - 7.0),
            np.zeros(len(patients)),
        ]
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def generate_synthetic_causal_data(
    patients: pd.DataFrame | None = None,
    *,
    n_patients: int = 500,
    random_state: int | None = 42,
) -> SyntheticCausalData:
    """Generate reproducible synthetic treatment assignments and HbA1c outcomes."""
    if patients is None:
        patients = SyntheticPatientGenerator(n_patients=n_patients, random_state=random_state).generate()
    else:
        SyntheticPatientGenerator(len(patients)).validate(patients)
    rng = np.random.default_rng(random_state)
    probabilities = _assignment_probabilities(patients)
    treatment_values = np.array([*TREATMENTS, CONTROL_TREATMENT], dtype=object)
    treatment = np.array(
        [treatment_values[rng.choice(len(treatment_values), p=probabilities[i])] for i in range(len(patients))],
        dtype=object,
    )

    hba1c = patients["hba1c"].to_numpy(dtype=float)
    bmi = patients["bmi"].to_numpy(dtype=float)
    age = patients["age"].to_numpy(dtype=float)
    egfr = patients["egfr"].to_numpy(dtype=float)
    glucose = patients["fasting_glucose"].to_numpy(dtype=float)
    reduction = np.zeros(len(patients), dtype=float)
    metformin = treatment == "Metformin"
    sglt2i = treatment == "SGLT2i"
    glp1 = treatment == "GLP-1"
    reduction[metformin] = 0.45 + 0.10 * np.clip(hba1c[metformin] - 6.5, 0, 5) / 5 + 0.03 * np.clip((120 - egfr[metformin]) / 105, 0, 1)
    reduction[sglt2i] = 0.55 + 0.12 * np.clip(hba1c[sglt2i] - 6.5, 0, 5) / 5 + 0.08 * np.clip((bmi[sglt2i] - 25) / 20, 0, 1)
    reduction[glp1] = 0.80 + 0.22 * np.clip(hba1c[glp1] - 6.5, 0, 5) / 5 + 0.16 * np.clip((bmi[glp1] - 25) / 20, 0, 1) - 0.04 * np.clip((age[glp1] - 65) / 20, 0, 1)
    outcome = hba1c - reduction + rng.normal(0, 0.28, len(patients)) + 0.0008 * (glucose - 150)

    result = patients.copy()
    result["treatment"] = treatment
    result["outcome_hba1c"] = outcome.round(4)
    return SyntheticCausalData(result[list(CAUSAL_DATA_COLUMNS)])   

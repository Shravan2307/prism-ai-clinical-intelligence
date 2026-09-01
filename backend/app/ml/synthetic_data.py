"""Focused, clearly synthetic patient data for the Type 2 Diabetes demo pathway.

This module is for ML development and testing only. It does not diagnose patients,
recommend treatment, or represent medically validated patient data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureRanges:
    """Inclusive bounds used by the generator and validator."""

    age: tuple[int, int] = (18, 85)
    bmi: tuple[float, float] = (18.0, 45.0)
    hba1c: tuple[float, float] = (5.5, 14.0)
    fasting_glucose: tuple[float, float] = (70.0, 350.0)
    egfr: tuple[float, float] = (15.0, 120.0)
    systolic_bp: tuple[float, float] = (90.0, 200.0)
    diastolic_bp: tuple[float, float] = (55.0, 125.0)


class SyntheticPatientGenerator:
    """Generate deterministic, diabetes-focused synthetic patient profiles."""

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "age",
        "bmi",
        "hba1c",
        "fasting_glucose",
        "egfr",
        "systolic_bp",
        "diastolic_bp",
        "current_meds",
    )
    APPROVED_MEDICATIONS: ClassVar[frozenset[str]] = frozenset(
        {"Metformin", "SGLT2i", "GLP-1", "ACEi", "ARB", "CCB"}
    )
    RANGES: ClassVar[FeatureRanges] = FeatureRanges()

    def __init__(self, n_patients: int = 500, random_state: int | None = 42) -> None:
        if not isinstance(n_patients, int) or isinstance(n_patients, bool) or n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        self.n_patients = n_patients
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        """Return a new DataFrame of synthetic patients."""
        rng = np.random.default_rng(self.random_state)
        n = self.n_patients

        age = np.rint(np.clip(rng.normal(58, 14, n), *self.RANGES.age)).astype(int)
        bmi = np.clip(rng.normal(30.5, 5.5, n), *self.RANGES.bmi).round(1)

        diabetic_group = rng.random(n) < 0.72
        hba1c = np.where(
            diabetic_group,
            rng.normal(8.5, 1.45, n),
            rng.normal(6.15, 0.42, n),
        )
        hba1c = np.clip(hba1c, *self.RANGES.hba1c).round(2)

        # Controlled noise preserves a positive association without making it exact.
        fasting_glucose = 25.5 * hba1c - 50 + rng.normal(0, 22, n)
        fasting_glucose = np.clip(fasting_glucose, *self.RANGES.fasting_glucose).round(1)

        renal_group = rng.choice(4, size=n, p=(0.48, 0.25, 0.19, 0.08))
        egfr_means = np.array([96.0, 70.0, 42.0, 22.0])
        egfr_sds = np.array([12.0, 8.0, 9.0, 4.0])
        egfr = np.clip(
            rng.normal(egfr_means[renal_group], egfr_sds[renal_group]),
            *self.RANGES.egfr,
        ).round(1)

        age_effect = (age - 50) * 0.42
        bmi_effect = (bmi - 25) * 0.72
        systolic_bp = np.clip(
            119 + age_effect + bmi_effect + rng.normal(0, 14, n),
            *self.RANGES.systolic_bp,
        ).round(1)
        diastolic_bp = np.clip(
            76 + (age - 50) * 0.12 + (bmi - 25) * 0.32 + rng.normal(0, 8, n),
            *self.RANGES.diastolic_bp,
        ).round(1)
        # Enforce the required ordering while retaining realistic variation.
        diastolic_bp = np.minimum(diastolic_bp, systolic_bp - 1.0)
        diastolic_bp = np.maximum(diastolic_bp, self.RANGES.diastolic_bp[0]).round(1)

        current_meds = [
            self._medications_for_patient(
                hba1c=float(hba1c[i]),
                systolic_bp=float(systolic_bp[i]),
                egfr=float(egfr[i]),
                rng=rng,
            )
            for i in range(n)
        ]

        df = pd.DataFrame(
            {
                "age": age,
                "bmi": bmi,
                "hba1c": hba1c,
                "fasting_glucose": fasting_glucose,
                "egfr": egfr,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "current_meds": current_meds,
            },
            columns=self.REQUIRED_COLUMNS,
        )
        self.validate(df)
        return df

    @staticmethod
    def _medications_for_patient(
        *, hba1c: float, systolic_bp: float, egfr: float, rng: np.random.Generator
    ) -> list[str]:
        """Assign plausible synthetic medication histories, not recommendations."""
        meds: list[str] = []
        diabetes_probability = float(np.clip(0.10 + (hba1c - 5.5) * 0.105, 0.10, 0.92))
        if rng.random() < diabetes_probability:
            meds.append("Metformin")
            if hba1c >= 8.0 and rng.random() < 0.45:
                meds.append("SGLT2i")
            if hba1c >= 9.5 and rng.random() < 0.25:
                meds.append("GLP-1")
        hypertension_probability = float(np.clip((systolic_bp - 125) / 65, 0.0, 0.82))
        if rng.random() < hypertension_probability:
            options = ["ACEi", "ARB", "CCB"]
            meds.append(options[int(rng.integers(0, len(options)))])
            if systolic_bp >= 170 and rng.random() < 0.22:
                remaining = [option for option in options if option not in meds]
                meds.append(remaining[int(rng.integers(0, len(remaining)))])
        # Lower eGFR changes the synthetic mix without encoding eligibility or advice.
        if egfr < 30 and "SGLT2i" in meds and rng.random() < 0.35:
            meds.remove("SGLT2i")
        return meds

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the generated schema and clinical bounds, raising clear errors."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if len(df) != self.n_patients:
            raise ValueError(f"Expected {self.n_patients} rows, found {len(df)}")
        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for column in self.REQUIRED_COLUMNS[:-1]:
            if df[column].isna().any():
                raise ValueError(f"NaN values found in required field: {column}")
        for column, bounds in (
            ("age", self.RANGES.age),
            ("bmi", self.RANGES.bmi),
            ("hba1c", self.RANGES.hba1c),
            ("fasting_glucose", self.RANGES.fasting_glucose),
            ("egfr", self.RANGES.egfr),
            ("systolic_bp", self.RANGES.systolic_bp),
            ("diastolic_bp", self.RANGES.diastolic_bp),
        ):
            if not df[column].between(*bounds).all():
                raise ValueError(f"{column} contains values outside {bounds}")
        if not (df["systolic_bp"] > df["diastolic_bp"]).all():
            raise ValueError("systolic_bp must be greater than diastolic_bp")
        for index, medications in df["current_meds"].items():
            if not isinstance(medications, list):
                raise ValueError(f"current_meds at row {index} must be a list")
            invalid = set(medications) - self.APPROVED_MEDICATIONS
            if invalid:
                raise ValueError(f"Invalid medications at row {index}: {sorted(invalid)}")
        return True


def get_training_data(
    n_patients: int = 500, random_state: int | None = 42
) -> pd.DataFrame:
    """Return the in-memory training DataFrame for the current process."""
    return SyntheticPatientGenerator(n_patients, random_state).generate()


def inspect_training_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight shape, schema, statistics, and medication frequencies."""
    SyntheticPatientGenerator(len(df)).validate(df)
    medication_counts: dict[str, int] = {}
    for medications in df["current_meds"]:
        for medication in medications:
            medication_counts[medication] = medication_counts.get(medication, 0) + 1
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "describe": df.drop(columns=["current_meds"]).describe(),
        "medication_frequency": medication_counts,
    }

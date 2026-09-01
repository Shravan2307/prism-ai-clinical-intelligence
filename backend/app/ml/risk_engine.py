"""Deterministic clinical risk assessment and care-navigation engine.

This is a synthetic-data clinical decision-support demonstration. It flags elevated
indicators and urgency only; it does not diagnose disease or prescribe treatment.
Production use would require validated thresholds and expert clinical review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .synthetic_data import SyntheticPatientGenerator


@dataclass(frozen=True)
class RiskThresholds:
    """Configurable prototype thresholds used for risk flagging, not diagnosis."""

    bmi_high: float = 30.0
    hba1c_elevated: float = 6.5
    hba1c_high: float = 9.0
    fasting_glucose_elevated: float = 126.0
    fasting_glucose_high: float = 250.0
    egfr_low: float = 60.0
    egfr_critical: float = 30.0
    systolic_bp_elevated: float = 140.0
    systolic_bp_high: float = 180.0
    diastolic_bp_elevated: float = 90.0
    diastolic_bp_high: float = 120.0
    older_age: float = 65.0


@dataclass(frozen=True)
class RiskFactor:
    name: str
    value: float | str
    severity: str
    explanation: str


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: float
    risk_level: str
    emergency: bool
    clinician_review_required: bool
    risk_factors: list[RiskFactor]
    care_pathway: str
    recommended_next_step: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class ClinicalRiskEngine:
    """Assess care-navigation urgency from the existing Epic 1 patient schema."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"

    CARE_PATHWAYS = {
        LOW: "ROUTINE_SELF_CARE",
        MODERATE: "ROUTINE_CLINICAL_REVIEW",
        HIGH: "PRIORITY_CLINICAL_REVIEW",
        EMERGENCY: "URGENT_EMERGENCY_CARE",
    }
    NEXT_STEPS = {
        LOW: "Continue routine health monitoring and maintain healthy lifestyle habits.",
        MODERATE: "Consider arranging a routine clinical evaluation to review the identified risk indicators.",
        HIGH: "Prompt clinical evaluation is recommended because multiple elevated risk indicators were detected.",
        EMERGENCY: "Seek urgent medical attention immediately. Do not rely on this AI system for emergency diagnosis or treatment.",
    }

    def __init__(self, thresholds: RiskThresholds | None = None) -> None:
        self.thresholds = thresholds or RiskThresholds()

    def assess(self, patient: pd.Series | dict[str, Any]) -> RiskAssessment:
        """Return deterministic risk, urgency, and care-navigation information."""
        patient_df = self._validated_patient_frame(patient)
        row = patient_df.iloc[0]
        factors = self._risk_factors(row)
        score = self._risk_score(row)
        emergency = self._is_emergency(row)
        level = self.EMERGENCY if emergency else self._level_for_score(score)
        review_required = level in {self.HIGH, self.EMERGENCY}
        pathway = self.CARE_PATHWAYS[level]
        explanation = self._explanation(level, score, factors)
        return RiskAssessment(
            risk_score=score,
            risk_level=level,
            emergency=emergency,
            clinician_review_required=review_required,
            risk_factors=factors,
            care_pathway=pathway,
            recommended_next_step=self.NEXT_STEPS[level],
            explanation=explanation,
        )

    def _validated_patient_frame(self, patient: pd.Series | dict[str, Any]) -> pd.DataFrame:
        if isinstance(patient, pd.Series):
            patient = patient.to_dict()
        if not isinstance(patient, dict):
            raise ValueError("patient must be a pandas Series or dictionary")
        try:
            patient_df = pd.DataFrame([patient])
        except Exception as exc:
            raise ValueError("Malformed patient input") from exc
        required = list(SyntheticPatientGenerator.REQUIRED_COLUMNS)
        missing = [column for column in required if column not in patient_df.columns]
        if missing:
            raise ValueError(f"Missing required patient fields: {missing}")
        selected = patient_df.loc[:, required]
        try:
            SyntheticPatientGenerator(1).validate(selected)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid patient input: {exc}") from exc
        numeric_values = selected.drop(columns=["current_meds"]).to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError("Patient numeric values must be finite")
        return selected

    def _risk_factors(self, row: pd.Series) -> list[RiskFactor]:
        t = self.thresholds
        factors: list[RiskFactor] = []
        if row["hba1c"] >= t.hba1c_high:
            factors.append(RiskFactor("Elevated HbA1c", float(row["hba1c"]), "high", f"HbA1c ({row['hba1c']:.2f}) is at or above the configured high-risk threshold of {t.hba1c_high:.2f}."))
        elif row["hba1c"] >= t.hba1c_elevated:
            factors.append(RiskFactor("Elevated HbA1c", float(row["hba1c"]), "moderate", f"HbA1c ({row['hba1c']:.2f}) is at or above the configured risk threshold of {t.hba1c_elevated:.2f}."))
        if row["fasting_glucose"] >= t.fasting_glucose_high:
            factors.append(RiskFactor("Elevated fasting glucose", float(row["fasting_glucose"]), "high", f"Fasting glucose ({row['fasting_glucose']:.1f}) is at or above the configured high-risk threshold of {t.fasting_glucose_high:.1f}."))
        elif row["fasting_glucose"] >= t.fasting_glucose_elevated:
            factors.append(RiskFactor("Elevated fasting glucose", float(row["fasting_glucose"]), "moderate", f"Fasting glucose ({row['fasting_glucose']:.1f}) is at or above the configured risk threshold of {t.fasting_glucose_elevated:.1f}."))
        if row["egfr"] <= t.egfr_critical:
            factors.append(RiskFactor("Reduced eGFR", float(row["egfr"]), "critical", f"eGFR ({row['egfr']:.1f}) is at or below the configured critical threshold of {t.egfr_critical:.1f}."))
        elif row["egfr"] <= t.egfr_low:
            factors.append(RiskFactor("Reduced eGFR", float(row["egfr"]), "high", f"eGFR ({row['egfr']:.1f}) is below the configured threshold of {t.egfr_low:.1f}."))
        if row["systolic_bp"] >= t.systolic_bp_high:
            factors.append(RiskFactor("Elevated systolic blood pressure", float(row["systolic_bp"]), "high", f"Systolic blood pressure ({row['systolic_bp']:.1f}) is at or above the configured high-risk threshold of {t.systolic_bp_high:.1f}."))
        elif row["systolic_bp"] >= t.systolic_bp_elevated:
            factors.append(RiskFactor("Elevated systolic blood pressure", float(row["systolic_bp"]), "moderate", f"Systolic blood pressure ({row['systolic_bp']:.1f}) is at or above the configured risk threshold of {t.systolic_bp_elevated:.1f}."))
        if row["diastolic_bp"] >= t.diastolic_bp_high:
            factors.append(RiskFactor("Elevated diastolic blood pressure", float(row["diastolic_bp"]), "high", f"Diastolic blood pressure ({row['diastolic_bp']:.1f}) is at or above the configured high-risk threshold of {t.diastolic_bp_high:.1f}."))
        elif row["diastolic_bp"] >= t.diastolic_bp_elevated:
            factors.append(RiskFactor("Elevated diastolic blood pressure", float(row["diastolic_bp"]), "moderate", f"Diastolic blood pressure ({row['diastolic_bp']:.1f}) is at or above the configured risk threshold of {t.diastolic_bp_elevated:.1f}."))
        if row["bmi"] >= t.bmi_high:
            factors.append(RiskFactor("Elevated BMI", float(row["bmi"]), "moderate", f"BMI ({row['bmi']:.1f}) is at or above the configured risk threshold of {t.bmi_high:.1f}."))
        if row["age"] >= t.older_age:
            factors.append(RiskFactor("Older age", float(row["age"]), "moderate", f"Age ({row['age']:.0f}) is at or above the configured older-age threshold of {t.older_age:.0f}."))
        return factors

    def _risk_score(self, row: pd.Series) -> float:
        t = self.thresholds
        points = 0.0
        points += 20 if row["hba1c"] >= t.hba1c_high else 12 if row["hba1c"] >= t.hba1c_elevated else 0
        points += 15 if row["fasting_glucose"] >= t.fasting_glucose_high else 10 if row["fasting_glucose"] >= t.fasting_glucose_elevated else 0
        points += 25 if row["egfr"] <= t.egfr_critical else 15 if row["egfr"] <= t.egfr_low else 0
        systolic_points = 20 if row["systolic_bp"] >= t.systolic_bp_high else 10 if row["systolic_bp"] >= t.systolic_bp_elevated else 0
        diastolic_points = 20 if row["diastolic_bp"] >= t.diastolic_bp_high else 10 if row["diastolic_bp"] >= t.diastolic_bp_elevated else 0
        points += max(systolic_points, diastolic_points)
        points += 10 if row["bmi"] >= t.bmi_high else 0
        points += 10 if row["age"] >= t.older_age else 0
        return round(float(np.clip(points, 0, 100)), 1)

    def _is_emergency(self, row: pd.Series) -> bool:
        t = self.thresholds
        return bool(
            row["egfr"] <= t.egfr_critical
            or row["hba1c"] >= 12.5
            or row["fasting_glucose"] >= 300
            or row["systolic_bp"] >= 190
            or row["diastolic_bp"] >= t.diastolic_bp_high
        )

    @staticmethod
    def _level_for_score(score: float) -> str:
        if score >= 75:
            return ClinicalRiskEngine.EMERGENCY
        if score >= 50:
            return ClinicalRiskEngine.HIGH
        if score >= 25:
            return ClinicalRiskEngine.MODERATE
        return ClinicalRiskEngine.LOW

    @staticmethod
    def _explanation(level: str, score: float, factors: list[RiskFactor]) -> str:
        if not factors:
            return f"The deterministic risk assessment score is {score:.1f}/100, with no configured elevated indicators detected. This is care-navigation information, not a diagnosis."
        names = ", ".join(factor.name for factor in factors)
        return f"The deterministic risk assessment score is {score:.1f}/100. Configured elevated indicators detected: {names}. Professional review should consider the patient information and clinical context; this output is not a diagnosis or prescription."

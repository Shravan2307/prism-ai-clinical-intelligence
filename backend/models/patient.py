"""API request models for patient analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator


class PatientRequest(BaseModel):
    """Validated patient representation accepted by the clinical API."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    age: int = Field(ge=18, le=90)
    bmi: float = Field(ge=15.0, le=60.0)
    hba1c: float = Field(ge=4.0, le=15.0)
    fasting_glucose: float = Field(ge=50.0, le=350.0)
    egfr: float = Field(ge=15.0, le=150.0)
    systolic_bp: float = Field(ge=90.0, le=220.0)
    diastolic_bp: float = Field(ge=50.0, le=130.0)
    current_meds: list[str] = Field(default_factory=list)

    @field_validator("current_meds")
    @classmethod
    def validate_meds(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("current_meds must contain non-empty strings")
        return value

    @field_validator("diastolic_bp")
    @classmethod
    def validate_pressure_order(cls, value: float, info: Any) -> float:
        systolic = info.data.get("systolic_bp")
        if systolic is not None and value >= systolic:
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        return value

    def to_patient_dict(self) -> dict[str, Any]:
        return self.model_dump()

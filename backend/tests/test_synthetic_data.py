import numpy as np
import pandas as pd
import pytest

from app.ml.synthetic_data import SyntheticPatientGenerator, get_training_data, inspect_training_data


def test_generate_returns_500_patients_by_default():
    df = SyntheticPatientGenerator().generate()
    assert len(df) == 500


def test_required_columns_exist_in_order():
    generator = SyntheticPatientGenerator()
    df = generator.generate()
    assert list(df.columns) == list(generator.REQUIRED_COLUMNS)


def test_numeric_ranges_and_no_missing_values():
    generator = SyntheticPatientGenerator()
    df = generator.generate()
    assert generator.validate(df) is True
    assert not df[list(generator.REQUIRED_COLUMNS[:-1])].isna().any().any()
    assert df.age.between(18, 85).all()
    assert df.bmi.between(18, 45).all()
    assert df.hba1c.between(5.5, 14).all()
    assert df.fasting_glucose.between(70, 350).all()
    assert df.egfr.between(15, 120).all()
    assert df.systolic_bp.between(90, 200).all()
    assert df.diastolic_bp.between(55, 125).all()


def test_blood_pressure_is_internally_consistent():
    df = SyntheticPatientGenerator().generate()
    assert (df.systolic_bp > df.diastolic_bp).all()


def test_medications_are_lists_and_use_only_approved_vocabulary():
    generator = SyntheticPatientGenerator()
    df = generator.generate()
    assert df.current_meds.map(lambda value: isinstance(value, list)).all()
    assert all(
        medication in generator.APPROVED_MEDICATIONS
        for medications in df.current_meds
        for medication in medications
    )


def test_same_seed_produces_identical_dataframe():
    left = SyntheticPatientGenerator(500, 42).generate()
    right = SyntheticPatientGenerator(500, 42).generate()
    pd.testing.assert_frame_equal(left, right)


def test_different_seeds_produce_different_dataframes():
    left = SyntheticPatientGenerator(500, 42).generate()
    right = SyntheticPatientGenerator(500, 99).generate()
    assert not left.equals(right)


def test_diabetes_focused_distribution_is_substantial():
    df = SyntheticPatientGenerator().generate()
    diabetes_range = (df.hba1c >= 6.5) & (df.fasting_glucose >= 126)
    assert diabetes_range.mean() >= 0.45
    assert df.hba1c.ge(6.5).sum() < len(df)


def test_hba1c_and_fasting_glucose_have_positive_relationship():
    df = SyntheticPatientGenerator().generate()
    assert df["hba1c"].corr(df["fasting_glucose"]) > 0.60


def test_validation_reports_missing_required_column():
    generator = SyntheticPatientGenerator()
    df = generator.generate().drop(columns=["egfr"])
    with pytest.raises(ValueError, match="Missing required columns"):
        generator.validate(df)


def test_validation_reports_invalid_medication():
    generator = SyntheticPatientGenerator()
    df = generator.generate()
    df.loc[0, "current_meds"] = ["NotApproved"]
    with pytest.raises(ValueError, match="Invalid medications"):
        generator.validate(df)


def test_in_memory_helper_and_inspection_helper():
    df = get_training_data()
    inspection = inspect_training_data(df)
    assert inspection["shape"] == (500, 8)
    assert inspection["columns"] == list(SyntheticPatientGenerator.REQUIRED_COLUMNS)
    assert isinstance(inspection["describe"], pd.DataFrame)
    assert isinstance(inspection["medication_frequency"], dict)

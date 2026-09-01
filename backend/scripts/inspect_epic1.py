from app.ml.synthetic_data import SyntheticPatientGenerator, get_training_data, inspect_training_data


df = get_training_data()
summary = inspect_training_data(df)
print("SHAPE", summary["shape"])
print("COLUMNS", summary["columns"])
print("SAMPLE")
print(df.head(5).to_string(index=False))
print("DESCRIBE")
print(summary["describe"].round(2).to_string())
print("MEDICATION_FREQUENCY", summary["medication_frequency"])
print("DIABETES_RANGE_PROPORTION", round(((df["hba1c"] >= 6.5) & (df["fasting_glucose"] >= 126)).mean(), 3))
print("HBA1C_GLUCOSE_CORRELATION", round(df["hba1c"].corr(df["fasting_glucose"]), 3))
print("VALID", SyntheticPatientGenerator().validate(df))

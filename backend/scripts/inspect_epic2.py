from app.ml.causal_data import generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine

causal_data = generate_synthetic_causal_data(random_state=42)
engine = CausalTreatmentEffectEngine(random_state=42).fit(causal_data)
patient = causal_data.X.iloc[0]
effects = engine.estimate_effects(patient, ["Metformin", "SGLT2i", "GLP-1"])
print("CAUSAL_DATA_SHAPE", causal_data.data.shape)
print("TREATMENT_COUNTS")
print(causal_data.T.value_counts().to_string())
print("PATIENT")
print(patient.to_string())
print("EFFECTS", effects)

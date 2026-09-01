from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.causal_data import generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine
from app.ml.treatment_ranker import TreatmentRankingEngine


causal_data = generate_synthetic_causal_data(random_state=42)
causal_engine = CausalTreatmentEffectEngine(random_state=42).fit(causal_data)
ranking_engine = TreatmentRankingEngine(causal_engine)
patient = causal_data.X.iloc[0]
safe_candidates = ["Metformin", "SGLT2i", "GLP-1"]
effects = causal_engine.estimate_effects(patient, safe_candidates)
result = ranking_engine.analyze(patient, safe_candidates)

print("EPIC 3 DEMO")
print("===========")
print("Safe candidates:")
for treatment in safe_candidates:
    print(f"- {treatment}")
print("\nEstimated HbA1c effects:")
for treatment, effect in effects.items():
    print(f"- {treatment}: {effect:.3f}")
print("\nTreatment ranking:")
for ranked in result["ranked_treatments"]:
    print(f"{ranked.rank}. {ranked.treatment} ({ranked.expected_hba1c_reduction:.3f})")
print(f"\nTop contributors for {result['ranked_treatments'][0].treatment}:")
for index, contribution in enumerate(
    ranking_engine.top_contributors(patient, result["ranked_treatments"][0].treatment, top_k=5),
    start=1,
):
    print(f"{index}. {contribution.feature}: {contribution.contribution:.4f}")
print("\nNOTE:")
print("These are synthetic model estimates for demonstration only.")
print("They are not medical recommendations.")

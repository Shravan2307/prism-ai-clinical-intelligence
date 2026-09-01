from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.risk_engine import ClinicalRiskEngine
from app.ml.synthetic_data import SyntheticPatientGenerator


patient = SyntheticPatientGenerator(n_patients=1, random_state=42).generate().iloc[0]
assessment = ClinicalRiskEngine().assess(patient)

print("EPIC 4 — CLINICAL RISK ASSESSMENT")
print("=================================")
print("\nPatient:")
print(f"Age: {patient['age']}")
print(f"BMI: {patient['bmi']}")
print(f"HbA1c: {patient['hba1c']}")
print(f"Fasting glucose: {patient['fasting_glucose']}")
print(f"eGFR: {patient['egfr']}")
print(f"Blood pressure: {patient['systolic_bp']} / {patient['diastolic_bp']}")
print(f"\nRisk score:\n{assessment.risk_score:.1f} / 100")
print(f"\nRisk level:\n{assessment.risk_level}")
print(f"\nEmergency:\n{'YES' if assessment.emergency else 'NO'}")
print("\nRisk factors:")
if assessment.risk_factors:
    for factor in assessment.risk_factors:
        print(f"- {factor.name}: {factor.explanation}")
else:
    print("- None detected by configured prototype thresholds")
print(f"\nCare pathway:\n{assessment.care_pathway}")
print(f"\nClinician review:\n{'REQUIRED' if assessment.clinician_review_required else 'NOT REQUIRED'}")
print(f"\nNext step:\n{assessment.recommended_next_step}")
print(f"\nExplanation:\n{assessment.explanation}")
print("\nNOTE:")
print("This is a synthetic-data clinical decision-support demonstration.")
print("It does not provide a diagnosis or prescription.")

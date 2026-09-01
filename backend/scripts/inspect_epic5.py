from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clinical.case_manager import CaseManager
from app.ml.risk_engine import ClinicalRiskEngine
from app.ml.synthetic_data import SyntheticPatientGenerator


patients = SyntheticPatientGenerator(n_patients=5, random_state=42).generate()
# Use synthetic profiles and controlled prototype indicators to demonstrate all queue priorities.
patients.loc[0, "fasting_glucose"] = 300.0
patients.loc[1, "age"] = 70
patients.loc[1, "bmi"] = 35.0
patients.loc[1, "hba1c"] = 9.5
patients.loc[1, "fasting_glucose"] = 260.0
patients.loc[1, "systolic_bp"] = 160.0
patients.loc[2, "hba1c"] = 7.0
patients.loc[2, "fasting_glucose"] = 140.0
patients.loc[2, "bmi"] = 32.0
patients.loc[3, "hba1c"] = 5.8
patients.loc[3, "fasting_glucose"] = 90.0
patients.loc[3, "bmi"] = 22.0
patients.loc[4, "hba1c"] = 6.0
patients.loc[4, "fasting_glucose"] = 100.0
patients.loc[4, "bmi"] = 24.0

risk_engine = ClinicalRiskEngine()
manager = CaseManager()
for index, row in patients.iterrows():
    manager.create_case(f"PATIENT-{int(index) + 1:03d}", risk_engine.assess(row))

queue = manager.get_doctor_queue()
opened = manager.get_case(queue[0]["case_id"])
manager.start_review(opened.case_id, "DOCTOR-001")
reviewed = manager.submit_review(
    opened.case_id,
    "DOCTOR-001",
    "ESCALATE",
    "Further clinical evaluation required.",
)

print("=" * 50)
print("EPIC 5 — DOCTOR WORKFLOW")
print("=" * 50)
print("\nCreating synthetic clinical cases...")
print(f"Cases created: {len(queue)}")
print("\nDOCTOR QUEUE")
print("-" * 50)
for index, item in enumerate(queue, start=1):
    print(f"{index}. {item['case_id']}")
    print(f"   Risk: {item['risk_level']}")
    print(f"   Priority: {item['priority']}")
    print(f"   Status: {item['status']}")
print("\nOPENING CASE")
print("-" * 50)
print(f"Case ID: {opened.case_id}")
print(f"Risk level: {opened.risk_level}")
print(f"Risk score: {opened.risk_score:.1f}")
print(f"Emergency: {'YES' if opened.emergency else 'NO'}")
print(f"Priority: {opened.priority}")
print(f"AI recommendation: {opened.ai_recommendation}")
print(f"Clinician review: {'REQUIRED' if opened.clinician_review_required else 'NOT REQUIRED'}")
print("\nDOCTOR REVIEW")
print("-" * 50)
print("Doctor: DOCTOR-001")
print("Decision: ESCALATE")
print("Notes: Further clinical evaluation required.")
print(f"New status: {reviewed.status}")
print("\nAUDIT HISTORY")
print("-" * 50)
for event in manager.get_audit_history(opened.case_id):
    print(event.action)
print("\n" + "=" * 50)
print("Synthetic clinical demonstration only.")
print("This system does not provide diagnosis or prescriptions.")
print("=" * 50)

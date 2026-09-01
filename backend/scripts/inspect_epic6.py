from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clinical.case_manager import CaseManager
from app.ml.causal_data import generate_synthetic_causal_data
from app.ml.causal_engine import CausalTreatmentEffectEngine
from app.ml.risk_engine import ClinicalRiskEngine
from app.ml.synthetic_data import SyntheticPatientGenerator
from app.ml.treatment_ranker import TreatmentRankingEngine
from app.reporting.health_report import HealthReportGenerator


causal_data = generate_synthetic_causal_data(random_state=42)
causal_engine = CausalTreatmentEffectEngine(random_state=42).fit(causal_data)
ranking_engine = TreatmentRankingEngine(causal_engine)
risk_engine = ClinicalRiskEngine()
case_manager = CaseManager()
report_generator = HealthReportGenerator()

# Choose a generated non-emergency profile for the analytical report demonstration.
patient = next(
    row for _, row in causal_data.X.iterrows()
    if not risk_engine.assess(row).emergency
)
risk_assessment = risk_engine.assess(patient)
case = case_manager.create_case("PATIENT-001", risk_assessment)
safe_candidates = ["Metformin", "SGLT2i"]
effects = causal_engine.estimate_effects(patient, safe_candidates)
ranked = ranking_engine.rank(effects, safe_candidates)
explanations = {
    item.treatment: ranking_engine.top_contributors(patient, item.treatment, top_k=3)
    for item in ranked
}
report = report_generator.generate(case, risk_assessment, ranked, explanations)

print("=" * 50)
print("EPIC 6 — EXPLAINABLE AI HEALTH REPORT")
print("=" * 50)
print(f"\nCase: {report.case_id}")
print(f"Risk level: {report.risk_level}")
print(f"Risk score: {report.risk_score:.1f}")
print(f"Emergency: {'YES' if report.emergency else 'NO'}")
print(f"Clinician review: {'REQUIRED' if report.clinician_review_required else 'NOT CURRENTLY REQUIRED'}")
print("\nPATIENT-FACING SUMMARY")
print("-" * 50)
print(report.summary)
print("\nKEY RISK INDICATORS")
print("-" * 50)
for indicator in report.key_risk_indicators or ["None detected by the current assessment"]:
    print(f"- {indicator}")
print("\nWHY THE SYSTEM FLAGGED THIS CASE")
print("-" * 50)
for explanation in report.explainability:
    print(f"- {explanation.feature}: {explanation.explanation}")
print("\nANALYTICAL TREATMENT EFFECTS")
print("-" * 50)
for finding in report.analytical_findings:
    print(f"{finding.rank}. {finding.treatment}: estimated HbA1c reduction {finding.expected_hba1c_reduction:.3f}")
print("These estimates are analytical only and are not prescriptions.")
print("\nCARE NAVIGATION")
print("-" * 50)
print(report.recommended_next_step)
print("\nDOCTOR STATUS")
print("-" * 50)
print(report.case_status)
print("\nLIMITATIONS")
print("-" * 50)
for limitation in report.limitations:
    print(f"- {limitation}")
print("\nDISCLAIMER")
print(report.disclaimer)

# Demonstrate the separate emergency report path.
emergency_patient = patient.copy()
emergency_patient["fasting_glucose"] = 300.0
emergency_risk = risk_engine.assess(emergency_patient)
emergency_case = case_manager.create_case("PATIENT-002", emergency_risk)
emergency_report = report_generator.generate(emergency_case, emergency_risk)
print("\nEMERGENCY DEMONSTRATION")
print("-" * 50)
print(f"Risk level: {emergency_report.risk_level}")
print(f"Emergency: {'YES' if emergency_report.emergency else 'NO'}")
print(f"Guidance: {emergency_report.emergency_guidance}")

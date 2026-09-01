export interface PatientRequest {
  age: number;
  bmi: number;
  hba1c: number;
  fasting_glucose: number;
  egfr: number;
  systolic_bp: number;
  diastolic_bp: number;
  current_meds: string[];
}

export interface RiskFactor {
  name: string;
  severity: string;
  value: number;
  threshold: number;
  explanation: string;
}

export interface RiskResult {
  level: string;
  score: number;
  factors: RiskFactor[];
}

export interface TriageResult {
  category: string;
  doctor_review_required: boolean;
  message: string;
}

export interface ClinicalExplanation {
  feature: string;
  contribution: number;
  direction: string;
  explanation: string;
}

export interface TreatmentRecommendation {
  treatment: string;
  estimated_effect: number;
  rank: number;
  explanations: ClinicalExplanation[];
}

export interface TreatmentAnalysis {
  ranked_options: TreatmentRecommendation[];
  label: string;
}

export interface DoctorCase {
  case_id: string;
  status: string;
  priority: string;
}

export interface HealthReport {
  report_id: string;
  case_id: string;
  patient_id: string;
  risk_level: string;
  risk_score: number;
  summary: string;
  key_risk_indicators: string[];
  explainability: ClinicalExplanation[];
  analytical_findings: TreatmentRecommendation[];
  recommended_next_step: string;
  clinician_review_required: boolean;
  emergency: boolean;
  emergency_guidance: string | null;
  limitations: string[];
  disclaimer: string;
  case_status: string;
  case_priority: string;
  ai_recommendation: string;
  doctor_decision: string | null;
}

export interface ClinicalAnalysisResponse {
  request_id: string;
  status: string;
  risk: RiskResult;
  triage: TriageResult;
  treatment_analysis: TreatmentAnalysis | null;
  doctor_case: DoctorCase | null;
  health_report: HealthReport;
}

export class ClinicalApiError extends Error {
  status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ClinicalApiError";
    this.status = status;
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export async function analyzePatient(
  patient: PatientRequest,
  signal?: AbortSignal,
): Promise<ClinicalAnalysisResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/patient/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
      signal,
    });

    if (!response.ok) {
      let detail = "The clinical analysis request could not be completed.";
      try {
        const payload = await response.json();
        if (response.status === 422 && Array.isArray(payload.detail)) {
          detail = payload.detail
            .map((item: { msg?: string }) => item.msg || "Invalid value")
            .join(" ");
        } else if (typeof payload.detail === "string") {
          detail = payload.detail;
        }
      } catch {
        // Keep the safe fallback message when the response is not JSON.
      }

      if (response.status === 404) detail = "The clinical analysis endpoint could not be found.";
      if (response.status >= 500) detail = "Clinical analysis could not be completed. Please try again.";
      throw new ClinicalApiError(detail, response.status);
    }

    return (await response.json()) as ClinicalAnalysisResponse;
  } catch (error) {
    if (error instanceof ClinicalApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ClinicalApiError("Unable to connect to the clinical analysis service.", null);
  }
}

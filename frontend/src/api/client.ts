/**
 * Typed API Client for Academic Intelligence Framework REST API.
 * Uses Vite proxy or direct backend URL http://127.0.0.1:8000.
 */

const BASE_URL = import.meta.env.VITE_API_URL || '';

export interface HealthResponse {
  status: string;
  device: string;
  seed: number;
  directories_initialized: Record<string, boolean>;
  timestamp_utc: string;
}

export interface MetaResponse {
  project_id: string;
  project_name: string;
  author: string;
  version: string;
  data_mode: 'SYNTHETIC' | 'REAL';
  synthetic_disclaimer: string;
  champion_model: {
    model_key: string;
    label: string;
    category: string;
    category_label: string;
    role: string;
    test_roc_auc: number;
    test_pr_auc: number;
    test_f2_score: number;
    lead_time_semesters: number;
    calibrated: boolean;
  };
  sequence_champion: {
    model_key: string;
    label: string;
    category: string;
    category_label: string;
    role: string;
    test_roc_auc: number;
    test_pr_auc: number;
    test_f2_score: number;
    lead_time_semesters: number;
    calibrated: boolean;
  };
  categories: Record<string, string>;
  n_registered_models: number;
  nfr1_latency_sla_seconds: number;
}

export interface LeaderboardModel {
  model_key: string;
  category: string;
  label: string;
  is_proposal_model: boolean;
  is_sequence_model: boolean;
  test_roc_auc: number;
  test_pr_auc: number;
  test_f2_score: number;
  test_brier_score: number;
  test_ece: number;
  test_lead_time: number;
  inference_time_ms: number;
  composite_rank: number;
  overall_rank: number;
  eligible: boolean;
  seed_std?: number;
  threshold?: number;
}

export interface LeaderboardResponse {
  data_source: string;
  champion: string;
  n_models: number;
  n_eligible: number;
  ranking_weights: Record<string, number>;
  eligibility_constraints: {
    max_inference_time_sec_per_student: number;
    max_seed_std: number;
  };
  models: LeaderboardModel[];
  category_champions: Record<string, {
    model_key: string;
    label: string;
    test_roc_auc: number;
    test_pr_auc: number;
    test_f2_score: number;
    composite_rank: number;
  }>;
}

export interface ModelDetailResponse {
  model_key: string;
  category: string;
  category_label: string;
  label: string;
  is_proposal_model: boolean;
  is_sequence_model: boolean;
  status: string;
  validation_metrics: Record<string, any>;
  stability_summary: Record<string, any>;
  calibration: {
    method?: string;
    brier_score_before?: number;
    brier_score_after?: number;
    brier_improvement_pct?: number;
    is_fitted?: boolean;
  };
  threshold: {
    strategy?: string;
    optimal_threshold?: number;
    metrics?: Record<string, number>;
  };
  best_params: Record<string, any>;
  training_curves: Array<{
    epoch?: number;
    train_loss?: number;
    val_loss?: number;
    train_auc?: number;
    val_auc?: number;
  }>;
  trials_summary: Record<string, any>;
  runtime_seconds: number;
  artifact_dir: string;
}

export interface ResearchQuestionAnswer {
  research_question: string;
  answer: string;
  sequence_champion: Record<string, any>;
  boosting_baseline: Record<string, any>;
  deltas: Record<string, number>;
  statistical_significance: Record<string, any>;
  practical_significance: string;
  honest_assessment: string;
  data_source: string;
}

export interface PlotCatalogItem {
  key: string;
  title: string;
  category: string;
  filename: string;
  url_path: string;
  description: string;
}

export interface DataSummaryResponse {
  data_mode: string;
  total_students: number;
  total_observations: number;
  prefix_prediction_samples: number;
  base_attrition_rate_pct: number;
  num_semesters: number;
  gpa_scale: string;
  variables: Array<{
    name: string;
    scale: string;
    description: string;
  }>;
  trajectory_archetypes: string[];
  plots_catalog: PlotCatalogItem[];
}

export interface GlobalFeatureImportanceItem {
  feature_name: string;
  display_name: string;
  domain: string;
  mean_abs_shap: number;
  relative_importance_pct: number;
  rank: number;
}

export interface GlobalExplainResponse {
  model_key: string;
  explainer_type: string;
  features: GlobalFeatureImportanceItem[];
  semester_attributions: Array<Record<string, any>>;
  faithfulness_audit: {
    test_samples_audited?: number;
    baseline_risk_mean?: number;
    neutralized_risk_mean?: number;
    absolute_risk_drop?: number;
    relative_risk_reduction_pct?: number;
    comprehensiveness_score?: number;
    is_faithful?: boolean;
    conclusion?: string;
  };
  stability_audit: {
    noise_std?: number;
    mean_spearman_rank_correlation?: number;
    stability_threshold?: number;
    is_stable?: boolean;
    conclusion?: string;
  };
}

export interface StudentExplainResponse {
  student_id: string;
  sample_id: string;
  window_semester: number;
  academic_risk_index: number;
  risk_band: 'Low' | 'Moderate' | 'High' | 'Critical';
  top_factors: Array<{
    name: string;
    display_name: string;
    direction: string;
    contribution: number;
    feature_value: number;
    unit: string;
  }>;
  explanation_text: string;
  model_name: string;
  calibrated: boolean;
  provenance_tag: string;
  timestamp_utc: string;
}

export interface FactorItem {
  name: string;
  display_name: string;
  direction: string;
  contribution: number;
  feature_value: number;
  unit: string;
}

export interface StudentPredictionResult {
  student_id: string;
  sample_id: string;
  window_semester: number;
  academic_risk_index: number;
  risk_band: 'Low' | 'Moderate' | 'High' | 'Critical';
  calibrated_probability: number;
  top_factors: FactorItem[];
  explanation_text: string;
  fusion_payload: Record<string, any>;
  inference_time_ms?: number;
  nfr1_compliant?: boolean;
  warnings?: string[];
}

export interface BatchUploadResult {
  success: boolean;
  total_students: number;
  total_records: number;
  summary: {
    total_students: number;
    total_records: number;
    average_risk_index: number;
    risk_band_counts: Record<string, number>;
  };
  predictions: StudentPredictionResult[];
  download_token: string;
  download_url_csv: string;
  download_url_json: string;
  inference_time_total_ms: number;
  inference_time_per_student_ms: number;
  nfr1_compliant: boolean;
  warnings: string[];
}

export interface ValidationErrorDetail {
  row?: number | null;
  column?: string | null;
  error_type: string;
  message: string;
}

// ── Generic Fetch Helper ─────────────────────────────────────────────────────

async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let errorBody: any = null;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }
      throw {
        status: res.status,
        statusText: res.statusText,
        detail: errorBody?.detail || errorBody || 'Network request failed',
      };
    }
    return (await res.json()) as T;
  } catch (err: any) {
    if (err.status) throw err;
    throw {
      status: 0,
      statusText: 'Connection Refused',
      detail: `Failed to connect to backend at ${url}. Ensure FastAPI is running (python run.py serve).`,
    };
  }
}

// ── API Methods ──────────────────────────────────────────────────────────────

export const api = {
  getHealth: () => apiFetch<HealthResponse>('/health'),
  getMeta: () => apiFetch<MetaResponse>('/meta'),
  getLeaderboard: () => apiFetch<LeaderboardResponse>('/leaderboard'),
  getModelDetails: (key: string) => apiFetch<ModelDetailResponse>(`/models/${encodeURIComponent(key)}`),
  getResearchAnswer: () => apiFetch<ResearchQuestionAnswer>('/research-answer'),
  getDataSummary: () => apiFetch<DataSummaryResponse>('/data/summary'),
  getGlobalExplain: () => apiFetch<GlobalExplainResponse>('/explain/global'),
  getStudentExplain: (id: string) => apiFetch<StudentExplainResponse>(`/explain/student/${encodeURIComponent(id)}`),
  
  getTemplateUrl: () => `${BASE_URL}/predict/template`,
  
  predictStudent: (payload: {
    student_id: string;
    semesters: Array<{
      semester: number;
      gpa?: number;
      exam_marks?: number;
      attendance_pct?: number;
      assignment_delay_days?: number;
      late_assignments_count?: number;
    }>;
  }) => apiFetch<StudentPredictionResult>('/predict/student', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }),

  uploadPredict: async (file: File): Promise<BatchUploadResult> => {
    const formData = new FormData();
    formData.append('file', file);
    const url = `${BASE_URL}/predict/upload`;

    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorBody: any = null;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }
      throw {
        status: res.status,
        detail: errorBody?.detail || errorBody || 'File upload failed',
      };
    }
    return res.json();
  },

  getDownloadUrl: (token: string, format: 'csv' | 'json' = 'csv') =>
    `${BASE_URL}/predict/download/${encodeURIComponent(token)}?format=${format}`,
};

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
  disclaimer: string;
}

export interface MetaResponse {
  project: string;
  model_version: string;
  training_date_utc: string;
  data_hash: string;
  is_synthetic: boolean;
  disclaimer: string;
  instrument_items: string[];
  latent_factors: string[];
  champions: {
    regression: string;
    classification: string;
  };
  status: string;
}

export interface LeaderboardEntry {
  rank: number;
  model_name: string;
  category: string;
  track: 'regression' | 'classification';
  feature_set: string;
  primary_metric: number;
  secondary_metric?: number;
  overfitting_gap?: number;
  ece?: number;
  inference_latency_ms?: number;
  model_size_kb?: number;
  composite_rank?: number;
  is_champion: boolean;
}

export interface LeaderboardResponse {
  track?: string;
  total_models: number;
  entries: LeaderboardEntry[];
}

export interface StatisticalComparison {
  champion: string;
  challenger: string;
  champion_mean: number;
  challenger_mean: number;
  wilcoxon_stat?: number | null;
  p_value: number;
  significant: boolean;
  alpha: number;
}

export interface ChampionTrackDetail {
  model_name: string;
  category: string;
  feature_set: string;
  selection_metric: string;
  primary_metric_value: number;
  overfitting_gap: number;
  high_risk_recall?: number;
  ece?: number;
  test_metrics: Record<string, number>;
  justification: string;
  best_params: Record<string, any>;
  best_per_category?: Record<string, string>;
  statistical_comparisons?: StatisticalComparison[];
}

export interface ChampionResponse {
  regression: ChampionTrackDetail;
  classification: ChampionTrackDetail;
  disclaimer: string;
}

export interface ReliabilityFactorDetail {
  factor: string;
  cronbach_alpha: number;
  mcdonald_omega?: number | null;
  passed: boolean;
  items: string[];
  item_total_correlations: Record<string, number>;
  alpha_if_item_deleted: Record<string, number>;
}

export interface ReliabilityResponse {
  overall_status: string;
  threshold: number;
  factors: Record<string, ReliabilityFactorDetail>;
  disclaimer: string;
}

export interface EDAResponse {
  sample_size: number;
  is_synthetic: boolean;
  pri_mean: number;
  pri_std: number;
  attrition_base_rate: number;
  risk_distribution: Record<string, number>;
  achieved_cronbach_alphas: Record<string, number>;
  available_plots: string[];
}

export interface GlobalShapResponse {
  track: string;
  champion_model: string;
  factor_importance: Array<{
    factor?: string;
    feature?: string;
    mean_abs_shap?: number;
    importance?: number;
    percentage?: number;
  }>;
  global_item_importance: Array<{
    item?: string;
    feature?: string;
    mean_abs_shap?: number;
    importance?: number;
  }>;
  sanity_checks_audit: {
    additivity_pass?: boolean;
    seed_stability_spearman?: number;
    permutation_agreement_spearman?: number;
    ground_truth_correlation?: number;
    notes?: string;
  };
  plot_paths: Record<string, string>;
  disclaimer: string;
}

export interface ValidationReport {
  status: 'PASSED' | 'FAILED' | 'WARNING';
  error_count: number;
  warning_count: number;
  messages: string[];
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  detected_columns: string[];
  suggested_mappings: Record<string, string>;
  mapping_confidences: Record<string, number>;
  validation_report: ValidationReport;
  has_outcomes: boolean;
  created_at: string;
  expires_at: string;
}

export interface ContributorDetail {
  feature?: string;
  factor?: string;
  shap_value: number;
  effect_direction?: string;
  feature_value?: number;
}

export interface StudentPrediction {
  study_id: string;
  pri_score: number;
  risk_category: 'Low' | 'Moderate' | 'High';
  class_probabilities: Record<string, number>;
  factor_scores: {
    stress: number;
    motivation: number;
    belongingness: number;
    self_efficacy: number;
    peer_support: number;
  };
  top_risk_contributors: ContributorDetail[];
  top_protective_contributors: ContributorDetail[];
  narrative_headline: string;
  narrative_drivers: string;
  narrative_buffers: string;
  narrative_actionable_advice?: string;
}

export interface CohortPredictionResponse {
  upload_id: string;
  model_used: string;
  track: string;
  students: StudentPrediction[];
  cohort_summary: {
    total_submitted: number;
    valid_records: number;
    rejected_records: number;
    high_risk_count: number;
    pri_mean: number;
    pri_median: number;
    pri_std: number;
    risk_distribution: Record<string, number>;
    risk_percentages: Record<string, number>;
    rejected_reasons?: any[];
    top_cohort_risk_drivers?: string[];
    // Fallback/compatibility aliases
    total_students?: number;
    average_pri?: number;
  };
  disclaimer: string;
}

export interface ModelDetailsResponse {
  name: string;
  track: string;
  category: string;
  feature_set: string;
  is_champion: boolean;
  best_params: Record<string, any>;
  train_metrics?: Record<string, number>;
  validation_metrics?: Record<string, number>;
  test_metrics?: Record<string, number>;
  overfitting_gap?: number;
  inference_latency_ms?: number;
  model_size_kb?: number;
  diagnostic_plots: string[];
}

export interface JobStatusResponse {
  job_id: string;
  dataset_type: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress_pct: number;
  message: string;
  created_at: string;
  updated_at: string;
  results?: {
    champion_model: string;
    champion_performance: string;
    leaderboard: any[];
    [key: string]: any;
  };
}

export interface DocItem {
  filename: string;
  title: string;
  path: string;
}

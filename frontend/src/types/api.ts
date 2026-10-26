export interface HealthResponse {
  status: string;
  uptime_seconds: number;
  version: string;
  environment: string;
}

export interface MetaResponse {
  model_name: string;
  model_version: string;
  data_banner: string;
  is_calibrated: boolean;
  calibration_method: string;
  training_date: string;
  framework_component: string;
  student_id_format: string;
  run_id?: string;
  timestamp?: string;
}

export interface LeaderboardEntry {
  model: string;
  track: string;
  category?: string;
  pr_auc?: number;
  roc_auc?: number;
  f1?: number;
  brier_score?: number;
  brier?: number;
  ece?: number;
  threshold?: number;
  brier_skill_score?: number;
  balanced_accuracy?: number;
  accuracy?: number;
  r2?: number;
  mae?: number;
  rmse?: number;
  composite_score?: number;
  rank?: number;
  train_time_s?: number;
  overfitting_gap?: number;
  parameters_count?: number;
  qualified?: boolean;
  status?: string;
  disqualification_reasons?: string[];
  split?: string;
  run_id?: string;
  timestamp?: string;
}

export interface SelectionDecision {
  selected_model: string;
  track: string;
  selection_rationale: string;
  parsimony_applied: boolean;
  top_candidate_delta?: number;
}

export interface LeaderboardResponse {
  track: string;
  total_models: number;
  selected_model: string;
  selection_decision: SelectionDecision;
  category_winners: Record<string, LeaderboardEntry>;
  leaderboard: LeaderboardEntry[];
  run_id?: string;
  timestamp?: string;
  split_used?: string;
}

export interface ModelHistoryResponse {
  model: string;
  epochs_or_iterations: (number | string)[];
  metrics: Record<string, any>;
}

export interface ModelMetricsResponse {
  model: string;
  track: string;
  splits: Record<string, any>[];
  curves: {
    roc?: { fpr: number[]; tpr: number[]; auc?: number };
    pr?: { precision: number[]; recall: number[]; auc?: number };
    calibration?: { prob_pred: number[]; prob_true: number[] };
  };
}

export interface AllSplitsResponse {
  records_count: number;
  data: Record<string, any>[];
}

export interface ShapGlobalFeature {
  feature: string;
  mean_abs_shap: number;
  rank: number;
}

export interface ShapGlobalResponse {
  base_value: number;
  feature_count: number;
  global_importance: ShapGlobalFeature[];
  disclaimer: string;
}

export interface ShapDriverItem {
  feature: string;
  description: string;
  points: number;
  shap_raw: number;
}

export interface ShapStudentResponse {
  participant_id: string;
  eri_score: number;
  risk_band: string;
  narrative: string;
  main_drivers: ShapDriverItem[];
  protective_factors: ShapDriverItem[];
  disclaimer: string;
}

export interface ShapStabilityResponse {
  top3_overlap: number;
  top5_overlap: number;
  spearman_rank_correlation: number;
  is_stable: boolean;
  unstable_features: string[];
  interpretation: string;
}

export interface CausalExposureSummary {
  exposure?: string;
  primary_adjusted_estimate?: Record<string, any>;
  unadjusted_estimate?: Record<string, any>;
  g_computation_estimate?: Record<string, any>;
  estimators_comparison?: Record<string, any>[] | Record<string, any>;
  ground_truth_recovery?: Record<string, any>;
  diagnostics?: Record<string, any>;
  narrative?: string | Record<string, any>;
}

export interface CausalSummaryResponse {
  metadata: Record<string, any>;
  exposures: Record<string, CausalExposureSummary>;
  disclaimer: string;
}

export interface CausalDiagnosticsResponse {
  balance_smd: Record<string, any>;
  exposure_diagnostics: Record<string, any>;
}

export interface CausalHeterogeneityResponse {
  status: string;
  gate_passed: boolean;
  n_samples: number;
  n_events: number;
  reason?: string;
  overall_cate_mean?: number;
  subgroups: Record<string, any>;
  disclaimer: string;
}

export interface DagNode {
  id: string;
  label: string;
  role: 'exposure' | 'confounder' | 'mediator' | 'outcome';
}

export interface DagEdge {
  source: string;
  target: string;
  description?: string;
}

export interface DagResponse {
  nodes: DagNode[];
  edges: DagEdge[];
  exposures: string[];
  mediators: string[];
  confounders: string[];
  primary_outcome: string;
  image_url: string;
}

export interface SchemaReport {
  total_rows_ingested: number;
  columns_matched_count: number;
  matched_features: string[];
  missing_features_imputed: string[];
  ignored_unknown_columns: string[];
  dropped_pii_columns: string[];
  has_outcome_column: boolean;
  generated_pseudonyms?: boolean;
}

export interface PredictionItem {
  participant_id: string;
  eri: number;
  risk_band: string;
  intervention_sla: string;
  recommended_action: string;
  narrative: string;
  data_quality_flags: number;
  confidence_note: string;
}

export interface CohortSummary {
  total_students: number;
  mean_eri: number;
  median_eri: number;
  std_eri?: number;
  risk_distribution?: Record<string, number>;
  risk_percentages?: Record<string, number>;
  top_cohort_drivers: Record<string, any>[];
  causal_summary_status: string;
}

export interface UploadPredictResponse {
  job_id: string;
  status: string;
  schema_report: SchemaReport;
  cohort_summary: CohortSummary;
  scored_students_sample: PredictionItem[];
  download_urls: Record<string, string>;
}

export interface FusionScoreItem {
  participant_id: string;
  eri: number;
  band: string;
  explanation_type: string;
  model_version: string;
  top_drivers: string[];
  narrative?: string;
}

export interface FusionScoreResponse {
  total_scored: number;
  explanation_type: string;
  scores: FusionScoreItem[];
}

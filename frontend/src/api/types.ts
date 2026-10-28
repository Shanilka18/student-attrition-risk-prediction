/**
 * TypeScript API definitions for the Multimodal Student Engagement Intelligence (MEI) Platform.
 * Matches docs/api/openapi.json and Pydantic response models in src/mei/api/models.py.
 * 
 * Component ID: KND_CDAP_IT_05
 */

// ---------------------------------------------------------------------------
// Health & Status
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  active_run_id: string;
}

export interface ChampionSummary {
  model_id: string;
  display_name: string;
  plain_name: string;
  category: string;
  test_f1?: number | null;
  test_auc_roc?: number | null;
  test_mcc?: number | null;
  cv_score: number;
  cv_f1: number;
  optimal_threshold: number;
  inference_time_ms: number;
  calibrator: string;
  band_verification_monotonic: boolean;
}

export interface StatusResponse {
  active_run_id: string;
  data_source: string;
  is_generated_sample: boolean;
  data_source_badge: string;
  n_students_total: int_or_number;
  n_students_train: int_or_number;
  n_students_test: int_or_number;
  trained_at: string;
  champion: ChampionSummary;
  track_b_champion: Record<string, any>;
  model_card_essentials: {
    component_id: string;
    project_name: string;
    bootstrap_95_ci: Record<string, [number, number]>;
    limitations: string[];
    sensitive_threshold: Record<string, any>;
  };
  bands?: {
    definitions: Record<
      string,
      { min: number; max: number; label: string; hex_color: string; action: string }
    >;
    test_holdout_verification: {
      bands: Array<{
        band_key: string;
        band_label: string;
        score_range: string;
        n_students: number;
        observed_at_risk_count: number;
        observed_at_risk_pct: number;
        mean_mei: number;
      }>;
      is_strictly_monotonic: boolean;
      observed_rates: number[];
      monotonicity_status: string;
    };
  };
}

type int_or_number = number;

// ---------------------------------------------------------------------------
// Leaderboard & Model Detail
// ---------------------------------------------------------------------------

export interface LeaderboardRow {
  rank: number;
  model_id: string;
  display_name: string;
  plain_name: string;
  category: string;
  feature_set: string;
  leaderboard_score: number;
  generalization_gap: number;
  overfit_status: "Good fit" | "Slight overfit" | "Overfit" | string;
  // Classification metrics
  val_f1_optimal?: number | null;
  val_auc_roc?: number | null;
  val_balanced_accuracy?: number | null;
  val_mcc?: number | null;
  val_precision?: number | null;
  val_recall?: number | null;
  val_brier_score?: number | null;
  val_ece?: number | null;
  train_f1_mean?: number | null;
  optimal_threshold?: number | null;
  // Regression metrics
  val_r2?: number | null;
  val_rmse?: number | null;
  val_mae?: number | null;
  val_spearman_rho?: number | null;
  val_within_10_pct?: number | null;
  train_r2_mean?: number | null;
  // Test metrics (null when quarantined / not evaluated)
  test_f1?: number | null;
  test_auc_roc?: number | null;
  test_mcc?: number | null;
  test_r2?: number | null;
  test_rmse?: number | null;
  test_spearman_rho?: number | null;
  // Timings & Flags
  training_time_s: number;
  inference_time_ms: number;
  is_champion: boolean;
  is_category_best: boolean;
  is_remediated: boolean;
}

export interface LeaderboardResponse {
  track: "classification" | "regression";
  active_run_id: string;
  total_models: number;
  metric_name: string;
  champion_id: string;
  rows: LeaderboardRow[];
}

export interface ModelDetailResponse {
  model_id: string;
  display_name: string;
  plain_name: string;
  category: string;
  track: string;
  feature_set: string;
  leaderboard_score: number;
  generalization_gap: number;
  overfit_status: string;
  status_explanation: string;
  hyperparameters: Record<string, any>;
  training_time_s: number;
  inference_time_ms: number;
  metrics: Record<string, any>;
  fold_metrics: Array<Record<string, any>>;
  confusion_matrix_raw?: number[][] | null;
  confusion_matrix_norm?: number[][] | null;
  threshold_table?: Array<Record<string, any>> | null;
  test_metrics?: Record<string, any> | null;
}

export interface ModelCurvesResponse {
  model_id: string;
  track: string;
  category: string;
  roc_fpr?: number[] | null;
  roc_tpr?: number[] | null;
  pr_precision?: number[] | null;
  pr_recall?: number[] | null;
  cal_prob_true?: number[] | null;
  cal_prob_pred?: number[] | null;
  epoch_curves?: Record<string, number[]> | null;
  best_epoch?: number | null;
  stage_2_start_epoch?: number | null;
  stage_2_used?: boolean | null;
  overfitting_zone?: Record<string, any> | any[] | null;
  iteration_losses?: Record<string, number[]> | null;
  learning_curves?: Record<string, any> | null;
  validation_curves?: Record<string, any> | null;
  remediation_comparison?: Record<string, any> | null;
}

export interface CompareModelsResponse {
  track: string;
  model_ids: string[];
  models: ModelDetailResponse[];
  metrics_comparison: Record<string, Record<string, any>>;
}

// ---------------------------------------------------------------------------
// Patterns & Insights
// ---------------------------------------------------------------------------

export interface EngagementPatternsResponse {
  active_run_id: string;
  hmm_k: number;
  clustering_k: number;
  state_names: Record<string, string>;
  transition_matrix: number[][];
  emission_means: number[][] | Record<string, number[]>;
  expected_dwell_times: Record<string, number>;
  pattern_distribution: Record<string, number>;
  level_distribution: Record<string, number>;
  mean_trajectories_by_pattern: Record<string, Record<string, number>>;
  risk_rates_by_pattern: Record<
    string,
    {
      n_students: number;
      at_risk_count: number;
      at_risk_pct: number;
      ci95_low: number;
      ci95_high: number;
    }
  >;
  clustering_comparison_table: Array<Record<string, any>>;
  ablation_benchmark: Record<string, any>;
}

export interface DriverItem {
  rank: number;
  feature: string;
  plain_name: string;
  indicator: string;
  period: string;
  mean_abs_shap: number;
  mean_abs_points: number;
}

export interface BeeswarmPoint {
  student_idx: number;
  feature: string;
  plain_name: string;
  shap_val: number;
  shap_points: number;
  feature_val: number;
}

export interface DependencePoint {
  feature_val: number;
  shap_points: number;
}

export interface RiskDriversResponse {
  active_run_id: string;
  baseline_probability: number;
  baseline_mei: number;
  validation_ranking: DriverItem[];
  test_ranking: DriverItem[];
  grouped_importance: {
    by_indicator: Record<string, { total_shap: number; share_pct: number }>;
    by_period: Record<string, { total_shap: number; share_pct: number }>;
    by_family: Record<string, { total_shap: number; share_pct: number }>;
  };
  shap_consistency: {
    n_folds: number;
    mean_spearman: number;
    std_spearman: number;
    kendall_w: number;
    mean_top10_jaccard: number;
    spearman_matrix: number[][];
  };
  cross_method_comparison: {
    correlations: {
      shap_vs_permutation: number;
      shap_vs_logistic: number;
      permutation_vs_logistic: number;
    };
  };
  pattern_importance: Record<
    string,
    {
      pattern_name: string;
      n_students: number;
      warning_small_sample: boolean;
      drivers: Array<{
        feature: string;
        plain_name: string;
        mean_shap: number;
        mean_shap_points: number;
      }>;
    }
  >;
  tree_hybrid_audit?: Record<string, any> | null;
  beeswarm_samples: BeeswarmPoint[];
  dependence_samples: Record<string, DependencePoint[]>;
}

// ---------------------------------------------------------------------------
// Survey Templates & Dictionary
// ---------------------------------------------------------------------------

export interface DictionaryItem {
  name: string;
  type: string;
  role: string;
  period?: number | null;
  indicator?: string | null;
  description: string;
  allowed_codes?: number[] | null;
  text_labels?: Record<string, string> | null;
}

export interface DictionaryResponse {
  version: string;
  num_columns: number;
  num_features: number;
  num_outcomes: number;
  columns: DictionaryItem[];
}

// ---------------------------------------------------------------------------
// Student Scoring & Validation
// ---------------------------------------------------------------------------

export interface ProblemRowDetail {
  row_index: number;
  student_id: string;
  reasons: string[];
}

export interface ColumnMatchDetail {
  required_key: string;
  plain_name: string;
  matched_column?: string | null;
  confidence_tier: "High" | "Check this" | "Not found" | string;
  confidence_score: number;
  sample_values: string[];
  status: "auto_accepted" | "needs_confirmation" | "unmapped" | string;
}

export interface DataQualitySummary {
  total_rows: number;
  scorable_rows: number;
  missing_answers_rows: number;
  out_of_range_rows: number;
  duplicate_rows: number;
  traffic_light: "GREEN" | "AMBER" | "RED" | string;
  message: string;
}

export interface FixableProblemItem {
  issue: string;
  affected_rows: number;
  system_action: string;
}

export interface ValidateUploadResponse {
  upload_id: string;
  filename: string;
  total_rows: number;
  detected_columns: string[];
  column_mapping: Record<string, string>;
  mapping_confidences: Record<string, number>;
  unmapped_required_columns: string[];
  unknown_columns: string[];
  suggestions: Record<string, Array<{ canonical: string; confidence: number }>>;
  traffic_light: "GREEN" | "AMBER" | "RED";
  scorable_rows_count: number;
  problematic_rows_count: number;
  sample_row_problems: ProblemRowDetail[];
  ready_to_score: boolean;
  column_matches?: ColumnMatchDetail[];
  quality_summary?: DataQualitySummary | null;
  fixable_problems?: FixableProblemItem[];
}

export interface RunScoringRequest {
  upload_id: string;
  mapping_overrides?: Record<string, string> | null;
}

export interface ScoredStudentRow {
  student_id: string;
  mei: number;
  band: "Low" | "Moderate" | "High";
  flag: "Yes" | "No";
  pattern: "Consistent" | "Declining" | "Sporadic" | "Improving";
  level: "Low" | "Medium" | "High";
  state_path: string;
  top_reasons: string;
  not_scored_reason?: string | null;
}

export interface ScoringOutcomeCheck {
  label: string;
  total_outcome_rows: number;
  actual_at_risk_count: number;
  predicted_at_risk_count: number;
  f1_score?: number | null;
  auc_roc?: number | null;
  recall?: number | null;
  precision?: number | null;
  balanced_accuracy?: number | null;
  confusion_matrix: {
    tp: number;
    fp: number;
    fn: number;
    tn: number;
  };
}

export interface RunScoringResponse {
  upload_id: string;
  total_rows: number;
  scored_count: number;
  unscored_count: number;
  risk_distribution: {
    Low: number;
    Moderate: number;
    High: number;
  };
  risk_distribution_pct: {
    Low: number;
    Moderate: number;
    High: number;
  };
  outcome_check?: ScoringOutcomeCheck | null;
}

export interface PaginatedScoredRows {
  upload_id: string;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  rows: ScoredStudentRow[];
}

export interface SingleStudentReason {
  text: string;
  direction: "risk_increasing" | "risk_decreasing";
  points: number;
  plain_name: string;
  feature: string;
  value: number;
}

export interface SingleScoreRequest {
  answers: Record<string, any>;
  student_id?: string;
}

export interface SingleScoreResponse {
  student_id: string;
  mei: number;
  band: string;
  flag: string;
  baseline_mei: number;
  points_difference: number;
  pattern: string;
  level: string;
  state_path: string;
  reasons: SingleStudentReason[];
}

export interface StudentDetailResponse {
  student_id: string;
  mei: number;
  band: string;
  flag: string;
  pattern: string;
  level: string;
  state_path: string;
  baseline_mei: number;
  points_difference: number;
  trajectories: {
    composite: number[];
    indicators: Record<string, number[]>;
  };
  reasons: SingleStudentReason[];
  plain_meanings: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Run Management
// ---------------------------------------------------------------------------

export interface RunSummary {
  run_id: string;
  is_active: boolean;
  data_source: string;
  trained_at: string;
  n_students: number;
  champion_model_id: string;
  champion_f1?: number | null;
  champion_cv_score?: number | null;
}

export interface RunsListResponse {
  active_run_id: string;
  total_runs: number;
  runs: RunSummary[];
}

// ---------------------------------------------------------------------------
// Background Jobs
// ---------------------------------------------------------------------------

export interface JobStatusResponse {
  job_id: string;
  job_type: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  result?: Record<string, any> | null;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// Retraining & Model Governance
// ---------------------------------------------------------------------------

export interface RetrainValidateResponse {
  upload_id: string;
  filename: string;
  total_rows: number;
  usable_rows: number;
  class_balance: {
    at_risk_count: number;
    not_at_risk_count: number;
    at_risk_pct: number;
  };
  missingness: {
    overall_missing_pct: number;
    mapped_questions_count: number;
  };
  warnings: string[];
  estimated_time_min: number;
  traffic_light: "GREEN" | "AMBER" | "RED" | string;
  is_refusal: boolean;
  refusal_reason: string | null;
  can_proceed: boolean;
  guards: {
    can_retrain: boolean;
    refusal_reason: string | null;
    traffic_light: string;
    is_small_cohort: boolean;
    cv_scheme: string;
    cv_folds_count: number;
    is_imbalanced: boolean;
    use_class_weights: boolean;
    disabled_models: string[];
    warnings: string[];
  };
}

export interface RetrainStartResponse {
  job_id: string;
  status: string;
  mode: string;
  dataset_name: string;
}

export interface RunComparisonResponse {
  run_a: {
    run_id: string;
    data_source: string;
    n_students: number;
    created_at: string;
    champion: Record<string, any>;
    leaderboard: LeaderboardRow[];
  };
  run_b: {
    run_id: string;
    data_source: string;
    n_students: number;
    created_at: string;
    champion: Record<string, any>;
    leaderboard: LeaderboardRow[];
  };
  comparison: {
    champion_changed: boolean;
    champion_a: string;
    champion_b: string;
    cv_score_diff: number;
    test_f1_diff: number;
    rank_changes: Array<{
      model_id: string;
      display_name: string;
      category: string;
      old_rank: number | null;
      new_rank: number;
      rank_delta: number | null;
      score: number;
    }>;
    verdict_summary: string;
    is_recommended: boolean;
  };
}

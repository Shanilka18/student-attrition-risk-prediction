"""Pydantic schemas and response models for the MEI FastAPI backend.

Strictly models real data structures produced by the MEI pipeline.
Zero hard-coded values or artificial mockup fields.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field("healthy", description="API health status")
    timestamp: str = Field(..., description="Current ISO 8601 server timestamp")
    version: str = Field("1.0.0", description="MEI API contract version")
    active_run_id: str = Field(..., description="Active pipeline run identifier")


class ChampionSummary(BaseModel):
    """Essential details of the frozen champion model."""

    model_id: str
    display_name: str
    plain_name: str
    category: str
    test_f1: Optional[float] = None
    test_auc_roc: Optional[float] = None
    test_mcc: Optional[float] = None
    cv_score: float
    cv_f1: float
    optimal_threshold: float
    inference_time_ms: float
    calibrator: str
    band_verification_monotonic: bool


class StatusResponse(BaseModel):
    """System and active run status."""

    active_run_id: str
    data_source: str
    is_generated_sample: bool
    data_source_badge: str
    n_students_total: int
    n_students_train: int
    n_students_test: int
    trained_at: str
    champion: ChampionSummary
    track_b_champion: Dict[str, Any]
    model_card_essentials: Dict[str, Any]
    bands: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Leaderboard & Models
# ---------------------------------------------------------------------------


class LeaderboardRow(BaseModel):
    """Single model benchmark row on leaderboard."""

    rank: int
    model_id: str
    display_name: str
    plain_name: str
    category: str
    feature_set: str
    leaderboard_score: float
    generalization_gap: float
    overfit_status: str
    # Classification metrics (optional if regression)
    val_f1_optimal: Optional[float] = None
    val_auc_roc: Optional[float] = None
    val_balanced_accuracy: Optional[float] = None
    val_mcc: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    val_brier_score: Optional[float] = None
    val_ece: Optional[float] = None
    train_f1_mean: Optional[float] = None
    optimal_threshold: Optional[float] = None
    # Regression metrics (optional if classification)
    val_r2: Optional[float] = None
    val_rmse: Optional[float] = None
    val_mae: Optional[float] = None
    val_spearman_rho: Optional[float] = None
    val_within_10_pct: Optional[float] = None
    train_r2_mean: Optional[float] = None
    # Test metrics (null when quarantined / not evaluated)
    test_f1: Optional[float] = None
    test_auc_roc: Optional[float] = None
    test_mcc: Optional[float] = None
    test_r2: Optional[float] = None
    test_rmse: Optional[float] = None
    test_spearman_rho: Optional[float] = None
    # Timings & Flags
    training_time_s: float
    inference_time_ms: float
    is_champion: bool = False
    is_category_best: bool = False
    is_remediated: bool = False

    @field_validator(
        "test_f1",
        "test_auc_roc",
        "test_mcc",
        "test_r2",
        "test_rmse",
        "test_spearman_rho",
        "val_ece",
        mode="before",
    )
    @classmethod
    def parse_optional_floats(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            import math

            return None if math.isnan(v) else float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None


class LeaderboardResponse(BaseModel):
    """Full leaderboard response for classification or regression."""

    track: str
    active_run_id: str
    total_models: int
    metric_name: str
    champion_id: str
    rows: List[LeaderboardRow]


class ModelDetailResponse(BaseModel):
    """Comprehensive performance and configuration detail for a single model."""

    model_id: str
    display_name: str
    plain_name: str
    category: str
    track: str
    feature_set: str
    leaderboard_score: float
    generalization_gap: float
    overfit_status: str
    status_explanation: str
    hyperparameters: Dict[str, Any]
    training_time_s: float
    inference_time_ms: float
    # Split metrics
    metrics: Dict[str, Any]
    fold_metrics: List[Dict[str, Any]]
    confusion_matrix_raw: Optional[List[List[int]]] = None
    confusion_matrix_norm: Optional[List[List[float]]] = None
    threshold_table: Optional[List[Dict[str, Any]]] = None
    test_metrics: Optional[Dict[str, Any]] = None


class ModelCurvesResponse(BaseModel):
    """Training, epoch, and metric curves for visualizations."""

    model_id: str
    track: str
    category: str
    # ROC, PR, and Calibration curves
    roc_fpr: Optional[List[float]] = None
    roc_tpr: Optional[List[float]] = None
    pr_precision: Optional[List[float]] = None
    pr_recall: Optional[List[float]] = None
    cal_prob_true: Optional[List[float]] = None
    cal_prob_pred: Optional[List[float]] = None
    # Deep model epoch curves
    epoch_curves: Optional[Dict[str, List[float]]] = None
    best_epoch: Optional[int] = None
    stage_2_start_epoch: Optional[int] = None
    stage_2_used: Optional[bool] = None
    overfitting_zone: Optional[Union[Dict[str, Any], List[Any]]] = None
    # Boosting iteration loss curves
    iteration_losses: Optional[Dict[str, List[float]]] = None
    # Learning curves & validation curves
    learning_curves: Optional[Dict[str, Any]] = None
    validation_curves: Optional[Dict[str, Any]] = None
    # Remediation before vs after
    remediation_comparison: Optional[Dict[str, Any]] = None


class CompareModelsResponse(BaseModel):
    """Side-by-side comparison payload for up to 4 models."""

    track: str
    model_ids: List[str]
    models: List[Dict[str, Any]]
    metrics_comparison: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Engagement Patterns & Insights
# ---------------------------------------------------------------------------


class EngagementPatternsResponse(BaseModel):
    """Engagement states, transition dynamics, and cohort patterns."""

    active_run_id: str
    hmm_k: int
    clustering_k: int
    state_names: Dict[str, str]
    transition_matrix: List[List[float]]
    emission_means: Union[List[List[float]], Dict[str, List[float]]]
    expected_dwell_times: Dict[str, float]
    pattern_distribution: Dict[str, int]
    level_distribution: Dict[str, int]
    mean_trajectories_by_pattern: Dict[str, Dict[str, float]]
    risk_rates_by_pattern: Dict[str, Dict[str, Any]]
    clustering_comparison_table: List[Dict[str, Any]]
    ablation_benchmark: Dict[str, Any]


# ---------------------------------------------------------------------------
# Risk Drivers & Explainability
# ---------------------------------------------------------------------------


class RiskDriversResponse(BaseModel):
    """Global, grouped, and cohort-stratified feature attribution drivers."""

    active_run_id: str
    baseline_probability: float
    baseline_mei: float
    validation_ranking: List[Dict[str, Any]]
    grouped_importance: Dict[str, Any]
    shap_consistency: Dict[str, Any]
    cross_method_comparison: Dict[str, Any]
    pattern_importance: Dict[str, Any]
    tree_hybrid_audit: Optional[Dict[str, Any]] = None
    beeswarm_samples: List[Dict[str, Any]]
    dependence_samples: Dict[str, List[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Survey Templates & Dictionary
# ---------------------------------------------------------------------------


class DictionaryItem(BaseModel):
    """Data dictionary column entry."""

    name: str
    type: str
    role: str
    period: Optional[int] = None
    indicator: Optional[str] = None
    description: str
    allowed_codes: Optional[List[int]] = None
    text_labels: Optional[Dict[str, str]] = None


class DictionaryResponse(BaseModel):
    """Complete data dictionary schema."""

    version: str
    num_columns: int
    num_features: int
    num_outcomes: int
    columns: List[DictionaryItem]


# ---------------------------------------------------------------------------
# Student Scoring & Validation
# ---------------------------------------------------------------------------


class ColumnMatchDetail(BaseModel):
    """Detailed matching status for each required survey item."""

    required_key: str
    plain_name: str
    matched_column: Optional[str] = None
    confidence_tier: str  # "High" | "Check this" | "Not found"
    confidence_score: float
    sample_values: List[str] = []
    status: str  # "auto_accepted" | "needs_confirmation" | "unmapped"


class DataQualitySummary(BaseModel):
    """Traffic light audit of uploaded tabular dataset."""

    total_rows: int
    scorable_rows: int
    missing_answers_rows: int
    out_of_range_rows: int
    duplicate_rows: int
    traffic_light: str  # "GREEN" | "AMBER" | "RED"
    message: str


class FixableProblemItem(BaseModel):
    """Structured remediation action for imperfect data."""

    issue: str
    affected_rows: int
    system_action: str


class ValidateUploadResponse(BaseModel):
    """Result of Stage 1-2 upload validation."""

    upload_id: str
    filename: str
    total_rows: int
    detected_columns: List[str]
    column_mapping: Dict[str, str]
    mapping_confidences: Dict[str, float]
    unmapped_required_columns: List[str]
    unknown_columns: List[str]
    suggestions: Dict[str, List[Dict[str, Any]]]
    traffic_light: str  # GREEN | AMBER | RED
    scorable_rows_count: int
    problematic_rows_count: int
    sample_row_problems: List[Dict[str, Any]]
    ready_to_score: bool
    column_matches: List[ColumnMatchDetail] = []
    quality_summary: Optional[DataQualitySummary] = None
    fixable_problems: List[FixableProblemItem] = []


class RunScoringRequest(BaseModel):
    """Request payload to trigger batch scoring on validated upload."""

    upload_id: str
    mapping_overrides: Optional[Dict[str, str]] = None


class ScoredStudentRow(BaseModel):
    """Scored student result row."""

    student_id: str
    mei: int
    band: str
    flag: str
    pattern: str
    level: str
    state_path: str
    top_reasons: str
    not_scored_reason: Optional[str] = None


class ScoringOutcomeCheck(BaseModel):
    """Validation block when uploaded file contains outcome questions."""

    label: str = "Model Evaluation on Uploaded File"
    total_outcome_rows: int
    actual_at_risk_count: int
    predicted_at_risk_count: int
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    confusion_matrix: Dict[str, int]


class RunScoringResponse(BaseModel):
    """Result summary of batch scoring execution."""

    upload_id: str
    total_rows: int
    scored_count: int
    unscored_count: int
    risk_distribution: Dict[str, int]
    risk_distribution_pct: Dict[str, float]
    outcome_check: Optional[ScoringOutcomeCheck] = None


class PaginatedScoredRows(BaseModel):
    """Paginated view of student scoring results."""

    upload_id: str
    total: int
    page: int
    page_size: int
    total_pages: int
    rows: List[ScoredStudentRow]


class SingleStudentReason(BaseModel):
    """Additive explanation reason for single student scoring."""

    text: str
    direction: str  # risk_increasing | risk_decreasing
    points: int
    plain_name: str
    feature: str
    value: float


class SingleScoreRequest(BaseModel):
    """Single student survey input for on-demand scoring."""

    answers: Dict[str, Any]
    student_id: Optional[str] = "STUDENT_INLINE"


class SingleScoreResponse(BaseModel):
    """Comprehensive scoring and attribution result for a single student."""

    student_id: str
    mei: int
    band: str
    flag: str
    baseline_mei: float
    points_difference: int
    pattern: str
    level: str
    state_path: str
    reasons: List[SingleStudentReason]


class StudentDetailResponse(BaseModel):
    """Comprehensive detail for a student row clicked in the results table."""

    student_id: str
    mei: int
    band: str
    flag: str
    pattern: str
    level: str
    state_path: str
    baseline_mei: float
    points_difference: int
    trajectories: Dict[
        str, Any
    ]  # {"composite": [p1, p2, p3, p4], "indicators": {ind: [p1, p2, p3, p4]}}
    reasons: List[SingleStudentReason]
    plain_meanings: Dict[str, str]


# ---------------------------------------------------------------------------
# Run Management
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Summary of an available experiment run."""

    run_id: str
    is_active: bool
    data_source: str
    trained_at: str
    n_students: int
    champion_model_id: str
    champion_f1: Optional[float] = None
    champion_cv_score: Optional[float] = None


class RunsListResponse(BaseModel):
    """List of all available runs in the project."""

    active_run_id: str
    total_runs: int
    runs: List[RunSummary]


# ---------------------------------------------------------------------------
# Background Jobs
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Status payload for asynchronous background task."""

    job_id: str
    job_type: str
    status: str  # queued | running | finished | failed | cancelled
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

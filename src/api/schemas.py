"""Pydantic schemas for the ERI FastAPI microservice.

Defines all request and response models for:
- Health and system metadata
- Model leaderboard and performance curves
- Explainability (SHAP global, local, stability)
- Applied Causal inference and DAG representation
- Batch scoring, upload handling, Fusion Framework scoring, and retraining jobs
"""

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health & Metadata Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    uptime_seconds: float = Field(..., example=120.5)
    version: str = Field(..., example="1.0.0")
    environment: str = Field(..., example="development")


class MetaResponse(BaseModel):
    model_name: str = Field(..., example="logistic_l2")
    model_version: str = Field(..., example="1.0.0")
    data_banner: str = Field(
        ...,
        example="SYNTHETIC - pipeline validation only",
        description="Mandatory banner flag for pipeline validation data",
    )
    is_calibrated: bool = Field(..., example=True)
    calibration_method: str = Field(..., example="isotonic")
    training_date: str = Field(..., example="2026-10-24")
    framework_component: str = Field(
        ...,
        example="Environmental & Financial Causal Risk Intelligence Framework",
    )
    student_id_format: str = Field(..., example="Pseudonymous (P-XXXXX / STU_XXXX)")
    run_id: str | None = Field(default=None, example="run_20261025_160000")
    timestamp: str | None = Field(default=None, example="2026-10-25T10:30:00Z")


# ---------------------------------------------------------------------------
# Leaderboard & Model Inspection Schemas
# ---------------------------------------------------------------------------


class LeaderboardEntry(BaseModel):
    model: str
    track: str
    category: str | None = None
    pr_auc: float | None = None
    roc_auc: float | None = None
    f1: float | None = None
    brier_score: float | None = None
    brier: float | None = None
    ece: float | None = None
    threshold: float | None = None
    brier_skill_score: float | None = None
    balanced_accuracy: float | None = None
    r2: float | None = None
    mae: float | None = None
    rmse: float | None = None
    composite_score: float | None = None
    rank: int | None = None
    qualified: bool | None = True
    status: str | None = "passed"
    disqualification_reasons: list[str] | None = Field(default_factory=list)
    split: str | None = "validation"
    run_id: str | None = None
    timestamp: str | None = None


class SelectionDecision(BaseModel):
    selected_model: str
    track: str
    selection_rationale: str
    parsimony_applied: bool
    top_candidate_delta: float | None = None


class LeaderboardResponse(BaseModel):
    track: str
    total_models: int
    selected_model: str
    selection_decision: dict[str, Any]
    category_winners: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    run_id: str | None = None
    timestamp: str | None = None
    split_used: str | None = "validation"


class ModelHistoryResponse(BaseModel):
    model: str
    epochs_or_iterations: list[int | float]
    metrics: dict[str, Any]


class ModelMetricsResponse(BaseModel):
    model: str
    track: str
    splits: list[dict[str, Any]]
    curves: dict[str, Any] = Field(
        default_factory=dict,
        description="ROC, PR, and Calibration curves if available",
    )


class AllSplitsResponse(BaseModel):
    records_count: int
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Explainability (SHAP) Schemas
# ---------------------------------------------------------------------------


class ShapGlobalFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    rank: int


class ShapGlobalResponse(BaseModel):
    base_value: float
    feature_count: int
    global_importance: list[ShapGlobalFeature]
    disclaimer: str = Field(
        default="This explains the model's score. It does not prove these factors caused the outcome."
    )


class ShapDriverItem(BaseModel):
    feature: str
    description: str
    points: int
    shap_raw: float


class ShapStudentResponse(BaseModel):
    participant_id: str
    eri_score: float
    risk_band: str
    narrative: str
    main_drivers: list[ShapDriverItem]
    protective_factors: list[ShapDriverItem]
    disclaimer: str = Field(
        default="This explains the model's score. It does not prove these factors caused the outcome."
    )


class ShapStabilityResponse(BaseModel):
    top3_overlap: float
    top5_overlap: float
    spearman_rank_correlation: float
    is_stable: bool
    unstable_features: list[str]
    interpretation: str


# ---------------------------------------------------------------------------
# Causal Analysis Schemas
# ---------------------------------------------------------------------------


class CausalExposureSummary(BaseModel):
    exposure: str | None = None
    primary_adjusted_estimate: dict[str, Any] | None = None
    unadjusted_estimate: dict[str, Any] | None = None
    g_computation_estimate: dict[str, Any] | None = None
    estimators_comparison: list[dict[str, Any]] | dict[str, Any] | None = None
    ground_truth_recovery: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    narrative: str | dict[str, Any] | None = None


class CausalSummaryResponse(BaseModel):
    metadata: dict[str, Any]
    exposures: dict[str, CausalExposureSummary]
    disclaimer: str = Field(
        default="Adjusted association under stated assumptions (observational data); not experimentally validated."
    )


class CausalDiagnosticsResponse(BaseModel):
    balance_smd: dict[str, Any]
    exposure_diagnostics: dict[str, Any]


class CausalHeterogeneityResponse(BaseModel):
    status: str
    gate_passed: bool
    n_samples: int
    n_events: int
    reason: str | None = None
    overall_cate_mean: float | None = None
    subgroups: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Adjusted association under stated assumptions (observational data); not experimentally validated."
    )


class DagNode(BaseModel):
    id: str
    label: str
    role: str = Field(..., description="exposure | confounder | outcome")


class DagEdge(BaseModel):
    source: str
    target: str
    description: str | None = None


class DagResponse(BaseModel):
    nodes: list[DagNode]
    edges: list[DagEdge]
    exposures: list[str]
    confounders: list[str]
    primary_outcome: str
    image_url: str


# ---------------------------------------------------------------------------
# Inference, Batch Upload & Fusion Framework Schemas
# ---------------------------------------------------------------------------


class SchemaReport(BaseModel):
    total_rows_ingested: int
    columns_matched_count: int
    matched_features: list[str]
    missing_features_imputed: list[str]
    ignored_unknown_columns: list[str]
    dropped_pii_columns: list[str]
    has_outcome_column: bool
    generated_pseudonyms: bool = True
    data_quality_report: dict[str, Any] | None = None


class PredictionItem(BaseModel):
    participant_id: str
    eri: float
    risk_band: str
    intervention_sla: str
    recommended_action: str
    narrative: str
    data_quality_flags: int
    confidence_note: str


class CohortSummary(BaseModel):
    total_students: int
    mean_eri: float
    median_eri: float
    std_eri: float | None = 0.0
    risk_distribution: dict[str, int] | None = Field(default_factory=dict)
    risk_percentages: dict[str, float] | None = Field(default_factory=dict)
    top_cohort_drivers: list[dict[str, Any]] = Field(default_factory=list)
    causal_summary_status: str


class UploadPredictResponse(BaseModel):
    job_id: str
    status: str
    schema_report: SchemaReport
    cohort_summary: CohortSummary
    scored_students_sample: list[PredictionItem]
    download_urls: dict[str, str]


class FusionScoreItem(BaseModel):
    participant_id: str
    eri: float
    band: str
    explanation_type: str = Field(default="model_explanation")
    model_version: str = Field(default="1.0.0")
    top_drivers: list[str] = Field(default_factory=list)
    narrative: str | None = None


class FusionScoreRequest(BaseModel):
    participant_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific Participant_IDs to score from the latest batch",
    )
    students: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional inline list of student questionnaire records to score directly",
    )


class FusionScoreResponse(BaseModel):
    total_scored: int
    explanation_type: str = "model_explanation"
    scores: list[FusionScoreItem]


# ---------------------------------------------------------------------------
# Retraining & Background Job Schemas
# ---------------------------------------------------------------------------


class RetrainRequest(BaseModel):
    outcome_column: str = Field(default="at_risk", example="at_risk")
    confirm: bool = Field(
        default=False,
        description="Must be explicitly True to replace production models and scalers",
    )
    dataset_path: str | None = Field(
        default=None,
        description="Path to uploaded or server-side training dataset. If omitted, uses default data.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str = Field(
        ..., example="completed"
    )  # queued | running | completed | failed
    progress_percent: int = Field(default=0, example=100)
    message: str = Field(default="")
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

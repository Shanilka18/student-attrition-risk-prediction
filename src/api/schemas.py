"""Pydantic schemas for the ERI FastAPI microservice.

Defines all request and response models for:
- Health and system metadata
- Model leaderboard and performance curves
- Explainability (SHAP global, local, stability)
- Applied Causal inference and DAG representation
- Batch scoring, upload handling, Fusion Framework scoring, and retraining jobs
"""

from typing import Dict, List, Optional, Any, Union
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
    run_id: Optional[str] = Field(default=None, example="run_20261025_160000")
    timestamp: Optional[str] = Field(default=None, example="2026-10-25T10:30:00Z")


# ---------------------------------------------------------------------------
# Leaderboard & Model Inspection Schemas
# ---------------------------------------------------------------------------

class LeaderboardEntry(BaseModel):
    model: str
    track: str
    category: Optional[str] = None
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    f1: Optional[float] = None
    brier_score: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    threshold: Optional[float] = None
    brier_skill_score: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    composite_score: Optional[float] = None
    rank: Optional[int] = None
    qualified: Optional[bool] = True
    status: Optional[str] = "passed"
    disqualification_reasons: Optional[List[str]] = Field(default_factory=list)
    split: Optional[str] = "validation"
    run_id: Optional[str] = None
    timestamp: Optional[str] = None


class SelectionDecision(BaseModel):
    selected_model: str
    track: str
    selection_rationale: str
    parsimony_applied: bool
    top_candidate_delta: Optional[float] = None


class LeaderboardResponse(BaseModel):
    track: str
    total_models: int
    selected_model: str
    selection_decision: Dict[str, Any]
    category_winners: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    run_id: Optional[str] = None
    timestamp: Optional[str] = None
    split_used: Optional[str] = "validation"


class ModelHistoryResponse(BaseModel):
    model: str
    epochs_or_iterations: List[Union[int, float]]
    metrics: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    model: str
    track: str
    splits: List[Dict[str, Any]]
    curves: Dict[str, Any] = Field(
        default_factory=dict,
        description="ROC, PR, and Calibration curves if available",
    )


class AllSplitsResponse(BaseModel):
    records_count: int
    data: List[Dict[str, Any]]


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
    global_importance: List[ShapGlobalFeature]
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
    main_drivers: List[ShapDriverItem]
    protective_factors: List[ShapDriverItem]
    disclaimer: str = Field(
        default="This explains the model's score. It does not prove these factors caused the outcome."
    )


class ShapStabilityResponse(BaseModel):
    top3_overlap: float
    top5_overlap: float
    spearman_rank_correlation: float
    is_stable: bool
    unstable_features: List[str]
    interpretation: str


# ---------------------------------------------------------------------------
# Causal Analysis Schemas
# ---------------------------------------------------------------------------

class CausalExposureSummary(BaseModel):
    exposure: Optional[str] = None
    primary_adjusted_estimate: Optional[Dict[str, Any]] = None
    unadjusted_estimate: Optional[Dict[str, Any]] = None
    g_computation_estimate: Optional[Dict[str, Any]] = None
    estimators_comparison: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ground_truth_recovery: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    narrative: Optional[Union[str, Dict[str, Any]]] = None


class CausalSummaryResponse(BaseModel):
    metadata: Dict[str, Any]
    exposures: Dict[str, CausalExposureSummary]
    disclaimer: str = Field(
        default="Adjusted association under stated assumptions (observational data); not experimentally validated."
    )


class CausalDiagnosticsResponse(BaseModel):
    balance_smd: Dict[str, Any]
    exposure_diagnostics: Dict[str, Any]


class CausalHeterogeneityResponse(BaseModel):
    status: str
    gate_passed: bool
    n_samples: int
    n_events: int
    reason: Optional[str] = None
    overall_cate_mean: Optional[float] = None
    subgroups: Dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Adjusted association under stated assumptions (observational data); not experimentally validated."
    )


class DagNode(BaseModel):
    id: str
    label: str
    role: str = Field(..., description="exposure | confounder | mediator | outcome")


class DagEdge(BaseModel):
    source: str
    target: str
    description: Optional[str] = None


class DagResponse(BaseModel):
    nodes: List[DagNode]
    edges: List[DagEdge]
    exposures: List[str]
    mediators: List[str]
    confounders: List[str]
    primary_outcome: str
    image_url: str


# ---------------------------------------------------------------------------
# Inference, Batch Upload & Fusion Framework Schemas
# ---------------------------------------------------------------------------

class SchemaReport(BaseModel):
    total_rows_ingested: int
    columns_matched_count: int
    matched_features: List[str]
    missing_features_imputed: List[str]
    ignored_unknown_columns: List[str]
    dropped_pii_columns: List[str]
    has_outcome_column: bool
    generated_pseudonyms: bool = True


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
    std_eri: Optional[float] = 0.0
    risk_distribution: Optional[Dict[str, int]] = Field(default_factory=dict)
    risk_percentages: Optional[Dict[str, float]] = Field(default_factory=dict)
    top_cohort_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    causal_summary_status: str


class UploadPredictResponse(BaseModel):
    job_id: str
    status: str
    schema_report: SchemaReport
    cohort_summary: CohortSummary
    scored_students_sample: List[PredictionItem]
    download_urls: Dict[str, str]


class FusionScoreItem(BaseModel):
    participant_id: str
    eri: float
    band: str
    explanation_type: str = Field(default="model_explanation")
    model_version: str = Field(default="1.0.0")
    top_drivers: List[str] = Field(default_factory=list)
    narrative: Optional[str] = None


class FusionScoreRequest(BaseModel):
    participant_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific Participant_IDs to score from the latest batch",
    )
    students: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional inline list of student questionnaire records to score directly",
    )


class FusionScoreResponse(BaseModel):
    total_scored: int
    explanation_type: str = "model_explanation"
    scores: List[FusionScoreItem]


# ---------------------------------------------------------------------------
# Retraining & Background Job Schemas
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    outcome_column: str = Field(default="student_attrition", example="student_attrition")
    confirm: bool = Field(
        default=False,
        description="Must be explicitly True to replace production models and scalers",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to uploaded or server-side training dataset. If omitted, uses default data.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str = Field(..., example="completed")  # queued | running | completed | failed
    progress_percent: int = Field(default=0, example=100)
    message: str = Field(default="")
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

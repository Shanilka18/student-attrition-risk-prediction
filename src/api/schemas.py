"""Pydantic data models for FastAPI validation, inference contracts, and Fusion Framework export."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. Questionnaire & Single Prediction Schemas (Backward Compatible)
# =====================================================================

class QuestionnaireInput(BaseModel):
    """Input payload representing 15-item Likert psychosocial questionnaire.

    Scale: 1 = Strongly Disagree to 5 = Strongly Agree.
    """

    study_id: str = Field(..., description="Anonymized participant identifier", json_schema_extra={"example": "STUDY_0042"})

    # Stress items (Risk)
    S1: int = Field(..., ge=1, le=5, description="Academic workload pressure")
    S2: int = Field(..., ge=1, le=5, description="Exam/assessment anxiety")
    S3: int = Field(..., ge=1, le=5, description="Financial or personal strain")

    # Motivation items (Protective)
    M1: int = Field(..., ge=1, le=5, description="Enthusiasm for chosen degree program")
    M2: int = Field(..., ge=1, le=5, description="Determination to complete studies")
    M3: int = Field(..., ge=1, le=5, description="Clear career aspirations")

    # Belongingness items (Protective)
    B1: int = Field(..., ge=1, le=5, description="Sense of belonging to university community")
    B2: int = Field(..., ge=1, le=5, description="Comfort engaging with faculty and campus")
    B3: int = Field(..., ge=1, le=5, description="Feeling accepted by peers")

    # Self-Efficacy items (Protective)
    SE1: int = Field(..., ge=1, le=5, description="Confidence in mastering complex coursework")
    SE2: int = Field(..., ge=1, le=5, description="Resilience in recovering from academic setbacks")
    SE3: int = Field(..., ge=1, le=5, description="Time management and self-study competence")

    # Peer Support items (Protective)
    PS1: int = Field(..., ge=1, le=5, description="Reliable study group or academic peer network")
    PS2: int = Field(..., ge=1, le=5, description="Emotional support from campus friends")
    PS3: int = Field(..., ge=1, le=5, description="Collaborative problem-solving with coursemates")


class FactorScores(BaseModel):
    """Psychosocial factor subscale scores on 1.0 to 5.0 raw scale."""

    stress: float = Field(..., description="Mean Stress score (1-5; higher = higher risk)")
    motivation: float = Field(..., description="Mean Motivation score (1-5; higher = higher protection)")
    belongingness: float = Field(..., description="Mean Belongingness score (1-5; higher = higher protection)")
    self_efficacy: float = Field(..., description="Mean Self-Efficacy score (1-5; higher = higher protection)")
    peer_support: float = Field(..., description="Mean Peer Support score (1-5; higher = higher protection)")


class FactorContribution(BaseModel):
    """SHAP attribution contribution for an individual factor or item."""

    feature_name: str
    impact: float = Field(..., description="SHAP value (positive increases risk, negative decreases risk)")
    interpretation: str


class PRIResponse(BaseModel):
    """Complete prediction and explanation payload for UI and single-student inspection."""

    study_id: str
    pri_score: float = Field(..., ge=0.0, le=100.0, description="Psychosocial Risk Index on 0-100 scale")
    risk_category: str = Field(..., description="'Low', 'Moderate', or 'High'")
    factor_scores: FactorScores
    top_risk_contributors: List[FactorContribution] = Field(default_factory=list)
    top_protective_contributors: List[FactorContribution] = Field(default_factory=list)
    disclaimer: str = Field(..., description="Ethical and legal research notice")


# =====================================================================
# 2. Metadata & Health Schemas
# =====================================================================

class MetaResponse(BaseModel):
    """Metadata detailing framework version, training date, hash, and status."""

    project: str
    model_version: str
    training_date_utc: str
    data_hash: str
    is_synthetic: bool
    disclaimer: str
    instrument_items: List[str]
    latent_factors: List[str]
    champions: Dict[str, str]
    status: str = "operational"


# =====================================================================
# 3. Leaderboard & Model Inspection Schemas
# =====================================================================

class LeaderboardEntry(BaseModel):
    """Single model benchmark evaluation record."""

    rank: int
    model_name: str
    category: str
    track: str
    feature_set: str
    primary_metric: float
    secondary_metric: Optional[float] = None
    overfitting_gap: Optional[float] = None
    ece: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    model_size_kb: Optional[float] = None
    composite_rank: Optional[int] = None
    is_champion: bool = False


class LeaderboardResponse(BaseModel):
    """Collection of benchmarked models for leaderboards."""

    track: Optional[str] = None
    total_models: int
    entries: List[LeaderboardEntry]


class ModelDetailsResponse(BaseModel):
    """Comprehensive performance profile, hyperparameters, and plots for a specific model."""

    name: str
    track: str
    category: str
    feature_set: str
    is_champion: bool
    best_params: Dict[str, Any]
    train_metrics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    overfitting_gap: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    model_size_kb: Optional[float] = None
    diagnostic_plots: List[str] = Field(default_factory=list)


# =====================================================================
# 4. Champion & Scientific Evaluation Schemas
# =====================================================================

class ChampionTrackDetail(BaseModel):
    """Specific champion model metadata."""

    model_name: str
    category: str
    feature_set: str
    selection_metric: str
    primary_metric_value: float
    overfitting_gap: float
    high_risk_recall: Optional[float] = None
    ece: Optional[float] = None
    test_metrics: Dict[str, Any]
    justification: str
    best_params: Dict[str, Any]


class ChampionResponse(BaseModel):
    """Current champion models across regression and classification tracks."""

    regression: ChampionTrackDetail
    classification: ChampionTrackDetail
    disclaimer: str


class ReliabilityFactorDetail(BaseModel):
    """Psychometric reliability profile for a latent factor."""

    factor: str
    cronbach_alpha: float
    mcdonald_omega: Optional[float] = None
    passed: bool
    items: List[str]
    item_total_correlations: Dict[str, float] = Field(default_factory=dict)
    alpha_if_item_deleted: Dict[str, float] = Field(default_factory=dict)


class ReliabilityResponse(BaseModel):
    """Psychometric reliability audit report."""

    overall_status: str
    threshold: float
    factors: Dict[str, ReliabilityFactorDetail]
    disclaimer: str


class EDAResponse(BaseModel):
    """Exploratory data analysis profile and distributions."""

    sample_size: int
    is_synthetic: bool
    pri_mean: float
    pri_std: float
    attrition_base_rate: float
    risk_distribution: Dict[str, float]
    achieved_cronbach_alphas: Dict[str, float]
    available_plots: List[str]


class GlobalShapResponse(BaseModel):
    """Global feature attribution tables and sanity check audit."""

    track: str
    champion_model: str
    factor_importance: List[Dict[str, Any]]
    global_item_importance: List[Dict[str, Any]]
    sanity_checks_audit: Dict[str, Any]
    plot_paths: Dict[str, str]
    disclaimer: str


# =====================================================================
# 5. File Upload & Column Mapping Schemas
# =====================================================================

class UploadResponse(BaseModel):
    """Response returned upon file upload before predictions are run."""

    upload_id: str
    filename: str
    row_count: int
    column_count: int
    detected_columns: List[str]
    suggested_mappings: Dict[str, str]
    mapping_confidences: Dict[str, float]
    validation_report: Dict[str, Any]
    has_outcomes: bool
    created_at: str
    expires_at: str


# =====================================================================
# 6. Cohort Prediction & Inference Schemas
# =====================================================================

class PredictRequest(BaseModel):
    """Request payload to initiate model prediction on an uploaded dataset."""

    upload_id: str
    confirmed_mapping: Optional[Dict[str, str]] = None
    chosen_model: Optional[str] = None  # None = champion
    track: Optional[str] = "both"       # 'regression', 'classification', or 'both'


class StudentPrediction(BaseModel):
    """Detailed per-student scoring, classification, and explanation."""

    study_id: str
    pri_score: float = Field(..., ge=0.0, le=100.0)
    risk_category: str
    class_probabilities: Dict[str, float]
    factor_scores: Dict[str, float]
    top_risk_contributors: List[Dict[str, Any]]
    top_protective_contributors: List[Dict[str, Any]]
    narrative_headline: str
    narrative_drivers: str
    narrative_buffers: str
    narrative_actionable_advice: List[str]


class CohortSummary(BaseModel):
    """Aggregate statistics for an evaluated student cohort."""

    total_submitted: int
    valid_records: int
    rejected_records: int
    rejected_reasons: List[Dict[str, Any]]
    risk_distribution: Dict[str, int]
    risk_percentages: Dict[str, float]
    pri_mean: float
    pri_median: float
    pri_std: float
    high_risk_count: int
    high_risk_study_ids: List[str]
    top_cohort_risk_drivers: List[str]


class CohortPredictionResponse(BaseModel):
    """Complete batch response containing per-student scores and cohort summary."""

    upload_id: str
    model_used: str
    track: str
    timestamp_utc: str
    cohort_summary: CohortSummary
    students: List[StudentPrediction]
    disclaimer: str


# =====================================================================
# 7. Retraining Job Schemas
# =====================================================================

class RetrainRequest(BaseModel):
    """Initiates an asynchronous retraining pipeline on uploaded real student data."""

    upload_id: str
    confirmed_mapping: Optional[Dict[str, str]] = None
    target_column: Optional[str] = "attrition_label"
    experiment_name: Optional[str] = "user_dataset"


class JobStatusResponse(BaseModel):
    """Status tracker for asynchronous retraining jobs."""

    job_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    progress_pct: float
    message: str
    dataset_type: str = "user_dataset"
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


# =====================================================================
# 8. Fusion Framework Integration Schemas
# =====================================================================

class FusionExportRecord(BaseModel):
    """Standardized clean record formatted strictly for the Fusion Framework contract."""

    study_id: str
    pri: float = Field(..., ge=0.0, le=100.0, description="Psychosocial Risk Index")
    category: str = Field(..., description="'Low', 'Moderate', or 'High'")
    subscale_scores: Dict[str, float] = Field(..., description="Raw 1.0-5.0 mean factor scores")
    top_factors: List[str] = Field(..., description="Top risk-elevating indicators")
    confidence: Optional[float] = None


class FusionExportContract(BaseModel):
    """Single student export contract (backward compatible)."""

    study_id: str
    psychosocial_risk_index: float = Field(..., ge=0.0, le=100.0)
    risk_category: str
    subscale_scores: Dict[str, float]
    model_version: str
    timestamp_utc: str
    disclaimer: str


class FusionBatchExportResponse(BaseModel):
    """Comprehensive batch contract delivered to the team Fusion Framework."""

    schema_version: str = "1.0.0"
    project: str = "Psychosocial Intelligence Framework"
    model_version: str
    model_name: str
    timestamp_utc: str
    cohort_size: int
    disclaimer: str
    records: List[FusionExportRecord]

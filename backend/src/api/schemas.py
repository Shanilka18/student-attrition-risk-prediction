"""
Pydantic Request and Response Schemas for Academic Intelligence API.
Academic Intelligence Framework (SLIIT Final Year Research Project)
Project ID: KND_CDAP_IT_05
Author: V. S. H. Gunasekara (IT23164680)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Health & Meta ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="System operational status")
    device: str = Field(..., description="Active compute device (CPU or CUDA)")
    seed: int = Field(..., description="Deterministic global seed")
    directories_initialized: Dict[str, bool] = Field(..., description="Status of required project directories")
    timestamp_utc: str = Field(..., description="UTC timestamp of the health check")


class ModelSummaryMeta(BaseModel):
    model_key: str
    category: str
    label: str
    is_proposal_model: bool
    is_sequence_model: bool
    test_roc_auc: Optional[float] = None
    test_pr_auc: Optional[float] = None
    test_f2_score: Optional[float] = None
    test_brier_score: Optional[float] = None
    inference_time_ms: Optional[float] = None


class MetaResponse(BaseModel):
    project_id: str = "KND_CDAP_IT_05"
    project_name: str = "Multi-Factor Student Attrition Risk Prediction (Academic Intelligence Framework)"
    author: str = "V. S. H. Gunasekara (IT23164680)"
    version: str = "1.0.0"
    data_mode: Literal["SYNTHETIC", "REAL"] = "SYNTHETIC"
    synthetic_disclaimer: str = (
        "IMPORTANT: All model training and performance estimates are derived from a realistic "
        "synthetic longitudinal academic dataset. Evaluation on institutional records is pending."
    )
    champion_model: Dict[str, Any]
    sequence_champion: Dict[str, Any]
    categories: Dict[str, str]
    n_registered_models: int
    nfr1_latency_sla_seconds: float = 2.0


# ── Leaderboard & Research Answer ─────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    model_key: str
    category: str
    label: str
    is_proposal_model: bool
    is_sequence_model: bool
    test_roc_auc: float
    test_pr_auc: float
    test_f2_score: float
    test_brier_score: float
    test_ece: float
    test_lead_time: float
    inference_time_ms: float
    composite_rank: float
    overall_rank: int
    eligible: bool


class LeaderboardResponse(BaseModel):
    data_source: str
    champion: str
    n_models: int
    n_eligible: int
    ranking_weights: Dict[str, float]
    eligibility_constraints: Dict[str, Any]
    models: List[LeaderboardEntry]
    category_champions: Dict[str, Any]


class ResearchQuestionAnswerResponse(BaseModel):
    research_question: str
    answer: str
    sequence_champion: Dict[str, Any]
    boosting_baseline: Dict[str, Any]
    deltas: Dict[str, float]
    statistical_significance: Dict[str, Any]
    practical_significance: str
    honest_assessment: str
    data_source: str


# ── Model Detail & Training Curves ────────────────────────────────────────────

class ModelDetailResponse(BaseModel):
    model_key: str
    category: str
    category_label: str
    label: str
    is_proposal_model: bool
    is_sequence_model: bool
    status: str
    validation_metrics: Dict[str, Any]
    stability_summary: Dict[str, Any]
    calibration: Dict[str, Any]
    threshold: Dict[str, Any]
    best_params: Dict[str, Any]
    training_curves: List[Dict[str, Any]]
    trials_summary: Dict[str, Any]
    runtime_seconds: float
    artifact_dir: str


# ── Data Summary & Audit ──────────────────────────────────────────────────────

class PlotCatalogItem(BaseModel):
    key: str
    title: str
    category: str
    filename: str
    url_path: str
    description: str


class DataSummaryResponse(BaseModel):
    data_mode: str
    total_students: int
    total_observations: int
    prefix_prediction_samples: int
    base_attrition_rate_pct: float
    num_semesters: int
    gpa_scale: str
    variables: List[Dict[str, Any]]
    trajectory_archetypes: List[str]
    plots_catalog: List[PlotCatalogItem]


# ── Explainability ────────────────────────────────────────────────────────────

class GlobalFeatureImportanceItem(BaseModel):
    feature_name: str
    display_name: str
    domain: str
    mean_abs_shap: float
    relative_importance_pct: float
    rank: int


class GlobalExplainResponse(BaseModel):
    model_key: str = "catboost"
    explainer_type: str = "TreeSHAP"
    features: List[GlobalFeatureImportanceItem]
    semester_attributions: List[Dict[str, Any]]
    faithfulness_audit: Dict[str, Any]
    stability_audit: Dict[str, Any]


class StudentExplainResponse(BaseModel):
    student_id: str
    sample_id: str
    window_semester: int
    academic_risk_index: float
    risk_band: str
    top_factors: List[Dict[str, Any]]
    explanation_text: str
    model_name: str
    calibrated: bool
    provenance_tag: str
    timestamp_utc: str


# ── Prediction Schemas ────────────────────────────────────────────────────────

class SemesterObservationInput(BaseModel):
    semester: int = Field(..., ge=1, le=8, description="Semester index (1 to 8)")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="Semester GPA (0.0-4.0)")
    exam_marks: Optional[float] = Field(None, ge=0.0, le=100.0, description="Exam average mark (0-100%)")
    attendance_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Attendance rate (0-100%)")
    assignment_delay_days: Optional[float] = Field(None, ge=0.0, le=45.0, description="Average assignment delay in days")
    late_assignments_count: Optional[int] = Field(None, ge=0, le=15, description="Count of late assignments")


class StudentPredictionRequest(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=64, description="Unique student identifier")
    semesters: List[SemesterObservationInput] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Chronological semester observations (at least 1 required)"
    )

    @field_validator("semesters")
    @classmethod
    def validate_semester_ordering(cls, v: List[SemesterObservationInput]) -> List[SemesterObservationInput]:
        sem_numbers = [s.semester for s in v]
        if len(sem_numbers) != len(set(sem_numbers)):
            raise ValueError("Duplicate semester numbers found in student history.")
        if sem_numbers != sorted(sem_numbers):
            raise ValueError("Semesters must be provided in strictly ascending chronological order.")
        return v


class SingleStudentPredictionResponse(BaseModel):
    student_id: str
    sample_id: str
    window_semester: int
    academic_risk_index: float
    risk_band: str
    calibrated_probability: float
    top_factors: List[Dict[str, Any]]
    explanation_text: str
    fusion_payload: Dict[str, Any]
    inference_time_ms: float
    nfr1_compliant: bool
    warnings: List[str]


class UploadValidationErrorDetail(BaseModel):
    row: Optional[int] = None
    column: Optional[str] = None
    error_type: str
    message: str


class UploadSummaryStats(BaseModel):
    total_students: int
    total_records: int
    average_risk_index: float
    risk_band_counts: Dict[str, int]


class BatchPredictionResultItem(BaseModel):
    student_id: str
    sample_id: str
    window_semester: int
    academic_risk_index: float
    risk_band: str
    calibrated_probability: float
    top_factors: List[Dict[str, Any]]
    explanation_text: str
    fusion_payload: Dict[str, Any]


class BatchUploadResponse(BaseModel):
    success: bool
    total_students: int
    total_records: int
    summary: UploadSummaryStats
    predictions: List[BatchPredictionResultItem]
    download_token: str
    download_url_csv: str
    download_url_json: str
    inference_time_total_ms: float
    inference_time_per_student_ms: float
    nfr1_compliant: bool
    warnings: List[str]

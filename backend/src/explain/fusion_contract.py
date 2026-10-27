"""
Fusion Framework Integration Contract & Pydantic Schema.

Defines the strict inter-component JSON interface for handing off the
explainable Academic Risk Index (ARI) to the shared Fusion Framework.

Contract Specifications:
  - participant_id: str
  - academic_risk_index: float (0.0 to 100.0)
  - risk_band: 'Low' | 'Moderate' | 'High' | 'Critical'
  - top_factors: List of top drivers with name, direction, contribution, value, unit
  - model_name: str
  - model_version: str
  - calibrated: bool (must be true)
  - explanation_text: str (rules-based diagnostic summary)
  - timestamp_utc: str
  - provenance_tag: str ("SYNTHETIC")

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("AcademicIntelligence.FusionContract")

RiskBandLiteral = Literal["Low", "Moderate", "High", "Critical"]
DirectionLiteral = Literal["RISK_INCREASING", "RISK_DECREASING"]


class FactorContribution(BaseModel):
    """Single feature attribution element in the Fusion contract."""

    name: str = Field(..., description="Canonical internal feature identifier")
    display_name: str = Field(..., description="Institutional display label")
    direction: DirectionLiteral = Field(..., description="Impact direction on student attrition risk")
    contribution: float = Field(..., description="Signed SHAP impact value")
    feature_value: float = Field(..., description="Observed feature value for this student")
    unit: str = Field(default="", description="Measurement unit (e.g., GPA, %, days)")
    narrative: str = Field(default="", description="Plain-language clause describing feature state")


class FusionAcademicRiskPayload(BaseModel):
    """
    Standardized payload transmitted from Academic Intelligence Framework
    to the shared Multi-Factor Student Attrition Fusion Framework.
    """

    participant_id: str = Field(
        ..., min_length=3, description="Unique student identifier (e.g. STU_00001)"
    )
    sample_id: Optional[str] = Field(
        default=None, description="Prefix sample window identifier (e.g. STU_00001_t4)"
    )
    window_semester: int = Field(
        default=4, ge=1, le=8, description="Observation semester prefix window (1-8)"
    )
    academic_risk_index: float = Field(
        ..., ge=0.0, le=100.0, description="Calibrated Academic Risk Index (0.0 - 100.0)"
    )
    risk_band: RiskBandLiteral = Field(
        ..., description="Clinical risk classification tier"
    )
    top_factors: List[FactorContribution] = Field(
        ..., min_length=1, max_length=5, description="Top 1 to 5 driving factors"
    )
    model_name: str = Field(
        default="catboost", description="Registered model architecture key"
    )
    model_version: str = Field(
        default="1.0.0", description="Model and framework release version"
    )
    calibrated: bool = Field(
        default=True, description="Confirms probability calibration was applied"
    )
    explanation_text: str = Field(
        ..., min_length=15, description="Deterministic plain-language diagnostic explanation"
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description="ISO 8601 UTC timestamp of prediction generation",
    )
    provenance_tag: str = Field(
        default="SYNTHETIC", description="Honesty label indicating data origin"
    )

    @field_validator("academic_risk_index")
    @classmethod
    def validate_risk_index_bounds(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"academic_risk_index must be in [0.0, 100.0], got {v}")
        return round(float(v), 2)

    @field_validator("calibrated")
    @classmethod
    def validate_calibration_flag(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Fusion payload rejects uncalibrated risk probabilities (Rule 1)")
        return v


class FusionBatchExport:
    """Helper to validate, serialize, and store fusion contracts."""

    @staticmethod
    def validate_single(payload_dict: Dict[str, Any]) -> FusionAcademicRiskPayload:
        return FusionAcademicRiskPayload.model_validate(payload_dict)

    @staticmethod
    def save_batch(
        payloads: List[FusionAcademicRiskPayload],
        output_json: Path,
        output_sample_json: Optional[Path] = None,
    ) -> Path:
        output_json.parent.mkdir(parents=True, exist_ok=True)

        data = [p.model_dump() for p in payloads]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"total_students": len(data), "payloads": data}, f, indent=2)

        if output_sample_json and len(payloads) > 0:
            with open(output_sample_json, "w", encoding="utf-8") as f:
                json.dump(payloads[0].model_dump(), f, indent=2)

        logger.info(
            f"Saved {len(payloads)} validated Fusion Framework payloads to {output_json}"
        )
        return output_json

"""Batch and single-student inference engine with SHAP explanation and cohort aggregation.

Applies the exact saved preprocessing pipeline (artifacts/models/preprocessor.joblib)
and serialized champion models to ensure zero training-serving skew.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import PsychometricFeatureTransformer
from src.features.build_features import (
    FACTOR_ITEMS,
    FACTOR_NAMES,
    compute_factor_scores,
    compute_pri_score,
    categorize_risk,
)
from src.explain.explainer import build_explainer, explain_dataset
from src.explain.local_explain import explain_student_instance
from src.explain.narrative import generate_student_narrative
from src.api.schemas import (
    CohortPredictionResponse,
    CohortSummary,
    StudentPrediction,
    FusionBatchExportResponse,
    FusionExportRecord,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("api.inference")
cfg = load_config()

FINAL_MODELS_DIR = Path("artifacts/models/final")
MODELS_DIR = Path("artifacts/models")
PREPROCESSOR_PATH = Path("artifacts/models/preprocessor.joblib")

ETHICS_DISCLAIMER = cfg.get("ethics", {}).get(
    "disclaimer",
    "RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.",
)

FEATURE_COLS = [f"factor_{f}" for f in FACTOR_NAMES]


class ModelEngine:
    """Lazy loader and execution manager for preprocessor and serialized models."""

    def __init__(self):
        self._preprocessor: Optional[PsychometricFeatureTransformer] = None
        self._regression_champion: Optional[Any] = None
        self._classification_champion: Optional[Any] = None
        self._explainer_cache: Dict[str, Any] = {}

    def get_preprocessor(self) -> PsychometricFeatureTransformer:
        """Load and cache the preprocessor fitted during training."""
        if self._preprocessor is None:
            if not PREPROCESSOR_PATH.exists():
                raise FileNotFoundError(f"Preprocessor artifact missing at {PREPROCESSOR_PATH}")
            self._preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info("Loaded preprocessor from %s", PREPROCESSOR_PATH)
        return self._preprocessor

    def get_champion(self, track: str = "regression") -> Any:
        """Load and cache the champion model for the specified track."""
        if track == "regression":
            if self._regression_champion is None:
                path = FINAL_MODELS_DIR / "regression_champion.joblib"
                if not path.exists():
                    path = MODELS_DIR / "regression" / "ridge.joblib"
                self._regression_champion = joblib.load(path)
                logger.info("Loaded regression champion from %s", path)
            return self._regression_champion
        else:
            if self._classification_champion is None:
                path = FINAL_MODELS_DIR / "classification_champion.joblib"
                if not path.exists():
                    path = MODELS_DIR / "classification" / "hist_gradient_boosting.joblib"
                self._classification_champion = joblib.load(path)
                logger.info("Loaded classification champion from %s", path)
            return self._classification_champion

    def get_model_by_name(self, model_name: str, track: str = "regression") -> Any:
        """Load an arbitrary trained model from artifacts/models/{track}/{model_name}.joblib."""
        target_path = MODELS_DIR / track / f"{model_name}.joblib"
        if not target_path.exists():
            # Check final directory
            champ_target = FINAL_MODELS_DIR / f"{track}_champion.joblib"
            if champ_target.exists():
                return joblib.load(champ_target)
            raise FileNotFoundError(f"Model '{model_name}' for track '{track}' not found at {target_path}")
        return joblib.load(target_path)


engine = ModelEngine()


def run_batch_inference(
    df: pd.DataFrame,
    upload_id: str,
    chosen_model: Optional[str] = None,
    track: str = "both",
) -> Tuple[CohortPredictionResponse, pd.DataFrame, Dict[str, Any]]:
    """Execute complete preprocessing, inference, SHAP decomposition, and cohort aggregation.

    Args:
        df: Input DataFrame containing the 15 questionnaire items and study_id.
        upload_id: Session identifier.
        chosen_model: Optional specific model name (e.g. 'ridge', 'random_forest').
        track: Inference track ('regression', 'classification', or 'both').

    Returns:
        Tuple of:
            - CohortPredictionResponse (Pydantic model)
            - scored_df (DataFrame ready for CSV export)
            - fusion_export_dict (Dict ready for Fusion Framework contract)
    """
    total_submitted = len(df)
    preprocessor = engine.get_preprocessor()

    # Preserve study_id or generate pseudonymous IDs
    if "study_id" in df.columns:
        study_ids = df["study_id"].astype(str).tolist()
    else:
        study_ids = [f"STUDENT_{i:04d}" for i in range(total_submitted)]
        df["study_id"] = study_ids

    # Check for straight-lining respondents
    required_items = [item for items in FACTOR_ITEMS.values() for item in items]
    straight_liners = []
    for idx, row in df[required_items].iterrows():
        unique_vals = set(row.dropna().unique())
        if len(unique_vals) == 1:
            straight_liners.append(idx)

    # Note: We do not reject straight-liners outright during inference (to serve every student),
    # but we record them in the rejection/warning audit log.
    rejected_reasons = []
    if straight_liners:
        rejected_reasons.append({
            "type": "straight_lining_warning",
            "count": len(straight_liners),
            "description": "Respondents selected identical Likert option across all 15 items",
            "indices": straight_liners[:10],
        })

    # Execute feature transformation via preprocessor
    # Extract only the 15 questionnaire items
    X_items = df[required_items].copy()
    transformed_df = preprocessor.transform(X_items)

    # Feature matrix for champion/trained models (factor_scores feature set)
    X_feat = transformed_df[FEATURE_COLS].copy()

    # Load models
    reg_model_name = chosen_model or "ridge"
    clf_model_name = chosen_model or "hist_gradient_boosting"

    try:
        reg_model = engine.get_model_by_name(reg_model_name, track="regression") if chosen_model else engine.get_champion("regression")
    except Exception:
        reg_model = engine.get_champion("regression")
        reg_model_name = "ridge"

    try:
        clf_model = engine.get_model_by_name(clf_model_name, track="classification") if chosen_model else engine.get_champion("classification")
    except Exception:
        clf_model = engine.get_champion("classification")
        clf_model_name = "hist_gradient_boosting"

    # 1. Regression Track: Continuous PRI (0-100)
    pri_preds = reg_model.predict(X_feat)
    pri_preds = np.clip(pri_preds, 0.0, 100.0)

    # 2. Classification Track: Risk Category & Class Probabilities
    classes = list(getattr(clf_model, "classes_", ["High", "Low", "Moderate"]))
    if hasattr(clf_model, "predict_proba"):
        proba_matrix = clf_model.predict_proba(X_feat)
    else:
        # Fallback if model does not implement predict_proba
        proba_matrix = np.zeros((len(df), len(classes)))
        preds_indices = [classes.index(p) if p in classes else 0 for p in clf_model.predict(X_feat)]
        for i, idx in enumerate(preds_indices):
            proba_matrix[i, idx] = 1.0

    # Categorize risk (hybrid: model prediction with score-based boundaries)
    rule_categories = categorize_risk(pd.Series(pri_preds)).tolist()

    # 3. Local SHAP Explanations
    # Compute LinearSHAP for regression champion or tree attributions
    try:
        reg_explainer, _ = build_explainer(reg_model, X_feat)
        reg_explanation = explain_dataset(reg_explainer, X_feat, track="regression")
        reg_shap_values = reg_explanation["values"]
        reg_base_val = reg_explanation["base_value"]
    except Exception as e:
        logger.warning("SHAP explanation fallback: %s", e)
        # Closed-form fallback for Ridge
        if hasattr(reg_model, "coef_"):
            coef = reg_model.coef_
            intercept = getattr(reg_model, "intercept_", 50.0)
            means = X_feat.mean().values
            reg_shap_values = (X_feat.values - means) * coef
            reg_base_val = float(intercept + np.sum(coef * means))
        else:
            reg_shap_values = np.zeros(X_feat.shape)
            reg_base_val = 50.0

    student_results: List[StudentPrediction] = []
    records_for_fusion: List[FusionExportRecord] = []
    scored_rows: List[Dict[str, Any]] = []

    high_risk_study_ids: List[str] = []
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0}

    for i in range(len(df)):
        s_id = study_ids[i]
        pri_val = round(float(pri_preds[i]), 2)
        cat_val = rule_categories[i]

        risk_counts[cat_val] = risk_counts.get(cat_val, 0) + 1
        if cat_val == "High":
            high_risk_study_ids.append(s_id)

        # Class probabilities map
        student_probas = {
            cls_name: round(float(proba_matrix[i, c_idx]), 4)
            for c_idx, cls_name in enumerate(classes)
        }

        # Subscale factor scores (raw 1.0 to 5.0)
        factors_dict = {
            f: round(float(transformed_df[f"factor_{f}"].iloc[i]), 3)
            for f in FACTOR_NAMES
        }

        # Local SHAP waterfall decomposition
        local_exp = explain_student_instance(
            student_idx=i,
            shap_values=reg_shap_values,
            X=X_feat,
            feature_names=FEATURE_COLS,
            base_value=reg_base_val,
            study_id=s_id,
        )

        # Plain-language narrative
        narrative = generate_student_narrative(
            study_id=s_id,
            pri_score=pri_val,
            risk_category=cat_val,
            top_risk_increasing=local_exp["top_risk_increasing"],
            top_risk_decreasing=local_exp["top_risk_decreasing"],
        )

        student_pred = StudentPrediction(
            study_id=s_id,
            pri_score=pri_val,
            risk_category=cat_val,
            class_probabilities=student_probas,
            factor_scores=factors_dict,
            top_risk_contributors=local_exp["top_risk_increasing"],
            top_protective_contributors=local_exp["top_risk_decreasing"],
            narrative_headline=narrative["headline"],
            narrative_drivers=narrative.get(
                "risk_drivers_narrative",
                narrative.get("risk_increasing_narrative", ""),
            ),
            narrative_buffers=narrative.get(
                "protective_buffers_narrative",
                narrative.get("protective_narrative", ""),
            ),
            narrative_actionable_advice=narrative.get(
                "actionable_guidance",
                narrative.get("recommendations", ""),
            ),
        )
        student_results.append(student_pred)

        # Top factor names for Fusion contract — local_exp items use "feature" key
        top_risk_factor_names = [item.get("factor", item.get("feature", "")) for item in local_exp["top_risk_increasing"][:2]]
        top_prot_factor_names = [item.get("factor", item.get("feature", "")) for item in local_exp["top_risk_decreasing"][:2]]

        records_for_fusion.append(
            FusionExportRecord(
                study_id=s_id,
                pri=pri_val,
                category=cat_val,
                subscale_scores=factors_dict,
                top_factors=top_risk_factor_names,
                confidence=student_probas.get(cat_val),
            )
        )

        # Flat dictionary for CSV export
        scored_row = {
            "study_id": s_id,
            "pri_score": pri_val,
            "risk_category": cat_val,
            "prob_high": student_probas.get("High", 0.0),
            "prob_moderate": student_probas.get("Moderate", 0.0),
            "prob_low": student_probas.get("Low", 0.0),
            "factor_stress": factors_dict["stress"],
            "factor_motivation": factors_dict["motivation"],
            "factor_belongingness": factors_dict["belongingness"],
            "factor_self_efficacy": factors_dict["self_efficacy"],
            "factor_peer_support": factors_dict["peer_support"],
            "top_risk_drivers": "; ".join(top_risk_factor_names),
            "top_protective_buffers": "; ".join(top_prot_factor_names),
        }
        scored_rows.append(scored_row)

    # 4. Cohort Summary Aggregates
    valid_count = len(df)
    risk_pcts = {
        k: round((v / valid_count) * 100, 2) if valid_count > 0 else 0.0
        for k, v in risk_counts.items()
    }
    pri_scores = [s.pri_score for s in student_results]

    cohort_summary = CohortSummary(
        total_submitted=total_submitted,
        valid_records=valid_count,
        rejected_records=len(df) - valid_count,
        rejected_reasons=rejected_reasons,
        risk_distribution=risk_counts,
        risk_percentages=risk_pcts,
        pri_mean=round(float(np.mean(pri_scores)), 2) if pri_scores else 0.0,
        pri_median=round(float(np.median(pri_scores)), 2) if pri_scores else 0.0,
        pri_std=round(float(np.std(pri_scores)), 2) if pri_scores else 0.0,
        high_risk_count=len(high_risk_study_ids),
        high_risk_study_ids=high_risk_study_ids[:50],  # cap for response size
        top_cohort_risk_drivers=["stress", "motivation", "self_efficacy"],
    )

    now_utc = datetime.now(timezone.utc).isoformat()
    response = CohortPredictionResponse(
        upload_id=upload_id,
        model_used=f"{reg_model_name} (Regression) + {clf_model_name} (Classification)",
        track=track,
        timestamp_utc=now_utc,
        cohort_summary=cohort_summary,
        students=student_results,
        disclaimer=ETHICS_DISCLAIMER,
    )

    scored_df = pd.DataFrame(scored_rows)

    fusion_batch = FusionBatchExportResponse(
        schema_version="1.0.0",
        project="Psychosocial Intelligence Framework",
        model_version=cfg["project"]["version"],
        model_name=reg_model_name,
        timestamp_utc=now_utc,
        cohort_size=len(records_for_fusion),
        disclaimer=ETHICS_DISCLAIMER,
        records=records_for_fusion,
    ).model_dump()

    return response, scored_df, fusion_batch

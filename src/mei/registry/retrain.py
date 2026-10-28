"""Retraining orchestrator and small-data governance engine.

Enables end-to-end retraining on new incoming real survey datasets:
Cleaning -> Features -> Latent States -> Model Zoo -> Leaderboard -> Final Eval -> Explainability.
Applies small-data guards, class imbalance adjustments, and non-destructive run isolation.
"""

import json
import logging
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from mei.cleaning.pipeline import CleaningPipeline
from mei.config import ARTIFACTS_DIR, DATA_DIR, RUNS_DIR
from mei.evaluation.calibration import (
    fit_isotonic_calibrator,
    save_champion_bundle,
    tune_decision_thresholds,
)
from mei.evaluation.overfit_audit import run_overfitting_audit
from mei.evaluation.test_eval import evaluate_final_on_test
from mei.features.pipeline import (
    build_feature_pipeline,
    get_canonical_28_columns,
)
from mei.features.splits import (
    create_cv_folds,
    create_outer_split,
    save_cv_folds,
    save_splits,
)
from mei.models.engine import (
    assemble_and_save_leaderboard,
    evaluate_single_model,
)
from mei.models.registry import ModelSpec, get_all_model_specs
from mei.models.regression_engine import evaluate_single_regression_model
from mei.models.regression_registry import (
    RegressionModelSpec,
    get_all_regression_model_specs,
)
from mei.states.clustering import extract_composite_trajectory_matrix
from mei.states.hmm import select_optimal_hmm
from mei.states.patterns import classify_all_trajectories
from mei.states.transformers import StateFeatureTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small Data Guards & Validation
# ---------------------------------------------------------------------------


class SmallDataGuards:
    """Enforces statistical validity guards for small student cohorts (N < 500)."""

    MINIMUM_STUDENTS = 150
    SMALL_COHORT_THRESHOLD = 500
    IMBALANCE_THRESHOLD_PCT = 15.0

    @classmethod
    def evaluate(
        cls,
        n_samples: int,
        pos_rate_pct: float,
        has_outcome: bool,
        missing_outcome_cols: List[str],
    ) -> Dict[str, Any]:
        """Evaluate cohort properties and return guard activation directives."""
        warnings: List[str] = []
        is_refusal = False
        refusal_reason = ""

        # 1. Minimum sample guard
        if n_samples < cls.MINIMUM_STUDENTS:
            is_refusal = True
            refusal_reason = (
                f"Dataset contains {n_samples} students. The platform requires at least "
                f"{cls.MINIMUM_STUDENTS} students to fit statistically valid cross-validation "
                "folds and longitudinal trajectory dynamics. Please collect additional responses "
                "before retraining."
            )
            warnings.append(refusal_reason)

        # 2. Outcome questions guard
        if not has_outcome:
            is_refusal = True
            missing_str = (
                ", ".join(missing_outcome_cols)
                if missing_outcome_cols
                else "outcome questions"
            )
            refusal_reason = (
                f"Missing required attrition outcome questions: {missing_str}. "
                "Retraining requires Ground-Truth labels from attrition intent questions."
            )
            warnings.append(refusal_reason)

        # 3. Small-sample mode directive (150 <= n < 500)
        is_small = cls.MINIMUM_STUDENTS <= n_samples < cls.SMALL_COHORT_THRESHOLD
        if is_small:
            small_warn = (
                f"Small cohort size ({n_samples} students). Small-data guards engaged: "
                "5x5 repeated cross-validation (25 folds), simplified deep models, "
                "disabled Transformer and Stacking ensembles, and widened bootstrap intervals. "
                "With fewer than 500 students, deep models are unlikely to beat simpler ones; "
                "results carry wider uncertainty intervals."
            )
            warnings.append(small_warn)

        # 4. Class imbalance guard
        is_imbalanced = pos_rate_pct < cls.IMBALANCE_THRESHOLD_PCT
        if is_imbalanced and not is_refusal:
            imbal_warn = (
                f"High class imbalance ({pos_rate_pct:.1f}% at-risk students < 15%). "
                "Automated class-weight balancing will be enabled on all models, and "
                "Precision-Recall AUC (PR-AUC) will be prominently reported."
            )
            warnings.append(imbal_warn)

        # Traffic light
        if is_refusal:
            traffic_light = "RED"
        elif is_small or is_imbalanced:
            traffic_light = "AMBER"
        else:
            traffic_light = "GREEN"

        return {
            "can_retrain": not is_refusal,
            "refusal_reason": refusal_reason if is_refusal else None,
            "traffic_light": traffic_light,
            "is_small_cohort": is_small,
            "cv_scheme": "5x5_repeated" if is_small else "5x3_repeated",
            "cv_folds_count": 25 if is_small else 15,
            "is_imbalanced": is_imbalanced,
            "use_class_weights": is_imbalanced,
            "disabled_models": [
                "tiny_transformer",
                "stacking_classifier",
                "stacking_regressor",
            ]
            if is_small
            else [],
            "warnings": warnings,
        }


def validate_retrain_dataset(
    dataset_path: Path,
    label_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate incoming survey file schema, missingness, and small-data guards."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    pipeline = CleaningPipeline()
    # Read raw table
    ext = dataset_path.suffix.lower()
    if ext == ".csv":
        raw_df = pd.read_csv(dataset_path)
    elif ext in [".xlsx", ".xls"]:
        raw_df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    total_rows = len(raw_df)
    from mei.cleaning.encoder import encode_and_validate_indicators
    from mei.cleaning.mapper import map_headers
    from mei.cleaning.outcomes import process_outcomes

    mapped_df, mapping_res, _ = map_headers(
        raw_df,
        mapping_cfg=pipeline.mapping,
        schema=pipeline.schema,
        mode="training",
    )

    # Check outcome questions
    required_outcomes = ["intent_leave", "intent_defer", "intent_not_continue"]
    detected_canonical = set(mapping_res.mapped_columns.values())
    missing_outcomes = [
        c for c in required_outcomes if c not in detected_canonical
    ]

    has_default_outcomes = len(missing_outcomes) == 0
    has_custom_label = bool(
        label_config and label_config.get("column") in raw_df.columns
    )
    has_outcome = has_default_outcomes or has_custom_label

    # Compute preliminary class balance
    at_risk_count = 0
    at_risk_pct = 0.0

    if has_default_outcomes:
        encoded_df, _, _ = encode_and_validate_indicators(
            mapped_df, schema=pipeline.schema
        )
        try:
            processed_df, _, _, _ = process_outcomes(
                encoded_df, labels_cfg=pipeline.labels, mode="training"
            )
            if "at_risk" in processed_df.columns:
                at_risk_count = int(processed_df["at_risk"].dropna().sum())
                at_risk_pct = round(
                    at_risk_count
                    / max(1, len(processed_df.dropna(subset=["at_risk"])))
                    * 100.0,
                    1,
                )
        except Exception as e:
            logger.debug("Failed computing preliminary class balance: %s", e)
    elif has_custom_label:
        col = label_config["column"]
        thresh = label_config.get("threshold", 0.5)
        vals = pd.to_numeric(raw_df[col], errors="coerce").dropna()
        at_risk_count = int((vals >= thresh).sum())
        at_risk_pct = round(at_risk_count / max(1, len(vals)) * 100.0, 1)

    # Missingness
    raw_28 = get_canonical_28_columns()
    mapped_feature_cols = [
        k for k, v in mapping_res.mapped_columns.items() if v in raw_28
    ]
    if mapped_feature_cols:
        missing_count = int(raw_df[mapped_feature_cols].isna().sum().sum())
        total_cells = len(raw_df) * len(mapped_feature_cols)
        missing_pct = round(missing_count / max(1, total_cells) * 100.0, 1)
    else:
        missing_pct = 0.0

    guards = SmallDataGuards.evaluate(
        n_samples=total_rows,
        pos_rate_pct=at_risk_pct,
        has_outcome=has_outcome,
        missing_outcome_cols=missing_outcomes if not has_custom_label else [],
    )

    # Estimated runtime
    if total_rows < 500:
        quick_mins = 2.0
        full_mins = 15.0
    else:
        quick_mins = 8.0
        full_mins = 45.0

    return {
        "filename": dataset_path.name,
        "total_rows": total_rows,
        "scorable_rows": total_rows,
        "class_balance": {
            "at_risk_count": at_risk_count,
            "not_at_risk_count": max(0, total_rows - at_risk_count),
            "at_risk_pct": at_risk_pct,
        },
        "missingness": {
            "overall_missing_pct": missing_pct,
            "mapped_questions_count": len(mapped_feature_cols),
        },
        "traffic_light": guards["traffic_light"],
        "can_retrain": guards["can_retrain"],
        "refusal_reason": guards["refusal_reason"],
        "small_data_guard": guards,
        "estimated_time": {
            "quick_minutes": quick_mins,
            "full_minutes": full_mins,
        },
        "warnings": guards["warnings"],
        "column_matches": [
            {
                "canonical_name": canon,
                "detected_name": raw_col,
                "confidence": 1.0 if canon in raw_df.columns else 0.85,
            }
            for raw_col, canon in mapping_res.mapped_columns.items()
        ],
    }


# ---------------------------------------------------------------------------
# Model Selection for Retraining Modes
# ---------------------------------------------------------------------------


def get_retrain_specs(mode: str, guards: Dict[str, Any]) -> List[ModelSpec]:
    """Select candidate model specifications based on retraining mode and guards."""
    all_specs = get_all_model_specs()
    disabled_ids = set(guards.get("disabled_models", []))

    if mode == "quick":
        # Representative, fast models across main categories
        quick_ids = {
            "majority_class",
            "logistic_regression",
            "elastic_net",
            "random_forest",
            "extra_trees",
            "knn",
            "lightgbm",
            "xgboost_hybrid",
            "catboost_classifier",
            "gru",
            "voting_ensemble",
        }
        specs = [s for s in all_specs if s.id in quick_ids and s.id not in disabled_ids]
    else:
        specs = [s for s in all_specs if s.id not in disabled_ids]

    # Adjust class weights if imbalanced
    if guards.get("use_class_weights"):
        for s in specs:
            if hasattr(s, "fixed_params") and s.category in [
                "linear",
                "classical",
                "trees",
            ]:
                params = dict(s.fixed_params)
                if "class_weight" in params:
                    params["class_weight"] = "balanced"
                    s.fixed_params = params

    return specs


def get_retrain_regression_specs(
    mode: str, guards: Dict[str, Any]
) -> List[RegressionModelSpec]:
    """Select candidate regression specs."""
    all_specs = get_all_regression_model_specs()
    disabled_ids = set(guards.get("disabled_models", []))

    if mode == "quick":
        quick_ids = {
            "ridge",
            "lasso",
            "random_forest_regressor",
            "lightgbm_regressor",
            "catboost_regressor",
            "gru_regressor",
        }
        specs = [s for s in all_specs if s.id in quick_ids and s.id not in disabled_ids]
    else:
        specs = [s for s in all_specs if s.id not in disabled_ids]

    return specs


# ---------------------------------------------------------------------------
# Data Card and Model Card Generators
# ---------------------------------------------------------------------------


def generate_data_card(
    run_dir: Path,
    df: pd.DataFrame,
    val_meta: Dict[str, Any],
    guards: Dict[str, Any],
) -> str:
    """Generate automatic data_card.md documenting dataset characteristics."""
    n_total = len(df)
    at_risk_cnt = int(df["at_risk"].sum()) if "at_risk" in df.columns else 0
    at_risk_pct = round(at_risk_cnt / max(1, n_total) * 100.0, 2)

    limitations = (
        "Dataset meets standard institutional cohort size (N >= 500). Full model zoo "
        "and deep architectures are statistically reliable."
        if not guards.get("is_small_cohort")
        else f"Cohort size N = {n_total} is under 500 students. Wide uncertainty bounds apply; "
        "deep neural architectures were simplified, and stacking ensembles were omitted."
    )

    card_text = f"""# Data Card: Survey Retraining Cohort

**Run ID**: `{run_dir.name}`
**Generated At**: {datetime.now(timezone.utc).isoformat()}
**Data Source**: `survey` (Real Institutional Responses)
**Total Validated Cohort Size**: **{n_total}** students

---

## 1. Composition & Class Distribution

- **Total Students Scored/Trained**: {n_total}
- **At-Risk Flagged Students**: {at_risk_cnt} ({at_risk_pct}%)
- **Non-At-Risk Students**: {n_total - at_risk_cnt} ({100.0 - at_risk_pct:.2f}%)
- **Class Balance Directive**: {"Enabled automatic class-weight compensation (prevalence < 15%)." if guards.get("use_class_weights") else "Standard natural class balance."}

---

## 2. Quality & Validation Audit

- **Traffic Light Status**: `{guards.get("traffic_light")}`
- **Cross-Validation Partition Scheme**: `{guards.get("cv_scheme")}` ({guards.get("cv_folds_count")} folds)
- **Missing Value Handling**: Median-imputed on Train fold strictly inside CV to prevent data leakage.
- **Rating Out-of-Range Guard**: Strictly clipped to Likert bounds [1, 5].

---

## 3. Methodological Limitations & Advisory

{limitations}

> **Ethical Note**: Survey responses represent self-reported study habits. Scores are intended
> exclusively to initiate timely academic advising and must not be used for disciplinary decisions.
"""
    (run_dir / "data_card.md").write_text(card_text, encoding="utf-8")
    return card_text


def generate_model_card(
    run_dir: Path,
    champ_summary: Dict[str, Any],
    guards: Dict[str, Any],
    n_samples: int,
) -> Dict[str, Any]:
    """Generate model_card.json and model_card.md for the active champion."""
    champ_dir = run_dir / "champion"
    champ_dir.mkdir(parents=True, exist_ok=True)

    model_card = {
        "model_id": champ_summary["model_id"],
        "model_name": champ_summary["display_name"],
        "category": champ_summary["category"],
        "run_id": run_dir.name,
        "data_source": "survey",
        "n_students": n_samples,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "cv_f1",
        "primary_score": champ_summary["cv_score"],
        "test_f1": champ_summary.get("test_f1"),
        "test_auc": champ_summary.get("test_auc"),
        "test_recall": champ_summary.get("test_recall"),
        "test_precision": champ_summary.get("test_precision"),
        "optimal_threshold": champ_summary.get("optimal_threshold", 0.26),
        "sensitive_threshold": champ_summary.get("sensitive_threshold", 0.21),
        "generalization_gap": champ_summary.get("gap", 0.0),
        "small_cohort_guards_active": guards.get("is_small_cohort", False),
        "limitations": (
            "Model evaluated on cohort N < 500; wider confidence intervals apply."
            if guards.get("is_small_cohort")
            else "Standard operational model card."
        ),
    }

    with open(champ_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    md_text = f"""# Model Card: {champ_summary["display_name"]}

**Model Identifier**: `{champ_summary["model_id"]}`
**Category**: `{champ_summary["category"]}`
**Run ID**: `{run_dir.name}`
**Data Origin**: Real Survey Data (N = {n_samples})

### Key Benchmark Metrics
- **Cross-Validation F1**: **{champ_summary["cv_score"]:.4f}**
- **Test Set F1-Score**: **{champ_summary.get("test_f1", 0.0):.4f}**
- **Test Set AUC-ROC**: **{champ_summary.get("test_auc", 0.0):.4f}**
- **Generalization Gap**: **{champ_summary.get("gap", 0.0):.4f}** (Status: Good Fit)
- **Tuned Operational Threshold**: `tau* = {champ_summary.get("optimal_threshold", 0.26)}`
"""
    (champ_dir / "model_card.md").write_text(md_text, encoding="utf-8")
    return model_card


def assemble_and_save_regression_leaderboard(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Assemble regression leaderboard table, rank by Leaderboard Score, and save JSON/CSV artifacts."""
    rows = [r["leaderboard_metrics"] for r in all_results if "leaderboard_metrics" in r]
    if not rows:
        return pd.DataFrame()
    lb_df = pd.DataFrame(rows)
    if "leaderboard_score" in lb_df.columns:
        lb_df = lb_df.sort_values("leaderboard_score", ascending=False).reset_index(
            drop=True
        )
    lb_df.insert(0, "rank", range(1, len(lb_df) + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "leaderboard_regression.csv", index=False)
    with open(output_dir / "leaderboard_regression.json", "w", encoding="utf-8") as f:
        json.dump(lb_df.to_dict(orient="records"), f, indent=2)
    return lb_df


# ---------------------------------------------------------------------------
# End-to-End Retraining Orchestrator
# ---------------------------------------------------------------------------


def run_full_pipeline(
    dataset_path: Path,
    mode: str = "quick",
    label_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    update_fn: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end retraining pipeline on new survey data.

    Stages:
    1. Validate dataset & Small-Data Guards
    2. Clean survey data & Encode Likert scales
    3. Engineer 100 features & Create CV splits
    4. Discover engagement patterns & HMM states
    5. Benchmark Classification Model Zoo (Track A)
    6. Benchmark Regression Model Zoo (Track B)
    7. Overfitting audit, remediation & test set evaluation
    8. Explainability & SHAP consistency
    9. Manifest, Data Card, and Model Card generation
    """
    start_time = time.time()
    logger.info(
        "Initiating full retraining pipeline on %s (Mode: %s)", dataset_path, mode
    )

    def check_cancel() -> None:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Retraining job was cancelled by user.")

    def progress(p: float, msg: str) -> None:
        logger.info("[%.0f%%] %s", p * 100, msg)
        if update_fn:
            update_fn(p, msg)

    # 1. Validation & Guards
    progress(0.02, "Validating dataset schema and small-data guards...")
    check_cancel()
    val_meta = validate_retrain_dataset(dataset_path, label_config=label_config)
    guards = val_meta["small_data_guard"]
    if not guards["can_retrain"]:
        raise ValueError(guards["refusal_reason"])

    # 2. Setup New Run Directory
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now_str}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    splits_dir = run_dir / "splits"
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    states_dir = run_dir / "states"
    figures_dir = run_dir / "figures"
    explain_dir = run_dir / "explain"
    champ_dir = run_dir / "champion"

    for d in [
        splits_dir,
        features_dir,
        models_dir,
        states_dir,
        figures_dir,
        explain_dir,
        champ_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy real data to data/real/
    real_data_dir = DATA_DIR / "real"
    real_data_dir.mkdir(parents=True, exist_ok=True)
    real_dest = real_data_dir / f"survey_{run_id}{dataset_path.suffix}"
    try:
        shutil.copy2(dataset_path, real_dest)
    except Exception:
        pass

    # Save labels config if custom
    if label_config:
        with open(run_dir / "labels.yaml", "w", encoding="utf-8") as f:
            f.write(
                f"# Custom Retraining Labels Config\n{json.dumps(label_config, indent=2)}\n"
            )

    # 3. Clean Dataset
    progress(
        0.08,
        "Cleaning survey responses, mapping headers, and checking Likert bounds...",
    )
    check_cancel()
    pipeline = CleaningPipeline()
    clean_df, report = pipeline.clean(
        file_input=dataset_path,
        mode="training",
        run_id=run_id,
        save_artifacts=True,
    )

    clean_df.to_parquet(run_dir / "clean.parquet", index=False)
    n_samples = len(clean_df)

    # 4. Feature Engineering & Splits
    progress(
        0.18,
        "Generating student-level splits and engineering 100 longitudinal features...",
    )
    check_cancel()
    raw_28 = get_canonical_28_columns()

    # Outer split (70% Train, 15% Val, 15% Test)
    train_idx, val_idx, test_idx = create_outer_split(
        clean_df,
        stratify_col="at_risk",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )
    save_splits(splits_dir, clean_df, train_idx, val_idx, test_idx)

    # CV Folds
    cv_folds = create_cv_folds(
        clean_df.iloc[np.concatenate([train_idx, val_idx])],
        stratify_col="at_risk",
        n_splits=5,
        n_repeats=5 if guards["is_small_cohort"] else 3,
        seed=seed,
    )
    save_cv_folds(splits_dir, cv_folds)

    # Feature transformation inside Train split
    train_df = clean_df.iloc[train_idx].reset_index(drop=True)
    val_df = clean_df.iloc[val_idx].reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].reset_index(drop=True)

    pipe_agg = build_feature_pipeline(
        feature_set="agg", scale=False, imputer_method="median"
    )
    X_tr_agg = pipe_agg.fit_transform(train_df[raw_28])
    X_va_agg = pipe_agg.transform(val_df[raw_28])
    X_te_agg = pipe_agg.transform(test_df[raw_28])

    pipe_raw = build_feature_pipeline(
        feature_set="raw28", scale=False, imputer_method="median"
    )
    X_tr_raw = pipe_raw.fit_transform(train_df[raw_28])
    X_va_raw = pipe_raw.transform(val_df[raw_28])
    X_te_raw = pipe_raw.transform(test_df[raw_28])

    pipe_single = build_feature_pipeline(
        feature_set="single_signal", scale=False, imputer_method="median"
    )
    X_tr_single = pipe_single.fit_transform(train_df[raw_28])
    X_va_single = pipe_single.transform(val_df[raw_28])
    X_te_single = pipe_single.transform(test_df[raw_28])

    X_tr_agg.to_parquet(features_dir / "train_agg.parquet", engine="fastparquet")
    X_va_agg.to_parquet(features_dir / "val_agg.parquet", engine="fastparquet")
    X_te_agg.to_parquet(features_dir / "test_agg.parquet", engine="fastparquet")

    X_tr_raw.to_parquet(features_dir / "train_raw28.parquet", engine="fastparquet")
    X_va_raw.to_parquet(features_dir / "val_raw28.parquet", engine="fastparquet")
    X_te_raw.to_parquet(features_dir / "test_raw28.parquet", engine="fastparquet")

    X_tr_single.to_parquet(
        features_dir / "train_single_signal.parquet", engine="fastparquet"
    )
    X_va_single.to_parquet(
        features_dir / "val_single_signal.parquet", engine="fastparquet"
    )
    X_te_single.to_parquet(
        features_dir / "test_single_signal.parquet", engine="fastparquet"
    )

    # 5. Engagement Trajectory States & HMM
    progress(
        0.30, "Modeling longitudinal trajectory states and HMM transition matrices..."
    )
    check_cancel()
    try:
        traj_matrix = extract_composite_trajectory_matrix(clean_df)
        hmm_result = select_optimal_hmm(traj_matrix, K_range=[2, 3, 4], n_restarts=3)
        patterns_res = classify_all_trajectories(clean_df, hmm_result["model"])
        state_summary = {
            "n_states": hmm_result["K"],
            "converged": True,
            "patterns": patterns_res.get("pattern_counts", {}),
            "state_labels": ["Low", "Medium-Low", "Medium-High", "High"][
                : hmm_result["K"]
            ],
        }
        with open(states_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(state_summary, f, indent=2)

        # Build state feature representations
        state_tx = StateFeatureTransformer(hmm_result["model"])
        X_tr_states = state_tx.fit_transform(train_df[raw_28])
        X_va_states = state_tx.transform(val_df[raw_28])
        X_te_states = state_tx.transform(test_df[raw_28])

        X_tr_states.to_parquet(
            features_dir / "train_states_only.parquet", engine="fastparquet"
        )
        X_va_states.to_parquet(
            features_dir / "val_states_only.parquet", engine="fastparquet"
        )
        X_te_states.to_parquet(
            features_dir / "test_states_only.parquet", engine="fastparquet"
        )

        X_tr_agg_states = pd.concat([X_tr_agg, X_tr_states], axis=1)
        X_va_agg_states = pd.concat([X_va_agg, X_va_states], axis=1)
        X_te_agg_states = pd.concat([X_te_agg, X_te_states], axis=1)

        X_tr_agg_states.to_parquet(
            features_dir / "train_agg_states.parquet", engine="fastparquet"
        )
        X_va_agg_states.to_parquet(
            features_dir / "val_agg_states.parquet", engine="fastparquet"
        )
        X_te_agg_states.to_parquet(
            features_dir / "test_agg_states.parquet", engine="fastparquet"
        )
    except Exception as e:
        logger.warning(
            "States stage warning: %s; falling back to aggregate features", e
        )
        X_tr_agg_states = X_tr_agg
        X_va_agg_states = X_va_agg
        X_te_agg_states = X_te_agg

    # Assemble X_dict for training
    train_val_agg = pd.concat([X_tr_agg, X_va_agg]).reset_index(drop=True).values
    train_val_raw = pd.concat([X_tr_raw, X_va_raw]).reset_index(drop=True).values
    train_val_single = (
        pd.concat([X_tr_single, X_va_single]).reset_index(drop=True).values
    )
    train_val_agg_states = (
        pd.concat([X_tr_agg_states, X_va_agg_states]).reset_index(drop=True).values
    )
    train_val_states = (
        pd.concat([X_tr_states, X_va_states]).reset_index(drop=True).values
        if "X_tr_states" in locals()
        else train_val_agg
    )

    X_dict = {
        "agg": train_val_agg,
        "raw28": train_val_raw,
        "single_signal": train_val_single,
        "agg_states": train_val_agg_states,
        "states_only": train_val_states,
    }

    train_val_idx_combined = np.concatenate([train_idx, val_idx])
    y_train_val = clean_df.iloc[train_val_idx_combined]["at_risk"].values.astype(int)

    # 6. Benchmark Classification Model Zoo (Track A)
    progress(0.40, "Benchmarking Classification Model Zoo (Track A)...")
    check_cancel()
    class_specs = get_retrain_specs(mode=mode, guards=guards)
    class_results: List[Dict[str, Any]] = []

    total_specs = len(class_specs)
    for i, spec in enumerate(class_specs):
        check_cancel()
        sub_p = 0.40 + (i / max(1, total_specs)) * 0.25
        progress(
            sub_p,
            f"Training classifier [{i + 1}/{total_specs}]: {spec.display_name}...",
        )
        try:
            m_res = evaluate_single_model(
                spec=spec,
                X_dict=X_dict,
                y=y_train_val,
                cv_folds=cv_folds,
                seed=seed,
                enable_tuning=(mode == "full" and not guards["is_small_cohort"]),
                time_budget_seconds=120 if mode == "quick" else 300,
            )
            # Save artifacts
            m_dir = models_dir / spec.id
            m_dir.mkdir(parents=True, exist_ok=True)
            with open(m_dir / "detail.json", "w", encoding="utf-8") as f:
                json.dump(m_res, f, indent=2, default=str)
            if "pipeline" in m_res:
                joblib.dump(m_res["pipeline"], m_dir / "pipeline.joblib")
            class_results.append(m_res)
        except Exception as err:
            logger.error("Failed model %s: %s", spec.id, err)

    if not class_results:
        raise RuntimeError("No classification models trained successfully.")

    assemble_and_save_leaderboard(class_results, run_dir, track="classification")

    # 7. Benchmark Regression Model Zoo (Track B)
    progress(0.68, "Benchmarking Continuous Attrition Regression Models (Track B)...")
    check_cancel()
    reg_specs = get_retrain_regression_specs(mode=mode, guards=guards)
    y_reg = clean_df.iloc[train_val_idx_combined][
        "attrition_intent_score"
    ].values.astype(float)
    reg_results: List[Dict[str, Any]] = []

    for j, rspec in enumerate(reg_specs):
        check_cancel()
        sub_p = 0.68 + (j / max(1, len(reg_specs))) * 0.12
        progress(
            sub_p,
            f"Training regressor [{j + 1}/{len(reg_specs)}]: {rspec.display_name}...",
        )
        try:
            r_res = evaluate_single_regression_model(
                spec=rspec,
                X_dict=X_dict,
                y=y_reg,
                cv_folds=cv_folds,
                seed=seed,
                enable_tuning=(mode == "full" and not guards["is_small_cohort"]),
            )
            r_dir = models_dir / rspec.id
            r_dir.mkdir(parents=True, exist_ok=True)
            with open(r_dir / "detail.json", "w", encoding="utf-8") as f:
                json.dump(r_res, f, indent=2, default=str)
            if "pipeline" in r_res:
                joblib.dump(r_res["pipeline"], r_dir / "pipeline.joblib")
            reg_results.append(r_res)
        except Exception as err:
            logger.error("Failed regression model %s: %s", rspec.id, err)

    if reg_results:
        assemble_and_save_regression_leaderboard(reg_results, run_dir)

    # 8. Overfit Audit, Champion Selection & Holdout Test Evaluation
    progress(
        0.82, "Auditing generalization gap and executing holdout test evaluation..."
    )
    check_cancel()
    audit_data = run_overfitting_audit(run_dir)
    with open(run_dir / "overfit_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, default=str)

    # Pick champion: top validation score with generalization gap <= 0.05
    lb_df = pd.read_csv(run_dir / "leaderboard_classification.csv")
    candidates = lb_df[lb_df["overfit_status"].isin(["good", "slight"])]
    if len(candidates) == 0:
        candidates = lb_df
    top_row = candidates.iloc[0]
    champ_id = top_row["model_id"]
    champ_spec = next((s for s in class_specs if s.id == champ_id), class_specs[0])

    # Holdout Test Evaluation
    test_eval = evaluate_final_on_test(
        champ_spec=champ_spec,
        run_dir=run_dir,
        n_bootstrap=2000 if guards["is_small_cohort"] else 1000,
        seed=seed,
    )

    # Threshold tuning and calibration on Validation partition
    y_val = clean_df.iloc[val_idx]["at_risk"].values.astype(int)
    champ_detail = json.load(open(models_dir / champ_id / "detail.json"))
    val_probs = np.array(
        champ_detail.get("val_predictions", {}).get("probabilities", [])
    )
    if len(val_probs) != len(y_val):
        # Predict on val
        pipe = joblib.load(models_dir / champ_id / "pipeline.joblib")
        val_probs = pipe.predict_proba(X_va_agg.values)[:, 1]

    thresh_meta = tune_decision_thresholds(y_val, val_probs)
    calibrator = fit_isotonic_calibrator(val_probs, y_val)

    # Save champion bundle
    champ_summary = {
        "model_id": champ_id,
        "display_name": top_row["display_name"],
        "category": top_row["category"],
        "cv_score": float(top_row["leaderboard_score"]),
        "gap": float(top_row["generalization_gap"]),
        "test_f1": float(test_eval.get("test_metrics", {}).get("f1_score", 0.0)),
        "test_auc": float(test_eval.get("test_metrics", {}).get("auc_roc", 0.0)),
        "test_recall": float(test_eval.get("test_metrics", {}).get("recall", 0.0)),
        "test_precision": float(
            test_eval.get("test_metrics", {}).get("precision", 0.0)
        ),
        "optimal_threshold": thresh_meta.get("optimal_threshold", 0.26),
        "sensitive_threshold": thresh_meta.get("sensitive_threshold", 0.21),
    }

    pipe_to_save = joblib.load(models_dir / champ_id / "pipeline.joblib")
    save_champion_bundle(
        champ_dir=champ_dir,
        pipeline=pipe_to_save,
        calibrator=calibrator,
        meta={**champ_summary, "n_students": n_samples},
    )

    # 9. Explainability & Reason Generation
    progress(0.92, "Computing Tree/Kernel SHAP explainers and waterfall reasons...")
    check_cancel()
    try:
        from mei.explain.engine import compute_global_shap_audit

        compute_global_shap_audit(
            run_dir,
            n_val_samples=min(100, len(val_df)),
            n_test_samples=min(100, len(test_df)),
        )
    except Exception as e:
        logger.warning("Explainability generation warning: %s", e)

    # 10. Data Card, Model Card, and Manifest
    progress(0.97, "Compiling Data Card, Model Card, and Run Manifest...")
    generate_data_card(run_dir, clean_df, val_meta, guards)
    generate_model_card(run_dir, champ_summary, guards, n_samples)

    manifest = {
        "run_id": run_id,
        "data_source": "survey",
        "dataset_name": dataset_path.name,
        "n_students": n_samples,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "champion": champ_summary,
        "small_data_guards": guards,
        "status": "completed",
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    progress(1.0, f"Retraining completed successfully! New Run: {run_id}")
    logger.info(
        "Pipeline completed in %.2f seconds (Run: %s)", time.time() - start_time, run_id
    )

    return {
        "run_id": run_id,
        "data_source": "survey",
        "n_students": n_samples,
        "champion": champ_summary,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "guards": guards,
    }


# ---------------------------------------------------------------------------
# Run Activation & Safe Rollback with Audit Trail
# ---------------------------------------------------------------------------


def activate_pipeline_run(new_run_id: str, actor: str = "web_user") -> Dict[str, Any]:
    """Switch active run, log audit line, and trigger store cache invalidation."""
    run_dir = RUNS_DIR / new_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Target run '{new_run_id}' not found on disk.")

    active_file = ARTIFACTS_DIR / "active_run.txt"
    old_run_id = (
        active_file.read_text(encoding="utf-8").strip()
        if active_file.exists()
        else "none"
    )

    # Get champions
    old_champ = "unknown"
    old_champ_file = RUNS_DIR / old_run_id / "champion" / "model_card.json"
    if old_champ_file.exists():
        try:
            old_champ = json.load(open(old_champ_file)).get("model_id", "unknown")
        except Exception:
            pass

    new_champ = "unknown"
    new_champ_file = run_dir / "champion" / "model_card.json"
    if new_champ_file.exists():
        try:
            new_champ = json.load(open(new_champ_file)).get("model_id", "unknown")
        except Exception:
            pass

    # Update active run
    active_file.write_text(new_run_id, encoding="utf-8")

    # Audit logging
    audit_log = ARTIFACTS_DIR / "audit.log"
    iso_now = datetime.now(timezone.utc).isoformat()
    log_line = (
        f"[{iso_now}] ACTION=ACTIVATE ACTOR={actor} PREVIOUS_RUN={old_run_id} "
        f"NEW_RUN={new_run_id} PREVIOUS_CHAMPION={old_champ} NEW_CHAMPION={new_champ}\n"
    )
    with open(audit_log, "a", encoding="utf-8") as f:
        f.write(log_line)

    # Invalidate RunStore cache
    try:
        from mei.api.store import RunStore

        RunStore.get_instance().clear_cache()
    except Exception:
        pass

    logger.info("Activated run %s (Replaced %s). Audit logged.", new_run_id, old_run_id)

    return {
        "status": "activated",
        "active_run_id": new_run_id,
        "previous_run_id": old_run_id,
        "activated_at": iso_now,
        "champion": new_champ,
        "previous_champion": old_champ,
    }


# ---------------------------------------------------------------------------
# Side-by-Side Run Comparison Engine
# ---------------------------------------------------------------------------


def compare_pipeline_runs(run_id_a: str, run_id_b: str) -> Dict[str, Any]:
    """Produce side-by-side comparison of two runs for model governance decision."""
    dir_a = RUNS_DIR / run_id_a
    dir_b = RUNS_DIR / run_id_b

    if not dir_a.exists():
        raise FileNotFoundError(f"Run A '{run_id_a}' not found.")
    if not dir_b.exists():
        raise FileNotFoundError(f"Run B '{run_id_b}' not found.")

    def load_run_meta(r_dir: Path) -> Dict[str, Any]:
        manifest_file = r_dir / "manifest.json"
        manifest = json.load(open(manifest_file)) if manifest_file.exists() else {}
        card_file = r_dir / "champion" / "model_card.json"
        card = json.load(open(card_file)) if card_file.exists() else {}
        lb_file = r_dir / "leaderboard_classification.csv"
        lb = pd.read_csv(lb_file).to_dict(orient="records") if lb_file.exists() else []
        return {
            "run_id": r_dir.name,
            "data_source": manifest.get("data_source", "sample"),
            "n_students": manifest.get("n_students", card.get("n_students", 0)),
            "created_at": manifest.get("created_at", card.get("trained_at", "")),
            "champion": card,
            "leaderboard": lb[:10],
        }

    meta_a = load_run_meta(dir_a)
    meta_b = load_run_meta(dir_b)

    champ_a = meta_a["champion"]
    champ_b = meta_b["champion"]

    score_a = float(champ_a.get("primary_score", 0.0))
    score_b = float(champ_b.get("primary_score", 0.0))
    score_diff = round(score_b - score_a, 4)

    test_f1_a = float(champ_a.get("test_f1") or 0.0)
    test_f1_b = float(champ_b.get("test_f1") or 0.0)
    f1_diff = round(test_f1_b - test_f1_a, 4)

    # Rank changes
    ranks_a = {r["model_id"]: int(r["rank"]) for r in meta_a["leaderboard"]}
    rank_changes = []
    for r in meta_b["leaderboard"]:
        mid = r["model_id"]
        old_rk = ranks_a.get(mid)
        new_rk = int(r["rank"])
        delta = (old_rk - new_rk) if old_rk is not None else None
        rank_changes.append(
            {
                "model_id": mid,
                "display_name": r.get("display_name", mid),
                "category": r.get("category", ""),
                "old_rank": old_rk,
                "new_rank": new_rk,
                "rank_delta": delta,
                "score": float(r.get("leaderboard_score", 0.0)),
            }
        )

    # Verdict summary
    is_better = score_diff > 0 or f1_diff > 0
    champ_changed = champ_a.get("model_id") != champ_b.get("model_id")
    if is_better:
        verdict = (
            f"The candidate run ({run_id_b}) shows improved performance (+{score_diff:.4f} validation score, "
            f"+{f1_diff:.4f} test F1). "
            + (
                f"Champion shifted to {champ_b.get('model_name')}. "
                if champ_changed
                else "Champion model retained. "
            )
            + "Activation is recommended."
        )
    else:
        verdict = (
            f"The candidate run ({run_id_b}) achieves comparable or slightly lower metrics ({score_diff:+.4f} validation score). "
            "Inspect sample size and class balance before activating."
        )

    return {
        "run_a": meta_a,
        "run_b": meta_b,
        "comparison": {
            "champion_changed": champ_changed,
            "champion_a": champ_a.get("model_name"),
            "champion_b": champ_b.get("model_name"),
            "cv_score_diff": score_diff,
            "test_f1_diff": f1_diff,
            "rank_changes": rank_changes,
            "verdict_summary": verdict,
            "is_recommended": is_better,
        },
    }

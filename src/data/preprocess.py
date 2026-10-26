"""Preprocessing, Missing Data Imputation, Cleaning, and Imbalance Pipeline.

Implements:
1. Cleaning: Schema validation -> straight-liner removal (configurable) -> outlier handling.
2. Imputation: Comparison of Median/Mode vs. Iterative MICE imputation (fitted strictly on train).
3. Production Preprocessing Pipeline: Sklearn ColumnTransformer / Pipeline saved with Joblib
   for reproducible inference on newly uploaded CSVs.
4. Class Imbalance: Configurable class weight computation and SMOTE oversampling
   enforced strictly within training folds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
# Explicitly enable IterativeImputer in scikit-learn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data.schema import validate_dataframe, REQUIRED_SURVEY_ITEMS
from src.features.build_features import (
    RAW_ITEM_COLS,
    FACTOR_COLS,
    RISK_FACTOR_COLS,
    INTERACTION_COLS,
    compute_factor_scores,
    compute_risk_oriented_factors,
    compute_interaction_features,
)
from src.utils.logger import get_logger

logger = get_logger("data.preprocess")


class PsychometricFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for psychometric factor and interaction engineering.

    Computes:
    - 5 latent factor domain scores
    - 5 risk-oriented factor scores
    - 5 non-linear interaction features
    All transformations are deterministic and leak-free.
    """

    def __init__(self, include_interactions: bool = True) -> None:
        self.include_interactions = include_interactions

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PsychometricFeatureTransformer":
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=RAW_ITEM_COLS)
        else:
            df = X.copy()

        df_factors = compute_factor_scores(df)
        df_risk = compute_risk_oriented_factors(df_factors)
        if self.include_interactions:
            df_full = compute_interaction_features(df_risk)
            return df_full
        return df_risk


def clean_dataset(
    df: pd.DataFrame,
    drop_straight_liners: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean survey dataset: validate schema, isolate straight-liners, and report audit info.

    Args:
        df: Input DataFrame.
        drop_straight_liners: Whether to remove unengaged respondents with zero item variance.
        validate: Whether to run schema validation check.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Audit summary dictionary).
    """
    initial_rows = len(df)
    audit: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "straight_liners_removed": 0,
        "validation_passed": True,
    }

    if validate:
        report = validate_dataframe(df, flag_straight_lining=True)
        audit["validation_passed"] = report.passed
        audit["validation_errors"] = report.errors
        audit["validation_warnings"] = report.warnings
        straight_liner_ids = report.straight_lining_respondents
    else:
        # Direct calculation of straight-lining
        items = [c for c in RAW_ITEM_COLS if c in df.columns]
        if len(items) == len(RAW_ITEM_COLS):
            std_rows = df[items].std(axis=1)
            straight_mask = (std_rows == 0.0) & df[items].notna().all(axis=1)
            straight_liner_ids = df.loc[straight_mask, "study_id"].tolist() if "study_id" in df.columns else []
        else:
            straight_liner_ids = []

    cleaned_df = df.copy()
    if drop_straight_liners and straight_liner_ids:
        if "study_id" in cleaned_df.columns:
            mask_sl = cleaned_df["study_id"].astype(str).isin([str(x) for x in straight_liner_ids])
            cleaned_df = cleaned_df[~mask_sl].reset_index(drop=True)
            audit["straight_liners_removed"] = int(mask_sl.sum())
            logger.info("Removed %d straight-lining respondents.", audit["straight_liners_removed"])

    audit["final_rows"] = len(cleaned_df)
    return cleaned_df, audit


def compare_imputation_strategies(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    items: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Empirically compare Median vs. Iterative (MICE) imputation fitted ONLY on train.

    Artificially masks values in val_df to evaluate mean absolute reconstruction error (MAE).

    Args:
        train_df: Training DataFrame (imputers fitted here ONLY).
        val_df: Validation DataFrame.
        items: Survey item column names.
        seed: Random seed.

    Returns:
        Dict[str, Any]: Comparison audit with reconstruction MAE per method.
    """
    if items is None:
        items = RAW_ITEM_COLS

    rng = np.random.default_rng(seed)
    train_sub = train_df[items].copy()
    val_sub = val_df[items].dropna().copy()

    if len(val_sub) < 20:
        return {"error": "Insufficient complete validation rows for masking evaluation."}

    # Artificially mask 10% of values in validation set to test reconstruction
    val_matrix = val_sub.values.copy()
    mask = rng.uniform(0, 1, size=val_matrix.shape) < 0.10
    ground_truth = val_matrix[mask]
    masked_val = val_matrix.copy()
    masked_val[mask] = np.nan

    # 1. Median Imputer
    median_imp = SimpleImputer(strategy="median")
    median_imp.fit(train_sub)
    df_masked = pd.DataFrame(masked_val, columns=items)
    recon_median = median_imp.transform(df_masked)
    mae_median = float(np.mean(np.abs(recon_median[mask] - ground_truth)))

    # 2. Iterative MICE Imputer
    mice_imp = IterativeImputer(max_iter=10, random_state=seed, initial_strategy="median")
    mice_imp.fit(train_sub)
    recon_mice = mice_imp.transform(df_masked)
    mae_mice = float(np.mean(np.abs(recon_mice[mask] - ground_truth)))

    logger.info("Imputation Comparison: Median MAE=%.4f | MICE MAE=%.4f", mae_median, mae_mice)

    return {
        "mae_median": round(mae_median, 4),
        "mae_mice": round(mae_mice, 4),
        "recommendation": "mice" if mae_mice < mae_median else "median",
    }


def build_preprocessing_pipeline(
    imputation_strategy: str = "median",
    include_scaler: bool = True,
    seed: int = 42,
) -> Pipeline:
    """Construct an end-to-end Scikit-Learn Pipeline for survey preprocessing.

    Pipeline structure:
    1. Imputer (Median or Iterative MICE)
    2. Psychometric Feature Transformer (calculates factor scores and interactions)
    3. StandardScaler (optional standard scaling)

    Args:
        imputation_strategy: 'median' or 'mice'.
        include_scaler: Whether to attach StandardScaler to continuous features.
        seed: Random seed for iterative methods.

    Returns:
        Pipeline: Unfitted Scikit-Learn Pipeline.
    """
    if imputation_strategy == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=seed, initial_strategy="median")
    else:
        imputer = SimpleImputer(strategy="median")

    steps: List[Tuple[str, Any]] = [
        ("imputer", imputer),
        ("psychometrics", PsychometricFeatureTransformer(include_interactions=True)),
    ]

    if include_scaler:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    output_path: str = "artifacts/models/preprocessor.joblib",
    imputation_strategy: str = "median",
    include_scaler: bool = False,
    seed: int = 42,
) -> Pipeline:
    """Fit complete preprocessing pipeline SOLELY on train_df and persist via joblib.

    Guarantees strict zero-leakage: validation/test data are never seen during fitting.

    Args:
        train_df: Training fold/split DataFrame.
        output_path: Path to serialized artifact.
        imputation_strategy: Imputation method ('median' or 'mice').
        include_scaler: Whether to include StandardScaler.
        seed: Random seed.

    Returns:
        Pipeline: Fitted Scikit-Learn Pipeline.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_preprocessing_pipeline(
        imputation_strategy=imputation_strategy,
        include_scaler=include_scaler,
        seed=seed,
    )

    X_train_raw = train_df[RAW_ITEM_COLS].copy()
    pipeline.fit(X_train_raw)

    joblib.dump(pipeline, out_file)
    logger.info("Fitted preprocessor pipeline saved to %s", out_file)
    return pipeline


def load_preprocessor(path: str = "artifacts/models/preprocessor.joblib") -> Pipeline:
    """Load serialized preprocessing pipeline for zero-leakage inference."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found: {p}")
    return joblib.load(p)


def get_class_weights_dict(y: pd.Series) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced classification.

    Args:
        y: Target series (e.g. binary attrition_label or 3-class risk_category).

    Returns:
        Dict[int, float]: Mapping of class label to inverse frequency weight.
    """
    classes = np.unique(y.dropna())
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.dropna().values)
    return {cls: round(float(w), 4) for cls, w in zip(classes, weights)}


def apply_smote_within_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling strictly within a cross-validation training fold.

    Prevents synthetic minority examples from bleeding into validation folds.

    Args:
        X_train: Training fold feature matrix.
        y_train: Training fold binary target.
        seed: Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (Resampled X_train, Resampled y_train).
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info("SMOTE applied: %d -> %d samples.", len(X_train), len(X_res))
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)
    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train


def main() -> None:
    """Execute complete Phase 3 Preprocessing, Reliability, and Splitting Pipeline."""
    from src.data.splits import create_stratified_splits, load_split
    from src.features.reliability import run_full_reliability_analysis, export_reliability_artifacts
    from src.features.build_features import engineer_all_features

    raw_path = Path("data/synthetic/synthetic_survey_3000.csv")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Synthetic survey not found at {raw_path}. Run Phase 2 first.")

    logger.info("Initiating Phase 3 Pipeline with %s...", raw_path)
    df_raw = pd.read_csv(raw_path)

    # 1. Cleaning & straight-liner removal
    df_clean, clean_audit = clean_dataset(df_raw, drop_straight_liners=True)
    logger.info("Cleaning Audit: %s", clean_audit)

    # 2. Stratified Splits (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = create_stratified_splits(
        df_clean,
        stratify_col="risk_category",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
        output_dir="data/processed",
    )

    # 3. Imputation Strategy Comparison (Median vs. MICE)
    imp_comparison = compare_imputation_strategies(train_df, val_df, seed=42)
    logger.info("Imputation Comparison Result: %s", imp_comparison)

    # 4. Fit & Save Preprocessing Pipeline (fitted strictly on train_df)
    preprocessor = fit_and_save_preprocessor(
        train_df,
        output_path="artifacts/models/preprocessor.joblib",
        imputation_strategy=imp_comparison.get("recommendation", "median"),
        include_scaler=False,
        seed=42,
    )

    # 5. Reliability Analysis on Training Data
    # First impute train items to compute reliability
    imputed_train_items = preprocessor.named_steps["imputer"].transform(train_df[RAW_ITEM_COLS])
    df_train_imputed = pd.DataFrame(imputed_train_items, columns=RAW_ITEM_COLS)

    rel_report = run_full_reliability_analysis(df_train_imputed, alpha_threshold=0.70)
    rel_report["cleaning_audit"] = clean_audit
    rel_report["imputation_comparison"] = imp_comparison

    # Export reliability artifacts
    saved_rel_plots = export_reliability_artifacts(
        rel_report,
        metrics_path="artifacts/metrics/reliability.json",
        plots_dir="artifacts/plots",
    )

    # 6. Feature Engineering for Train and Val
    train_eng, scaler = engineer_all_features(df_train_imputed, fit_scaler=True)
    imputed_val_items = preprocessor.named_steps["imputer"].transform(val_df[RAW_ITEM_COLS])
    df_val_imputed = pd.DataFrame(imputed_val_items, columns=RAW_ITEM_COLS)
    val_eng, _ = engineer_all_features(df_val_imputed, scaler=scaler, fit_scaler=False)

    print("\n" + "=" * 65)
    print("PHASE 3 PREPROCESSING & RELIABILITY PIPELINE COMPLETE!")
    print(f"Cleaned Samples:        {len(df_clean)} (Removed {clean_audit['straight_liners_removed']} straight-liners)")
    print(f"Splits Generated:       Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Locked)")
    print(f"Imputation Evaluation:  Median MAE={imp_comparison.get('mae_median')}, MICE MAE={imp_comparison.get('mae_mice')}")
    print(f"Selected Imputer:       {imp_comparison.get('recommendation', 'median').upper()}")
    print("\nCronbach's Alpha Internal Consistency Summary (Train Set):")
    print("-" * 65)
    for f_name, f_data in rel_report["factors"].items():
        print(f"  • {f_name.replace('_', ' ').title():<18}: Alpha = {f_data['cronbach_alpha']:.4f} | Omega = {f_data['mcdonald_omega']:.4f} | [{f_data['status']}]")
    print("-" * 65)
    print(f"All Factors Passed Threshold (>=0.70): {rel_report['all_factors_passed']}")
    print(f"Saved Preprocessor Pipeline: artifacts/models/preprocessor.joblib")
    print(f"Saved Reliability JSON:      artifacts/metrics/reliability.json")
    print(f"Saved Plots:                 {saved_rel_plots}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

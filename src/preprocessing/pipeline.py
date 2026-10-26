"""Strict in-fold preprocessing pipelines for tabular attrition prediction.

Guarantees:
1. Leakage Prevention: ColumnTransformer is fitted strictly inside training folds.
2. Feature Set Boundary: Restricts inputs to environmental/financial features and documented
   baseline covariates (baseline GPA band, baseline engagement score). Mediators, identifiers,
   fairness audit variables, and targets are strictly prohibited.
3. Dual Feature Views:
   (a) Tree-View: Imputed, ordinal/one-hot encoded, NO scaling (for XGBoost, LightGBM, CatBoost).
   (b) Scaled-View: Imputed, encoded, and RobustScaler transformed (for Linear/Deep/KNN).
4. No NaNs: Guaranteed complete transformation with missingness indicators preserved.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.utils.logging_config import setup_logger

logger = setup_logger("pipeline")

# ==============================================================================
# FEATURE SET BOUNDARY SPECIFICATION (PROJECT_CONTEXT.md)
# ==============================================================================

NUMERIC_FEATURES = [
    "commute_time_min",
    "commute_cost_lkr_week",
    "work_hours_per_week",
    "dependents_count",
    "baseline_engagement_score",
]

# Explicit ordinal hierarchies
ORDINAL_CATEGORIES = {
    "household_income_bracket": [1, 2, 3, 4, 5, 6],
    "financial_stress": [1, 2, 3, 4, 5],
    "internet_reliability": [1, 2, 3, 4, 5],
    "internet_affordability": [1, 2, 3, 4, 5],
    "device_quality": [1, 2, 3, 4, 5],
    "year_of_study": [1, 2, 3, 4],
    "baseline_gpa_band": [
        "Band 1 (< 2.00)",
        "Band 2 (2.00 - 2.49)",
        "Band 3 (2.50 - 2.99)",
        "Band 4 (3.00 - 3.49)",
        "Band 5 (3.50 - 4.00)",
    ],
}

ORDINAL_FEATURES = list(ORDINAL_CATEGORIES.keys())

NOMINAL_FEATURES = [
    "district_type",
    "housing_type",
    "device_type",
    "faculty",
]

BINARY_FEATURES = [
    "quiet_study_space",
    "part_time_employment",
    "scholarship_bursary",
]

ALL_ALLOWED_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES

# Forbidden columns that must NEVER enter predictive modeling features
FORBIDDEN_COLUMNS = {
    # Identifiers
    "participant_id",
    # Fairness auditing attribute (omitted from predictive features)
    "gender",
    # Intermediate Mediators (MUST NOT enter adjustment set or predictive features)
    "later_gpa_decline",
    "later_engagement_decline",
    # Derived policy exposures (analyzed in Phase 6 Causal track)
    "financial_hardship",
    "long_commute",
    "inadequate_digital_access",
    "heavy_part_time_work",
    # Target Outcomes
    "attrition",
    "intention_to_withdraw",
    "latent_risk_score",
    # Generative metadata (Bayes ceiling benchmark only)
    "true_probability",
}


def validate_feature_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict feature set boundary.

    Drops forbidden variables (mediators, outcomes, IDs, gender) and asserts
    that only approved environmental/financial features remain.
    """
    detected_forbidden = [col for col in df.columns if col in FORBIDDEN_COLUMNS]
    if detected_forbidden:
        logger.warning(
            f"Enforcing feature set boundary: Dropping forbidden variables: {detected_forbidden}"
        )
        df_filtered = df.drop(columns=detected_forbidden)
    else:
        df_filtered = df.copy()

    # Verify that only allowed features remain
    remaining_cols = set(df_filtered.columns)
    unauthorized = remaining_cols - set(ALL_ALLOWED_FEATURES)
    if unauthorized:
        raise ValueError(
            f"UNAUTHORIZED FEATURE DETECTED: {unauthorized}. "
            f"Only documented environmental, financial, and baseline covariates may enter the model."
        )

    return df_filtered


def build_column_transformer(scaled: bool = False) -> ColumnTransformer:
    """Construct Scikit-Learn ColumnTransformer for mixed tabular features.

    Parameters
    ----------
    scaled : bool
        If True, appends RobustScaler to numerical features (scaled-view).
        If False, preserves raw numerical scales (tree-view).

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer instance.
    """
    # 1. Numeric Pipeline
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scaled:
        numeric_steps.append(("scaler", RobustScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # 2. Ordinal Pipeline (handle string/int categories, map to integers with imputer)
    ordinal_categories_list = [ORDINAL_CATEGORIES[col] for col in ORDINAL_FEATURES]
    ordinal_pipe = Pipeline([
        (
            "encoder",
            OrdinalEncoder(
                categories=ordinal_categories_list,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])

    # 3. Nominal Pipeline (Most-frequent imputer + One-Hot Encoding)
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 4. Binary Pipeline (Most-frequent imputer)
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("ord", ordinal_pipe, ORDINAL_FEATURES),
            ("nom", nominal_pipe, NOMINAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return transformer


class LeakageProofPreprocessor(BaseEstimator, TransformerMixin):
    """Wrapper ensuring ColumnTransformer is fitted strictly within training partitions.

    Guarantees that fit() raises an error if invoked outside an isolated training fold.
    """

    def __init__(self, view_type: str = "tree"):
        """Initialize preprocessor.

        Parameters
        ----------
        view_type : str
            Either 'tree' (unscaled) or 'scaled' (RobustScaler applied).
        """
        assert view_type in {"tree", "scaled"}, f"view_type must be 'tree' or 'scaled', got {view_type}"
        self.view_type = view_type
        self.is_scaled = (view_type == "scaled")
        self.transformer = build_column_transformer(scaled=self.is_scaled)
        self.is_fitted = False
        self.fitted_on: Optional[str] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None, is_train_fold: bool = False):
        """Fit preprocessor strictly on training fold data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : Optional[Any]
            Target vector (unused during unsupervised transformation).
        is_train_fold : bool
            Safety assertion flag. Must be set to True.
        """
        if not is_train_fold:
            msg = (
                "DATA LEAKAGE PREVENTION VIOLATION: "
                "Attempted to fit preprocessor without confirming training fold isolation! "
                "You must pass is_train_fold=True to confirm this call occurs inside a training fold."
            )
            logger.error(msg)
            raise ValueError(msg)

        X_valid = validate_feature_boundary(X)
        self.transformer.fit(X_valid)
        self.is_fitted = True
        self.fitted_on = "training_fold"
        try:
            self.feature_names_out_ = list(self.transformer.get_feature_names_out())
        except AttributeError:
            self.feature_names_out_ = None

        logger.info(f"Preprocessor ({self.view_type}-view) successfully fitted on {len(X)} training samples.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using parameters learned exclusively from training fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Complete numerical array with zero NaNs.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted! Call fit() inside training fold first.")

        X_valid = validate_feature_boundary(X)
        transformed = self.transformer.transform(X_valid)

        # Assertion: Transformed matrix must have zero NaNs
        nan_count = int(np.isnan(transformed).sum())
        if nan_count > 0:
            raise ValueError(f"Transformation produced {nan_count} NaNs in feature matrix!")

        return transformed

    def fit_transform(self, X: pd.DataFrame, y=None, is_train_fold: bool = False) -> np.ndarray:
        """Fit and transform on training fold data."""
        self.fit(X, y=y, is_train_fold=is_train_fold)
        return self.transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return list of transformed feature column names."""
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        return [f"feature_{i}" for i in range(len(self.transformer.transformers_))]


def get_tree_pipeline() -> LeakageProofPreprocessor:
    """Return preprocessor for Tree-based algorithms (XGBoost, LightGBM, CatBoost, RF)."""
    return LeakageProofPreprocessor(view_type="tree")


def get_scaled_pipeline() -> LeakageProofPreprocessor:
    """Return preprocessor for Linear/Deep/Distance algorithms (Logistic Regression, SVM, KNN, TabNet, MLP)."""
    return LeakageProofPreprocessor(view_type="scaled")

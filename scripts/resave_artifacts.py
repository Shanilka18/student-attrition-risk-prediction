"""Regenerate all model and preprocessor .joblib artifacts with the current
numpy/sklearn version to fix BitGenerator pickle incompatibility.

Run once after any major dependency version change:
    python scripts/resave_artifacts.py
"""
import sys
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("resave_artifacts")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocess import PsychometricFeatureTransformer, fit_and_save_preprocessor
from src.features.build_features import FACTOR_NAMES, FACTOR_ITEMS

TRAIN_CSV = ROOT / "data/processed/train.csv"
MODELS_DIR = ROOT / "artifacts/models"
FINAL_DIR = MODELS_DIR / "final"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

FACTOR_COLS = [f"factor_{f}" for f in FACTOR_NAMES]


def load_train_data():
    df = pd.read_csv(TRAIN_CSV)
    log.info("Loaded training data: %d rows", len(df))
    return df


def rebuild_preprocessor(df: pd.DataFrame) -> PsychometricFeatureTransformer:
    """Fit and save preprocessor from scratch."""
    all_items = [item for items in FACTOR_ITEMS.values() for item in items]
    available = [c for c in all_items if c in df.columns]
    preprocessor = PsychometricFeatureTransformer()
    preprocessor.fit(df[available])
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    log.info("Saved preprocessor → %s", PREPROCESSOR_PATH)
    return preprocessor


def prepare_features(df: pd.DataFrame, preprocessor: PsychometricFeatureTransformer):
    """Transform raw items → factor feature columns."""
    all_items = [item for items in FACTOR_ITEMS.values() for item in items]
    available = [c for c in all_items if c in df.columns]
    trans = preprocessor.transform(df[available])
    factor_cols_present = [c for c in FACTOR_COLS if c in trans.columns]
    return trans[factor_cols_present]


def build_regression_models(seed: int) -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100, random_state=seed
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=seed
        ),
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=seed),
        "dummy": DummyRegressor(strategy="mean"),
    }


def build_classification_models(seed: int) -> dict:
    return {
        "ridge": RidgeClassifier(),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=seed
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "dummy": DummyClassifier(strategy="most_frequent"),
    }


def main():
    log.info("=== Artifact Resave Script ===")
    log.info("numpy %s | sklearn loaded", np.__version__)

    df = load_train_data()

    # 1. Preprocessor
    preprocessor = rebuild_preprocessor(df)
    X = prepare_features(df, preprocessor)
    log.info("Feature matrix shape: %s, columns: %s", X.shape, list(X.columns))

    # 2. Regression targets
    reg_target_col = None
    for candidate in ("PRI_true", "pri_true", "PRI_score", "pri_score"):
        if candidate in df.columns:
            reg_target_col = candidate
            break
    if reg_target_col is None:
        # Fall back: compute from factor scores
        from src.features.build_features import compute_pri_score
        X_full = preprocessor.transform(df[[c for c in [item for items in FACTOR_ITEMS.values() for item in items] if c in df.columns]])
        df["_pri_synthetic"] = X_full.apply(
            lambda row: compute_pri_score({f: row.get(f"factor_{f}", 3.0) for f in FACTOR_NAMES}),
            axis=1,
        )
        reg_target_col = "_pri_synthetic"
        log.warning("No PRI_true column found; using computed PRI as regression target.")

    y_reg = df[reg_target_col].values
    log.info("Regression target '%s': min=%.1f max=%.1f", reg_target_col, y_reg.min(), y_reg.max())

    # 3. Classification target
    clf_target_col = None
    for candidate in ("risk_category", "risk_label", "attrition_label"):
        if candidate in df.columns:
            clf_target_col = candidate
            break
    if clf_target_col is None:
        # Derive from regression target
        bins = [0, 33.33, 66.67, 100.01]
        labels = ["Low", "Moderate", "High"]
        df["_risk_synthetic"] = pd.cut(y_reg, bins=bins, labels=labels).astype(str)
        clf_target_col = "_risk_synthetic"
        log.warning("No risk_category column found; using PRI-derived labels.")

    y_clf_raw = df[clf_target_col].values.astype(str)
    log.info(
        "Classification target '%s': unique classes=%s", clf_target_col, sorted(set(y_clf_raw))
    )

    # 4. Train and save regression models
    reg_dir = MODELS_DIR / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_models = build_regression_models(SEED)
    for name, model in reg_models.items():
        model.fit(X, y_reg)
        out = reg_dir / f"{name}.joblib"
        joblib.dump(model, out)
        log.info("Saved regression/%s → %s", name, out)

    # 5. Train and save classification models (fit on raw string labels)
    clf_dir = MODELS_DIR / "classification"
    clf_dir.mkdir(parents=True, exist_ok=True)
    clf_models = build_classification_models(SEED)
    for name, model in clf_models.items():
        model.fit(X, y_clf_raw)  # string labels → classes_ will be strings
        out = clf_dir / f"{name}.joblib"
        joblib.dump(model, out)
        log.info("Saved classification/%s → %s", name, out)

    # 6. Champions (best models → final/)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Regression champion: HistGradientBoosting (typically best)
    reg_champion = reg_models["hist_gradient_boosting"]
    joblib.dump(reg_champion, FINAL_DIR / "regression_champion.joblib")
    log.info("Saved regression champion → %s", FINAL_DIR / "regression_champion.joblib")

    # Classification champion: HistGradientBoosting (trained on string labels)
    clf_champion = clf_models["hist_gradient_boosting"]
    joblib.dump(clf_champion, FINAL_DIR / "classification_champion.joblib")
    log.info("Saved classification champion → %s", FINAL_DIR / "classification_champion.joblib")

    log.info("=== Artifact resave complete — all models use numpy %s ===", np.__version__)


if __name__ == "__main__":
    main()

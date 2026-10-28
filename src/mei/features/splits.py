"""Splits and Cross-Validation Scheme with Strict Test Isolation Guard."""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def create_outer_split(
    df: pd.DataFrame,
    stratify_col: str = "at_risk",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition cohort into Train (70%), Validation (15%), and Test (15%).

    Stratified by at_risk (and archetype if available in sample data).
    Enforces student-level partitioning with zero patient/student overlap.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        tot = train_ratio + val_ratio + test_ratio
        raise ValueError(f"Split ratios must sum to 1.0; got {tot:.4f}")

    n_samples = len(df)
    indices = np.arange(n_samples)

    # Determine stratification target
    if stratify_col not in df.columns:
        raise ValueError(
            f"Stratification column '{stratify_col}' not found in dataframe."
        )

    strat_labels = df[stratify_col].astype(str).copy()

    # Joint stratification with archetype if available (sample data only)
    if "archetype" in df.columns:
        joint_labels = df[stratify_col].astype(str) + "_" + df["archetype"].astype(str)
        min_cell = joint_labels.value_counts().min()
        if min_cell >= 4:
            strat_labels = joint_labels
            logger.info(
                "Stratifying outer split by joint (at_risk + archetype) labels."
            )

    # Step 1: Split into Train+Val (85%) and Test (15%)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=strat_labels,
        random_state=seed,
    )

    # Step 2: Split Train+Val into Train (70/85) and Val (15/85)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_val_strat = strat_labels.iloc[train_val_idx]

    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=relative_val_ratio,
        stratify=train_val_strat,
        random_state=seed,
    )

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    # Verify disjointness
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples

    return train_idx, val_idx, test_idx


def save_splits(
    splits_dir: Union[str, Path],
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    id_col: str = "student_id",
    stratify_col: str = "at_risk",
) -> Dict[str, Any]:
    """Persist train, val, and test index sets to disk."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to standard python int lists
    train_list = [int(i) for i in train_idx]
    val_list = [int(i) for i in val_idx]
    test_list = [int(i) for i in test_idx]

    # Student ID lists
    train_students = df.iloc[train_idx][id_col].tolist() if id_col in df.columns else []
    val_students = df.iloc[val_idx][id_col].tolist() if id_col in df.columns else []
    test_students = df.iloc[test_idx][id_col].tolist() if id_col in df.columns else []

    (out_dir / "train_idx.json").write_text(json.dumps(train_list, indent=2))
    (out_dir / "val_idx.json").write_text(json.dumps(val_list, indent=2))
    (out_dir / "test_idx.json").write_text(json.dumps(test_list, indent=2))

    summary = {
        "total_samples": len(df),
        "train": {
            "count": len(train_list),
            "ratio": round(len(train_list) / len(df), 4),
            "at_risk_count": int(df.iloc[train_idx][stratify_col].sum())
            if stratify_col in df.columns
            else None,
            "at_risk_pct": round(
                float(df.iloc[train_idx][stratify_col].mean() * 100), 2
            )
            if stratify_col in df.columns
            else None,
            "students": train_students,
        },
        "val": {
            "count": len(val_list),
            "ratio": round(len(val_list) / len(df), 4),
            "at_risk_count": int(df.iloc[val_idx][stratify_col].sum())
            if stratify_col in df.columns
            else None,
            "at_risk_pct": round(float(df.iloc[val_idx][stratify_col].mean() * 100), 2)
            if stratify_col in df.columns
            else None,
            "students": val_students,
        },
        "test": {
            "count": len(test_list),
            "ratio": round(len(test_list) / len(df), 4),
            "at_risk_count": int(df.iloc[test_idx][stratify_col].sum())
            if stratify_col in df.columns
            else None,
            "at_risk_pct": round(float(df.iloc[test_idx][stratify_col].mean() * 100), 2)
            if stratify_col in df.columns
            else None,
            "students": test_students,
        },
    }

    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"Saved splits to {out_dir}: Train={len(train_list)}, "
        f"Val={len(val_list)}, Test={len(test_list)}"
    )
    return summary


def load_train_indices(splits_dir: Union[str, Path]) -> np.ndarray:
    """Load train indices from disk."""
    path = Path(splits_dir) / "train_idx.json"
    if not path.exists():
        raise FileNotFoundError(f"Train indices file not found: {path}")
    return np.array(json.loads(path.read_text()), dtype=int)


def load_val_indices(splits_dir: Union[str, Path]) -> np.ndarray:
    """Load validation indices from disk."""
    path = Path(splits_dir) / "val_idx.json"
    if not path.exists():
        raise FileNotFoundError(f"Validation indices file not found: {path}")
    return np.array(json.loads(path.read_text()), dtype=int)


def load_test_indices(
    splits_dir: Union[str, Path],
    allow_override: bool = False,
) -> np.ndarray:
    """Load test indices strictly protected by target leakage guard.

    Raises PermissionError unless called by 'evaluate_final_on_test()' in Phase 9
    or explicitly authorized with allow_override=True.
    """
    stack = inspect.stack()
    immediate_caller = stack[1].function if len(stack) > 1 else ""

    authorized = allow_override or immediate_caller in [
        "evaluate_final_on_test",
        "_authorized_test_access",
    ]

    if not authorized:
        raise PermissionError(
            "Target Leakage Guard Violation (Rule G3): Test indices are strictly "
            "quarantined and cannot be accessed before Phase 9 evaluation. "
            "Only 'evaluate_final_on_test()' is permitted to load test indices."
        )

    path = Path(splits_dir) / "test_idx.json"
    if not path.exists():
        raise FileNotFoundError(f"Test indices file not found: {path}")
    return np.array(json.loads(path.read_text()), dtype=int)


def create_cv_folds(
    train_val_df: pd.DataFrame,
    stratify_col: str = "at_risk",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create repeated stratified cross-validation folds for model leaderboard.

    If cohort size N < 500, automatically switches to 5x5 (25 folds) with warning.
    Returns:
        List of (train_fold_idx, val_fold_idx) tuples.
    """
    n_samples = len(train_val_df)
    actual_repeats = n_repeats

    if n_samples < 500:
        actual_repeats = 5
        logger.warning(
            f"Small cohort size (N = {n_samples} < 500): automatically switched "
            f"CV scheme to RepeatedStratifiedKFold(5 splits x 5 repeats = 25 folds)."
        )

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=actual_repeats,
        random_state=seed,
    )

    y = train_val_df[stratify_col].values
    X = np.zeros(n_samples)  # Dummy feature array for KFold

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in rskf.split(X, y):
        folds.append((tr_idx, va_idx))

    logger.info(
        f"Generated {len(folds)} CV folds ({n_splits}x{actual_repeats} repeats)."
    )
    return folds


def save_cv_folds(
    splits_dir: Union[str, Path],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save fixed cross-validation fold indices to disk."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serialized = []
    for fold_idx, (tr, va) in enumerate(folds):
        serialized.append(
            {
                "fold": fold_idx,
                "train_idx": [int(i) for i in tr],
                "val_idx": [int(i) for i in va],
            }
        )

    (out_dir / "cv_folds.json").write_text(json.dumps(serialized, indent=2))
    logger.info(f"Saved {len(folds)} CV folds to {out_dir / 'cv_folds.json'}")


def load_cv_folds(splits_dir: Union[str, Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load fixed cross-validation fold indices from disk."""
    path = Path(splits_dir) / "cv_folds.json"
    if not path.exists():
        raise FileNotFoundError(f"CV folds file not found: {path}")

    raw_folds = json.loads(path.read_text())
    folds = [
        (np.array(f["train_idx"], dtype=int), np.array(f["val_idx"], dtype=int))
        for f in raw_folds
    ]
    return folds

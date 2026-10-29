"""Reproducible outer and repeated cross-validation splits for EFI features."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root


@dataclass(frozen=True)
class FeatureSplits:
    """In-memory partition indices and their immutable run artifact."""

    data: pd.DataFrame
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    path: Path
    warning: str | None = None


def create_feature_splits(
    data: pd.DataFrame,
    run_id: str,
    output_root: str | Path | None = None,
    seed: int = 42,
    student_col: str = "student_id",
    target_col: str = "at_risk",
) -> FeatureSplits:
    """Create stratified 70/15/15 participant splits and shared repeated folds.

    Indices refer to positional rows in the returned, reset-index data frame.
    Test positions are written once and are not exposed through a loader here.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "run_id may contain only letters, digits, underscores, and hyphens"
        )
    missing = {student_col, target_col} - set(data.columns)
    if missing:
        raise ValueError(f"Split input is missing required columns: {sorted(missing)}")
    frame = data.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("Cannot create splits from an empty dataset")
    if frame[[student_col, target_col]].isna().any().any():
        raise ValueError("Student identifiers and target labels must not be missing")
    if frame[student_col].duplicated().any():
        raise ValueError(
            "Expected one cleaned response per student before creating splits"
        )
    labels = frame[target_col]
    if labels.nunique() != 2:
        raise ValueError("Stratified splitting requires both at-risk classes")
    if labels.value_counts().min() < 3:
        raise ValueError("Each at-risk class must contain at least three students")

    positions = np.arange(len(frame))
    train_validation, test = train_test_split(
        positions,
        test_size=0.15,
        random_state=seed,
        stratify=labels.to_numpy(),
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        stratify=labels.iloc[train_validation].to_numpy(),
    )

    leaderboard_size = len(train_validation)
    n_splits = 5
    n_repeats = 5 if leaderboard_size < 500 else 3
    small_sample_warning = None
    if leaderboard_size < 500:
        small_sample_warning = (
            f"Only {leaderboard_size} students are available for leaderboard CV; "
            "using repeated stratified 5-fold CV with 5 repeats."
        )
        warnings.warn(small_sample_warning, UserWarning, stacklevel=2)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    folds: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    train_validation = np.asarray(train_validation, dtype=int)
    y_train_validation = labels.iloc[train_validation].to_numpy()
    for fold_train, fold_validation in cv.split(train_validation, y_train_validation):
        folds.append(
            (
                tuple(int(value) for value in train_validation[fold_train]),
                tuple(int(value) for value in train_validation[fold_validation]),
            )
        )

    project = load_yaml(get_project_root() / "config" / "project.yaml")
    data_source = project.get("project", {}).get("data_source")
    if data_source not in {"sample", "survey"}:
        raise ValueError(
            "config/project.yaml project.data_source must be sample or survey"
        )

    root = get_project_root() if output_root is None else Path(output_root)
    run_dir = root / "artifacts" / "runs" / run_id / "splits"
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = run_dir / "indices.json"
    payload: dict[str, Any] = {
        "metadata": {
            "run_id": run_id,
            "data_source": data_source,
            "seed": seed,
            "total_students": len(frame),
            "train_students": len(train),
            "validation_students": len(validation),
            "test_students": len(test),
            "leaderboard_students": leaderboard_size,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "warning": small_sample_warning,
        },
        "train_indices": [int(value) for value in train],
        "validation_indices": [int(value) for value in validation],
        "test_indices": [int(value) for value in test],
        "folds": [
            {
                "train_indices": list(fold_train),
                "validation_indices": list(fold_validation),
            }
            for fold_train, fold_validation in folds
        ],
    }
    try:
        with split_path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Split indices already exist for run {run_id}; choose a new run_id"
        ) from exc

    return FeatureSplits(
        data=frame,
        train_indices=tuple(int(value) for value in train),
        validation_indices=tuple(int(value) for value in validation),
        test_indices=tuple(int(value) for value in test),
        folds=tuple(folds),
        path=split_path,
        warning=small_sample_warning,
    )

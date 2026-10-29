"""Tests for participant splits and single-use held-out access."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.final_evaluation import evaluate_final_on_test
from src.preprocessing.splits import create_feature_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _labelled_students(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [f"S{index:05d}" for index in range(size)],
            "at_risk": np.arange(size) % 2,
        }
    )


def test_splits_are_stratified_disjoint_and_saved_once(tmp_path):
    data = _labelled_students(1000)
    result = create_feature_splits(data, "split_test", tmp_path, seed=13)

    assert len(result.train_indices) == 700
    assert len(result.validation_indices) == 150
    assert len(result.test_indices) == 150
    assert len(result.folds) == 15
    assert set(result.train_indices).isdisjoint(result.validation_indices)
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert set(result.validation_indices).isdisjoint(result.test_indices)
    for indices in (
        result.train_indices,
        result.validation_indices,
        result.test_indices,
    ):
        assert 0.45 <= result.data.iloc[list(indices)]["at_risk"].mean() <= 0.55
    for train_fold, validation_fold in result.folds:
        assert set(train_fold).isdisjoint(validation_fold)
        assert set(train_fold).union(validation_fold) == set(
            result.train_indices + result.validation_indices
        )

    saved = json.loads(result.path.read_text(encoding="utf-8"))
    assert saved["test_indices"] == list(result.test_indices)
    assert len(saved["folds"]) == 15
    with pytest.raises(FileExistsError, match="already exist"):
        create_feature_splits(data, "split_test", tmp_path, seed=13)


def test_small_leaderboard_uses_five_repeats_and_warns(tmp_path):
    with pytest.warns(UserWarning, match="using repeated stratified 5-fold"):
        result = create_feature_splits(_labelled_students(400), "small_split", tmp_path)

    assert len(result.folds) == 25
    assert result.warning is not None


def test_final_evaluation_reads_only_test_rows_once(tmp_path):
    data = _labelled_students(100)
    result = create_feature_splits(data, "final_test", tmp_path)
    observed: list[str] = []

    output = evaluate_final_on_test(
        result.data,
        "final_test",
        lambda test_frame: observed.extend(test_frame["student_id"].tolist())
        or len(test_frame),
        tmp_path,
    )

    assert output == len(result.test_indices)
    assert set(observed) == set(
        result.data.iloc[list(result.test_indices)]["student_id"]
    )
    with pytest.raises(PermissionError, match="already been accessed"):
        evaluate_final_on_test(
            result.data, "final_test", lambda frame: len(frame), tmp_path
        )


def test_persisted_test_positions_have_only_one_reader():
    source_root = PROJECT_ROOT / "src"
    artifact_paths = []
    read_paths = []
    for path in source_root.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        owns_index_artifact = "indices.json" in source
        if owns_index_artifact:
            artifact_paths.append(relative)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                owns_index_artifact
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "read_text"
            ):
                read_paths.append(relative)

    assert set(artifact_paths) == {
        "src/preprocessing/splits.py",
        "src/evaluation/final_evaluation.py",
    }
    assert read_paths == ["src/evaluation/final_evaluation.py"]

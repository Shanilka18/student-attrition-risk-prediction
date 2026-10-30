"""
Tests for Leakage-Safe Splits, Preprocessing, Feature Alignment, and Imbalance Handlers.
Component: Academic Intelligence Framework (IT23164680)
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from backend.src.config import config
from backend.src.data.splits import assert_zero_student_overlap
from backend.src.features.preprocessor import AcademicPreprocessor
from backend.src.features.imbalance import (
    compute_balanced_class_weights,
    compute_pytorch_pos_weight,
    apply_smote_tabular,
    BinaryFocalLoss
)


def test_split_zero_student_overlap():
    """Verify that student lists in data/splits have strictly zero intersection."""
    train_path = config.splits_dir / "train_students.json"
    val_path = config.splits_dir / "val_students.json"
    test_path = config.splits_dir / "test_students.json"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Splits not yet generated on disk. Run make preprocess first.")

    with open(train_path, "r", encoding="utf-8") as f:
        train_sids = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_sids = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_sids = json.load(f)

    # Must raise no exception
    assert_zero_student_overlap(train_sids, val_sids, test_sids)

    total = len(train_sids) + len(val_sids) + len(test_sids)
    assert 0.68 <= len(train_sids) / total <= 0.72
    assert 0.13 <= len(val_sids) / total <= 0.17
    assert 0.13 <= len(test_sids) / total <= 0.17


def test_temporal_split_zero_overlap():
    """Verify that temporal historical and future cohort student sets have zero intersection."""
    trn_path = config.splits_dir / "temporal_train_students.json"
    tst_path = config.splits_dir / "temporal_test_students.json"

    if not (trn_path.exists() and tst_path.exists()):
        pytest.skip("Temporal splits not yet generated.")

    with open(trn_path, "r", encoding="utf-8") as f:
        trn_sids = set(json.load(f))
    with open(tst_path, "r", encoding="utf-8") as f:
        tst_sids = set(json.load(f))

    assert len(trn_sids.intersection(tst_sids)) == 0


def test_preprocessor_no_leakage():
    """Verify preprocessor calculates medians strictly on train and does not alter them during transform."""
    train_df = pd.DataFrame({
        "student_id": ["STU_1", "STU_1"],
        "semester": [1, 2],
        "gpa": [3.0, 3.2],
        "exam_marks": [70.0, 75.0],
        "attendance_pct": [80.0, 85.0],
        "assignment_delay_days": [1.0, 0.5],
        "late_assignments_count": [0, 0]
    })

    test_df = pd.DataFrame({
        "student_id": ["STU_99"],
        "semester": [1],
        "gpa": [np.nan],  # Missing value in test set
        "exam_marks": [20.0],
        "attendance_pct": [30.0],
        "assignment_delay_days": [10.0],
        "late_assignments_count": [4]
    })

    prep = AcademicPreprocessor()
    prep.fit(train_df)

    # Median GPA on train is (3.0 + 3.2)/2 = 3.1
    assert abs(prep.train_medians["gpa"] - 3.1) < 1e-4

    # Transform test set
    transformed_test = prep.transform(test_df)

    # Imputed test GPA must be exactly the train median (3.1), with zero influence from test values
    assert abs(transformed_test["gpa"].iloc[0] - 3.1) < 1e-4
    assert prep.train_medians["gpa"] == 3.1


def test_feature_alignment_across_tabular_and_sequence():
    """Verify sample IDs and binary targets match 1-to-1 between tabular and sequence tensors."""
    tab_path = config.processed_data_dir / "tabular_features.csv"
    seq_path = config.processed_data_dir / "sequence_tensors.pt"

    if not (tab_path.exists() and seq_path.exists()):
        pytest.skip("Processed features not yet built on disk.")

    df_tab = pd.read_csv(tab_path)
    seq_data = torch.load(seq_path, weights_only=False)

    assert len(df_tab) == seq_data["tensors"].shape[0]
    assert list(df_tab["sample_id"]) == seq_data["sample_ids"]
    assert np.array_equal(df_tab["target"].values, seq_data["targets"].numpy())


def test_sequence_mask_correctness():
    """Verify that padding mask strictly matches sequence lengths and zero padding."""
    seq_path = config.processed_data_dir / "sequence_tensors.pt"
    if not seq_path.exists():
        pytest.skip("Sequence tensors not yet generated.")

    seq_data = torch.load(seq_path, weights_only=False)
    tensors = seq_data["tensors"]
    mask = seq_data["padding_mask"]
    lengths = seq_data["seq_lengths"]

    n_samples = tensors.shape[0]
    for i in range(min(n_samples, 200)):
        seq_len = lengths[i].item()
        # Active positions must be True
        assert mask[i, :seq_len].all()
        # Padded positions must be False
        if seq_len < mask.shape[1]:
            assert not mask[i, seq_len:].any()
            # Padded tensor entries must be exactly zero
            assert (tensors[i, seq_len:, :] == 0.0).all()


def test_imbalance_handlers():
    """Verify class weights, SMOTE, and FocalLoss operate correctly."""
    y = np.array([0, 0, 0, 0, 1])
    weights = compute_balanced_class_weights(y)
    assert weights[1] > weights[0]

    pos_w = compute_pytorch_pos_weight(y)
    assert pos_w.item() == 4.0

    # Focal loss
    fl = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    targets = torch.tensor([1, 0])
    loss = fl(logits, targets)
    assert loss.item() > 0.0

    # SMOTE
    X = pd.DataFrame({"f1": [1.0, 1.2, 0.9, 1.1, 5.0, 5.1], "f2": [2.0, 2.1, 1.9, 2.2, 8.0, 8.2]})
    y_imbal = np.array([0, 0, 0, 0, 1, 1])
    X_res, y_res = apply_smote_tabular(X, y_imbal, seed=42, k_neighbors=1)
    assert len(X_res) > len(X)
    assert np.sum(y_res == 1) == np.sum(y_res == 0)

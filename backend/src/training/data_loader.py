"""
Dataset Loader & Grouped Cross-Validation Partitioner for Model Training.
Prepares aligned tabular, static, and sequence partitions strictly according to
student ID splits (no row leakage, frozen test partition untouched).

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from backend.src.config import config

logger = logging.getLogger("AcademicIntelligence.DataLoader")


class TrainingDataLoader:
    """
    Central manager for loading and slicing training/validation data
    across tabular, static, and 3D sequence representations.
    """

    def __init__(self, data_dir: Optional[Path] = None, splits_dir: Optional[Path] = None):
        self.processed_dir = data_dir or config.processed_data_dir
        self.splits_dir = splits_dir or config.splits_dir

        self._tabular_df: Optional[pd.DataFrame] = None
        self._static_df: Optional[pd.DataFrame] = None
        self._seq_payload: Optional[Dict[str, Any]] = None
        self._train_sids: Optional[List[str]] = None
        self._val_sids: Optional[List[str]] = None

    def load_splits(self) -> Tuple[List[str], List[str]]:
        """Loads frozen train and validation student ID lists."""
        if self._train_sids is None:
            with open(self.splits_dir / "train_students.json", "r", encoding="utf-8") as f:
                self._train_sids = json.load(f)
            with open(self.splits_dir / "val_students.json", "r", encoding="utf-8") as f:
                self._val_sids = json.load(f)
            logger.info(
                f"Loaded student split lists: Train={len(self._train_sids)} students, "
                f"Val={len(self._val_sids)} students (Test split deliberately excluded)"
            )
        return self._train_sids, self._val_sids

    def load_tabular_data(self) -> Dict[str, Any]:
        """
        Loads tabular longitudinal features (22 engineered columns).
        Returns partitioned X_train, y_train, X_val, y_val, and metadata.
        """
        train_sids, val_sids = self.load_splits()
        if self._tabular_df is None:
            csv_path = self.processed_dir / "tabular_features.csv"
            self._tabular_df = pd.read_csv(csv_path)

        df = self._tabular_df
        meta_cols = ["sample_id", "student_id", "window_t", "target"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        train_mask = df["student_id"].isin(train_sids)
        val_mask = df["student_id"].isin(val_sids)

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)

        return {
            "X_train": train_df[feature_cols].values.astype(np.float32),
            "y_train": train_df["target"].values.astype(np.int64),
            "student_ids_train": train_df["student_id"].values,
            "sample_ids_train": train_df["sample_id"].values,
            "X_val": val_df[feature_cols].values.astype(np.float32),
            "y_val": val_df["target"].values.astype(np.int64),
            "student_ids_val": val_df["student_id"].values,
            "sample_ids_val": val_df["sample_id"].values,
            "feature_names": feature_cols,
        }

    def load_static_data(self) -> Dict[str, Any]:
        """
        Loads static snapshot comparator features (last_gpa, last_attendance_pct).
        """
        train_sids, val_sids = self.load_splits()
        if self._static_df is None:
            csv_path = self.processed_dir / "static_features.csv"
            self._static_df = pd.read_csv(csv_path)

        df = self._static_df
        feature_cols = ["last_gpa", "last_attendance_pct"]

        train_mask = df["student_id"].isin(train_sids)
        val_mask = df["student_id"].isin(val_sids)

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)

        return {
            "X_train": train_df[feature_cols].values.astype(np.float32),
            "y_train": train_df["target"].values.astype(np.int64),
            "student_ids_train": train_df["student_id"].values,
            "sample_ids_train": train_df["sample_id"].values,
            "X_val": val_df[feature_cols].values.astype(np.float32),
            "y_val": val_df["target"].values.astype(np.int64),
            "student_ids_val": val_df["student_id"].values,
            "sample_ids_val": val_df["sample_id"].values,
            "feature_names": feature_cols,
        }

    def load_sequence_data(self) -> Dict[str, Any]:
        """
        Loads 3D sequence tensors, padding masks, and lengths.
        Returns sliced dict payloads for train and validation partitions.
        """
        train_sids, val_sids = self.load_splits()
        if self._seq_payload is None:
            pt_path = self.processed_dir / "sequence_tensors.pt"
            self._seq_payload = torch.load(pt_path, map_location="cpu", weights_only=False)

        student_ids = np.array(self._seq_payload["student_ids"])
        train_indices = np.where(np.isin(student_ids, train_sids))[0]
        val_indices = np.where(np.isin(student_ids, val_sids))[0]

        t_idx = torch.tensor(train_indices, dtype=torch.long)
        v_idx = torch.tensor(val_indices, dtype=torch.long)

        train_dict = {
            "tensors": self._seq_payload["tensors"][t_idx],
            "padding_mask": self._seq_payload["padding_mask"][t_idx],
            "seq_lengths": self._seq_payload["seq_lengths"][t_idx],
        }
        val_dict = {
            "tensors": self._seq_payload["tensors"][v_idx],
            "padding_mask": self._seq_payload["padding_mask"][v_idx],
            "seq_lengths": self._seq_payload["seq_lengths"][v_idx],
        }

        y_train = self._seq_payload["targets"][t_idx].numpy()
        y_val = self._seq_payload["targets"][v_idx].numpy()

        return {
            "X_train": train_dict,
            "y_train": y_train,
            "student_ids_train": student_ids[train_indices],
            "X_val": val_dict,
            "y_val": y_val,
            "student_ids_val": student_ids[val_indices],
            "feature_names": self._seq_payload["feature_names"],
        }

    def get_data_for_model(self, model_key: str, category: str) -> Dict[str, Any]:
        """
        Routes data loading based on model category:
          Category F -> static snapshot data (2 features)
          Category E -> 3D padded sequence tensors & masks
          Category A, B, C, D -> 22 engineered tabular features
        """
        if category == "F":
            return self.load_static_data()
        elif category == "E":
            return self.load_sequence_data()
        else:
            return self.load_tabular_data()

    def get_train_cv_folds(self, n_splits: int = 3, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Constructs grouped, stratified cross-validation folds strictly within train_students.
        Guarantees zero student overlap between CV train and CV val folds.
        """
        train_sids, _ = self.load_splits()
        tab_data = self.load_tabular_data()
        student_ids_train = tab_data["student_ids_train"]
        y_train = tab_data["y_train"]

        # Student-level outcome mapping for stratification
        unique_sids, first_indices = np.unique(student_ids_train, return_index=True)
        student_labels = y_train[first_indices]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []

        for trn_sid_idx, val_sid_idx in skf.split(unique_sids, student_labels):
            fold_trn_sids = set(unique_sids[trn_sid_idx])
            fold_val_sids = set(unique_sids[val_sid_idx])

            sample_trn_mask = np.isin(student_ids_train, list(fold_trn_sids))
            sample_val_mask = np.isin(student_ids_train, list(fold_val_sids))

            folds.append((
                np.where(sample_trn_mask)[0],
                np.where(sample_val_mask)[0]
            ))

        return folds

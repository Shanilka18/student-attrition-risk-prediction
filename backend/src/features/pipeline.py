"""
Master Feature Pipeline: Zero-Leakage Longitudinal Data Preparation.
Orchestrates grouped stratified splits, temporal validation partitions,
AcademicPreprocessor fitting (STRICTLY on train fold), tabular trend feature extraction,
static comparator snapshots, and 3D sequence tensors.

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

from backend.src.config import config
from backend.src.data.splits import create_stratified_group_splits, create_temporal_split
from backend.src.features.preprocessor import AcademicPreprocessor
from backend.src.features.builder import (
    build_tabular_engineered_features,
    build_static_snapshot_features,
    build_sequence_tensors,
)
from backend.src.evaluation.static_baseline import evaluate_static_snapshot_baseline

logger = logging.getLogger("AcademicIntelligence.FeaturePipeline")


def run_feature_pipeline(
    df_long: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    df_task: pd.DataFrame,
    seed: Optional[int] = None,
    max_window: int = 6,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_cv_folds: int = 5,
    evaluate_static_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Executes the unified zero-leakage feature generation pipeline.
    Uses active config paths (can be run-scoped/isolated or canonical).
    """
    active_seed = seed if seed is not None else config.global_seed
    splits_dir = config.splits_dir
    models_dir = config.models_dir
    processed_dir = config.processed_data_dir
    metrics_dir = config.metrics_dir

    for d in [splits_dir, models_dir, processed_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate Leakage-Safe Grouped Splits
    logger.info("Step 1: Generating student-grouped stratified splits...")
    split_stats = create_stratified_group_splits(
        df_outcomes=df_outcomes,
        output_dir=splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_cv_folds=n_cv_folds,
        seed=active_seed,
    )

    # Attempt temporal split, with graceful fallback if cohort structure is absent/insufficient
    temporal_stats = None
    try:
        if "cohort_year" in df_outcomes.columns and df_outcomes["cohort_year"].nunique() >= 2:
            years = sorted(df_outcomes["cohort_year"].unique())
            train_years = tuple(years[:-1])
            test_year = int(years[-1])
            temporal_stats = create_temporal_split(
                df_outcomes, splits_dir, train_years=train_years, test_year=test_year
            )
        else:
            logger.info("Temporal split skipped: single cohort or cohort_year column unavailable. Providing fallback references.")
            with open(splits_dir / "train_students.json", "r", encoding="utf-8") as f:
                tr_s = json.load(f)
            with open(splits_dir / "test_students.json", "r", encoding="utf-8") as f:
                te_s = json.load(f)
            with open(splits_dir / "temporal_train_students.json", "w", encoding="utf-8") as f:
                json.dump(tr_s, f, indent=2)
            with open(splits_dir / "temporal_test_students.json", "w", encoding="utf-8") as f:
                json.dump(te_s, f, indent=2)
    except Exception as e:
        logger.warning(f"Temporal split fallback activated due to: {e}")
        with open(splits_dir / "train_students.json", "r", encoding="utf-8") as f:
            tr_s = json.load(f)
        with open(splits_dir / "test_students.json", "r", encoding="utf-8") as f:
            te_s = json.load(f)
        with open(splits_dir / "temporal_train_students.json", "w", encoding="utf-8") as f:
            json.dump(tr_s, f, indent=2)
        with open(splits_dir / "temporal_test_students.json", "w", encoding="utf-8") as f:
            json.dump(te_s, f, indent=2)

    with open(splits_dir / "train_students.json", "r", encoding="utf-8") as f:
        train_sids = json.load(f)
    with open(splits_dir / "test_students.json", "r", encoding="utf-8") as f:
        test_sids = json.load(f)

    # 2. Fit Preprocessor STRICTLY on Training Students
    logger.info("Step 2: Fitting AcademicPreprocessor on training partition records...")
    df_train_long = df_long[df_long["student_id"].isin(train_sids)]
    preprocessor = AcademicPreprocessor()
    preprocessor.fit(df_train_long)
    prep_path = models_dir / "preprocessor.joblib"
    preprocessor.save(prep_path)
    logger.info(f"Saved fitted preprocessor to: {prep_path}")

    # Transform all records using fitted preprocessor
    df_clean_long = preprocessor.transform(df_long)

    # 3. Build Tabular Engineered Features
    logger.info("Step 3: Extracting longitudinal trend features (Tabular)...")
    df_tabular = build_tabular_engineered_features(df_clean_long, df_task)
    tab_csv_path = processed_dir / "tabular_features.csv"
    df_tabular.to_csv(tab_csv_path, index=False)
    try:
        df_tabular.to_parquet(processed_dir / "tabular_features.parquet", index=False)
    except Exception:
        pass
    logger.info(f"Saved tabular features to: {tab_csv_path} (Shape: {df_tabular.shape})")

    # 4. Build Static Snapshot Features
    logger.info("Step 4: Building static snapshot baseline features...")
    df_static = build_static_snapshot_features(df_tabular)
    static_csv_path = processed_dir / "static_features.csv"
    df_static.to_csv(static_csv_path, index=False)
    logger.info(f"Saved static snapshot features to: {static_csv_path}")

    # 5. Build Sequence Tensors for PyTorch Models
    logger.info("Step 5: Building 3D padded sequence tensors and masks...")
    effective_max_window = min(max_window, int(df_clean_long["semester"].max()))
    seq_payload = build_sequence_tensors(
        df_clean_long, df_task, preprocessor, max_window=effective_max_window
    )
    seq_path = processed_dir / "sequence_tensors.pt"
    torch.save(seq_payload, seq_path)
    logger.info(f"Saved sequence tensors to: {seq_path} (Shape: {seq_payload['tensors'].shape})")

    # 6. Static Snapshot Baseline Benchmark
    static_results = None
    if evaluate_static_baseline:
        logger.info("Step 6: Running static snapshot baseline benchmark...")
        try:
            static_results = evaluate_static_snapshot_baseline(
                df_static,
                train_sids,
                test_sids,
                metrics_dir / "static_snapshot_benchmark.json",
                seed=active_seed,
            )
        except Exception as e:
            logger.warning(f"Static baseline benchmark skipped: {e}")

    return {
        "split_stats": split_stats,
        "temporal_stats": temporal_stats,
        "tabular_shape": df_tabular.shape,
        "sequence_tensor_shape": list(seq_payload["tensors"].shape),
        "static_results": static_results,
    }

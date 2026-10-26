"""Data Splitting and Leakage Prevention Module.

Implements:
- Stratified Train / Validation / Test partitioning (70% / 15% / 15%) with fixed seed.
- Stratified K-Fold and Repeated Stratified K-Fold cross-validation generators.
- Strict anti-leakage runtime guard: test set access is sealed and raises an exception
  unless explicitly unlocked by final evaluation routines with allow_test=True.
"""

from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("data.splits")

PROCESSED_DIR = Path("data/processed")


class SealedTestAccessError(RuntimeError):
    """Exception raised when unauthorized code attempts to load the sealed holdout test set."""
    pass


def create_stratified_splits(
    df: pd.DataFrame,
    stratify_col: str = "risk_category",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into stratified train, validation, and holdout test splits.

    Args:
        df: Input DataFrame containing features and stratification column.
        stratify_col: Column name to balance across splits (e.g. 'risk_category' or 'attrition_label').
        train_size: Proportion allocated to training set (default 0.70).
        val_size: Proportion allocated to validation set (default 0.15).
        test_size: Proportion allocated to sealed test set (default 0.15).
        seed: Random seed for exact reproducibility.
        output_dir: Directory where partitioned CSV files will be saved.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), "Split proportions must sum to 1.0"
    set_seed(seed)

    logger.info(
        "Creating stratified splits (%.0f%% train, %.0f%% val, %.0f%% test) on '%s' (seed=%d)...",
        train_size * 100, val_size * 100, test_size * 100, stratify_col, seed,
    )

    stratify_target = df[stratify_col] if stratify_col in df.columns else None

    # Step 1: Separate holdout test set (e.g. 15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_target,
        random_state=seed,
        shuffle=True,
    )

    # Step 2: Separate train and validation sets from remaining pool
    relative_val_size = val_size / (train_size + val_size)
    stratify_train_val = train_val_df[stratify_col] if stratify_col in train_val_df.columns else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=stratify_train_val,
        random_state=seed,
        shuffle=True,
    )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_path = out_path / "train.csv"
        val_path = out_path / "val.csv"
        test_path = out_path / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Write a lock metadata file next to test.csv
        lock_meta = out_path / "test.lock"
        with open(lock_meta, "w", encoding="utf-8") as f:
            f.write("SEALED TEST SET - Touch only once during final evaluation of the shortlisted model.\n")

        logger.info(
            "Splits persisted to %s (Train: %d, Val: %d, Test: %d)",
            out_path, len(train_df), len(val_df), len(test_df),
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_split(
    split_name: str,
    processed_dir: str = "data/processed",
    allow_test: bool = False,
) -> pd.DataFrame:
    """Load a specific partitioned split with mandatory anti-leakage guards.

    Args:
        split_name: Name of split to load ('train', 'val', or 'test').
        processed_dir: Path to directory containing processed splits.
        allow_test: Explicit security flag. Must be set to True to load 'test'.

    Returns:
        pd.DataFrame: Loaded dataset split.

    Raises:
        SealedTestAccessError: If attempting to load 'test' while allow_test is False.
        FileNotFoundError: If the split file does not exist on disk.
    """
    split_clean = split_name.lower().strip()
    if split_clean == "test" and not allow_test:
        raise SealedTestAccessError(
            "TEST SET ACCESS VIOLATION: The holdout test set is sealed to prevent data leakage. "
            "It must not be loaded during feature selection, model training, or hyperparameter tuning. "
            "To access the test set for final model evaluation only, pass allow_test=True explicitly."
        )

    file_map = {
        "train": "train.csv",
        "val": "val.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }
    if split_clean not in file_map:
        raise ValueError(f"Unknown split name '{split_name}'. Valid options: 'train', 'val', 'test'")

    path = Path(processed_dir) / file_map[split_clean]
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run create_stratified_splits first.")

    logger.info("Loading '%s' split from %s (allow_test=%s)", split_clean, path, allow_test)
    return pd.read_csv(path)


def get_stratified_cv(
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> StratifiedKFold:
    """Return configured StratifiedKFold or RepeatedStratifiedKFold generator.

    Args:
        n_splits: Number of cross-validation folds. Default 5.
        n_repeats: Number of repetitions. If 1, returns StratifiedKFold; if >1, RepeatedStratifiedKFold.
        seed: Random state.

    Returns:
        Stratified cross-validation splitter.
    """
    if n_repeats > 1:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

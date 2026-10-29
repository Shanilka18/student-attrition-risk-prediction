"""Participant-level grouped stratified splitting with sealed test set enforcement.

Partitions:
- 70% Train
- 15% Validation
- 15% Test (Strictly sealed until Phase 4 finalist benchmark)

Guarantees:
1. Group-aware separation: All rows for a participant reside strictly within one split.
2. Stratification: Attrition class balance is preserved identically across train, val, and test.
3. Serialization of indices to data/processed/split_ids.json.
4. Sealed test set enforcer raising PermissionError on premature access.
5. In-fold 5-fold x 3-repeat StratifiedGroupKFold cross-validation generator.
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.paths import get_data_dir, ensure_dir
from src.utils.logging_config import setup_logger

logger = setup_logger("split")


class SealedTestAccessViolation(PermissionError):
    """Raised when an unauthorized routine attempts to access the sealed test set."""
    pass


def create_grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "participant_id",
    stratify_col: str = "attrition",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Partition dataset into Train, Validation, and Test splits with grouping and stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset to split.
    group_col : str
        Group identifier column (default 'participant_id').
    stratify_col : str
        Target column for stratification (default 'attrition').
    train_frac : float
        Proportion for training set (default 0.70).
    val_frac : float
        Proportion for validation set (default 0.15).
    test_frac : float
        Proportion for held-out test set (default 0.15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary containing index lists for 'train', 'val', and 'test'.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-5, "Split fractions must sum to 1.0"
    logger.info(
        f"Partitioning {len(df)} rows into Train ({train_frac*100:.0f}%), "
        f"Val ({val_frac*100:.0f}%), Test ({test_frac*100:.0f}%) using seed={seed}..."
    )

    rng = np.random.default_rng(seed)

    # 1. Aggregate to unique participant level
    grouped = df.groupby(group_col).agg({
        stratify_col: "first",
    }).reset_index()

    n_groups = len(grouped)
    y_groups = grouped[stratify_col].values
    groups = grouped[group_col].values

    # 2. Stratified split for test set (15%)
    n_test_splits = int(round(1.0 / test_frac))  # ~7 splits for ~15% test
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)

    train_val_group_idx, test_group_idx = next(sgkf_test.split(groups, y_groups, groups))
    test_participants = set(groups[test_group_idx])

    # 3. Stratified split of remaining 85% into Train (70/85 = 82.35%) and Val (15/85 = 17.65%)
    train_val_groups = groups[train_val_group_idx]
    train_val_y = y_groups[train_val_group_idx]

    val_relative_frac = val_frac / (train_frac + val_frac)  # ~0.1765
    n_val_splits = int(round(1.0 / val_relative_frac))  # ~6 splits
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed + 1)

    train_sub_idx, val_sub_idx = next(sgkf_val.split(train_val_groups, train_val_y, train_val_groups))
    train_participants = set(train_val_groups[train_sub_idx])
    val_participants = set(train_val_groups[val_sub_idx])

    # Verify zero participant leakage across sets
    assert train_participants.isdisjoint(val_participants), "Train and Val participant overlap detected!"
    assert train_participants.isdisjoint(test_participants), "Train and Test participant overlap detected!"
    assert val_participants.isdisjoint(test_participants), "Val and Test participant overlap detected!"

    # 4. Map back to original row indices
    train_indices = df[df[group_col].isin(train_participants)].index.tolist()
    val_indices = df[df[group_col].isin(val_participants)].index.tolist()
    test_indices = df[df[group_col].isin(test_participants)].index.tolist()

    # Log stratification balance
    for name, idxs in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        subset_y = df.loc[idxs, stratify_col]
        prev = subset_y.mean() * 100.0
        logger.info(f"[{name} Split] Rows: {len(idxs)} ({len(idxs)/len(df)*100:.1f}%) | Attrition: {prev:.2f}%")

    split_dict = {
        "metadata": {
            "seed": seed,
            "total_rows": len(df),
            "total_participants": n_groups,
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
            "test_rows": len(test_indices),
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    return split_dict


def get_cv_folds(
    df: pd.DataFrame,
    indices: List[int],
    group_col: str = "participant_id",
    stratify_col: str = "attrition",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate repeated stratified group cross-validation folds within specified subset.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        (train_fold_indices, val_fold_indices)
    """
    sub_df = df.iloc[indices].reset_index(drop=True)
    groups = sub_df[group_col].values
    y = sub_df[stratify_col].values

    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat * 100)
        for train_fold_idx, val_fold_idx in sgkf.split(sub_df, y, groups):
            # Map back to original index positions
            train_orig = np.array(indices)[train_fold_idx]
            val_orig = np.array(indices)[val_fold_idx]
            yield train_orig, val_orig


def get_sealed_test_set(
    allow_test_access: bool = False,
    caller_description: str = "Unspecified",
) -> pd.DataFrame:
    """Access the sealed held-out test partition.

    Enforces project hard rule: Held-out test set is evaluated ONCE only
    for finalist models at the conclusion of the research benchmark.

    Parameters
    ----------
    allow_test_access : bool
        Explicit boolean authorization flag. Must be set to True.
    caller_description : str
        Description of calling script or module for security auditing.

    Returns
    -------
    pd.DataFrame
        Held-out test set.

    Raises
    ------
    SealedTestAccessViolation
        If allow_test_access is False.
    """
    if not allow_test_access:
        msg = (
            "SECURITY VIOLATION (DATA LEAKAGE PREVENTION): "
            "Attempted unauthorized access to sealed test set! "
            f"Caller: '{caller_description}'. "
            "Under PROJECT_CONTEXT.md hard rules, the test set may ONLY be opened "
            "ONCE during final benchmark evaluation. Set allow_test_access=True "
            "only from src/evaluation/final_evaluation.py."
        )
        logger.error(msg)
        raise SealedTestAccessViolation(msg)

    processed_dir = get_data_dir("processed")
    test_parquet = processed_dir / "test.parquet"
    test_csv = processed_dir / "test.csv"
    if test_parquet.is_file():
        logger.warning(
            f"!!! [TEST SET ACCESS GRANTED] !!! Opening sealed test set (parquet) for: {caller_description}"
        )
        return pd.read_parquet(test_parquet)
    elif test_csv.is_file():
        logger.warning(
            f"!!! [TEST SET ACCESS GRANTED] !!! Opening sealed test set (csv) for: {caller_description}"
        )
        return pd.read_csv(test_csv)
    else:
        raise FileNotFoundError(f"Sealed test file not found in {processed_dir}. Run split.py first.")


def load_split_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val datasets (and test with sealed enforcement)."""
    processed_dir = get_data_dir("processed")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    # Test set must be accessed via get_sealed_test_set
    return train_df, val_df, None


def run_splitting_pipeline() -> Tuple[Path, Dict[str, Any]]:
    """Execute data splitting pipeline and serialize split indices and partitions."""
    interim_dir = get_data_dir("interim")
    processed_dir = get_data_dir("processed")

    clean_path = interim_dir / "clean.csv"
    if not clean_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_path}. Run cleaner.py first.")

    df = pd.read_csv(clean_path)
    split_dict = create_grouped_stratified_split(df, seed=42)

    # Save split IDs
    split_ids_path = processed_dir / "split_ids.json"
    with open(split_ids_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    # Save physical partitions in CSV and Parquet
    train_sub = df.iloc[split_dict["train_indices"]]
    val_sub = df.iloc[split_dict["val_indices"]]
    test_sub = df.iloc[split_dict["test_indices"]]

    train_sub.to_csv(processed_dir / "train.csv", index=False)
    val_sub.to_csv(processed_dir / "val.csv", index=False)
    test_sub.to_csv(processed_dir / "test.csv", index=False)

    try:
        train_sub.to_parquet(processed_dir / "train.parquet", index=False)
        val_sub.to_parquet(processed_dir / "val.parquet", index=False)
        test_sub.to_parquet(processed_dir / "test.parquet", index=False)
    except Exception as exc:
        logger.warning(f"Parquet engine unavailable ({exc}); saved as CSV.")

    logger.info(f"Split indices saved to: {split_ids_path}")
    logger.info(f"Partitions (train, val, test) created in: {processed_dir}")
    return split_ids_path, split_dict


if __name__ == "__main__":
    split_p, s_dict = run_splitting_pipeline()
    print(f"\n[SUCCESS] Splitting pipeline completed:")
    print(f"  Split IDs JSON: {split_p}")
    print(f"  Train samples:  {s_dict['metadata']['train_rows']}")
    print(f"  Val samples:    {s_dict['metadata']['val_rows']}")
    print(f"  Test samples:   {s_dict['metadata']['test_rows']} (SEALED)\n")

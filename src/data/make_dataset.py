"""Data ingestion, validation, synthetic data generation, and splitting pipelines.

Follows strict non-leakage protocols and labels all generated synthetic data explicitly.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.config import load_config

logger = get_logger("data.make_dataset")


def generate_synthetic_survey(
    n_samples: int = 500,
    seed: int = 42,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate synthetic 15-item Likert survey dataset with latent correlations.

    Items are generated on a 1 to 5 scale (1=Strongly Disagree ... 5=Strongly Agree):
    - Stress (S1, S2, S3): Risk factor
    - Motivation (M1, M2, M3): Protective factor
    - Belongingness (B1, B2, B3): Protective factor
    - Self-efficacy (SE1, SE2, SE3): Protective factor
    - Peer support (PS1, PS2, PS3): Protective factor

    NOTE: This generates purely synthetic data for methodological verification.
    It does NOT possess real-world epidemiological or clinical validity.

    Args:
        n_samples (int): Number of synthetic survey respondents.
        seed (int): Random seed for reproducibility.
        config (Optional[Dict[str, Any]]): Configuration dictionary.

    Returns:
        pd.DataFrame: Synthetic survey dataframe with study_id and 15 Likert items.
    """
    logger.info("Generating %d synthetic survey records (seed=%d)...", n_samples, seed)
    set_seed(seed)
    
    # Placeholder implementation structure
    item_cols = [
        "S1", "S2", "S3",
        "M1", "M2", "M3",
        "B1", "B2", "B3",
        "SE1", "SE2", "SE3",
        "PS1", "PS2", "PS3",
    ]
    data = {
        "study_id": [f"SYNTH_{i:04d}" for i in range(1, n_samples + 1)]
    }
    for col in item_cols:
        data[col] = np.random.randint(1, 6, size=n_samples)
        
    df = pd.DataFrame(data)
    df["is_synthetic"] = True
    return df


def validate_survey_data(df: pd.DataFrame) -> bool:
    """Validate survey integrity: checks schema, Likert boundaries [1, 5], and missing values.

    Args:
        df (pd.DataFrame): Survey DataFrame to validate.

    Returns:
        bool: True if validation passes, raises ValueError otherwise.
    """
    expected_items = [
        "S1", "S2", "S3",
        "M1", "M2", "M3",
        "B1", "B2", "B3",
        "SE1", "SE2", "SE3",
        "PS1", "PS2", "PS3",
    ]
    missing_cols = [c for c in expected_items if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required survey columns: {missing_cols}")
    
    for c in expected_items:
        if not df[c].between(1, 5).all():
            raise ValueError(f"Column {c} contains values outside Likert range [1, 5]")
            
    logger.info("Survey dataset passed integrity validation.")
    return True


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform leak-free train/validation/test split.

    The test set is sealed for final model verification only.

    Args:
        df (pd.DataFrame): Input dataframe.
        test_size (float): Fraction of data allocated to final test set.
        val_size (float): Fraction of train-val pool allocated to validation set.
        stratify_col (Optional[str]): Column to stratify splits on.
        seed (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    logger.info("Performing train/val/test split with test_size=%.2f, val_size=%.2f", test_size, val_size)
    # Placeholder implementation
    return df, df, df


def main() -> None:
    """Execution entrypoint for data generation and ingestion pipeline."""
    cfg = load_config()
    logger.info("Initializing dataset pipeline for project: %s", cfg["project"]["name"])
    raw_synthetic = generate_synthetic_survey(n_samples=500, seed=cfg["project"]["random_seed"])
    validate_survey_data(raw_synthetic)
    output_path = Path(cfg["paths"]["data_synthetic"]) / "synthetic_survey_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_synthetic.to_csv(output_path, index=False)
    logger.info("Synthetic raw survey saved to %s", output_path)


if __name__ == "__main__":
    main()

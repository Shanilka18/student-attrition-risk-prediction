"""
Configuration and Reproducibility Manager.
Academic Intelligence Framework (IT23164680)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import os
import random
import numpy as np
import torch
import yaml

# Configure default logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AcademicIntelligence.Config")


def get_project_root() -> Path:
    """Returns the absolute root directory of the research project."""
    # backend/src/config.py -> parents[2] is project root
    return Path(__file__).resolve().parents[2]


def load_raw_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def detect_compute_device(requested_device: str = "auto") -> torch.device:
    """
    Detects and logs available compute acceleration (CUDA / MPS / CPU).
    Falls back gracefully if hardware acceleration is unavailable.
    """
    req = requested_device.lower()
    device_str = "cpu"

    if req == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Compute Acceleration Detected: NVIDIA GPU ({device_name})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            logger.info("Compute Acceleration Detected: Apple Silicon MPS")
        else:
            device_str = "cpu"
            logger.info("Compute Device: Standard CPU (No hardware accelerator detected)")
    elif req == "cuda":
        if torch.cuda.is_available():
            device_str = "cuda"
            logger.info(f"Compute Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
    else:
        device_str = "cpu"
        logger.info("Compute Device set to CPU.")

    return torch.device(device_str)


def set_global_seed(seed: int = 42) -> None:
    """
    Enforces deterministic reproducibility across Python, NumPy, and PyTorch.
    Ensures non-negotiable rule #1: One global seed in config, fixed seeds everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic global seed set to: {seed}")


class AppConfig:
    """Central typed configuration wrapper."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._raw = config_dict or load_raw_config()
        self.root_dir = get_project_root()
        
        # Project metadata
        proj = self._raw.get("project", {})
        self.project_name: str = proj.get("name", "academic-intelligence-framework")
        self.project_id: str = proj.get("id", "KND_CDAP_IT_05")
        self.version: str = proj.get("version", "1.0.0")
        self.global_seed: int = int(proj.get("global_seed", 42))
        
        # Paths resolved as absolute Path objects
        p = self._raw.get("paths", {})
        self.raw_data_dir: Path = self.root_dir / p.get("raw_data_dir", "data/raw")
        self.synthetic_data_dir: Path = self.root_dir / p.get("synthetic_data_dir", "data/synthetic")
        self.processed_data_dir: Path = self.root_dir / p.get("processed_data_dir", "data/processed")
        self.splits_dir: Path = self.root_dir / p.get("splits_dir", "data/splits")
        self.uploads_dir: Path = self.root_dir / p.get("uploads_dir", "data/uploads")
        self.models_dir: Path = self.root_dir / p.get("models_dir", "artifacts/models")
        self.metrics_dir: Path = self.root_dir / p.get("metrics_dir", "artifacts/metrics")
        self.leaderboard_dir: Path = self.root_dir / p.get("leaderboard_dir", "artifacts/leaderboard")
        self.plots_dir: Path = self.root_dir / p.get("plots_dir", "artifacts/plots")
        self.explanations_dir: Path = self.root_dir / p.get("explanations_dir", "artifacts/explanations")
        self.docs_dir: Path = self.root_dir / p.get("docs_dir", "docs")
        self.schema_dir: Path = self.root_dir / p.get("schema_dir", "data/schema")

        # Data & Sequences
        d = self._raw.get("data", {})
        self.cohort_size: int = d.get("cohort_size", 1000)
        self.num_semesters: int = d.get("num_semesters", 4)
        self.min_semesters: int = d.get("min_semesters", 2)
        self.target_attrition_rate: float = d.get("target_attrition_rate", 0.20)
        self.noise_level: float = d.get("noise_level", 0.18)
        self.group_column: str = d.get("group_column", "participant_id")
        self.target_column: str = d.get("target_column", "attrition_label")
        self.sequence_features: List[str] = d.get("sequence_features", [
            "gpa", "exam_mark_avg", "attendance_rate", "submission_delay_avg_days"
        ])

        # Splits
        s = self._raw.get("splits", {})
        self.test_ratio: float = s.get("test_ratio", 0.20)
        self.val_ratio: float = s.get("val_ratio", 0.20)
        self.n_cv_splits: int = s.get("n_cv_splits", 5)
        self.split_strategy: str = s.get("strategy", "stratified_group")

        # Compute Device & Seeds
        self.device = detect_compute_device(proj.get("device", "auto"))
        set_global_seed(self.global_seed)


# Singleton instance
config = AppConfig()

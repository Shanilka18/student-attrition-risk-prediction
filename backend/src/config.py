"""
Configuration and Reproducibility Manager.
Academic Intelligence Framework (IT23164680)
"""

from contextlib import contextmanager
import json
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
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
        self.uploads_dir: Path = self.root_dir / p.get("uploads_dir", "data/uploads")
        self.docs_dir: Path = self.root_dir / p.get("docs_dir", "docs")
        self.schema_dir: Path = self.root_dir / p.get("schema_dir", "data/schema")

        self._thread_local = threading.local()
        self._canonical_models_dir: Path = self.root_dir / p.get("models_dir", "artifacts/models")
        self._canonical_metrics_dir: Path = self.root_dir / p.get("metrics_dir", "artifacts/metrics")
        self._canonical_leaderboard_dir: Path = self.root_dir / p.get("leaderboard_dir", "artifacts/leaderboard")
        self._canonical_plots_dir: Path = self.root_dir / p.get("plots_dir", "artifacts/plots")
        self._canonical_explanations_dir: Path = self.root_dir / p.get("explanations_dir", "artifacts/explanations")
        self._canonical_processed_data_dir: Path = self.root_dir / p.get("processed_data_dir", "data/processed")
        self._canonical_splits_dir: Path = self.root_dir / p.get("splits_dir", "data/splits")

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

    @property
    def models_dir(self) -> Path:
        return getattr(self._thread_local, "models_dir", self._canonical_models_dir)

    @models_dir.setter
    def models_dir(self, val: Path) -> None:
        self._thread_local.models_dir = val

    @property
    def metrics_dir(self) -> Path:
        return getattr(self._thread_local, "metrics_dir", self._canonical_metrics_dir)

    @metrics_dir.setter
    def metrics_dir(self, val: Path) -> None:
        self._thread_local.metrics_dir = val

    @property
    def leaderboard_dir(self) -> Path:
        return getattr(self._thread_local, "leaderboard_dir", self._canonical_leaderboard_dir)

    @leaderboard_dir.setter
    def leaderboard_dir(self, val: Path) -> None:
        self._thread_local.leaderboard_dir = val

    @property
    def plots_dir(self) -> Path:
        return getattr(self._thread_local, "plots_dir", self._canonical_plots_dir)

    @plots_dir.setter
    def plots_dir(self, val: Path) -> None:
        self._thread_local.plots_dir = val

    @property
    def explanations_dir(self) -> Path:
        return getattr(self._thread_local, "explanations_dir", self._canonical_explanations_dir)

    @explanations_dir.setter
    def explanations_dir(self, val: Path) -> None:
        self._thread_local.explanations_dir = val

    @property
    def processed_data_dir(self) -> Path:
        return getattr(self._thread_local, "processed_data_dir", self._canonical_processed_data_dir)

    @processed_data_dir.setter
    def processed_data_dir(self, val: Path) -> None:
        self._thread_local.processed_data_dir = val

    @property
    def splits_dir(self) -> Path:
        return getattr(self._thread_local, "splits_dir", self._canonical_splits_dir)

    @splits_dir.setter
    def splits_dir(self, val: Path) -> None:
        self._thread_local.splits_dir = val

    @property
    def is_isolated(self) -> bool:
        """Returns True if the current thread/execution is redirected to an isolated user run."""
        return getattr(self._thread_local, "is_isolated", False)

    @property
    def active_job_id(self) -> Optional[str]:
        """Returns active isolated job ID if currently running in isolation."""
        return getattr(self._thread_local, "active_job_id", None)

    @contextmanager
    def isolated_run(self, job_id: str) -> Generator[Path, None, None]:
        """
        Context manager redirecting all output directories to an isolated path:
        artifacts/user_dataset/<job_id>/
        Thread-safe: only the calling thread is redirected to the isolated workspace.
        Guarantees synthetic thesis benchmarks and production models are never overwritten.
        """
        user_base_dir = self.root_dir / "artifacts" / "user_dataset" / job_id
        
        # Override paths to isolated job directory for this thread
        self._thread_local.models_dir = user_base_dir / "models"
        self._thread_local.metrics_dir = user_base_dir / "metrics"
        self._thread_local.leaderboard_dir = user_base_dir / "leaderboard"
        self._thread_local.plots_dir = user_base_dir / "plots"
        self._thread_local.explanations_dir = user_base_dir / "explanations"
        self._thread_local.processed_data_dir = user_base_dir / "processed"
        self._thread_local.splits_dir = user_base_dir / "splits"
        self._thread_local.is_isolated = True
        self._thread_local.active_job_id = job_id
        
        for d in [
            self._thread_local.models_dir,
            self._thread_local.metrics_dir,
            self._thread_local.leaderboard_dir,
            self._thread_local.plots_dir,
            self._thread_local.explanations_dir,
            self._thread_local.processed_data_dir,
            self._thread_local.splits_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Entered isolated run context for job_id={job_id}: {user_base_dir}")

        try:
            yield user_base_dir
        finally:
            self._thread_local.models_dir = self._canonical_models_dir
            self._thread_local.metrics_dir = self._canonical_metrics_dir
            self._thread_local.leaderboard_dir = self._canonical_leaderboard_dir
            self._thread_local.plots_dir = self._canonical_plots_dir
            self._thread_local.explanations_dir = self._canonical_explanations_dir
            self._thread_local.processed_data_dir = self._canonical_processed_data_dir
            self._thread_local.splits_dir = self._canonical_splits_dir
            self._thread_local.is_isolated = False
            self._thread_local.active_job_id = None
            logger.info(f"Exited isolated run context. Restored canonical paths.")

    def save_latest_user_dataset_metadata(self, meta: Dict[str, Any]) -> Path:
        """Saves artifacts/user_dataset/latest.json summary after a successful retrain job."""
        out_path = self.root_dir / "artifacts" / "user_dataset" / "latest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved latest user dataset metadata to: {out_path}")
        return out_path

    @property
    def active_run_path(self) -> Path:
        return self.root_dir / "artifacts" / "active_run.json"

    def get_active_run(self) -> Dict[str, Any]:
        """Returns metadata for the currently active scoring model run (REAL or SYNTHETIC)."""
        if self.active_run_path.exists():
            try:
                with open(self.active_run_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading active_run.json: {e}")
        return {"data_mode": "SYNTHETIC", "run_id": None}

    def set_active_run(self, run_meta: Dict[str, Any]) -> None:
        """Sets artifacts/active_run.json pointer to activate a retrained model run."""
        self.active_run_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.active_run_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        logger.info(f"Updated active scoring run pointer to: {run_meta.get('run_id')} ({run_meta.get('champion')})")

    def clear_active_run(self) -> None:
        """Rolls back to canonical thesis synthetic benchmark by deleting active_run.json."""
        if self.active_run_path.exists():
            try:
                self.active_run_path.unlink()
                logger.info("Cleared active_run.json. Rolled back to canonical SYNTHETIC thesis benchmark.")
            except Exception as e:
                logger.warning(f"Failed to delete active_run.json: {e}")


# Singleton instance
config = AppConfig()

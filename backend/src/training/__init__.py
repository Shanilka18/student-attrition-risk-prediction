"""
Training and Validation Engine Package.
Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

from backend.src.training.calibration import ProbabilityCalibrator
from backend.src.training.data_loader import TrainingDataLoader
from backend.src.training.engine import train_and_validate_model
from backend.src.training.metrics import compute_validation_metrics
from backend.src.training.orchestrator import run_training_pipeline
from backend.src.training.threshold import optimize_decision_threshold
from backend.src.training.tuner import tune_model_hyperparameters

__all__ = [
    "run_training_pipeline",
    "train_and_validate_model",
    "TrainingDataLoader",
    "ProbabilityCalibrator",
    "optimize_decision_threshold",
    "compute_validation_metrics",
    "tune_model_hyperparameters",
]

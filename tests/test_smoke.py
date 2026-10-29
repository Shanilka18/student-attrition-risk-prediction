"""Smoke tests for the ERI Research Repository.

Verifies:
1. Mutually compatible imports for all 22 required packages.
2. Global seed determinism across NumPy and PyTorch.
3. Integrity and schema compliance of configs (config.yaml, schema.yaml, leaderboard_weights.yaml).
4. Dynamic project directory paths and folder existence.
5. Utility functions (Timer, Logger, Path helpers).
"""

import importlib
import sys

import pytest


# Verify python version is 3.11
def test_python_version():
    """Verify runtime is Python 3.11."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (major, minor) == (3, 11), f"Expected Python 3.11, found {major}.{minor}"


# 1. Package Import Verification
@pytest.mark.parametrize(
    "pkg_name,module_name",
    [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
        ("torch", "torch"),
        ("pytorch-tabnet", "pytorch_tabnet"),
        ("shap", "shap"),
        ("econml", "econml"),
        ("dowhy", "dowhy"),
        ("statsmodels", "statsmodels.api"),
        ("optuna", "optuna"),
        ("joblib", "joblib"),
        ("pytest", "pytest"),
        ("openpyxl", "openpyxl"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-multipart", "multipart"),
        ("pydantic", "pydantic"),
        ("matplotlib", "matplotlib.pyplot"),
        ("pyyaml", "yaml"),
    ],
)
def test_package_imports(pkg_name, module_name):
    """Verify each of the 22 required libraries imports without error."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Failed to import package '{pkg_name}' ({module_name}): {exc}")


# 2. Seed Determinism & Reproducibility
def test_seed_determinism():
    """Verify set_seed ensures deterministic sampling across libraries."""
    import numpy as np
    import torch

    from src.utils.seed import set_seed

    set_seed(42)
    val_np_1 = np.random.uniform(0, 1, size=(5,))
    val_torch_1 = torch.rand(5)

    set_seed(42)
    val_np_2 = np.random.uniform(0, 1, size=(5,))
    val_torch_2 = torch.rand(5)

    np.testing.assert_allclose(
        val_np_1, val_np_2, err_msg="NumPy random numbers did not match with seed 42"
    )
    assert torch.equal(
        val_torch_1, val_torch_2
    ), "PyTorch random tensors did not match with seed 42"


# 3. Config Loading & Integrity
def test_config_loader():
    """Verify config.yaml loads properly with required keys."""
    from src.utils.config_loader import load_config

    cfg = load_config()
    assert isinstance(cfg, dict), "Config must be a dictionary"
    required_sections = [
        "project",
        "reproducibility",
        "paths",
        "data",
        "split",
        "outcomes",
        "eri_scale",
        "causal",
        "leakage_control",
    ]
    for section in required_sections:
        assert section in cfg, f"Missing required config section: {section}"

    assert cfg["reproducibility"]["seed"] == 42
    assert cfg["data"]["synthetic"]["n_samples"] == 2000
    assert cfg["split"]["cv_folds"] == 5
    assert cfg["split"]["cv_repeats"] == 3


# 4. Schema Integrity & Role Verification
def test_schema_loader():
    """Verify the canonical Environmental & Financial schema and shared outcomes."""
    from src.utils.config_loader import load_schema

    schema = load_schema()
    assert len(schema["items"]) == 17
    assert {"attrition_intent_score", "at_risk"} <= set(schema["outcomes"])
    assert all(
        schema["outcomes"][name]["role"] == "outcome"
        for name in ("attrition_intent_score", "at_risk")
    )


# 5. Leaderboard Weights Loading
def test_leaderboard_weights_loader():
    """Verify leaderboard_weights.yaml loads with valid composite weights."""
    from src.utils.config_loader import load_leaderboard_weights

    weights = load_leaderboard_weights()
    assert "classification_weights" in weights
    cw = weights["classification_weights"]
    total_cls_weight = sum(item["weight"] for item in cw.values())
    assert (
        abs(total_cls_weight - 1.0) < 1e-4
    ), f"Classification weights must sum to 1.0, got {total_cls_weight}"

    assert "model_categories" in weights
    cat_ids = [c["id"] for c in weights["model_categories"]]
    assert "linear_shallow" in cat_ids
    assert "tree_ensembles" in cat_ids
    assert "deep_learning" in cat_ids


# 6. Paths Resolution & Directory Existence
def test_paths_helpers():
    """Verify path helper returns valid directories."""
    from src.utils.paths import (
        get_artifacts_dir,
        get_configs_dir,
        get_data_dir,
        get_docs_dir,
        get_project_root,
    )

    root = get_project_root()
    assert root.is_dir()
    assert (root / "PROJECT_CONTEXT.md").is_file()

    for stage in ["raw", "interim", "processed", "synthetic"]:
        d = get_data_dir(stage)
        assert d.is_dir(), f"Data stage dir does not exist: {d}"

    assert get_artifacts_dir().is_dir()
    assert get_configs_dir().is_dir()
    assert get_docs_dir().is_dir()


# 7. Timer & Profiling Utility
def test_timer_utility():
    """Verify Timer context manager and timed decorator."""
    import time

    from src.utils.timer import Timer, timed

    with Timer("Test block") as t:
        time.sleep(0.01)
    assert t.elapsed_seconds >= 0.005

    @timed("Decorated function")
    def sample_func():
        return 42

    result = sample_func()
    assert result == 42


# 8. Logger Setup
def test_logger_setup():
    """Verify standardized logger instantiates and writes messages without exception."""
    from src.utils.logging_config import setup_logger

    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    logger.info("Smoke test logger message.")

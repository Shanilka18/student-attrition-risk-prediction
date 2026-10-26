"""Smoke tests for the ERI Research Repository.

Verifies:
1. Mutually compatible imports for all 22 required packages.
2. Global seed determinism across NumPy and PyTorch.
3. Integrity and schema compliance of configs (config.yaml, schema.yaml, leaderboard_weights.yaml).
4. Dynamic project directory paths and folder existence.
5. Utility functions (Timer, Logger, Path helpers).
"""

import sys
import pytest
from pathlib import Path

# Verify python version is 3.11
def test_python_version():
    """Verify runtime is Python 3.11."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (major, minor) == (3, 11), f"Expected Python 3.11, found {major}.{minor}"


# 1. Package Import Verification
@pytest.mark.parametrize(
    "pkg_name,import_statement",
    [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("scikit-learn", "import sklearn"),
        ("xgboost", "import xgboost as xgb"),
        ("lightgbm", "import lightgbm as lgb"),
        ("catboost", "import catboost as cb"),
        ("torch", "import torch"),
        ("pytorch-tabnet", "import pytorch_tabnet"),
        ("shap", "import shap"),
        ("econml", "import econml"),
        ("dowhy", "import dowhy"),
        ("statsmodels", "import statsmodels.api as sm"),
        ("optuna", "import optuna"),
        ("joblib", "import joblib"),
        ("pytest", "import pytest"),
        ("openpyxl", "import openpyxl"),
        ("fastapi", "import fastapi"),
        ("uvicorn", "import uvicorn"),
        ("python-multipart", "import multipart"),
        ("pydantic", "import pydantic"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("pyyaml", "import yaml"),
    ],
)
def test_package_imports(pkg_name, import_statement):
    """Verify each of the 22 required libraries imports without error."""
    exec_scope = {}
    try:
        exec(import_statement, exec_scope)
    except Exception as exc:
        pytest.fail(f"Failed to import package '{pkg_name}' with command '{import_statement}': {exc}")


# 2. Seed Determinism & Reproducibility
def test_seed_determinism():
    """Verify set_seed ensures deterministic sampling across libraries."""
    from src.utils.seed import set_seed
    import numpy as np
    import torch

    set_seed(42)
    val_np_1 = np.random.uniform(0, 1, size=(5,))
    val_torch_1 = torch.rand(5)

    set_seed(42)
    val_np_2 = np.random.uniform(0, 1, size=(5,))
    val_torch_2 = torch.rand(5)

    np.testing.assert_allclose(val_np_1, val_np_2, err_msg="NumPy random numbers did not match with seed 42")
    assert torch.equal(val_torch_1, val_torch_2), "PyTorch random tensors did not match with seed 42"


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
    """Verify schema.yaml defines all columns with valid roles and descriptions."""
    from src.utils.config_loader import load_schema

    schema = load_schema()
    assert "columns" in schema, "Schema must have a 'columns' list"
    cols = schema["columns"]
    assert len(cols) >= 20, f"Expected at least 20 columns in schema, got {len(cols)}"

    valid_roles = {"id", "feature", "exposure", "covariate", "mediator", "outcome", "demographic_audit"}
    col_names = set()
    found_roles = set()

    for col in cols:
        assert "name" in col, f"Column missing 'name': {col}"
        assert "type" in col, f"Column '{col.get('name')}' missing 'type'"
        assert "role" in col, f"Column '{col.get('name')}' missing 'role'"
        assert col["role"] in valid_roles, f"Column '{col['name']}' has invalid role '{col['role']}'"
        assert "description" in col, f"Column '{col['name']}' missing 'description'"

        col_names.add(col["name"])
        found_roles.add(col["role"])

    # Verify key specific variables exist
    expected_vars = [
        "participant_id",
        "household_income_bracket",
        "financial_stress",
        "commute_time_min",
        "housing_type",
        "baseline_gpa_band",
        "later_gpa_decline",
        "attrition",
        "latent_risk_score",
    ]
    for ev in expected_vars:
        assert ev in col_names, f"Expected variable '{ev}' not found in schema"

    # Verify mediators are specifically tagged
    mediator_cols = [c["name"] for c in cols if c["role"] == "mediator"]
    assert "later_gpa_decline" in mediator_cols
    assert "later_engagement_decline" in mediator_cols


# 5. Leaderboard Weights Loading
def test_leaderboard_weights_loader():
    """Verify leaderboard_weights.yaml loads with valid composite weights."""
    from src.utils.config_loader import load_leaderboard_weights

    weights = load_leaderboard_weights()
    assert "classification_weights" in weights
    cw = weights["classification_weights"]
    total_cls_weight = sum(item["weight"] for item in cw.values())
    assert abs(total_cls_weight - 1.0) < 1e-4, f"Classification weights must sum to 1.0, got {total_cls_weight}"

    assert "model_categories" in weights
    cat_ids = [c["id"] for c in weights["model_categories"]]
    assert "linear_shallow" in cat_ids
    assert "tree_ensembles" in cat_ids
    assert "deep_learning" in cat_ids


# 6. Paths Resolution & Directory Existence
def test_paths_helpers():
    """Verify path helper returns valid directories."""
    from src.utils.paths import (
        get_project_root,
        get_data_dir,
        get_artifacts_dir,
        get_configs_dir,
        get_docs_dir,
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
    from src.utils.timer import Timer, timed
    import time

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

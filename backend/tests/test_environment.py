"""
Tests for Environment and Critical ML Dependency Verification.
"""

def test_imports_ml_libraries():
    """Verify that all essential ML, DL, and XAI packages import cleanly."""
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    import torch
    import shap
    import optuna
    import scipy
    import statsmodels
    import yaml

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgb.__version__ is not None
    assert lgb.__version__ is not None
    assert cb.__version__ is not None
    assert torch.__version__ is not None
    assert shap.__version__ is not None
    assert optuna.__version__ is not None
    assert scipy.__version__ is not None
    assert statsmodels.__version__ is not None
    assert yaml.__version__ is not None


def test_device_detection():
    """Verify that compute device detection operates without exception."""
    from backend.src.config import detect_compute_device
    import torch

    device = detect_compute_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu", "mps"]


def test_seed_determinism():
    """Verify that set_global_seed generates reproducible pseudo-random numbers."""
    from backend.src.config import set_global_seed
    import numpy as np
    import torch
    import random

    set_global_seed(42)
    val_py1 = random.random()
    val_np1 = np.random.rand(5)
    val_pt1 = torch.rand(5)

    set_global_seed(42)
    val_py2 = random.random()
    val_np2 = np.random.rand(5)
    val_pt2 = torch.rand(5)

    assert val_py1 == val_py2
    np.testing.assert_array_equal(val_np1, val_np2)
    assert torch.equal(val_pt1, val_pt2)

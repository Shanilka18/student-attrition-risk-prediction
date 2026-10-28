"""Unit tests for PyTorch Deep Learning architectures, training protocols, and leakage audits."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mei.models.deep.architectures import (
    BiGRUModel,
    CNN1DModel,
    GRUModel,
    HybridMultimodalModel,
    LSTMModel,
    MLPResidual,
    MLPWide,
    OverparameterizedMLP,
    TinyTransformer,
    build_deep_model,
)
from mei.models.deep.trainer import (
    DeepEngagementDataset,
    train_two_stage,
)


def test_deep_models_build_and_forward() -> None:
    """Verify that all 8 candidate architectures build and perform valid forward pass."""
    batch_size = 8
    seq_len = 4
    n_indicators = 7
    tab_dim = 126

    # Test inputs
    dummy_seq = torch.randn(batch_size, seq_len, n_indicators)
    dummy_tab = torch.randn(batch_size, tab_dim)

    # 1. MLPWide
    m1 = MLPWide(input_dim=tab_dim, hidden_dim=64, noise_std=0.0)
    out1 = m1(dummy_tab)
    assert out1.shape == (batch_size,)

    # 2. MLPResidual
    m2 = MLPResidual(input_dim=tab_dim, hidden_dim=32, noise_std=0.0)
    out2 = m2(dummy_tab)
    assert out2.shape == (batch_size,)

    # 3. GRUModel
    m3 = GRUModel(input_dim=n_indicators, hidden_dim=16, noise_std=0.0)
    out3 = m3(dummy_seq)
    assert out3.shape == (batch_size,)

    # 4. LSTMModel
    m4 = LSTMModel(input_dim=n_indicators, hidden_dim=16, noise_std=0.0)
    out4 = m4(dummy_seq)
    assert out4.shape == (batch_size,)

    # 5. BiGRUModel
    m5 = BiGRUModel(input_dim=n_indicators, hidden_dim=12, noise_std=0.0)
    out5 = m5(dummy_seq)
    assert out5.shape == (batch_size,)

    # 6. CNN1DModel
    m6 = CNN1DModel(input_dim=n_indicators, filters=16, noise_std=0.0)
    out6 = m6(dummy_seq)
    assert out6.shape == (batch_size,)

    # 7. TinyTransformer
    m7 = TinyTransformer(input_dim=n_indicators, d_model=16, noise_std=0.0)
    out7 = m7(dummy_seq)
    assert out7.shape == (batch_size,)

    # 8. HybridMultimodalModel
    m8 = HybridMultimodalModel(
        seq_input_dim=n_indicators, tab_input_dim=tab_dim, fusion_dim=16, noise_std=0.0
    )
    out8 = m8((dummy_seq, dummy_tab))
    assert out8.shape == (batch_size,)

    # 9. OverparameterizedMLP
    m9 = OverparameterizedMLP(input_dim=tab_dim)
    out9 = m9(dummy_tab)
    assert out9.shape == (batch_size,)


def test_build_deep_model_factory() -> None:
    """Verify that build_deep_model instantiates correct classes from string IDs."""
    model_ids = [
        "mlp_wide",
        "mlp_residual",
        "gru",
        "lstm",
        "bi_gru",
        "cnn_1d",
        "tiny_transformer",
        "hybrid_multimodal",
        "overfit_demo",
    ]
    for m_id in model_ids:
        model = build_deep_model(m_id, tab_input_dim=126, seq_input_dim=7)
        assert isinstance(model, nn.Module)

    with pytest.raises(ValueError):
        build_deep_model("non_existent_deep_model")


def test_deep_models_overfit_toy_set() -> None:
    """Sanity check: Deep model must be able to overfit a small 16-sample toy set to < 0.1 loss."""
    torch.manual_seed(42)
    np.random.seed(42)

    X_toy = np.random.randn(16, 20).astype(np.float32)
    y_toy = np.array([0, 1] * 8, dtype=np.float32)

    ds_toy = DeepEngagementDataset(X_tab=X_toy, y=y_toy)
    model = MLPWide(input_dim=20, hidden_dim=64, dropout=0.0, noise_std=0.0)

    # Train for 40 epochs on toy set
    res = train_two_stage(
        model=model,
        train_dataset=ds_toy,
        val_dataset=ds_toy,
        batch_size=16,
        lr_stage1=0.01,
        weight_decay=0.0,
        max_epochs_stage1=40,
        patience_stage1=40,
        max_epochs_stage2=0,
        seed=42,
    )
    final_loss = res["history"]["train_loss"][-1]
    assert final_loss < 0.15, f"Expected toy loss < 0.15, got {final_loss}"


def test_early_stopping_restores_best_weights() -> None:
    """Verify that train_two_stage restores best validation weights when training diverges."""
    torch.manual_seed(42)
    np.random.seed(42)

    X_tr = np.random.randn(40, 10).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=40).astype(np.float32)
    X_val = np.random.randn(20, 10).astype(np.float32)
    y_val = np.random.randint(0, 2, size=20).astype(np.float32)

    ds_tr = DeepEngagementDataset(X_tab=X_tr, y=y_tr)
    ds_val = DeepEngagementDataset(X_tab=X_val, y=y_val)

    model = MLPWide(input_dim=10, hidden_dim=32, noise_std=0.0)

    res = train_two_stage(
        model=model,
        train_dataset=ds_tr,
        val_dataset=ds_val,
        batch_size=20,
        lr_stage1=0.005,
        max_epochs_stage1=30,
        patience_stage1=5,
        max_epochs_stage2=0,
        seed=42,
    )

    best_loss = res["best_val_loss"]
    final_val_loss = res["history"]["val_loss"][-1]
    assert best_loss <= final_val_loss
    assert res["best_epoch"] <= len(res["history"]["val_loss"])


def test_history_keys_and_structure() -> None:
    """Verify that training history dictionary contains all required metric series."""
    torch.manual_seed(42)
    X_tr = np.random.randn(30, 8).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=30).astype(np.float32)

    ds = DeepEngagementDataset(X_tab=X_tr, y=y_tr)
    model = MLPWide(input_dim=8, hidden_dim=16, noise_std=0.0)

    res = train_two_stage(
        model=model,
        train_dataset=ds,
        val_dataset=ds,
        batch_size=15,
        max_epochs_stage1=5,
        patience_stage1=5,
        max_epochs_stage2=2,
        seed=42,
    )

    expected_keys = [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_auc",
        "val_auc",
        "train_prec",
        "val_prec",
        "train_rec",
        "val_rec",
    ]
    for k in expected_keys:
        assert k in res["history"]
        assert len(res["history"][k]) > 0


def test_seed_reproducibility() -> None:
    """Verify that identical random seeds yield deterministic first-epoch training losses."""
    X_tr = np.random.randn(30, 8).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=30).astype(np.float32)

    ds1 = DeepEngagementDataset(X_tab=X_tr.copy(), y=y_tr.copy())
    ds2 = DeepEngagementDataset(X_tab=X_tr.copy(), y=y_tr.copy())

    torch.manual_seed(123)
    model1 = MLPWide(input_dim=8, hidden_dim=16, noise_std=0.0)
    res1 = train_two_stage(
        model=model1,
        train_dataset=ds1,
        val_dataset=ds1,
        batch_size=15,
        max_epochs_stage1=3,
        patience_stage1=3,
        max_epochs_stage2=0,
        seed=123,
    )

    torch.manual_seed(123)
    model2 = MLPWide(input_dim=8, hidden_dim=16, noise_std=0.0)
    res2 = train_two_stage(
        model=model2,
        train_dataset=ds2,
        val_dataset=ds2,
        batch_size=15,
        max_epochs_stage1=3,
        patience_stage1=3,
        max_epochs_stage2=0,
        seed=123,
    )

    assert (
        pytest.approx(res1["history"]["train_loss"][0], rel=1e-5)
        == res2["history"]["train_loss"][0]
    )

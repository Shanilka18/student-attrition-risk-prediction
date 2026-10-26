"""Unit tests for model reproducibility and architectures."""

import pytest
import numpy as np
from src.utils.seed import set_seed
from src.models.architectures import TabularMLP


def test_seed_reproducibility():
    """Verify set_seed produces reproducible pseudo-random numbers."""
    set_seed(42)
    seq1 = np.random.rand(5)
    set_seed(42)
    seq2 = np.random.rand(5)
    np.testing.assert_array_equal(seq1, seq2)


def test_tabular_mlp_instantiation():
    """Verify TabularMLP initializes with correct dimensions."""
    try:
        import torch
        model = TabularMLP(input_dim=15, hidden_dims=[32, 16], output_dim=1)
        dummy_input = torch.randn(4, 15)
        output = model(dummy_input)
        assert output.shape == (4, 1)
    except ImportError:
        pytest.skip("PyTorch not installed in current environment")

"""Reproducibility utilities for random seed management.

Ensures deterministic execution across Python random, NumPy, PyTorch (CPU & CUDA),
and system hash seeds according to PROJECT_CONTEXT.md hard rules.
"""

import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Set global random seeds for deterministic execution across libraries.

    Parameters
    ----------
    seed : int
        The integer random seed to apply globally (default is 42).
    deterministic_torch : bool
        Whether to enforce deterministic algorithms in PyTorch.

    Returns
    -------
    int
        The seed value that was set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Enforce deterministic PyTorch operations where available
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass
    except ImportError:
        pass

    return seed

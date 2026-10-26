"""Global random seed setter to ensure end-to-end experimental reproducibility."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across standard library, NumPy, and PyTorch (if installed).

    Args:
        seed (int): Integer seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

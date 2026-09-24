"""Universal reproducibility module.

Sets random seeds across Python's standard library, NumPy, and PyTorch
(including CUDA backends and cuDNN determinism flags) to ensure consistent results.
"""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Enforce reproducibility across all components of the system.

    Sets seeds for Python's `random`, `os.environ['PYTHONHASHSEED']`,
    `numpy.random`, and `torch` (including CPU, GPU, and cuDNN).

    Args:
        seed: The integer seed value to apply universally. Defaults to 42.

    Example:
        >>> from src.utils.seed import seed_everything
        >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

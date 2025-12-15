"""Deterministic seeding utilities for reproducible pipeline execution."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic_ops: bool = True) -> np.random.Generator:
    """Set all random seeds for deterministic execution across libraries.

    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator (modern API)
    - PyTorch (if available) with deterministic ops
    - PYTHONHASHSEED environment variable

    Args:
        seed: Random seed value
        deterministic_ops: Enable deterministic operations in PyTorch

    Returns:
        Seeded NumPy random number generator
    """
    # Set Python hash seed (must be done before import)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set Python random seed
    random.seed(seed)

    # Create and return NumPy RNG (modern API)
    rng = np.random.default_rng(seed)

    # Set PyTorch seeds if available
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms for reproducibility
        if deterministic_ops and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    except ImportError:
        pass  # PyTorch not available, skip
    except Exception:
        # Catch any PyTorch-specific errors but don't fail
        pass

    return rng


def env_fingerprint() -> dict[str, Any]:
    """Capture environment fingerprint for reproducibility tracking.

    Returns:
        Dictionary with environment metadata including:
        - Python version and platform
        - Package versions (NumPy, etc.)
        - Git SHA (best effort)
        - CPU/hardware flags
    """

    def git_sha() -> str:
        """Get current git SHA, return 'unknown' if not available."""
        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                timeout=5,
            )
            return result.strip()
        except Exception:
            return "unknown"

    # Core environment info
    fingerprint = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "numpy_version": np.__version__,
    }

    # Add package versions if available
    try:
        import pandas as pd

        fingerprint["pandas_version"] = pd.__version__
    except ImportError:
        pass

    try:
        import scipy

        fingerprint["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    try:
        import torch

        fingerprint["torch_version"] = torch.__version__
        fingerprint["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        fingerprint["torch_version"] = None
        fingerprint["cuda_available"] = False

    # CPU information
    try:
        fingerprint["cpu_count"] = os.cpu_count()
        fingerprint["machine"] = platform.machine()
    except Exception:
        pass

    return fingerprint

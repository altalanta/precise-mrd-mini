"""Determinism utilities for reproducible pipeline execution."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def hash_array(array: np.ndarray, precision: int = 6) -> str:
    """Compute SHA256 hash of numpy array for determinism testing."""
    if array.dtype != np.float64:
        array = array.astype(np.float64)

    # Round to specified precision to avoid floating point noise
    rounded = np.round(array, decimals=precision)

    # Convert to bytes for hashing
    bytes_data = rounded.tobytes()

    return hashlib.sha256(bytes_data).hexdigest()

"""Determinism utilities for reproducible pipeline execution."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import warnings
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for deterministic execution.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (if available)
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not available, skip


def hash_array(array: np.ndarray, precision: int = 6) -> str:
    """Compute SHA256 hash of numpy array for determinism testing.
    
    Args:
        array: NumPy array to hash
        precision: Decimal precision for rounding (to avoid floating point noise)
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    if array.dtype != np.float64:
        array = array.astype(np.float64)

    # Round to specified precision to avoid floating point noise
    rounded = np.round(array, decimals=precision)

    # Convert to bytes for hashing
    bytes_data = rounded.tobytes()

    return hashlib.sha256(bytes_data).hexdigest()


def get_git_sha() -> str:
    """Get current git SHA for reproducibility tracking.
    
    Returns:
        Git SHA string, or "unknown" if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


@contextmanager
def determinism_context(seed: int) -> Generator[dict[str, Any], None, None]:
    """Context manager for deterministic execution with metadata tracking.
    
    This context manager:
    1. Sets all random seeds
    2. Captures git SHA and other metadata
    3. Yields metadata dictionary
    4. Can be used to ensure deterministic execution blocks
    
    Args:
        seed: Random seed value
        
    Yields:
        Dictionary with reproducibility metadata
        
    Example:
        >>> with determinism_context(42) as metadata:
        ...     result = some_stochastic_function()
        ...     print(f"Git SHA: {metadata['git_sha']}")
    """
    # Set seeds
    set_all_seeds(seed)
    
    # Capture metadata
    import platform
    from datetime import datetime
    
    metadata = {
        "seed": seed,
        "git_sha": get_git_sha(),
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    
    # Add PyTorch version if available
    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False
    
    # Temporarily suppress specific warnings that can affect determinism
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        yield metadata


def validate_determinism(
    func, 
    args: tuple = (), 
    kwargs: dict | None = None,
    seed: int = 42,
    n_runs: int = 3,
    hash_precision: int = 6
) -> dict[str, Any]:
    """Validate that a function produces deterministic results.
    
    Args:
        func: Function to test for determinism
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function  
        seed: Random seed to use
        n_runs: Number of runs to compare
        hash_precision: Precision for array hashing
        
    Returns:
        Dictionary with validation results
    """
    if kwargs is None:
        kwargs = {}
    
    results = []
    hashes = []
    
    for i in range(n_runs):
        with determinism_context(seed):
            result = func(*args, **kwargs)
            results.append(result)
            
            # Try to hash the result if it's an array-like object
            try:
                if hasattr(result, '__array__'):
                    array_result = np.asarray(result)
                    hash_val = hash_array(array_result, precision=hash_precision)
                elif isinstance(result, (list, tuple)):
                    # Try to convert to array
                    array_result = np.asarray(result)
                    hash_val = hash_array(array_result, precision=hash_precision)
                else:
                    # For other types, use string representation
                    hash_val = hashlib.sha256(str(result).encode()).hexdigest()
                
                hashes.append(hash_val)
            except Exception as e:
                hashes.append(f"error_{i}: {str(e)}")
    
    # Check if all hashes are identical
    is_deterministic = len(set(hashes)) == 1
    
    return {
        "is_deterministic": is_deterministic,
        "n_runs": n_runs,
        "seed": seed,
        "hashes": hashes,
        "unique_hashes": len(set(hashes)),
        "first_hash": hashes[0] if hashes else None,
    }

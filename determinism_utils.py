from pathlib import Path

import numpy as np
import torch

from .io import PipelineIO


def set_global_seed(seed: int, deterministic_ops: bool = True) -> np.random.Generator:
    """Set the global random seed for reproducible operations."""
    rng = np.random.default_rng(seed)
    if deterministic_ops:
        # Set numpy's random seed
        np.random.seed(seed)
        # Set torch's random seed
        torch.manual_seed(seed)
        # Set torch's deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return rng


def create_hash_manifest(file_paths: list[str]) -> dict[str, str]:
    """Create a manifest of SHA256 hashes for a list of files."""
    manifest_data = {}
    for f_path in file_paths:
        f_name = Path(f_path).name
        manifest_data[f_name] = PipelineIO.calculate_sha256(f_path)

    return manifest_data


class ArtifactManager:
    """A simple manager to track artifacts and their hashes."""

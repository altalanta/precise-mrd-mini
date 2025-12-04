"""Utility functions for the precise MRD pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
import yaml


class PipelineIO:
    """A collection of static methods for pipeline I/O operations."""

    @staticmethod
    def calculate_sha256(file_path: str) -> str:
        """Calculate the SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: str):
        """Save a pandas DataFrame to a parquet file."""
        df.to_parquet(path, index=False)

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        """Load a pandas DataFrame from a parquet file."""
        return pd.read_parquet(path)

    @staticmethod
    def save_json(data: dict, path: str):
        """Save a dictionary to a JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(path: str) -> dict:
        """Load a dictionary from a JSON file."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_yaml(data: dict, path: str):
        """Save a dictionary to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def load_yaml(path: str) -> dict:
        """Load a dictionary from a YAML file."""
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        """Ensure directory exists, create if necessary."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj


def validate_repository_state() -> list[str]:
    """Validate that git repository is in a consistent state.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check if we're in a git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        errors.append("Not in a git repository or git not available")
        return errors

    # Check for uncommitted changes that might affect reproducibility
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        if result.stdout.strip():
            # Only warn about source code changes, not generated files
            lines = result.stdout.strip().split("\n")
            source_changes = [
                line
                for line in lines
                if not any(
                    path in line for path in ["data/", "reports/", ".snakemake/"]
                )
            ]
            if source_changes:
                errors.append(
                    f"Repository has uncommitted source changes: {source_changes}"
                )
    except Exception as e:
        errors.append(f"Could not check git status: {e}")

    return errors


def get_package_versions() -> dict[str, str]:
    """Get the versions of all installed packages.

    Returns:
        Dictionary mapping package names to their installed versions.
    """
    return {dist.metadata["Name"]: dist.version for dist in distributions()}

"""Utility functions for the precise MRD pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


class PipelineIO:
    """Input/output utilities for pipeline artifacts."""

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        """Ensure directory exists, create if necessary."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj

    @staticmethod
    def save_json(data: Dict[str, Any], path: str | Path) -> None:
        """Save data as JSON file."""
        PipelineIO.ensure_dir(Path(path).parent)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)


def validate_repository_state() -> List[str]:
    """Validate that git repository is in a consistent state.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check if we're in a git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, check=True, timeout=5
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        errors.append("Not in a git repository or git not available")
        return errors

    # Check for uncommitted changes that might affect reproducibility
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5
        )
        if result.stdout.strip():
            # Only warn about source code changes, not generated files
            lines = result.stdout.strip().split('\n')
            source_changes = [
                line for line in lines
                if not any(path in line for path in ['data/', 'reports/', '.snakemake/'])
            ]
            if source_changes:
                errors.append(f"Repository has uncommitted source changes: {source_changes}")
    except Exception as e:
        errors.append(f"Could not check git status: {e}")

    return errors

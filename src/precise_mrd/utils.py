"""Utility helpers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACT_FILENAMES = {
    "simulate": "simulated_reads.parquet",
    "collapse": "collapsed_umis.parquet",
    "error_model": "error_model.parquet",
    "call": "mrd_calls.parquet",
    "metrics": "metrics.json",
    "lineage": "lineage.json",
    "report_md": "report.md",
    "report_html": "report.html",
    "lod_grid": "lod_grid.parquet",
    "roc_png": "roc_curve.png",
    "pr_png": "pr_curve.png",
    "config": "config.json",
    "run_context": "run_context.json",
}


class PipelineIO:
    """Helper for reading/writing artifacts with deterministic paths."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        if key not in ARTIFACT_FILENAMES:
            msg = f"unknown artifact key: {key}"
            raise KeyError(msg)
        return self.base_dir / ARTIFACT_FILENAMES[key]

    def write_parquet(self, key: str, df: pd.DataFrame) -> Path:
        path = self.path(key)
        df.to_parquet(path, index=False)
        return path

    def read_parquet(self, key: str) -> pd.DataFrame:
        path = self.path(key)
        return pd.read_parquet(path)

    def write_json(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path(key)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": os.environ.get("PYTHON_VERSION", "unknown"),
        "cwd": str(Path.cwd()),
    }


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_json_ready(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    return data

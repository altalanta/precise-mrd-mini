"""QC metric helpers for MRD pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def compute_qc_metrics(family_df: pd.DataFrame, locus_df: pd.DataFrame) -> Dict[str, float]:
    """Compute simple QC metrics from collapsed data."""
    coverage = float(locus_df["total_count"].sum())
    duplication = float(1 - family_df["family_size"].mean() / family_df["family_size"].max()) if not family_df.empty else 0.0
    on_target = float(len(locus_df) / max(len(family_df), 1))
    mean_family = float(family_df["family_size"].mean()) if not family_df.empty else 0.0
    return {
        "total_coverage": coverage,
        "duplication_rate": duplication,
        "on_target_fraction": on_target,
        "mean_family_size": mean_family,
    }


def save_qc_metrics(
    family_path: Path,
    locus_path: Path,
    output_path: Path = Path("tmp/qc_metrics.json"),
) -> Dict[str, float]:
    family_df = pd.read_parquet(family_path)
    locus_df = pd.read_parquet(locus_path)
    metrics = compute_qc_metrics(family_df, locus_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    return metrics

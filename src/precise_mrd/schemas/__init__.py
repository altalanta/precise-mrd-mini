"""Schema validators for pipeline artifacts."""

from __future__ import annotations

import polars as pl
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Schema:
    name: str
    schema: pl.Schema

    def validate(self, frame: pd.DataFrame) -> None:
        try:
            pl.DataFrame(frame).cast(self.schema)
        except Exception as exc:  # pragma: no cover - polars details vary
            msg = f"{self.name} schema validation failed: {exc}"
            raise ValueError(msg) from exc


SIMULATED_READS_SCHEMA = Schema(
    name="simulated_reads",
    schema=pl.Schema(
        {
            "sample_id": pl.Utf8,
            "sample_type": pl.Utf8,
            "variant_id": pl.Utf8,
            "allele_fraction": pl.Float64,
            "depth": pl.Int64,
            "umi_id": pl.Utf8,
            "read_index": pl.Int64,
            "allele": pl.Utf8,
        }
    ),
)

COLLAPSED_UMI_SCHEMA = Schema(
    name="collapsed_umis",
    schema=pl.Schema(
        {
            "sample_id": pl.Utf8,
            "sample_type": pl.Utf8,
            "variant_id": pl.Utf8,
            "allele_fraction": pl.Float64,
            "depth": pl.Int64,
            "umi_id": pl.Utf8,
            "family_size": pl.Int64,
            "alt_reads": pl.Int64,
            "ref_reads": pl.Int64,
        }
    ),
)

ERROR_MODEL_SCHEMA = Schema(
    name="error_model",
    schema=pl.Schema(
        {
            "variant_id": pl.Utf8,
            "depth": pl.Int64,
            "alpha": pl.Float64,
            "beta": pl.Float64,
            "mean_error": pl.Float64,
        }
    ),
)

MRD_CALL_SCHEMA = Schema(
    name="mrd_calls",
    schema=pl.Schema(
        {
            "sample_id": pl.Utf8,
            "variant_id": pl.Utf8,
            "depth": pl.Int64,
            "alt_reads": pl.Int64,
            "total_reads": pl.Int64,
            "pvalue": pl.Float64,
            "detected": pl.Boolean,
            "truth_positive": pl.Boolean,
            "allele_fraction": pl.Float64,
        }
    ),
)

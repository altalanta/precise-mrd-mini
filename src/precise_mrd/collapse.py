"""UMI collapsing stage."""

from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .schemas import COLLAPSED_UMI_SCHEMA, SIMULATED_READS_SCHEMA


def collapse_umis(reads: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Collapse per-read data into UMI consensus counts."""

    SIMULATED_READS_SCHEMA.validate(reads)
    umi_cfg = config.umi

    grouped = (
        reads.groupby(
            ["sample_id", "sample_type", "variant_id", "allele_fraction", "depth", "umi_id"],
            sort=False,
        )["allele"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={"ALT": "alt_reads", "REF": "ref_reads"})
        .reset_index()
    )

    if "alt_reads" not in grouped:
        grouped["alt_reads"] = 0
    if "ref_reads" not in grouped:
        grouped["ref_reads"] = 0

    grouped["family_size"] = grouped["alt_reads"] + grouped["ref_reads"]
    collapsed = grouped[grouped["family_size"] >= umi_cfg.min_family_size].copy()

    COLLAPSED_UMI_SCHEMA.validate(collapsed)
    return collapsed

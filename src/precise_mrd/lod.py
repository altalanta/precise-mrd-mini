"""LoD helper functions."""

from __future__ import annotations

import pandas as pd


def lod_grid(collapsed: pd.DataFrame) -> pd.DataFrame:
    def _compute(frame: pd.DataFrame) -> pd.Series:
        detection_rate = (frame["alt_reads"] > 0).mean()
        mean_alt = frame["alt_reads"].mean()
        return pd.Series({"detection_rate": detection_rate, "mean_alt_reads": mean_alt})

    grouped = (
        collapsed.groupby(["variant_id", "depth", "allele_fraction"], as_index=False)
        .apply(_compute)
        .reset_index(drop=True)
    )
    grouped.sort_values(["variant_id", "depth", "allele_fraction"], inplace=True)
    return grouped

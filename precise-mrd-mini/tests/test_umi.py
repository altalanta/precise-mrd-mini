from __future__ import annotations

import pandas as pd

from mrd import umi


def test_collapse_simple() -> None:
    data = pd.DataFrame(
        [
            {"patient_id": "P", "chrom": "chr1", "pos": 1, "umi": "AAAA", "start": 10, "sequence": "ACGTACGT", "true_alt": 1},
            {"patient_id": "P", "chrom": "chr1", "pos": 1, "umi": "AAAT", "start": 10, "sequence": "ACGTACGT", "true_alt": 1},
            {"patient_id": "P", "chrom": "chr1", "pos": 1, "umi": "CCCC", "start": 10, "sequence": "ACGTACGT", "true_alt": 0},
            {"patient_id": "P", "chrom": "chr1", "pos": 1, "umi": "CCCC", "start": 10, "sequence": "ACGTACGT", "true_alt": 0},
        ]
    )
    family_df, locus_df = umi.collapse_reads(data, min_family_size=2, max_distance=1)
    assert len(family_df) == 2
    assert int(family_df.loc[family_df["umi_representative"] == "AAAA", "alt_count"].iloc[0]) == 2
    row = locus_df.iloc[0]
    assert row["alt_count"] == 2
    assert row["total_count"] == 4

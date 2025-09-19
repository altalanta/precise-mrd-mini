from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrd import simulate


def test_simulation_outputs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """patient_id: TEST
n_variants: 5
depth_mean: 50
umi_family_geom_p: 0.5
base_error_rate: 0.0001
vaf_true: [0.0, 0.1]
panel_error_alpha_beta: [1.5, 3000]
"""
    )
    reads_path = tmp_path / "reads.parquet"
    variants_path = tmp_path / "variants.csv"
    simulate.run(config_path=config, output_reads=reads_path, variants_csv=variants_path, seed=42)

    variants = pd.read_csv(variants_path)
    reads = pd.read_parquet(reads_path)
    assert len(variants) == 5
    assert {"umi", "sequence", "true_alt"}.issubset(reads.columns)

    second_reads_path = tmp_path / "reads2.parquet"
    simulate.run(config_path=config, output_reads=second_reads_path, variants_csv=variants_path, seed=42)
    reads_repeat = pd.read_parquet(second_reads_path)
    pd.testing.assert_frame_equal(reads.sort_index(axis=1), reads_repeat.sort_index(axis=1))

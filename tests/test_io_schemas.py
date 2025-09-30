"""Test I/O schemas and data validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precise_mrd.determinism_utils import set_all_seeds
from precise_mrd.schemas import COLLAPSED_UMI_SCHEMA


def test_collapsed_schema_validation():
    """Validate collapsed UMI schema structure and data types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run a minimal pipeline to generate collapsed data
        from precise_mrd.config import load_config
        from precise_mrd.collapse import collapse_umis
        from precise_mrd.rng import choose_rng
        from precise_mrd.simulate import simulate_reads

        # Create minimal config
        config_path = Path(tmp_dir) / "test_config.yaml"
        config_path.write_text(
            """
run_id: "schema_test"
seed: 7

simulation:
  allele_fractions: [0.01]
  umi_depths: [100]
  n_replicates: 2
  n_bootstrap: 10
  
umi:
  min_family_size: 2
  max_family_size: 10
  max_edit_distance: 1
  quality_threshold: 20
  consensus_threshold: 0.6

error_model:
  contexts: ["ACG"]
  base_error_rate: 1e-4
  contamination_rate: 1e-3

stats:
  test_type: "poisson"
  alpha: 0.05
  fdr_method: "benjamini_hochberg"
  min_alt_umi: 1

filters:
  strand_bias_threshold: 0.01
  end_repair_filter: false
  end_repair_distance: 10
  end_repair_contexts: ["G>T"]

lod:
  detection_threshold: 0.95
  confidence_level: 0.95
  lob_replicates: 10

qc:
  min_total_umi_families: 1
  max_contamination_rate: 0.1
  min_depth_per_locus: 1

report:
  include_plots: false

output:
  save_intermediate: true
  compression: "gzip"
  decimal_places: 6
"""
        )

        set_all_seeds(7)
        cfg = load_config(config_path)
        rng = choose_rng(7)

        simulated = simulate_reads(cfg, rng)
        collapsed = collapse_umis(simulated.reads, cfg)

        # Test schema validation passes
        COLLAPSED_UMI_SCHEMA.validate(collapsed)

        # Test column presence
        expected_columns = {
            "sample_id",
            "sample_type",
            "variant_id",
            "allele_fraction",
            "depth",
            "umi_id",
            "family_size",
            "alt_reads",
            "ref_reads",
        }
        assert set(collapsed.columns) == expected_columns

        # Test data types
        assert collapsed["family_size"].dtype in [np.int32, np.int64]
        assert collapsed["alt_reads"].dtype in [np.int32, np.int64]
        assert collapsed["ref_reads"].dtype in [np.int32, np.int64]
        assert collapsed["allele_fraction"].dtype in [np.float32, np.float64]
        assert collapsed["depth"].dtype in [np.int32, np.int64]

        # Test no NaNs in critical columns
        assert not collapsed["alt_reads"].isna().any()
        assert not collapsed["ref_reads"].isna().any()
        assert not collapsed["family_size"].isna().any()
        assert not collapsed["variant_id"].isna().any()

        # Test logical constraints
        assert (collapsed["family_size"] >= cfg.umi.min_family_size).all()
        assert (collapsed["alt_reads"] >= 0).all()
        assert (collapsed["ref_reads"] >= 0).all()
        assert (collapsed["alt_reads"] + collapsed["ref_reads"] == collapsed["family_size"]).all()


def test_collapsed_umi_structure():
    """Test collapsed UMI data structure properties."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from precise_mrd.config import load_config
        from precise_mrd.collapse import collapse_umis
        from precise_mrd.rng import choose_rng
        from precise_mrd.simulate import simulate_reads

        # Create test config
        config_path = Path(tmp_dir) / "structure_test.yaml"
        config_path.write_text(
            """
run_id: "structure_test"
seed: 42

simulation:
  allele_fractions: [0.005]
  umi_depths: [50]
  n_replicates: 1
  n_bootstrap: 5
  
umi:
  min_family_size: 2
  consensus_threshold: 0.6

error_model:
  contexts: ["ACG"]
  base_error_rate: 1e-4

stats:
  test_type: "poisson"
  alpha: 0.05
  min_alt_umi: 1

filters:
  strand_bias_threshold: 0.01
  end_repair_filter: false

lod:
  detection_threshold: 0.95
  lob_replicates: 5

qc:
  min_total_umi_families: 1
  max_contamination_rate: 0.1

report:
  include_plots: false

output:
  save_intermediate: false
"""
        )

        set_all_seeds(42)
        cfg = load_config(config_path)
        rng = choose_rng(42)

        simulated = simulate_reads(cfg, rng)
        collapsed = collapse_umis(simulated.reads, cfg)

        # Test that we have data
        assert len(collapsed) > 0, "Collapsed data should not be empty"

        # Test UMI IDs are unique per sample/variant
        grouped = collapsed.groupby(["sample_id", "variant_id", "umi_id"]).size()
        assert grouped.max() == 1, "UMI IDs should be unique per sample/variant"

        # Test sample types are expected values
        sample_types = set(collapsed["sample_type"].unique())
        expected_types = {"case", "control"}
        assert sample_types.issubset(
            expected_types
        ), f"Unexpected sample types: {sample_types - expected_types}"

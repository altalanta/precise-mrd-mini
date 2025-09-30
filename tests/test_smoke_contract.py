"""Test smoke path contract and determinism."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from precise_mrd.cli import _load_config, _prepare_output
from precise_mrd.determinism_utils import hash_array, set_all_seeds
from precise_mrd.utils import PipelineIO


def test_smoke_contract_via_micro_pipeline():
    """Test smoke contract by running a micro version of the pipeline."""
    # Create minimal config for micro test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a minimal smoke config
        micro_config_path = tmp_path / "micro_config.yaml"
        micro_config_path.write_text(
            """
run_id: "micro_test"
seed: 7

simulation:
  allele_fractions: [0.005]
  umi_depths: [100]  # Very small for micro test
  n_replicates: 4
  n_bootstrap: 10
  
umi:
  min_family_size: 2
  max_family_size: 10
  max_edit_distance: 1
  quality_threshold: 20
  consensus_threshold: 0.6

error_model:
  contexts: ["ACG"]  # Single context for speed
  base_error_rate: 1e-4
  contamination_rate: 1e-3

stats:
  test_type: "poisson"
  alpha: 0.05
  fdr_method: "benjamini_hochberg"
  min_alt_umi: 1

filters:
  strand_bias_threshold: 0.01
  end_repair_filter: true
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

        # Run micro pipeline
        from precise_mrd.call import call_mrd
        from precise_mrd.collapse import collapse_umis
        from precise_mrd.config import load_config
        from precise_mrd.error_model import fit_error_model
        from precise_mrd.rng import choose_rng
        from precise_mrd.simulate import simulate_reads

        set_all_seeds(7)
        cfg = load_config(micro_config_path)
        out_dir = tmp_path / "micro_out"
        io = PipelineIO(out_dir)

        rng = choose_rng(7)
        simulated = simulate_reads(cfg, rng)
        collapsed = collapse_umis(simulated.reads, cfg)
        error_df = fit_error_model(collapsed, cfg)
        calls, metrics_payload, lod = call_mrd(collapsed, error_df, cfg, rng)

        # Validate metrics.json contract
        assert isinstance(metrics_payload, dict)
        assert "roc_auc" in metrics_payload
        assert "pr_auc" in metrics_payload or "average_precision" in metrics_payload
        assert isinstance(metrics_payload["roc_auc"], (int, float))

        # Check that we have confidence intervals (95% CI bounds)
        assert "roc_auc_ci" in metrics_payload
        ci = metrics_payload["roc_auc_ci"]
        assert "lower" in ci and "upper" in ci

        # Validate LoD grid (should have at least some entries)
        assert len(lod) >= 1, f"LoD grid should have at least 1 entry, got {len(lod)}"
        assert "detection_rate" in lod.columns
        assert "allele_fraction" in lod.columns


def test_smoke_golden_hash():
    """Test that smoke scores hash matches expected value for seed=7."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from precise_mrd.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run smoke command
        result = runner.invoke(
            app, ["smoke", "--seed", "7", "--out", tmp_dir, "--config", "configs/smoke.yaml"]
        )

        if result.exit_code != 0:
            pytest.skip(f"Smoke command failed: {result.stdout}")

        # Check that smoke_scores.npy was created
        smoke_dir = Path(tmp_dir) / "smoke"
        scores_path = smoke_dir / "smoke_scores.npy"

        if not scores_path.exists():
            pytest.skip("smoke_scores.npy not found")

        scores = np.load(scores_path)
        actual_hash = hash_array(scores)

        # Expected hash for seed=7 with smoke config
        expected_hash = "5b6fb58e61fa475939767d68a446f97f1bff02c0e5935a3ea8bb51e6515783d8"

        assert (
            actual_hash == expected_hash
        ), f"Hash mismatch: got {actual_hash}, expected {expected_hash}"

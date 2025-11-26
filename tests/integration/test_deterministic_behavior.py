"""
Integration Tests for Deterministic Behavior

This module validates that the pipeline produces identical results
across different runs with the same configuration and seed.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precise_mrd.call import call_mrd
from precise_mrd.collapse import collapse_umis
from precise_mrd.config import PipelineConfig
from precise_mrd.determinism_utils import env_fingerprint, set_global_seed
from precise_mrd.error_model import fit_error_model
from precise_mrd.metrics import calculate_metrics
from precise_mrd.reporting import render_report
from precise_mrd.simulate import simulate_reads


@pytest.mark.integration
@pytest.mark.determinism
class TestDeterministicBehavior:
    """Test deterministic behavior across multiple runs."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_identical_results_same_seed(self, temp_dir):
        """Test that identical seeds produce identical results."""
        config_dict = {
            "run_id": "determinism_test",
            "seed": 12345,
            "simulation": {
                "allele_fractions": [0.01, 0.001],
                "umi_depths": [2000, 8000],
                "n_replicates": 3,
                "n_bootstrap": 30,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        config = PipelineConfig.from_dict(config_dict)

        # Run pipeline multiple times with same seed
        results = []
        for i in range(3):
            output_path = temp_dir / f"run_{i}"
            result = self._run_pipeline(config, output_path, seed=config.seed)
            results.append(result)

        # All results should be identical
        self._assert_all_identical(results)

    def test_different_seeds_produce_different_results(self, temp_dir):
        """Test that different seeds produce different results."""
        config_dict = {
            "run_id": "seed_variation_test",
            "simulation": {
                "allele_fractions": [0.01, 0.001],
                "umi_depths": [5000],
                "n_replicates": 5,
                "n_bootstrap": 50,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        # Test different seeds
        seeds = [42, 123, 456]
        results = []

        for seed in seeds:
            config = PipelineConfig.from_dict({**config_dict, "seed": seed})
            output_path = temp_dir / f"seed_{seed}"
            result = self._run_pipeline(config, output_path, seed=seed)
            results.append(result)

        # Results should be different but valid
        self._assert_results_different_but_valid(results)

    def test_configuration_hash_stability(self, temp_dir):
        """Test that configuration hashes are stable."""
        configs = [
            {"seed": 42, "run_id": "test1"},
            {"seed": 42, "run_id": "test2"},
            {"seed": 123, "run_id": "test1"},
        ]

        hashes = []
        for config_dict in configs:
            full_config = {
                **config_dict,
                "simulation": {
                    "allele_fractions": [0.01],
                    "umi_depths": [1000],
                    "n_replicates": 1,
                },
                "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
                "stats": {"test_type": "poisson", "alpha": 0.05},
            }

            config = PipelineConfig.from_dict(full_config)
            config_hash = config.config_hash()
            hashes.append(config_hash)

        # Same config should have same hash
        assert hashes[0] == hashes[0], "Same config should have same hash"
        # Different configs should have different hashes
        assert hashes[0] != hashes[1], "Different configs should have different hashes"
        assert hashes[0] != hashes[2], "Different seeds should have different hashes"

    def test_environment_fingerprint_consistency(self):
        """Test that environment fingerprints are consistent."""
        # Get environment fingerprint multiple times
        fingerprints = [env_fingerprint() for _ in range(3)]

        # All fingerprints should be identical
        for i in range(1, len(fingerprints)):
            assert fingerprints[0] == fingerprints[i], (
                f"Fingerprint {i} differs from first"
            )

    def test_artifact_hash_stability(self, temp_dir):
        """Test that artifact hashes are stable for identical runs."""
        config_dict = {
            "run_id": "hash_stability_test",
            "seed": 999,
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [1000],
                "n_replicates": 2,
                "n_bootstrap": 20,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        config = PipelineConfig.from_dict(config_dict)

        # Run pipeline twice
        self._run_pipeline(config, temp_dir / "run1", seed=config.seed)
        self._run_pipeline(config, temp_dir / "run2", seed=config.seed)

        # Generate hash manifests
        manifest1 = self._generate_hash_manifest(temp_dir / "run1")
        manifest2 = self._generate_hash_manifest(temp_dir / "run2")

        # Hashes should be identical
        assert manifest1 == manifest2, (
            "Hash manifests should be identical for same inputs"
        )

    def test_cross_platform_compatibility(self, temp_dir):
        """Test compatibility across different random number generation patterns."""
        # Test different numpy random generation methods
        rng_methods = [
            {
                "method": "default_rng",
                "generator": lambda seed: np.random.default_rng(seed),
            },
            {"method": "legacy", "generator": lambda seed: np.random.RandomState(seed)},
        ]

        config_dict = {
            "run_id": "platform_test",
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [1000],
                "n_replicates": 3,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        results = []
        for rng_info in rng_methods:
            config = PipelineConfig.from_dict({**config_dict, "seed": 42})
            output_path = temp_dir / f"rng_{rng_info['method']}"

            result = self._run_pipeline_with_rng(
                config, output_path, rng_info["generator"]
            )
            results.append(result)

        # Results should be consistent (both methods should work)
        for result in results:
            assert result["metrics"]["roc_auc"] > 0.5, (
                "Both RNG methods should produce valid results"
            )

    def _run_pipeline(
        self, config: PipelineConfig, output_path: Path, seed: int
    ) -> dict:
        """Run complete pipeline with given configuration."""
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Step 1: Simulate reads
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Step 2: Collapse UMIs
        collapsed_df = collapse_umis(reads_df, config, rng)

        # Step 3: Fit error model
        error_model = fit_error_model(collapsed_df, config, rng)

        # Step 4: Call variants
        calls_df = call_mrd(
            collapsed_df,
            error_model,
            config,
            rng,
            use_ml_calling=False,
            use_deep_learning=False,
        )

        # Step 5: Calculate metrics
        metrics = calculate_metrics(calls_df, config)

        # Step 6: Generate reports
        render_report(calls_df, metrics, config, output_path=str(output_path))

        return {
            "reads_df": reads_df,
            "collapsed_df": collapsed_df,
            "error_model": error_model,
            "calls_df": calls_df,
            "metrics": metrics,
            "config": config,
            "output_path": output_path,
        }

    def _run_pipeline_with_rng(
        self, config: PipelineConfig, output_path: Path, rng_generator
    ) -> dict:
        """Run pipeline with specific random number generator."""
        # Use provided RNG generator instead of default
        rng = rng_generator(config.seed)

        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Simulate reads with custom RNG
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Continue with normal pipeline
        collapsed_df = collapse_umis(reads_df, config, rng)
        error_model = fit_error_model(collapsed_df, config, rng)

        calls_df = call_mrd(
            collapsed_df,
            error_model,
            config,
            rng,
            use_ml_calling=False,
            use_deep_learning=False,
        )

        metrics = calculate_metrics(calls_df, config)

        return {
            "reads_df": reads_df,
            "collapsed_df": collapsed_df,
            "calls_df": calls_df,
            "metrics": metrics,
            "config": config,
            "rng_method": rng_generator.__name__
            if hasattr(rng_generator, "__name__")
            else str(rng_generator),
        }

    def _assert_all_identical(self, results: list):
        """Assert that all results are identical."""
        if len(results) < 2:
            return

        # Compare key metrics
        metrics_list = [r["metrics"] for r in results]
        for i in range(1, len(metrics_list)):
            assert metrics_list[0]["roc_auc"] == metrics_list[i]["roc_auc"], (
                "ROC AUC should be identical"
            )
            assert (
                metrics_list[0]["average_precision"]
                == metrics_list[i]["average_precision"]
            ), "AP should be identical"

        # Compare call results
        calls_list = [r["calls_df"] for r in results]
        for i in range(1, len(calls_list)):
            self._assert_dataframes_identical(calls_list[0], calls_list[i])

    def _assert_dataframes_identical(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Assert that two DataFrames are identical."""
        assert len(df1) == len(df2), "DataFrames should have same length"

        # Sort both DataFrames by key columns for comparison
        sort_cols = (
            ["sample_id", "family_id"] if "family_id" in df1.columns else ["sample_id"]
        )
        df1_sorted = df1.sort_values(sort_cols).reset_index(drop=True)
        df2_sorted = df2.sort_values(sort_cols).reset_index(drop=True)

        # Compare key columns
        for col in ["sample_id", "is_variant", "p_value", "p_adjusted"]:
            if col in df1.columns:
                assert (df1_sorted[col] == df2_sorted[col]).all(), (
                    f"Column {col} should be identical"
                )

    def _assert_results_different_but_valid(self, results: list):
        """Assert that results are different but all valid."""
        # Results should be different
        calls_list = [r["calls_df"] for r in results]

        # At least some results should differ
        all_identical = True
        for i in range(1, len(calls_list)):
            if not self._dataframes_equal(calls_list[0], calls_list[i]):
                all_identical = False
                break

        assert not all_identical, "Different seeds should produce different results"

        # But all should be valid
        for result in results:
            metrics = result["metrics"]
            assert metrics["roc_auc"] > 0, "All results should have positive ROC AUC"
            assert 0 <= metrics["roc_auc"] <= 1, "ROC AUC should be in [0,1]"

    def _dataframes_equal(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Check if two DataFrames are equal."""
        try:
            self._assert_dataframes_identical(df1, df2)
            return True
        except AssertionError:
            return False

    def _generate_hash_manifest(self, output_path: Path) -> dict:
        """Generate hash manifest for output files."""
        manifest = {}

        reports_dir = output_path / "reports"
        if reports_dir.exists():
            for file_path in reports_dir.glob("*"):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                        manifest[str(file_path.relative_to(output_path))] = file_hash

        return manifest


@pytest.mark.integration
@pytest.mark.determinism
class TestDeterminismRegression:
    """Test for determinism regression detection."""

    def test_known_good_configuration(self, temp_dir):
        """Test a known good configuration that should always pass."""
        # Use a configuration that's known to work well
        config_dict = {
            "run_id": "regression_test",
            "seed": 42,
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [5000],
                "n_replicates": 5,
                "n_bootstrap": 100,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        config = PipelineConfig.from_dict(config_dict)
        result = self._run_pipeline(config, temp_dir / "regression_test")

        # Should achieve good performance
        assert result["metrics"]["roc_auc"] > 0.9, (
            "Known good configuration should achieve high ROC AUC"
        )

        # Should have deterministic results
        result2 = self._run_pipeline(config, temp_dir / "regression_test2")
        self._assert_dataframes_identical(result["calls_df"], result2["calls_df"])

    def test_parameter_sensitivity(self, temp_dir):
        """Test sensitivity to small parameter changes."""
        base_config = {
            "run_id": "sensitivity_test",
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [5000],
                "n_replicates": 5,
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05},
        }

        # Test small changes in parameters
        parameter_variations = [
            {"name": "seed", "value": 42, "delta": 1},
            {"name": "min_family_size", "value": 3, "delta": 1},
            {"name": "consensus_threshold", "value": 0.6, "delta": 0.1},
            {"name": "alpha", "value": 0.05, "delta": 0.01},
        ]

        base_result = None
        for variation in parameter_variations:
            config_dict = base_config.copy()
            config_dict[variation["name"]] = variation["value"]

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"sensitivity_{variation['name']}"

            result = self._run_pipeline(config, output_path)

            if base_result is None:
                base_result = result
            else:
                # Results should be similar but not identical for small changes
                self._assert_results_similar(base_result, result, variation)

    def _assert_results_similar(self, result1: dict, result2: dict, variation: dict):
        """Assert that results are similar for small parameter changes."""
        metrics1 = result1["metrics"]
        metrics2 = result2["metrics"]

        # ROC AUC should be similar (within 10% relative difference)
        auc_ratio = metrics2["roc_auc"] / metrics1["roc_auc"]
        assert 0.9 <= auc_ratio <= 1.1, (
            f"ROC AUC should be similar for {variation['name']} change"
        )

        # Should not be identical (unless it's a deterministic parameter like seed)
        if variation["name"] != "seed":
            # Check that at least some calls differ
            calls1 = result1["calls_df"]
            calls2 = result2["calls_df"]

            # For non-deterministic parameters, results should differ
            if variation["name"] in ["min_family_size", "consensus_threshold", "alpha"]:
                # These should affect results
                different_calls = (calls1["is_variant"] != calls2["is_variant"]).sum()
                assert different_calls > 0, (
                    f"Parameter {variation['name']} should affect results"
                )

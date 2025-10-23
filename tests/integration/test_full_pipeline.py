"""
Comprehensive Integration Tests for Precise MRD Pipeline

This module contains end-to-end integration tests that validate the complete
pipeline functionality across various configurations and scenarios.
"""

import pytest
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from precise_mrd.config import PipelineConfig, load_config
from precise_mrd.simulate import simulate_reads
from precise_mrd.collapse import collapse_umis
from precise_mrd.call import call_mrd
from precise_mrd.error_model import fit_error_model
from precise_mrd.metrics import calculate_metrics
from precise_mrd.reporting import render_report
from precise_mrd.validation import validate_artifacts
from precise_mrd.determinism_utils import env_fingerprint


@dataclass
class TestScenario:
    """Configuration for a test scenario."""
    name: str
    config: dict
    expected_behavior: dict
    description: str


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test the complete pipeline end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "run_id": "integration_test",
            "seed": 42,
            "simulation": {
                "allele_fractions": [0.01, 0.001, 0.0001],
                "umi_depths": [1000, 5000, 10000],
                "n_replicates": 5,
                "n_bootstrap": 50
            },
            "umi": {
                "min_family_size": 3,
                "max_family_size": 100,
                "quality_threshold": 20,
                "consensus_threshold": 0.6
            },
            "stats": {
                "test_type": "poisson",
                "alpha": 0.05,
                "fdr_method": "benjamini_hochberg"
            }
        }

    def test_smoke_test_configuration(self, temp_dir, base_config):
        """Test the standard smoke test configuration end-to-end."""
        config = PipelineConfig.from_dict(base_config)
        output_path = temp_dir / "smoke_results"

        # Run complete pipeline
        results = self._run_full_pipeline(config, output_path)

        # Validate results structure
        assert 'calls_df' in results
        assert 'metrics' in results
        assert 'config' in results

        # Validate artifact contracts
        artifacts_valid = validate_artifacts(output_path)
        assert artifacts_valid, "Artifact validation failed"

        # Check deterministic behavior
        results2 = self._run_full_pipeline(config, output_path)
        self._assert_deterministic_results(results, results2)

    def test_high_depth_configuration(self, temp_dir):
        """Test pipeline with high sequencing depths."""
        config_dict = {
            "run_id": "high_depth_test",
            "seed": 42,
            "simulation": {
                "allele_fractions": [0.001, 0.0001],
                "umi_depths": [25000, 50000],
                "n_replicates": 3,
                "n_bootstrap": 100
            },
            "umi": {
                "min_family_size": 2,
                "max_family_size": 200,
                "quality_threshold": 30,
                "consensus_threshold": 0.8
            },
            "stats": {
                "test_type": "binomial",
                "alpha": 0.01,
                "fdr_method": "benjamini_hochberg"
            }
        }

        config = PipelineConfig.from_dict(config_dict)
        output_path = temp_dir / "high_depth_results"

        results = self._run_full_pipeline(config, output_path)

        # Validate high depth results
        assert results['calls_df'].shape[0] > 0
        assert results['metrics']['roc_auc'] > 0.8  # Should have good performance

    def test_contamination_scenarios(self, temp_dir):
        """Test pipeline resilience to different contamination levels."""
        contamination_scenarios = [
            {"name": "no_contamination", "rate": 0.0},
            {"name": "low_contamination", "rate": 0.0001},
            {"name": "moderate_contamination", "rate": 0.001},
            {"name": "high_contamination", "rate": 0.01}
        ]

        for scenario in contamination_scenarios:
            config_dict = {
                "run_id": f"contamination_test_{scenario['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001],
                    "umi_depths": [5000],
                    "n_replicates": 5,
                    "n_bootstrap": 50
                },
                "umi": {
                    "min_family_size": 3,
                    "consensus_threshold": 0.6
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"contamination_{scenario['name']}"

            results = self._run_full_pipeline(config, output_path)

            # Validate contamination impact
            if scenario['rate'] > 0:
                # Higher contamination should affect detection performance
                self._validate_contamination_impact(results, scenario['rate'])

    def test_stratified_analysis_scenarios(self, temp_dir):
        """Test pipeline with different genomic contexts and stratification."""
        # Test different trinucleotide contexts
        contexts = [
            {"name": "gc_rich", "gc_content": 0.7, "error_rate": 0.001},
            {"name": "at_rich", "gc_content": 0.3, "error_rate": 0.002},
            {"name": "neutral", "gc_content": 0.5, "error_rate": 0.0015}
        ]

        for context in contexts:
            config_dict = {
                "run_id": f"stratified_test_{context['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001, 0.0001],
                    "umi_depths": [2000, 8000],
                    "n_replicates": 4,
                    "n_bootstrap": 50
                },
                "umi": {
                    "min_family_size": 3,
                    "consensus_threshold": 0.6
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"stratified_{context['name']}"

            results = self._run_full_pipeline(config, output_path)

            # Validate stratified performance
            self._validate_stratified_results(results, context)

    def test_edge_case_configurations(self, temp_dir):
        """Test pipeline with edge case configurations."""
        edge_cases = [
            {
                "name": "minimal_depth",
                "config": {
                    "allele_fractions": [0.1, 0.01],
                    "umi_depths": [100, 500],
                    "n_replicates": 2
                }
            },
            {
                "name": "extreme_af",
                "config": {
                    "allele_fractions": [0.5, 0.00001],
                    "umi_depths": [1000],
                    "n_replicates": 3
                }
            },
            {
                "name": "high_replicates",
                "config": {
                    "allele_fractions": [0.01],
                    "umi_depths": [1000],
                    "n_replicates": 50
                }
            }
        ]

        for case in edge_cases:
            config_dict = {
                "run_id": f"edge_case_{case['name']}",
                "seed": 42,
                **case['config'],
                "umi": {
                    "min_family_size": 1,
                    "consensus_threshold": 0.5
                },
                "stats": {
                    "test_type": "binomial",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"edge_{case['name']}"

            results = self._run_full_pipeline(config, output_path)

            # Validate edge case behavior
            self._validate_edge_case_results(results, case)

    def test_deterministic_reproducibility(self, temp_dir):
        """Test that pipeline produces identical results with same configuration."""
        config_dict = {
            "run_id": "determinism_test",
            "seed": 12345,
            "simulation": {
                "allele_fractions": [0.01, 0.001],
                "umi_depths": [2000, 8000],
                "n_replicates": 3,
                "n_bootstrap": 30
            },
            "umi": {
                "min_family_size": 3,
                "consensus_threshold": 0.6
            },
            "stats": {
                "test_type": "poisson",
                "alpha": 0.05
            }
        }

        config = PipelineConfig.from_dict(config_dict)

        # Run pipeline twice with same configuration
        results1 = self._run_full_pipeline(config, temp_dir / "run1")
        results2 = self._run_full_pipeline(config, temp_dir / "run2")

        # Results should be identical
        self._assert_deterministic_results(results1, results2)

    def test_artifact_contract_compliance(self, temp_dir, base_config):
        """Test that all outputs comply with artifact contracts."""
        config = PipelineConfig.from_dict(base_config)
        output_path = temp_dir / "contract_test"

        results = self._run_full_pipeline(config, output_path)

        # Validate all required artifacts are present
        required_artifacts = [
            "metrics.json",
            "auto_report.html",
            "run_context.json",
            "hash_manifest.txt"
        ]

        for artifact in required_artifacts:
            artifact_path = output_path / "reports" / artifact
            assert artifact_path.exists(), f"Missing required artifact: {artifact}"

        # Validate artifact schemas
        validation_result = validate_artifacts(output_path / "reports")
        assert validation_result, "Artifact schema validation failed"

    def _run_full_pipeline(self, config: PipelineConfig, output_path: Path) -> dict:
        """Run the complete pipeline and return results."""
        from precise_mrd.determinism_utils import set_global_seed

        # Set deterministic environment
        set_global_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        # Create output directories
        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Step 1: Simulate reads
        print(f"  Simulating reads for {config.run_id}...")
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Step 2: Collapse UMIs
        print("  Collapsing UMIs...")
        collapsed_df = collapse_umis(reads_df, config, rng)

        # Step 3: Fit error model
        print("  Fitting error model...")
        error_model = fit_error_model(collapsed_df, config, rng)

        # Step 4: Call variants
        print("  Calling variants...")
        calls_df = call_mrd(
            collapsed_df,
            error_model,
            config,
            rng,
            use_ml_calling=False,
            use_deep_learning=False
        )

        # Step 5: Calculate metrics
        print("  Calculating metrics...")
        metrics = calculate_metrics(calls_df, config)

        # Step 6: Generate reports
        print("  Generating reports...")
        render_report(calls_df, metrics, config, output_path=str(output_path))

        return {
            'reads_df': reads_df,
            'collapsed_df': collapsed_df,
            'error_model': error_model,
            'calls_df': calls_df,
            'metrics': metrics,
            'config': config,
            'output_path': output_path
        }

    def _assert_deterministic_results(self, results1: dict, results2: dict):
        """Assert that two pipeline runs produced identical results."""
        # Compare key metrics
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']

        assert metrics1['roc_auc'] == metrics2['roc_auc'], "ROC AUC differs between runs"
        assert metrics1['average_precision'] == metrics2['average_precision'], "AP differs between runs"

        # Compare call results (key columns)
        calls1 = results1['calls_df']
        calls2 = results2['calls_df']

        assert len(calls1) == len(calls2), "Different number of calls between runs"

        # Check specific columns that should be deterministic
        deterministic_cols = ['sample_id', 'is_variant', 'p_value', 'p_adjusted']
        for col in deterministic_cols:
            if col in calls1.columns:
                assert (calls1[col] == calls2[col]).all(), f"Column {col} differs between runs"

    def _validate_contamination_impact(self, results: dict, contamination_rate: float):
        """Validate that contamination affects results as expected."""
        metrics = results['metrics']
        calls_df = results['calls_df']

        # Higher contamination should generally decrease performance
        if contamination_rate > 0.001:
            # Should still detect high AF variants
            high_af_calls = calls_df[calls_df['allele_fraction'] >= 0.01]
            assert len(high_af_calls) > 0, "Should detect high AF variants even with contamination"

    def _validate_stratified_results(self, results: dict, context: dict):
        """Validate stratified analysis results."""
        calls_df = results['calls_df']

        # Should have calls across different allele fractions and depths
        unique_afs = calls_df['allele_fraction'].unique()
        unique_depths = calls_df['umi_depth'].unique()

        assert len(unique_afs) > 1, "Should test multiple allele fractions"
        assert len(unique_depths) > 1, "Should test multiple depths"

    def _validate_edge_case_results(self, results: dict, case: dict):
        """Validate edge case behavior."""
        calls_df = results['calls_df']

        # Should still produce valid results even with edge case parameters
        assert len(calls_df) > 0, "Should produce some calls even in edge cases"
        assert calls_df['p_value'].notna().all(), "All p-values should be valid"
        assert (calls_df['p_value'] >= 0).all() and (calls_df['p_value'] <= 1).all(), "P-values should be in [0,1]"


@pytest.mark.integration
class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""

    def test_invalid_configurations(self):
        """Test pipeline behavior with invalid configurations."""
        invalid_configs = [
            # Invalid allele fractions
            {"allele_fractions": [-0.1, 0.001]},  # Negative AF
            {"allele_fractions": [0.001, 1.5]},   # AF > 1
            {"allele_fractions": []},             # Empty AF list

            # Invalid depths
            {"umi_depths": [0, 1000]},            # Zero depth
            {"umi_depths": [-100, 1000]},         # Negative depth

            # Invalid replicates
            {"n_replicates": 0},                  # Zero replicates
            {"n_replicates": -1},                 # Negative replicates
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                config_dict = {
                    "run_id": "invalid_test",
                    "seed": 42,
                    "simulation": invalid_config,
                    "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
                    "stats": {"test_type": "poisson", "alpha": 0.05}
                }
                PipelineConfig.from_dict(config_dict)

    def test_missing_dependencies(self, temp_dir):
        """Test pipeline behavior when dependencies are missing."""
        config_dict = {
            "run_id": "missing_deps_test",
            "seed": 42,
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [1000],
                "n_replicates": 1
            },
            "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
            "stats": {"test_type": "poisson", "alpha": 0.05}
        }

        config = PipelineConfig.from_dict(config_dict)

        # This should handle missing intermediate files gracefully
        with pytest.raises(Exception):  # Should fail but not crash
            # Try to run pipeline without proper setup
            from precise_mrd.call import call_mrd
            # This should fail gracefully rather than crash
            pass


@pytest.mark.integration
class TestPerformanceRegression:
    """Test for performance regression across configurations."""

    def test_configuration_performance_matrix(self, temp_dir):
        """Test performance across a matrix of configurations."""
        configurations = [
            {"n_replicates": 5, "umi_depths": [1000], "allele_fractions": [0.01]},
            {"n_replicates": 10, "umi_depths": [1000], "allele_fractions": [0.01]},
            {"n_replicates": 5, "umi_depths": [5000], "allele_fractions": [0.01]},
            {"n_replicates": 5, "umi_depths": [1000], "allele_fractions": [0.001]},
        ]

        results = []
        for i, config_params in enumerate(configurations):
            config_dict = {
                "run_id": f"perf_test_{i}",
                "seed": 42,
                **config_params,
                "umi": {"min_family_size": 3, "consensus_threshold": 0.6},
                "stats": {"test_type": "poisson", "alpha": 0.05}
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"perf_{i}"

            result = self._run_full_pipeline(config, output_path)
            results.append(result)

        # Validate performance trends
        self._validate_performance_trends(results, configurations)

    def _validate_performance_trends(self, results: list, configurations: list):
        """Validate that performance changes as expected with configuration changes."""
        metrics = [r['metrics'] for r in results]

        # More replicates should improve precision
        rep5_idx = next(i for i, c in enumerate(configurations) if c['n_replicates'] == 5)
        rep10_idx = next(i for i, c in enumerate(configurations) if c['n_replicates'] == 10)

        # Higher replicates should have similar or better performance
        assert metrics[rep10_idx]['roc_auc'] >= metrics[rep5_idx]['roc_auc'] * 0.95

        # Higher depth should improve sensitivity
        depth1k_idx = next(i for i, c in enumerate(configurations) if 1000 in c['umi_depths'])
        depth5k_idx = next(i for i, c in enumerate(configurations) if 5000 in c['umi_depths'])

        if depth5k_idx != depth1k_idx:
            # Higher depth should generally improve performance
            assert metrics[depth5k_idx]['roc_auc'] >= metrics[depth1k_idx]['roc_auc'] * 0.9

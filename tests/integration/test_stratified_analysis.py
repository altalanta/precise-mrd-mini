"""
Integration Tests for Stratified Analysis

This module tests the pipeline's stratified analysis capabilities across
different genomic contexts and sequencing conditions.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from precise_mrd.config import PipelineConfig
from precise_mrd.simulate import simulate_reads
from precise_mrd.collapse import collapse_umis
from precise_mrd.call import call_mrd
from precise_mrd.error_model import fit_error_model
from precise_mrd.metrics import calculate_metrics
from precise_mrd.determinism_utils import set_global_seed


@pytest.mark.integration
@pytest.mark.stratified
class TestStratifiedAnalysis:
    """Test stratified analysis across different genomic contexts."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_trinucleotide_context_stratification(self, temp_dir):
        """Test pipeline performance across different trinucleotide contexts."""
        contexts = [
            {"name": "cpg_islands", "gc_content": 0.8, "error_rate": 0.0005, "context_bias": 2.0},
            {"name": "coding_regions", "gc_content": 0.6, "error_rate": 0.001, "context_bias": 1.0},
            {"name": "intergenic", "gc_content": 0.4, "error_rate": 0.002, "context_bias": 0.5},
            {"name": "repetitive", "gc_content": 0.3, "error_rate": 0.003, "context_bias": 0.3}
        ]

        results = []
        for context in contexts:
            config_dict = {
                "run_id": f"context_{context['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001, 0.0001],
                    "umi_depths": [2000, 10000],
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
            output_path = temp_dir / f"context_{context['name']}"

            result = self._run_stratified_pipeline(config, output_path, context)
            results.append(result)

        # Validate context-specific performance
        self._validate_context_performance(results, contexts)

    def test_depth_stratification(self, temp_dir):
        """Test pipeline performance across different sequencing depths."""
        depth_configs = [
            {"depths": [500, 1000], "name": "low_depth"},
            {"depths": [2000, 5000], "name": "medium_depth"},
            {"depths": [10000, 25000], "name": "high_depth"},
            {"depths": [50000, 100000], "name": "ultra_high_depth"}
        ]

        for depth_config in depth_configs:
            config_dict = {
                "run_id": f"depth_{depth_config['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001, 0.0001],
                    "umi_depths": depth_config['depths'],
                    "n_replicates": 3,
                    "n_bootstrap": 50
                },
                "umi": {
                    "min_family_size": 2,
                    "consensus_threshold": 0.6
                },
                "stats": {
                    "test_type": "binomial",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"depth_{depth_config['name']}"

            result = self._run_stratified_pipeline(config, output_path, depth_config)
            results.append(result)

        # Validate depth-dependent performance
        self._validate_depth_performance(results, depth_configs)

    def test_allele_frequency_stratification(self, temp_dir):
        """Test pipeline performance across different allele frequency ranges."""
        af_configs = [
            {"afs": [0.1, 0.05, 0.01], "name": "high_af"},
            {"afs": [0.01, 0.005, 0.001], "name": "medium_af"},
            {"afs": [0.001, 0.0005, 0.0001], "name": "low_af"},
            {"afs": [0.0001, 0.00005, 0.00001], "name": "ultra_low_af"}
        ]

        for af_config in af_configs:
            config_dict = {
                "run_id": f"af_{af_config['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": af_config['afs'],
                    "umi_depths": [5000, 15000],
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
            output_path = temp_dir / f"af_{af_config['name']}"

            result = self._run_stratified_pipeline(config, output_path, af_config)
            results.append(result)

        # Validate AF-dependent performance
        self._validate_af_performance(results, af_configs)

    def test_multi_dimensional_stratification(self, temp_dir):
        """Test pipeline with multiple stratification dimensions simultaneously."""
        # Test combination of context, depth, and AF
        multi_strata = [
            {
                "name": "gc_rich_high_af",
                "context": {"gc_content": 0.7, "error_rate": 0.0005},
                "depths": [10000, 25000],
                "afs": [0.01, 0.005]
            },
            {
                "name": "at_rich_low_af",
                "context": {"gc_content": 0.3, "error_rate": 0.002},
                "depths": [5000, 15000],
                "afs": [0.001, 0.0005]
            }
        ]

        for stratum in multi_strata:
            config_dict = {
                "run_id": f"multi_{stratum['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": stratum['afs'],
                    "umi_depths": stratum['depths'],
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
            output_path = temp_dir / f"multi_{stratum['name']}"

            result = self._run_stratified_pipeline(config, output_path, stratum)
            results.append(result)

        # Validate multi-dimensional stratification
        self._validate_multi_stratification(results, multi_strata)

    def _run_stratified_pipeline(self, config: PipelineConfig, output_path: Path, stratum_info: dict) -> dict:
        """Run pipeline with stratified analysis."""
        set_global_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Simulate reads with context-specific characteristics
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Add context information to reads if available
        if 'context' in stratum_info:
            reads_df = self._add_context_info(reads_df, stratum_info['context'], rng)

        # Continue with normal pipeline
        collapsed_df = collapse_umis(reads_df, config, rng)
        error_model = fit_error_model(collapsed_df, config, rng)

        calls_df = call_mrd(
            collapsed_df,
            error_model,
            config,
            rng,
            use_ml_calling=False,
            use_deep_learning=False
        )

        metrics = calculate_metrics(calls_df, config)

        return {
            'reads_df': reads_df,
            'collapsed_df': collapsed_df,
            'calls_df': calls_df,
            'metrics': metrics,
            'config': config,
            'stratum_info': stratum_info
        }

    def _add_context_info(self, reads_df: pd.DataFrame, context: dict, rng: np.random.Generator) -> pd.DataFrame:
        """Add genomic context information to reads."""
        enhanced_df = reads_df.copy()

        # Add context-specific error rates
        base_error_rate = context.get('error_rate', 0.001)

        # Add some variation based on context
        n_reads = len(enhanced_df)
        context_error_rates = rng.normal(
            loc=base_error_rate,
            scale=base_error_rate * 0.2,
            size=n_reads
        )

        # Ensure error rates are positive
        context_error_rates = np.maximum(context_error_rates, 0.0001)

        enhanced_df['context_error_rate'] = context_error_rates
        enhanced_df['gc_content'] = context.get('gc_content', 0.5)

        return enhanced_df

    def _validate_context_performance(self, results: list, contexts: list):
        """Validate performance differences across genomic contexts."""
        # Extract metrics for each context
        context_metrics = {}
        for result, context in zip(results, contexts):
            context_metrics[context['name']] = result['metrics']

        # GC-rich contexts should generally perform better
        gc_rich_idx = next(i for i, c in enumerate(contexts) if c['name'] == 'cpg_islands')
        intergenic_idx = next(i for i, c in enumerate(contexts) if c['name'] == 'intergenic')

        gc_rich_auc = context_metrics[contexts[gc_rich_idx]['name']]['roc_auc']
        intergenic_auc = context_metrics[contexts[intergenic_idx]['name']]['roc_auc']

        # GC-rich should perform at least as well as intergenic
        assert gc_rich_auc >= intergenic_auc * 0.9, "GC-rich context should perform well"

    def _validate_depth_performance(self, results: list, depth_configs: list):
        """Validate performance scaling with sequencing depth."""
        # Extract metrics for each depth configuration
        depth_metrics = {}
        for result, config in zip(results, depth_configs):
            depth_metrics[config['name']] = result['metrics']

        # Higher depth should generally improve performance
        low_depth_idx = next(i for i, c in enumerate(depth_configs) if c['name'] == 'low_depth')
        high_depth_idx = next(i for i, c in enumerate(depth_configs) if c['name'] == 'high_depth')

        low_auc = depth_metrics[depth_configs[low_depth_idx]['name']]['roc_auc']
        high_auc = depth_metrics[depth_configs[high_depth_idx]['name']]['roc_auc']

        # Higher depth should improve performance
        assert high_auc >= low_auc * 1.1, "Higher depth should improve performance"

    def _validate_af_performance(self, results: list, af_configs: list):
        """Validate performance across allele frequency ranges."""
        # Extract metrics for each AF configuration
        af_metrics = {}
        for result, config in zip(results, af_configs):
            af_metrics[config['name']] = result['metrics']

        # Higher AF should generally perform better
        high_af_idx = next(i for i, c in enumerate(af_configs) if c['name'] == 'high_af')
        low_af_idx = next(i for i, c in enumerate(af_configs) if c['name'] == 'low_af')

        high_auc = af_metrics[af_configs[high_af_idx]['name']]['roc_auc']
        low_auc = af_metrics[af_configs[low_af_idx]['name']]['roc_auc']

        # Higher AF should perform better
        assert high_auc >= low_auc * 1.2, "Higher AF should perform significantly better"

    def _validate_multi_stratification(self, results: list, multi_strata: list):
        """Validate multi-dimensional stratification results."""
        for result, stratum in zip(results, multi_strata):
            calls_df = result['calls_df']

            # Should have calls across multiple dimensions
            unique_afs = calls_df['allele_fraction'].unique()
            unique_depths = calls_df['umi_depth'].unique()

            assert len(unique_afs) > 1, "Should test multiple allele fractions"
            assert len(unique_depths) > 1, "Should test multiple depths"

            # Should have reasonable performance even in challenging strata
            metrics = result['metrics']
            assert metrics['roc_auc'] > 0.7, f"Should maintain good performance in {stratum['name']}"


@pytest.mark.integration
@pytest.mark.stratified
class TestPowerAnalysisStratification:
    """Test power analysis across different strata."""

    def test_power_by_depth_stratum(self, temp_dir):
        """Test statistical power across different depth strata."""
        depth_strata = [
            {"min_depth": 500, "max_depth": 1000, "name": "low_depth"},
            {"min_depth": 5000, "max_depth": 10000, "name": "medium_depth"},
            {"min_depth": 25000, "max_depth": 50000, "name": "high_depth"}
        ]

        for stratum in depth_strata:
            config_dict = {
                "run_id": f"power_depth_{stratum['name']}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.005, 0.001, 0.0005],
                    "umi_depths": [stratum['min_depth'], stratum['max_depth']],
                    "n_replicates": 10,
                    "n_bootstrap": 100
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
            output_path = temp_dir / f"power_depth_{stratum['name']}"

            result = self._run_power_analysis(config, output_path, stratum)

            # Validate power analysis results
            self._validate_power_by_depth(result, stratum)

    def _run_power_analysis(self, config: PipelineConfig, output_path: Path, stratum_info: dict) -> dict:
        """Run power analysis for a given stratum."""
        set_global_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Run multiple replicates for power analysis
        power_results = []
        for rep in range(config.simulation.n_replicates):
            rep_config = config.copy()
            rep_config.seed = config.seed + rep * 1000
            rep_config.run_id = f"{config.run_id}_rep_{rep}"

            reads_df = simulate_reads(rep_config, rng, output_path=str(output_path))
            collapsed_df = collapse_umis(reads_df, rep_config, rng)
            error_model = fit_error_model(collapsed_df, rep_config, rng)

            calls_df = call_mrd(
                collapsed_df,
                error_model,
                rep_config,
                rng,
                use_ml_calling=False,
                use_deep_learning=False
            )

            power_results.append(calls_df)

        # Aggregate power analysis results
        combined_calls = pd.concat(power_results, ignore_index=True)

        # Calculate power metrics
        power_metrics = self._calculate_power_metrics(combined_calls, config)

        return {
            'power_results': power_results,
            'combined_calls': combined_calls,
            'power_metrics': power_metrics,
            'config': config,
            'stratum_info': stratum_info
        }

    def _calculate_power_metrics(self, calls_df: pd.DataFrame, config: PipelineConfig) -> dict:
        """Calculate power analysis metrics."""
        # Group by allele fraction and depth
        power_by_stratum = {}

        for af in config.simulation.allele_fractions:
            for depth in config.simulation.umi_depths:
                stratum_calls = calls_df[
                    (calls_df['allele_fraction'] == af) &
                    (calls_df['umi_depth'] == depth)
                ]

                if len(stratum_calls) > 0:
                    # Calculate detection power
                    detection_rate = stratum_calls['is_variant'].mean()
                    sensitivity = stratum_calls['significant'].mean() if 'significant' in stratum_calls.columns else detection_rate

                    power_by_stratum[f"af_{af}_depth_{depth}"] = {
                        'detection_rate': detection_rate,
                        'sensitivity': sensitivity,
                        'n_samples': len(stratum_calls),
                        'mean_p_value': stratum_calls['p_value'].mean()
                    }

        return power_by_stratum

    def _validate_power_by_depth(self, result: dict, stratum: dict):
        """Validate power analysis results for depth stratum."""
        power_metrics = result['power_metrics']

        # Higher depths should have higher power
        if stratum['name'] == 'high_depth':
            # Should achieve good power even for low AF
            low_af_keys = [k for k in power_metrics.keys() if 'af_0.001' in k or 'af_0.0005' in k]
            for key in low_af_keys:
                metrics = power_metrics[key]
                assert metrics['sensitivity'] > 0.7, f"Should achieve good power in high depth stratum for {key}"

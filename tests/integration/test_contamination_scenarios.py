"""
Integration Tests for Contamination Analysis

This module tests the pipeline's robustness to contamination and validates
contamination detection and mitigation strategies.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from precise_mrd.config import PipelineConfig
from precise_mrd.simulate import simulate_reads
from precise_mrd.sim.contamination import simulate_contamination
from precise_mrd.collapse import collapse_umis
from precise_mrd.call import call_mrd
from precise_mrd.error_model import fit_error_model
from precise_mrd.metrics import calculate_metrics
from precise_mrd.determinism_utils import set_global_seed


@pytest.mark.integration
@pytest.mark.contamination
class TestContaminationIntegration:
    """Test contamination effects on the full pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_index_hopping_simulation(self, temp_dir):
        """Test pipeline resilience to index hopping contamination."""
        # Test different index hopping rates
        hop_rates = [0.0, 0.0001, 0.001, 0.01]

        results = []
        for hop_rate in hop_rates:
            config_dict = {
                "run_id": f"index_hopping_{hop_rate}",
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
            output_path = temp_dir / f"hop_{hop_rate}"

            result = self._run_pipeline_with_contamination(
                config, output_path, contamination_type="index_hopping", rate=hop_rate
            )
            results.append(result)

        # Validate contamination impact
        self._validate_index_hopping_impact(results, hop_rates)

    def test_sample_carryover_contamination(self, temp_dir):
        """Test pipeline resilience to sample carryover contamination."""
        carryover_rates = [0.0, 0.00001, 0.0001, 0.001]

        for rate in carryover_rates:
            config_dict = {
                "run_id": f"carryover_{rate}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001, 0.0001],
                    "umi_depths": [10000],
                    "n_replicates": 3
                },
                "umi": {
                    "min_family_size": 2,
                    "consensus_threshold": 0.7
                },
                "stats": {
                    "test_type": "binomial",
                    "alpha": 0.01
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"carryover_{rate}"

            result = self._run_pipeline_with_contamination(
                config, output_path, contamination_type="carryover", rate=rate
            )

            # Validate carryover impact
            self._validate_carryover_impact(result, rate)

    def test_multi_sample_contamination(self, temp_dir):
        """Test contamination in multiplexed samples."""
        # Simulate 8-plex with varying contamination
        n_samples = 8
        contamination_matrix = np.random.uniform(0, 0.001, (n_samples, n_samples))

        config_dict = {
            "run_id": "multiplex_test",
            "seed": 42,
            "simulation": {
                "allele_fractions": [0.01],
                "umi_depths": [2000],
                "n_replicates": 4
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
        output_path = temp_dir / "multiplex"

        result = self._run_multiplex_pipeline(config, output_path, contamination_matrix)

        # Validate multiplex performance
        self._validate_multiplex_results(result, contamination_matrix)

    def test_umi_based_contamination_detection(self, temp_dir):
        """Test UMI-based methods for contamination detection."""
        # Test different UMI family size thresholds
        family_thresholds = [1, 3, 5, 10]

        for threshold in family_thresholds:
            config_dict = {
                "run_id": f"umi_threshold_{threshold}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01, 0.001],
                    "umi_depths": [5000],
                    "n_replicates": 5
                },
                "umi": {
                    "min_family_size": threshold,
                    "consensus_threshold": 0.6
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"umi_{threshold}"

            result = self._run_pipeline_with_contamination(
                config, output_path, contamination_type="index_hopping", rate=0.001
            )

            # Validate UMI filtering effectiveness
            self._validate_umi_filtering(result, threshold)

    def test_background_subtraction_methods(self, temp_dir):
        """Test different background subtraction approaches."""
        subtraction_methods = ["none", "blank_control", "negative_control", "estimated_rate"]

        for method in subtraction_methods:
            config_dict = {
                "run_id": f"subtraction_{method}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01],
                    "umi_depths": [5000],
                    "n_replicates": 5
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
            output_path = temp_dir / f"subtraction_{method}"

            result = self._run_pipeline_with_background_subtraction(
                config, output_path, method, background_rate=0.001
            )

            # Validate subtraction method effectiveness
            self._validate_subtraction_method(result, method)

    def _run_pipeline_with_contamination(self, config: PipelineConfig, output_path: Path,
                                       contamination_type: str, rate: float) -> dict:
        """Run pipeline with simulated contamination."""
        set_global_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        # Create output directories
        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Simulate reads with contamination
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Apply contamination based on type
        if contamination_type == "index_hopping":
            contaminated_df = self._apply_index_hopping(reads_df, rate, rng)
        elif contamination_type == "carryover":
            contaminated_df = self._apply_carryover(reads_df, rate, rng)
        else:
            contaminated_df = reads_df

        # Continue with normal pipeline
        collapsed_df = collapse_umis(contaminated_df, config, rng)
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
            'contaminated_df': contaminated_df,
            'collapsed_df': collapsed_df,
            'calls_df': calls_df,
            'metrics': metrics,
            'config': config,
            'contamination_rate': rate,
            'contamination_type': contamination_type
        }

    def _apply_index_hopping(self, reads_df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
        """Apply index hopping contamination to reads."""
        contaminated_df = reads_df.copy()

        # Identify reads that should be contaminated
        n_reads = len(contaminated_df)
        n_hopped = int(n_reads * rate)

        if n_hopped > 0:
            # Randomly select reads to hop
            hop_indices = rng.choice(n_reads, n_hopped, replace=False)

            # Simulate hopping by changing sample IDs
            unique_samples = contaminated_df['sample_id'].unique()
            for idx in hop_indices:
                current_sample = contaminated_df.at[idx, 'sample_id']
                other_samples = [s for s in unique_samples if s != current_sample]
                if other_samples:
                    new_sample = rng.choice(other_samples)
                    contaminated_df.at[idx, 'sample_id'] = new_sample

        return contaminated_df

    def _apply_carryover(self, reads_df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
        """Apply sample carryover contamination to reads."""
        contaminated_df = reads_df.copy()

        # Simulate carryover by adding low-level contamination from other samples
        n_reads = len(contaminated_df)
        n_carryover = int(n_reads * rate)

        if n_carryover > 0:
            # Create carryover reads
            unique_samples = contaminated_df['sample_id'].unique()

            for _ in range(n_carryover):
                # Choose a random sample to contaminate
                target_sample = rng.choice(unique_samples)
                source_samples = [s for s in unique_samples if s != target_sample]
                source_sample = rng.choice(source_samples)

                # Create carryover read
                carryover_read = contaminated_df[contaminated_df['sample_id'] == source_sample].iloc[0].copy()
                carryover_read['sample_id'] = target_sample
                carryover_read['family_id'] = f"carryover_{rng.integers(1000000)}"

                contaminated_df = pd.concat([contaminated_df, pd.DataFrame([carryover_read])], ignore_index=True)

        return contaminated_df

    def _validate_index_hopping_impact(self, results: list, hop_rates: list):
        """Validate that index hopping affects results as expected."""
        # Extract metrics for each rate
        metrics_by_rate = {}
        for result, rate in zip(results, hop_rates):
            metrics_by_rate[rate] = result['metrics']

        # Higher hop rates should generally decrease performance
        for i in range(1, len(hop_rates)):
            if hop_rates[i] > hop_rates[i-1]:
                # Performance should not improve significantly with higher contamination
                current_auc = metrics_by_rate[hop_rates[i]]['roc_auc']
                prev_auc = metrics_by_rate[hop_rates[i-1]]['roc_auc']
                assert current_auc <= prev_auc * 1.1, f"Unexpected performance improvement with contamination rate {hop_rates[i]}"

    def _validate_carryover_impact(self, result: dict, rate: float):
        """Validate carryover contamination impact."""
        calls_df = result['calls_df']

        # Should still detect true variants even with carryover
        true_variants = calls_df[
            (calls_df['allele_fraction'] >= 0.01) &
            (calls_df['is_variant'] == True)
        ]

        # Should have some true positive detections
        assert len(true_variants) > 0, f"Should detect true variants even with carryover rate {rate}"

    def _validate_multiplex_results(self, result: dict, contamination_matrix: np.ndarray):
        """Validate multiplexed sample results."""
        calls_df = result['calls_df']

        # Should have results for multiple samples
        unique_samples = calls_df['sample_id'].unique()
        assert len(unique_samples) > 1, "Should have results for multiple samples"

        # Each sample should have some calls
        for sample in unique_samples:
            sample_calls = calls_df[calls_df['sample_id'] == sample]
            assert len(sample_calls) > 0, f"Sample {sample} should have calls"

    def _validate_umi_filtering(self, result: dict, threshold: int):
        """Validate UMI-based contamination filtering."""
        calls_df = result['calls_df']
        collapsed_df = result['collapsed_df']

        # Higher thresholds should result in fewer but higher quality calls
        n_calls = len(calls_df)
        mean_family_size = collapsed_df['family_size'].mean()

        # Should have reasonable number of calls
        assert n_calls > 0, f"Should have calls even with family size threshold {threshold}"

        # Mean family size should be above threshold
        assert mean_family_size >= threshold * 0.8, f"Mean family size should be above threshold {threshold}"

    def _validate_subtraction_method(self, result: dict, method: str):
        """Validate background subtraction method effectiveness."""
        calls_df = result['calls_df']

        # Should still produce valid statistical results
        assert calls_df['p_value'].notna().all(), "All p-values should be valid"
        assert (calls_df['p_value'] >= 0).all(), "P-values should be non-negative"
        assert (calls_df['p_value'] <= 1).all(), "P-values should be <= 1"


@pytest.mark.integration
@pytest.mark.contamination
class TestContaminationMitigation:
    """Test contamination mitigation strategies."""

    def test_umi_deduplication_effectiveness(self, temp_dir):
        """Test UMI deduplication for contamination removal."""
        # Test with and without UMI deduplication
        dedup_methods = ["none", "family", "consensus"]

        for method in dedup_methods:
            config_dict = {
                "run_id": f"dedup_{method}",
                "seed": 42,
                "simulation": {
                    "allele_fractions": [0.01],
                    "umi_depths": [5000],
                    "n_replicates": 3
                },
                "umi": {
                    "min_family_size": 2,
                    "consensus_threshold": 0.6
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.05
                }
            }

            config = PipelineConfig.from_dict(config_dict)
            output_path = temp_dir / f"dedup_{method}"

            result = self._run_pipeline_with_deduplication(
                config, output_path, deduplication_method=method
            )

            # Validate deduplication effectiveness
            self._validate_deduplication_method(result, method)

    def _run_pipeline_with_deduplication(self, config: PipelineConfig, output_path: Path,
                                       deduplication_method: str) -> dict:
        """Run pipeline with specified deduplication method."""
        set_global_seed(config.seed)
        rng = np.random.default_rng(config.seed)

        (output_path / "reports").mkdir(parents=True, exist_ok=True)

        # Simulate reads with contamination
        reads_df = simulate_reads(config, rng, output_path=str(output_path))

        # Apply deduplication based on method
        if deduplication_method == "family":
            # More aggressive family-based filtering
            config.umi.min_family_size = max(config.umi.min_family_size, 3)
        elif deduplication_method == "consensus":
            # More aggressive consensus filtering
            config.umi.consensus_threshold = max(config.umi.consensus_threshold, 0.8)

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
            'deduplication_method': deduplication_method
        }

    def _validate_deduplication_method(self, result: dict, method: str):
        """Validate deduplication method effectiveness."""
        calls_df = result['calls_df']
        collapsed_df = result['collapsed_df']

        # Should have fewer but higher quality calls with more aggressive deduplication
        n_calls = len(calls_df)
        mean_quality = collapsed_df['quality_score'].mean()
        mean_consensus = collapsed_df['consensus_agreement'].mean()

        assert n_calls > 0, f"Should have calls with deduplication method {method}"
        assert mean_quality > 0, "Should have positive quality scores"
        assert mean_consensus > 0, "Should have positive consensus scores"

"""MRD calling and statistical testing module."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

from .config import PipelineConfig
from .ml_models import GradientBoostedVariantCaller, EnsembleVariantCaller
from .deep_learning_models import CNNLSTMModel, HybridModel, DeepLearningVariantCaller
from .data_schemas import (
    CollapsedUmisSchema,
    ErrorModelSchema,
    StatisticalCallsSchema,
    MLCallsSchema,
    DLCallsSchema,
)
import pandera as pa


def poisson_test(observed: int, expected: float) -> float:
    """Two-tailed Poisson test."""
    if expected <= 0:
        return 1.0
    
    # Calculate two-tailed p-value
    left_tail = stats.poisson.cdf(observed, expected)
    right_tail = stats.poisson.sf(observed - 1, expected)
    
    # Two-tailed: double the smaller tail, but cap at 1.0
    p_value = 2 * min(left_tail, right_tail)
    return min(p_value, 1.0)


def binomial_test(successes: int, trials: int, p: float) -> float:
    """Two-tailed binomial test."""
    if trials == 0:
        return 1.0
    if p <= 0:
        # Special case: if p=0 and we observed successes, it's very significant
        return 0.0 if successes > 0 else 1.0
    if p >= 1:
        # Special case: if p=1 and we observed failures, it's very significant  
        return 0.0 if successes < trials else 1.0
    return stats.binomtest(successes, trials, p, alternative='two-sided').pvalue


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (vectorized implementation)."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    if m == 0:
        return np.array([], dtype=bool), np.array([])

    # TODO: Implement proper idempotence later
    # For now, always apply full BH procedure

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH procedure for rejection (vectorized)
    # Find the largest k where sorted_p[k] <= (k+1)/m * alpha
    k_values = np.arange(m) + 1  # k = 1, 2, ..., m
    thresholds = k_values / m * alpha
    valid_k = sorted_p <= thresholds

    # Find the maximum k where condition holds (rightmost True)
    if np.any(valid_k):
        max_k = np.max(np.where(valid_k)[0])
        rejected = np.zeros(m, dtype=bool)
        rejected[sorted_indices[:max_k + 1]] = True
    else:
        rejected = np.zeros(m, dtype=bool)

    # Adjusted p-values (Benjamini-Hochberg step-up) - vectorized
    adjusted_p = sorted_p * m / k_values

    # Ensure monotonicity (decreasing from right to left) - vectorized
    # Use cumulative minimum from the right
    adjusted_p = np.minimum.accumulate(adjusted_p[::-1])[::-1]

    # Clip to [0, 1]
    adjusted_p = np.clip(adjusted_p, 0, 1)

    # Reorder to original order
    final_adjusted = np.zeros_like(p_values)
    final_adjusted[sorted_indices] = adjusted_p

    return rejected, final_adjusted


@pa.check_input(pa.DataFrameSchema(CollapsedUmisSchema.to_schema().columns,-
                                 filter_ignore_na=True,
                                 strict=False), "collapsed_df")
@pa.check_input(pa.DataFrameSchema(ErrorModelSchema.to_schema().columns,-
                                 filter_ignore_na=True,
                                 strict=False), "error_model_df")
def call_mrd(
    collapsed_df: pd.DataFrame,
    error_model_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    use_ml_calling: bool = False,
    ml_model_type: str = 'ensemble',
    use_deep_learning: bool = False,
    dl_model_type: str = 'cnn_lstm'
) -> pd.DataFrame:
    """Perform MRD calling with statistical testing or ML-based approaches.

    Args:
        collapsed_df: DataFrame with collapsed UMI data
        error_model_df: DataFrame with error model
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        use_ml_calling: Whether to use ML-based variant calling
        ml_model_type: Type of ML model to use ('ensemble', 'xgboost', 'lightgbm', 'gbm')
        use_deep_learning: Whether to use deep learning-based variant calling
        dl_model_type: Type of deep learning model to use ('cnn_lstm', 'hybrid', 'transformer')

    Returns:
        DataFrame with MRD calls and statistics
    """

    # Use ML-based variant calling if requested
    if use_ml_calling:
        print(f"  Using enhanced ML-based variant calling ({ml_model_type})...")

        # Choose model based on type
        if ml_model_type == 'ensemble':
            ml_caller = EnsembleVariantCaller(config)
        else:
            ml_caller = GradientBoostedVariantCaller(config, ml_model_type)

        # Train the model
        if ml_model_type == 'ensemble':
            training_results = ml_caller.train_ensemble(collapsed_df, rng)
        else:
            training_results = ml_caller.train_model(collapsed_df, rng)

        # Get ML predictions
        ml_probabilities = ml_caller.predict_variants(collapsed_df)

        # Use optimal threshold from training
        optimal_threshold = training_results.get('optimal_threshold', np.median(ml_probabilities))
        ml_calls = (ml_probabilities > optimal_threshold).astype(int)

        # Get feature importance for reporting
        feature_importance = ml_caller.get_feature_importance()

        print(f"  Ensemble trained with {len(ml_caller.models)} models")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Top features: {list(feature_importance.keys())[:3]}")

        # Create results DataFrame (vectorized)
        results_df = pd.DataFrame({
            'sample_id': collapsed_df['sample_id'],
            'family_id': collapsed_df['family_id'],
            'family_size': collapsed_df['family_size'],
            'quality_score': collapsed_df['quality_score'],
            'consensus_agreement': collapsed_df['consensus_agreement'],
            'passes_quality': collapsed_df['passes_quality'],
            'passes_consensus': collapsed_df['passes_consensus'],
            'is_variant': ml_calls,
            'p_value': 1.0 - ml_probabilities,  # Convert probability to p-value-like score
            'ml_probability': ml_probabilities,
            'ml_threshold': optimal_threshold,
            'calling_method': 'ml_ensemble',
            'config_hash': config.config_hash()
        })

        df = MLCallsSchema.validate(results_df)

    # Use deep learning-based variant calling if requested
    elif use_deep_learning:
        print(f"  Using deep learning-based variant calling ({dl_model_type})...")

        # Initialize deep learning caller
        dl_caller = DeepLearningVariantCaller(config, dl_model_type)

        # Train the model
        training_results = dl_caller.train_model(collapsed_df, rng)

        # Get deep learning predictions
        dl_probabilities = dl_caller.predict_variants(collapsed_df)

        # Use optimal threshold from training
        optimal_threshold = training_results.get('optimal_threshold', np.median(dl_probabilities))
        dl_calls = (dl_probabilities > optimal_threshold).astype(int)

        # Get model summary for reporting
        model_summary = dl_caller.get_model_summary()

        print(f"  Deep learning model trained: {model_summary}")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")

        # Create results DataFrame (vectorized)
        results_df = pd.DataFrame({
            'sample_id': collapsed_df['sample_id'],
            'family_id': collapsed_df['family_id'],
            'family_size': collapsed_df['family_size'],
            'quality_score': collapsed_df['quality_score'],
            'consensus_agreement': collapsed_df['consensus_agreement'],
            'passes_quality': collapsed_df['passes_quality'],
            'passes_consensus': collapsed_df['passes_consensus'],
            'is_variant': dl_calls,
            'p_value': 1.0 - dl_probabilities,  # Convert probability to p-value-like score
            'dl_probability': dl_probabilities,
            'dl_threshold': optimal_threshold,
            'calling_method': f'dl_{dl_model_type}',
            'config_hash': config.config_hash()
        })

        df = DLCallsSchema.validate(results_df)

    else:
        # Default: Use statistical testing
        print("  Using statistical variant calling...")

        stats_config = config.stats

        # Handle case where stats config might be a dict
        if isinstance(stats_config, dict):
            test_type = stats_config['test_type']
            alpha = stats_config['alpha']
            fdr_method = stats_config['fdr_method']
        else:
            test_type = stats_config.test_type
            alpha = stats_config.alpha
            fdr_method = stats_config.fdr_method

        # Group by sample for statistical testing
        call_data = []

        for sample_id in collapsed_df['sample_id'].unique():
            sample_data = collapsed_df[collapsed_df['sample_id'] == sample_id]

            if len(sample_data) == 0:
                continue

            # Get sample metadata
            allele_fraction = sample_data['allele_fraction'].iloc[0]

            # Count variants and total families
            n_variants = sample_data['is_variant'].sum()
            n_total = len(sample_data)

            if n_total == 0:
                continue

            # Get expected error rate (average across contexts)
            mean_error_rate = error_model_df['error_rate'].mean()
            expected_variants = n_total * mean_error_rate

            # Perform statistical test
            if test_type == "poisson":
                p_value = poisson_test(n_variants, expected_variants)
            elif test_type == "binomial":
                p_value = binomial_test(n_variants, n_total, mean_error_rate)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            # Calculate effect size
            if expected_variants > 0:
                fold_change = n_variants / expected_variants
            else:
                fold_change = float('inf') if n_variants > 0 else 1.0

            # Quality metrics
            mean_quality = sample_data['quality_score'].mean()
            mean_consensus = sample_data['consensus_agreement'].mean()

            call_data.append({
                'sample_id': sample_id,
                'allele_fraction': allele_fraction,
                'n_variants': n_variants,
                'n_total': n_total,
                'variant_fraction': n_variants / n_total if n_total > 0 else 0,
                'expected_variants': expected_variants,
                'fold_change': fold_change,
                'p_value': p_value,
                'mean_quality': mean_quality,
                'mean_consensus': mean_consensus,
                'test_type': test_type,
                'config_hash': config.config_hash(),
            })

        if not call_data:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(call_data)

            # Apply multiple testing correction
            p_values = df['p_value'].values
            rejected, adjusted_p = benjamini_hochberg_correction(
                p_values,
                alpha
            )

            df['p_adjusted'] = adjusted_p
            df['significant'] = rejected
            df['alpha'] = alpha
            df['fdr_method'] = fdr_method
            df = StatisticalCallsSchema.validate(df)

    # Save results if output path specified
    if output_path and not df.empty:
        df.to_parquet(output_path, index=False)

    return df
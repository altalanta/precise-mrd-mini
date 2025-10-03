"""MRD calling and statistical testing module."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

from .config import PipelineConfig


def poisson_test(observed: int, expected: float) -> float:
    """Two-tailed Poisson test."""
    if expected <= 0:
        return 1.0
    return 2 * min(
        stats.poisson.cdf(observed, expected),
        stats.poisson.sf(observed - 1, expected)
    )


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
    return stats.binom_test(successes, trials, p, alternative='two-sided')


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    if m == 0:
        return np.array([], dtype=bool), np.array([])
    
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # BH procedure
    rejected = np.zeros(m, dtype=bool)
    
    for i in range(m - 1, -1, -1):
        if sorted_p[i] <= (i + 1) / m * alpha:
            rejected[sorted_indices[:i + 1]] = True
            break
    
    # Adjusted p-values (Benjamini-Hochberg step-up)
    adjusted_p = np.zeros_like(sorted_p)
    for i in range(m):
        adjusted_p[i] = sorted_p[i] * m / (i + 1)
    
    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    # Clip to [0, 1]
    adjusted_p = np.clip(adjusted_p, 0, 1)
    
    # Reorder to original order
    final_adjusted = np.zeros_like(p_values)
    final_adjusted[sorted_indices] = adjusted_p
    
    return rejected, final_adjusted


def call_mrd(
    collapsed_df: pd.DataFrame,
    error_model_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Perform MRD calling with statistical testing.
    
    Args:
        collapsed_df: DataFrame with collapsed UMI data
        error_model_df: DataFrame with error model
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        
    Returns:
        DataFrame with MRD calls and statistics
    """
    
    stats_config = config.stats
    
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
        if stats_config.test_type == "poisson":
            p_value = poisson_test(n_variants, expected_variants)
        elif stats_config.test_type == "binomial":
            p_value = binomial_test(n_variants, n_total, mean_error_rate)
        else:
            raise ValueError(f"Unknown test type: {stats_config.test_type}")
        
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
            'test_type': stats_config.test_type,
            'config_hash': config.config_hash(),
        })
    
    if not call_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(call_data)
    
    # Apply multiple testing correction
    p_values = df['p_value'].values
    rejected, adjusted_p = benjamini_hochberg_correction(
        p_values, 
        stats_config.alpha
    )
    
    df['p_adjusted'] = adjusted_p
    df['significant'] = rejected
    df['alpha'] = stats_config.alpha
    df['fdr_method'] = stats_config.fdr_method
    
    if output_path:
        df.to_parquet(output_path, index=False)
    
    return df
"""Simulation module for generating synthetic ctDNA/UMI data."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .config import PipelineConfig


def simulate_reads(
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Simulate synthetic UMI reads for ctDNA analysis.

    Args:
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results

    Returns:
        DataFrame with simulated read data
    """

    # Extract simulation parameters
    sim_config = config.simulation

    # Handle case where simulation config might be a dict
    if isinstance(sim_config, dict):
        allele_fractions = sim_config['allele_fractions']
        umi_depths = sim_config['umi_depths']
        n_replicates = sim_config['n_replicates']
    else:
        allele_fractions = sim_config.allele_fractions
        umi_depths = sim_config.umi_depths
        n_replicates = sim_config.n_replicates

    # Handle UMI config
    umi_config = config.umi
    if isinstance(umi_config, dict):
        max_family_size = umi_config['max_family_size']
    else:
        max_family_size = umi_config.max_family_size

    # Generate synthetic data based on configuration - VECTORIZED APPROACH
    n_variants = len(allele_fractions)
    n_depths = len(umi_depths)
    total_samples = n_variants * n_depths * n_replicates

    # Create condition grid using meshgrid for vectorization
    af_grid, depth_grid, rep_grid = np.meshgrid(
        allele_fractions, umi_depths, np.arange(n_replicates), indexing='ij'
    )

    # Flatten grids for vectorized operations
    af_flat = af_grid.flatten()
    depth_flat = depth_grid.flatten()
    rep_flat = rep_grid.flatten()

    # Vectorized background error rates
    background_rates = rng.uniform(1e-5, 1e-3, total_samples)

    # Vectorized family sizes (Poisson distributed)
    family_sizes = rng.poisson(lam=5, size=total_samples)
    family_sizes = np.clip(family_sizes, 1, max_family_size)

    # Vectorized variant simulation
    # True positives based on allele fraction
    n_true_variants = rng.binomial(depth_flat, af_flat)

    # False positives from background errors
    n_false_positives = rng.binomial(
        depth_flat - n_true_variants,
        background_rates
    )

    # Vectorized quality scores
    quality_scores = rng.normal(25, 5, total_samples)
    quality_scores = np.clip(quality_scores, 10, 40)

    # Create sample IDs
    sample_ids = np.arange(total_samples)

    # Build DataFrame directly from arrays (much faster than list append)
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'allele_fraction': af_flat,
        'target_depth': depth_flat,
        'replicate': rep_flat,
        'n_families': depth_flat,  # Each sample represents depth families
        'n_true_variants': n_true_variants,
        'n_false_positives': n_false_positives,
        'background_rate': background_rates,
        'mean_family_size': family_sizes,  # Simplified for performance
        'mean_quality': quality_scores,    # Simplified for performance
        'config_hash': config.config_hash(),
    })

    if output_path:
        df.to_parquet(output_path, index=False)

    return df
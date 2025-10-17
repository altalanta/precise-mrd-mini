"""Simulation module for generating synthetic ctDNA/UMI data."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .config import PipelineConfig
from .performance import IntelligentCache, ChunkedProcessor, profile_function, CacheStrategy


@profile_function
def simulate_reads(
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    use_cache: bool = True,
    chunked_processing: bool = False
) -> pd.DataFrame:
    """Simulate synthetic UMI reads for ctDNA analysis.

    Args:
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        use_cache: Whether to use intelligent caching
        chunked_processing: Whether to use chunked processing for large datasets

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

    # Check cache for existing results
    if use_cache:
        cache = IntelligentCache(memory_limit_mb=100, disk_cache_dir=".cache")
        cache_key = f"simulate_{config.config_hash()}_{len(allele_fractions)}_{len(umi_depths)}_{n_replicates}"

        cached_result = cache.get(cache_key)
        if cached_result is not None:
            print(f"  Using cached simulation results: {cache_key}")
            if output_path:
                cached_result.to_parquet(output_path, index=False)
            return cached_result

    # Generate synthetic data based on configuration - OPTIMIZED APPROACH
    n_variants = len(allele_fractions)
    n_depths = len(umi_depths)
    total_samples = n_variants * n_depths * n_replicates

    print(f"  Simulating {total_samples:,} samples with {n_variants} AF levels, {n_depths} depths, {n_replicates} replicates")

    # Use chunked processing for large datasets
    if chunked_processing and total_samples > 50000:
        print("  Using chunked processing for large dataset")
        processor = ChunkedProcessor(chunk_size=10000)

        def simulate_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
            """Simulate a chunk of samples."""
            chunk_size = len(chunk_df)
            chunk_allele_fractions = chunk_df['allele_fraction'].values
            chunk_depths = chunk_df['target_depth'].values

            # Vectorized operations for this chunk
            background_rates = rng.uniform(1e-5, 1e-3, chunk_size)
            family_sizes = rng.poisson(lam=5, size=chunk_size)
            family_sizes = np.clip(family_sizes, 1, max_family_size)

            # True positives and false positives
            n_true_variants = rng.binomial(chunk_depths, chunk_allele_fractions)
            n_false_positives = rng.binomial(chunk_depths - n_true_variants, background_rates)

            # Quality scores
            quality_scores = rng.normal(25, 5, chunk_size)
            quality_scores = np.clip(quality_scores, 10, 40)

            # Update chunk DataFrame
            chunk_df['n_true_variants'] = n_true_variants
            chunk_df['n_false_positives'] = n_false_positives
            chunk_df['background_rate'] = background_rates
            chunk_df['mean_family_size'] = family_sizes
            chunk_df['mean_quality'] = quality_scores
            chunk_df['config_hash'] = config.config_hash

            return chunk_df

        # Create condition grid for chunking
        condition_data = []
        sample_id = 0
        for af in allele_fractions:
            for depth in umi_depths:
                for rep in range(n_replicates):
                    condition_data.append({
                        'sample_id': sample_id,
                        'allele_fraction': af,
                        'target_depth': depth,
                        'replicate': rep,
                    })
                    sample_id += 1

        condition_df = pd.DataFrame(condition_data)

        # Process in chunks
        df = processor.process_dataframe_chunks(
            condition_df,
            simulate_chunk,
            lambda chunks: pd.concat(chunks, ignore_index=True)
        )
    else:
        # Standard vectorized approach for smaller datasets
        print("  Using vectorized processing")

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

    # Cache the result for future use
    if use_cache:
        cache.put(cache_key, df, CacheStrategy.HYBRID, ttl=3600)  # Cache for 1 hour

    if output_path:
        df.to_parquet(output_path, index=False)

    return df
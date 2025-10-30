"""Simulation module for generating synthetic ctDNA/UMI data."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pandera as pa

from .config import PipelineConfig
from .data_schemas import SimulatedReadsSchema


@pa.check_output(SimulatedReadsSchema)
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
    
    # Generate synthetic data based on configuration
    n_variants = len(allele_fractions)
    n_depths = len(umi_depths)
    total_samples = n_variants * n_depths * n_replicates
    
    # Create grid of conditions
    data = []
    sample_id = 0
    
    for af in allele_fractions:
        for depth in umi_depths:
            for rep in range(n_replicates):
                # Simulate UMI families
                n_families = depth
                
                # Background error rate (trinucleotide context dependent)
                background_rate = rng.uniform(1e-5, 1e-3)
                
                # Generate reads per family (Poisson distributed)
                family_sizes = rng.poisson(lam=5, size=n_families)
                family_sizes = np.clip(family_sizes, 1, max_family_size)
                
                # Simulate variant calls
                # True positives based on allele fraction
                n_true_variants = rng.binomial(n_families, af)
                
                # False positives from background errors
                n_false_positives = rng.binomial(
                    n_families - n_true_variants, 
                    background_rate
                )
                
                # Quality scores (higher for true variants)
                quality_scores = rng.normal(25, 5, n_families)
                quality_scores = np.clip(quality_scores, 10, 40)
                
                sample_data = {
                    'sample_id': sample_id,
                    'allele_fraction': af,
                    'target_depth': depth,
                    'replicate': rep,
                    'n_families': n_families,
                    'n_true_variants': n_true_variants,
                    'n_false_positives': n_false_positives,
                    'background_rate': background_rate,
                    'mean_family_size': np.mean(family_sizes),
                    'mean_quality': np.mean(quality_scores),
                    'config_hash': config.config_hash(),
                }
                data.append(sample_data)
                sample_id += 1
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_parquet(output_path, index=False)
    
    return df
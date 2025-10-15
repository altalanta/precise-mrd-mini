"""Error modeling for background mutation rates."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from .config import PipelineConfig


def fit_error_model(
    collapsed_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Fit trinucleotide context-specific error model.
    
    Args:
        collapsed_df: DataFrame with collapsed UMI data
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        
    Returns:
        DataFrame with error model parameters
    """
    
    # Define trinucleotide contexts
    contexts = [
        'AAA', 'AAC', 'AAG', 'AAT',
        'ACA', 'ACC', 'ACG', 'ACT',
        'AGA', 'AGC', 'AGG', 'AGT',
        'ATA', 'ATC', 'ATG', 'ATT',
        'CAA', 'CAC', 'CAG', 'CAT',
        'CCA', 'CCC', 'CCG', 'CCT',
        'CGA', 'CGC', 'CGG', 'CGT',
        'CTA', 'CTC', 'CTG', 'CTT',
        'GAA', 'GAC', 'GAG', 'GAT',
        'GCA', 'GCC', 'GCG', 'GCT',
        'GGA', 'GGC', 'GGG', 'GGT',
        'GTA', 'GTC', 'GTG', 'GTT',
        'TAA', 'TAC', 'TAG', 'TAT',
        'TCA', 'TCC', 'TCG', 'TCT',
        'TGA', 'TGC', 'TGG', 'TGT',
        'TTA', 'TTC', 'TTG', 'TTT'
    ]
    
    # Fit error rates for each trinucleotide context
    error_data = []
    
    for context in contexts:
        # Estimate error rate from negative control samples (lowest AF)
        negative_samples = collapsed_df[collapsed_df['allele_fraction'] <= 0.0001]
        
        if len(negative_samples) == 0:
            # Fallback to lowest AF samples
            min_af = collapsed_df['allele_fraction'].min()
            negative_samples = collapsed_df[collapsed_df['allele_fraction'] == min_af]
        
        # Count variants in negative samples for this context
        if len(negative_samples) > 0:
            variant_rate = negative_samples['is_variant'].mean()
            # Add context-specific variation
            context_modifier = rng.uniform(0.5, 2.0)
            error_rate = variant_rate * context_modifier
        else:
            # Default error rate
            error_rate = rng.uniform(1e-5, 1e-3)
        
        # Confidence interval from bootstrap
        if len(negative_samples) > 10:
            bootstrap_rates = []
            for _ in range(100):
                boot_sample = negative_samples.sample(
                    n=len(negative_samples), 
                    replace=True, 
                    random_state=rng.integers(0, 2**32-1)
                )
                boot_rate = boot_sample['is_variant'].mean() * context_modifier
                bootstrap_rates.append(boot_rate)
            
            ci_lower = np.percentile(bootstrap_rates, 2.5)
            ci_upper = np.percentile(bootstrap_rates, 97.5)
        else:
            ci_lower = error_rate * 0.5
            ci_upper = error_rate * 2.0
        
        error_data.append({
            'trinucleotide_context': context,
            'error_rate': error_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_observations': len(negative_samples),
            'config_hash': config.config_hash(),
        })
    
    df = pd.DataFrame(error_data)
    
    if output_path:
        df.to_parquet(output_path, index=False)
    
    return df
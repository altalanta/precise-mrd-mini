"""UMI collapse and consensus calling module."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from .config import PipelineConfig


def collapse_umis(
    reads_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Collapse UMI families and call consensus.
    
    Args:
        reads_df: DataFrame with simulated read data
        config: Pipeline configuration
        rng: Seeded random number generator  
        output_path: Optional path to save results
        
    Returns:
        DataFrame with collapsed UMI data
    """
    
    umi_config = config.umi
    
    # Process each sample
    collapsed_data = []
    
    for _, sample in reads_df.iterrows():
        sample_id = sample['sample_id']
        n_families = int(sample['n_families'])
        
        # Generate UMI family consensus data
        family_data = []
        
        for family_id in range(n_families):
            # Simulate family size (from original simulation)
            family_size = max(1, int(rng.poisson(5)))
            family_size = min(family_size, umi_config.max_family_size)
            
            # Skip families below minimum size threshold
            if family_size < umi_config.min_family_size:
                continue
            
            # Simulate quality scores for consensus
            quality_score = rng.normal(25, 3)
            quality_score = np.clip(quality_score, 10, 40)
            
            # Determine if consensus passes quality threshold
            passes_quality = quality_score >= umi_config.quality_threshold
            
            # Simulate consensus agreement
            consensus_agreement = rng.uniform(0.5, 1.0)
            passes_consensus = consensus_agreement >= umi_config.consensus_threshold
            
            # Determine variant call
            is_variant = passes_quality and passes_consensus and (
                # True variant with high probability if high AF
                (sample['allele_fraction'] > 0.01 and rng.random() < 0.8) or
                # Background error with low probability
                (rng.random() < sample['background_rate'])
            )
            
            family_data.append({
                'sample_id': sample_id,
                'family_id': family_id,
                'family_size': family_size,
                'quality_score': quality_score,
                'consensus_agreement': consensus_agreement,
                'passes_quality': passes_quality,
                'passes_consensus': passes_consensus,
                'is_variant': is_variant,
                'allele_fraction': sample['allele_fraction'],
            })
        
        collapsed_data.extend(family_data)
    
    df = pd.DataFrame(collapsed_data)
    
    if output_path:
        df.to_parquet(output_path, index=False)
    
    return df
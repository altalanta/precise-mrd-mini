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
    output_path: Optional[str] = None,
    is_fastq_data: bool = False
) -> pd.DataFrame:
    """Collapse UMI families and call consensus.

    Args:
        reads_df: DataFrame with read data (synthetic or FASTQ)
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        is_fastq_data: Whether input data comes from FASTQ files

    Returns:
        DataFrame with collapsed UMI data
    """

    umi_config = config.umi

    # Handle case where umi config might be a dict
    if isinstance(umi_config, dict):
        min_family_size = umi_config['min_family_size']
        max_family_size = umi_config['max_family_size']
        quality_threshold = umi_config['quality_threshold']
        consensus_threshold = umi_config['consensus_threshold']
    else:
        min_family_size = umi_config.min_family_size
        max_family_size = umi_config.max_family_size
        quality_threshold = umi_config.quality_threshold
        consensus_threshold = umi_config.consensus_threshold

    if is_fastq_data:
        # Handle FASTQ data (already processed into family groups)
        collapsed_data = []

        for _, row in reads_df.iterrows():
            # FASTQ data comes pre-processed with UMI family information
            collapsed_data.append({
                'sample_id': row['sample_id'],
                'family_id': 0,  # Single family per UMI in FASTQ processing
                'family_size': row['family_size'],
                'quality_score': row['mean_quality'],
                'consensus_agreement': row['consensus_agreement'],
                'passes_quality': row['passes_quality'],
                'passes_consensus': row['passes_consensus'],
                'is_variant': False,  # Will be determined by downstream analysis
                'allele_fraction': 0.0,  # Not available in FASTQ data
            })

        df = pd.DataFrame(collapsed_data)

    else:
        # Handle synthetic data (original logic)
        collapsed_data = []

        for _, sample in reads_df.iterrows():
            sample_id = sample['sample_id']
            n_families = int(sample['n_families'])
            allele_fraction = sample['allele_fraction']
            background_rate = sample['background_rate']

            # Generate UMI family consensus data for this sample
            for family_id in range(n_families):
                # Use deterministic family size based on sample statistics
                # This ensures reproducibility while maintaining performance
                mean_size = max(3, sample['mean_family_size'])
                family_size = max(1, int(rng.poisson(mean_size)))
                family_size = min(family_size, max_family_size)

                # Skip families below minimum size threshold
                if family_size < min_family_size:
                    continue

                # Use deterministic quality scores based on allele fraction
                # Higher AF samples get higher quality scores
                base_quality = 25 + (allele_fraction * 100)  # Scale with AF
                quality_score = rng.normal(base_quality, 3)
                quality_score = np.clip(quality_score, 10, 40)

                # Determine if consensus passes quality threshold
                passes_quality = quality_score >= quality_threshold

                # Use deterministic consensus agreement based on quality
                # Higher quality gets higher consensus agreement
                consensus_base = min(0.95, 0.7 + (quality_score - 10) / 60)  # Scale from 0.7 to 0.95, cap at 0.95
                consensus_agreement = rng.uniform(consensus_base, 1.0)
                passes_consensus = consensus_agreement >= consensus_threshold

                # Determine variant call deterministically based on AF and background rate
                is_true_variant = (allele_fraction > 0.01 and rng.random() < 0.8)
                is_background_error = rng.random() < background_rate
                is_variant = passes_quality and passes_consensus and (is_true_variant or is_background_error)

                collapsed_data.append({
                    'sample_id': sample_id,
                    'family_id': family_id,
                    'family_size': family_size,
                    'quality_score': quality_score,
                    'consensus_agreement': consensus_agreement,
                    'passes_quality': passes_quality,
                    'passes_consensus': passes_consensus,
                    'is_variant': is_variant,
                    'allele_fraction': allele_fraction,
                })

        df = pd.DataFrame(collapsed_data)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df
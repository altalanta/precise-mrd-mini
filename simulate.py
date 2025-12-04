"""Simulation module for generating synthetic ctDNA/UMI data."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pandera as pa

from .config import PipelineConfig
from .data_schemas import SimulatedReadsSchema

# =============================================================================
# Constants for read simulation
# =============================================================================

# Background error rate bounds (typical sequencing error range)
MIN_BACKGROUND_ERROR_RATE: float = 1e-5
MAX_BACKGROUND_ERROR_RATE: float = 1e-3

# Poisson distribution parameter for family sizes
DEFAULT_FAMILY_SIZE_LAMBDA: int = 5

# Quality score distribution parameters (Phred scale)
QUALITY_SCORE_MEAN: float = 25.0
QUALITY_SCORE_STD: float = 5.0
MIN_QUALITY_SCORE: float = 10.0
MAX_QUALITY_SCORE: float = 40.0


@pa.check_output(SimulatedReadsSchema)
def simulate_reads(
    config: PipelineConfig, rng: np.random.Generator, output_path: str | None = None
) -> pd.DataFrame:
    """Simulate synthetic UMI reads for ctDNA (circulating tumor DNA) analysis.

    Generates synthetic sequencing data that mimics real UMI-tagged ctDNA samples,
    enabling validation and benchmarking of MRD detection pipelines without
    requiring real patient data.

    The simulation creates a grid of conditions based on allele fractions and
    sequencing depths, with multiple replicates for statistical power. Each
    simulated sample includes realistic UMI family structures and quality scores.

    Args:
        config: Pipeline configuration containing simulation parameters including
            allele_fractions, umi_depths, n_replicates, and UMI settings.
        rng: Seeded numpy random number generator for reproducible simulations.
            Use np.random.default_rng(seed) to create one.
        output_path: Optional path to save results as a Parquet file.
            If None, results are only returned in memory.

    Returns:
        DataFrame containing simulated read data with columns:
            - sample_id: Unique identifier for each simulated sample
            - allele_fraction: True variant allele frequency for this sample
            - target_depth: Requested sequencing depth (number of UMI families)
            - replicate: Replicate number for this allele_fraction/depth combination
            - n_families: Actual number of UMI families generated
            - n_true_variants: Number of true positive variant calls
            - n_false_positives: Number of false positive calls from background errors
            - background_rate: Simulated background error rate
            - mean_family_size: Average reads per UMI family
            - mean_quality: Average Phred quality score
            - config_hash: Hash of the configuration for reproducibility tracking

    Raises:
        pandera.errors.SchemaError: If output DataFrame fails schema validation.

    Example:
        >>> config = load_config("config.yaml")
        >>> rng = np.random.default_rng(42)
        >>> reads_df = simulate_reads(config, rng, "output/reads.parquet")
    """

    # Extract simulation parameters
    sim_config = config.simulation

    # Handle case where simulation config might be a dict
    if isinstance(sim_config, dict):
        allele_fractions = sim_config["allele_fractions"]
        umi_depths = sim_config["umi_depths"]
        n_replicates = sim_config["n_replicates"]
    else:
        allele_fractions = sim_config.allele_fractions
        umi_depths = sim_config.umi_depths
        n_replicates = sim_config.n_replicates

    # Handle UMI config
    umi_config = config.umi
    if isinstance(umi_config, dict):
        max_family_size = umi_config["max_family_size"]
    else:
        max_family_size = umi_config.max_family_size

    # Generate synthetic data based on configuration

    # Create grid of conditions
    conditions = list(
        product(
            range(len(allele_fractions)), range(len(umi_depths)), range(n_replicates)
        )
    )

    data = []
    sample_id = 0

    for af_idx, depth_idx, rep_idx in conditions:
        af = allele_fractions[af_idx]
        depth = umi_depths[depth_idx]
        rep = rep_idx

        # Simulate UMI families
        n_families = depth

        # Background error rate (trinucleotide context dependent)
        background_rate = rng.uniform(
            MIN_BACKGROUND_ERROR_RATE, MAX_BACKGROUND_ERROR_RATE
        )

        # Generate reads per family (Poisson distributed)
        family_sizes = rng.poisson(lam=DEFAULT_FAMILY_SIZE_LAMBDA, size=n_families)
        family_sizes = np.clip(family_sizes, 1, max_family_size)

        # Simulate variant calls
        # True positives based on allele fraction
        n_true_variants = rng.binomial(n_families, af)

        # False positives from background errors
        n_false_positives = rng.binomial(n_families - n_true_variants, background_rate)

        # Quality scores (higher for true variants)
        quality_scores = rng.normal(QUALITY_SCORE_MEAN, QUALITY_SCORE_STD, n_families)
        quality_scores = np.clip(quality_scores, MIN_QUALITY_SCORE, MAX_QUALITY_SCORE)

        sample_data = {
            "sample_id": sample_id,
            "allele_fraction": af,
            "target_depth": depth,
            "replicate": rep,
            "n_families": n_families,
            "n_true_variants": n_true_variants,
            "n_false_positives": n_false_positives,
            "background_rate": background_rate,
            "mean_family_size": np.mean(family_sizes),
            "mean_quality": np.mean(quality_scores),
            "config_hash": config.config_hash(),
        }
        data.append(sample_data)
        sample_id += 1

    df = pd.DataFrame(data)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df

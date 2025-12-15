"""UMI collapse and consensus calling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa

from .config import PipelineConfig
from .data_schemas import CollapsedUmisSchema, SimulatedReadsSchema
from .performance import parallel_timing_decorator

# Optional Dask import for parallel processing
try:
    import dask.dataframe as dd
    from dask import delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# =============================================================================
# Constants for UMI collapse simulation
# =============================================================================

# Quality score bounds (Phred scale)
MIN_QUALITY_SCORE: int = 10
MAX_QUALITY_SCORE: int = 40
BASE_QUALITY_SCORE: int = 25
QUALITY_SCORE_STD: float = 3.0

# Quality-to-AF scaling factor: higher AF samples get higher quality scores
QUALITY_AF_SCALE: float = 100.0

# Consensus agreement bounds
MIN_CONSENSUS_AGREEMENT: float = 0.7
MAX_CONSENSUS_AGREEMENT: float = 0.95

# Minimum mean family size fallback
DEFAULT_MIN_MEAN_FAMILY_SIZE: int = 3

# Variant calling thresholds
TRUE_VARIANT_AF_THRESHOLD: float = 0.01  # AF above which true variants are expected
TRUE_VARIANT_PROBABILITY: float = (
    0.8  # Probability of calling true variant when AF > threshold
)

# Default number of Dask partitions
DEFAULT_N_PARTITIONS: int = 4


@pa.check_input(
    pa.DataFrameSchema(SimulatedReadsSchema.to_schema().columns, strict=False),
)
@pa.check_output(CollapsedUmisSchema)
def collapse_umis(
    reads_df: pd.DataFrame | dd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: str | None = None,
    is_fastq_data: bool = False,
    use_parallel: bool = False,
    n_partitions: int = None,
) -> pd.DataFrame:
    """Collapse UMI families and call consensus.

    Args:
        reads_df: DataFrame with read data (synthetic or FASTQ)
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        is_fastq_data: Whether input data comes from FASTQ files
        use_parallel: Whether to use parallel processing with Dask
        n_partitions: Number of partitions for parallel processing

    Returns:
        DataFrame with collapsed UMI data
    """

    # Use parallel processing if requested and Dask is available
    if use_parallel and DASK_AVAILABLE and isinstance(reads_df, pd.DataFrame):
        return _collapse_umis_parallel(
            reads_df,
            config,
            rng,
            output_path,
            is_fastq_data,
            n_partitions,
        )

    # Fall back to original implementation for other cases
    return _collapse_umis_sequential(reads_df, config, rng, output_path, is_fastq_data)


def _collapse_umis_sequential(
    reads_df: pd.DataFrame | dd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: str | None = None,
    is_fastq_data: bool = False,
) -> pd.DataFrame:
    """Sequential UMI collapse implementation."""

    umi_config = config.umi

    # Handle case where umi config might be a dict
    if isinstance(umi_config, dict):
        min_family_size = umi_config["min_family_size"]
        max_family_size = umi_config["max_family_size"]
        quality_threshold = umi_config["quality_threshold"]
        consensus_threshold = umi_config["consensus_threshold"]
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
            collapsed_data.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": 0,  # Single family per UMI in FASTQ processing
                    "family_size": row["family_size"],
                    "quality_score": row["mean_quality"],
                    "consensus_agreement": row["consensus_agreement"],
                    "passes_quality": row["passes_quality"],
                    "passes_consensus": row["passes_consensus"],
                    "is_variant": False,  # Will be determined by downstream analysis
                    "allele_fraction": 0.0,  # Not available in FASTQ data
                },
            )

        df = pd.DataFrame(collapsed_data)

    else:
        # Handle synthetic data (original logic)
        collapsed_data = []

        for _, sample in reads_df.iterrows():
            sample_id = sample["sample_id"]
            n_families = int(sample["n_families"])
            allele_fraction = sample["allele_fraction"]
            background_rate = sample["background_rate"]

            # Generate UMI family consensus data for this sample
            for family_id in range(n_families):
                # Use deterministic family size based on sample statistics
                # This ensures reproducibility while maintaining performance
                mean_size = max(
                    DEFAULT_MIN_MEAN_FAMILY_SIZE,
                    sample["mean_family_size"],
                )
                family_size = max(1, int(rng.poisson(mean_size)))
                family_size = min(family_size, max_family_size)

                # Skip families below minimum size threshold
                if family_size < min_family_size:
                    continue

                # Use deterministic quality scores based on allele fraction
                # Higher AF samples get higher quality scores
                base_quality = BASE_QUALITY_SCORE + (allele_fraction * QUALITY_AF_SCALE)
                quality_score = rng.normal(base_quality, QUALITY_SCORE_STD)
                quality_score = np.clip(
                    quality_score,
                    MIN_QUALITY_SCORE,
                    MAX_QUALITY_SCORE,
                )

                # Determine if consensus passes quality threshold
                passes_quality = quality_score >= quality_threshold

                # Use deterministic consensus agreement based on quality
                # Higher quality gets higher consensus agreement
                # Scale from MIN_CONSENSUS_AGREEMENT to MAX_CONSENSUS_AGREEMENT
                quality_range = MAX_QUALITY_SCORE - MIN_QUALITY_SCORE
                consensus_base = min(
                    MAX_CONSENSUS_AGREEMENT,
                    MIN_CONSENSUS_AGREEMENT
                    + (quality_score - MIN_QUALITY_SCORE) / quality_range,
                )
                consensus_agreement = rng.uniform(consensus_base, 1.0)
                passes_consensus = consensus_agreement >= consensus_threshold

                # Determine variant call deterministically based on AF and background rate
                is_true_variant = (
                    allele_fraction > TRUE_VARIANT_AF_THRESHOLD
                    and rng.random() < TRUE_VARIANT_PROBABILITY
                )
                is_background_error = rng.random() < background_rate
                is_variant = (
                    passes_quality
                    and passes_consensus
                    and (is_true_variant or is_background_error)
                )

                collapsed_data.append(
                    {
                        "sample_id": sample_id,
                        "family_id": family_id,
                        "family_size": family_size,
                        "quality_score": quality_score,
                        "consensus_agreement": consensus_agreement,
                        "passes_quality": passes_quality,
                        "passes_consensus": passes_consensus,
                        "is_variant": is_variant,
                        "allele_fraction": allele_fraction,
                    },
                )

        df = pd.DataFrame(collapsed_data)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df


@delayed
@parallel_timing_decorator
def _process_sample_group(
    group_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Process a group of samples for parallel UMI collapse."""
    # This is essentially the inner loop of the original sequential implementation
    # but operates on a subset of samples

    umi_config = config.umi
    if isinstance(umi_config, dict):
        min_family_size = umi_config["min_family_size"]
        max_family_size = umi_config["max_family_size"]
        quality_threshold = umi_config["quality_threshold"]
        consensus_threshold = umi_config["consensus_threshold"]
    else:
        min_family_size = umi_config.min_family_size
        max_family_size = umi_config.max_family_size
        quality_threshold = umi_config.quality_threshold
        consensus_threshold = umi_config.consensus_threshold

    collapsed_data = []

    for _, sample in group_df.iterrows():
        sample_id = sample["sample_id"]
        n_families = int(sample["n_families"])
        allele_fraction = sample["allele_fraction"]
        background_rate = sample["background_rate"]

        # Generate UMI family consensus data for this sample
        for family_id in range(n_families):
            # Use deterministic family size based on sample statistics
            mean_size = max(DEFAULT_MIN_MEAN_FAMILY_SIZE, sample["mean_family_size"])
            family_size = max(1, int(rng.poisson(mean_size)))
            family_size = min(family_size, max_family_size)

            # Skip families below minimum size threshold
            if family_size < min_family_size:
                continue

            # Use deterministic quality scores based on allele fraction
            base_quality = BASE_QUALITY_SCORE + (allele_fraction * QUALITY_AF_SCALE)
            quality_score = rng.normal(base_quality, QUALITY_SCORE_STD)
            quality_score = np.clip(quality_score, MIN_QUALITY_SCORE, MAX_QUALITY_SCORE)

            passes_quality = quality_score >= quality_threshold

            quality_range = MAX_QUALITY_SCORE - MIN_QUALITY_SCORE
            consensus_base = min(
                MAX_CONSENSUS_AGREEMENT,
                MIN_CONSENSUS_AGREEMENT
                + (quality_score - MIN_QUALITY_SCORE) / quality_range,
            )
            consensus_agreement = rng.uniform(consensus_base, 1.0)
            passes_consensus = consensus_agreement >= consensus_threshold

            is_true_variant = (
                allele_fraction > TRUE_VARIANT_AF_THRESHOLD
                and rng.random() < TRUE_VARIANT_PROBABILITY
            )
            is_background_error = rng.random() < background_rate
            is_variant = (
                passes_quality
                and passes_consensus
                and (is_true_variant or is_background_error)
            )

            collapsed_data.append(
                {
                    "sample_id": sample_id,
                    "family_id": family_id,
                    "family_size": family_size,
                    "quality_score": quality_score,
                    "consensus_agreement": consensus_agreement,
                    "passes_quality": passes_quality,
                    "passes_consensus": passes_consensus,
                    "is_variant": is_variant,
                    "allele_fraction": allele_fraction,
                },
            )

    return pd.DataFrame(collapsed_data)


@parallel_timing_decorator
def _collapse_umis_parallel(
    reads_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: str | None = None,
    is_fastq_data: bool = False,
    n_partitions: int = None,
) -> pd.DataFrame:
    """Parallel UMI collapse implementation using Dask."""

    if is_fastq_data:
        # For FASTQ data, parallelization is simpler since each row is independent
        # We can use Dask's map_partitions for this
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for parallel processing but is not available",
            )

        ddf = dd.from_pandas(reads_df, npartitions=n_partitions or DEFAULT_N_PARTITIONS)

        def process_fastq_partition(partition_df):
            collapsed_data = []
            for _, row in partition_df.iterrows():
                collapsed_data.append(
                    {
                        "sample_id": row["sample_id"],
                        "family_id": 0,
                        "family_size": row["family_size"],
                        "quality_score": row["mean_quality"],
                        "consensus_agreement": row["consensus_agreement"],
                        "passes_quality": row["passes_quality"],
                        "passes_consensus": row["passes_consensus"],
                        "is_variant": False,
                        "allele_fraction": 0.0,
                    },
                )
            return pd.DataFrame(collapsed_data)

        result_ddf = ddf.map_partitions(process_fastq_partition)
        df = result_ddf.compute()

    else:
        # For synthetic data, we need to group by samples and process each group in parallel
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for parallel processing but is not available",
            )

        # Split dataframe into chunks for parallel processing
        n_partitions = n_partitions or min(4, len(reads_df))

        # Create a sample index for grouping
        sample_groups = []
        samples_per_partition = max(1, len(reads_df) // n_partitions)

        for i in range(0, len(reads_df), samples_per_partition):
            group_df = reads_df.iloc[i : i + samples_per_partition].copy()
            sample_groups.append(group_df)

        # Process each group in parallel using Dask delayed
        delayed_results = []
        for group_df in sample_groups:
            delayed_result = _process_sample_group(group_df, config, rng)
            delayed_results.append(delayed_result)

        # Compute all results in parallel
        from dask import compute

        parallel_results = compute(*delayed_results)

        # Combine results
        df = pd.concat(parallel_results, ignore_index=True)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df

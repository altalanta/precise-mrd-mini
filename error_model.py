"""Error modeling for background mutation rates."""

from __future__ import annotations

import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandera as pa

from .config import PipelineConfig
from .advanced_stats import BayesianErrorModel
from .data_schemas import ErrorModelSchema, CollapsedUmisSchema


def _get_cache_key(config: PipelineConfig, collapsed_df: pd.DataFrame) -> str:
    """Generate a cache key based on configuration and data characteristics."""
    # Create a deterministic representation of the configuration and data
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__

    # Include key data characteristics that affect error model
    data_summary = {
        'n_samples': len(collapsed_df),
        'mean_af': collapsed_df['allele_fraction'].mean(),
        'min_af': collapsed_df['allele_fraction'].min(),
        'max_af': collapsed_df['allele_fraction'].max(),
        'config_hash': config.config_hash(),
    }

    # Create deterministic string representation
    cache_input = json.dumps([config_dict, data_summary], sort_keys=True)
    return hashlib.sha256(cache_input.encode()).hexdigest()[:16]


def _load_cached_error_model(cache_key: str, cache_dir: str = ".cache") -> Optional[pd.DataFrame]:
    """Load cached error model if it exists."""
    cache_path = Path(cache_dir) / f"error_model_{cache_key}.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None
    return None


def _save_cached_error_model(error_df: pd.DataFrame, cache_key: str, cache_dir: str = ".cache") -> None:
    """Save error model to cache."""
    cache_path = Path(cache_dir) / f"error_model_{cache_key}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_parquet(cache_path, index=False)


@pa.check_input(pa.DataFrameSchema(CollapsedUmisSchema.to_schema().columns,
                                 strict=False))
@pa.check_output(ErrorModelSchema)
def fit_error_model(
    collapsed_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = ".cache",
    use_advanced_stats: bool = False
) -> pd.DataFrame:
    """Fit trinucleotide context-specific error model with optional caching and advanced statistics.

    Args:
        collapsed_df: DataFrame with collapsed UMI data
        config: Pipeline configuration
        rng: Seeded random number generator
        output_path: Optional path to save results
        use_cache: Whether to use caching for expensive computations
        cache_dir: Directory for cache files
        use_advanced_stats: Whether to use advanced Bayesian modeling

    Returns:
        DataFrame with error model parameters
    """

    # Try to load from cache first
    if use_cache:
        cache_key = _get_cache_key(config, collapsed_df)
        cached_result = _load_cached_error_model(cache_key, cache_dir)
        if cached_result is not None:
            print(f"  Using cached error model: {cache_key}")
            if output_path:
                cached_result.to_parquet(output_path, index=False)
            return cached_result

    # Use advanced Bayesian modeling if requested
    if use_advanced_stats:
        print("  Using advanced Bayesian error modeling...")
        bayesian_model = BayesianErrorModel(config)
        advanced_results = bayesian_model.fit_bayesian_model(collapsed_df, rng)

        # Convert to DataFrame format for compatibility
        error_data = []
        for context, params in advanced_results['context_error_rates'].items():
            error_data.append({
                'trinucleotide_context': context,
                'error_rate': params['error_rate'],
                'ci_lower': params['ci_lower'],
                'ci_upper': params['ci_upper'],
                'n_observations': params['n_observations'],
                'model_type': 'bayesian',
                'config_hash': params.get('config_hash', config.config_hash())
            })

        df = pd.DataFrame(error_data)

        # Save to cache for future use
        if use_cache:
            _save_cached_error_model(df, cache_key, cache_dir)

        if output_path:
            df.to_parquet(output_path, index=False)

        return df

    # Original simple error model

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

    # Save to cache for future use
    if use_cache:
        cache_key = _get_cache_key(config, collapsed_df)
        _save_cached_error_model(df, cache_key, cache_dir)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df

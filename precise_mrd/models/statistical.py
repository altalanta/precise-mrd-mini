"""Statistical models for variant calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import VariantCaller

if TYPE_CHECKING:
    from ..config import PipelineConfig


class StatisticalVariantCaller(VariantCaller):
    """Variant caller using statistical tests (Poisson, Binomial)."""

    def predict(self, collapsed_df: pd.DataFrame, error_model_df: pd.DataFrame) -> pd.DataFrame:
        """Perform MRD calling with statistical testing."""
        stats_config = self.config.stats
        test_type = stats_config.test_type
        alpha = stats_config.alpha
        fdr_method = stats_config.fdr_method

        call_data = []
        for sample_id in collapsed_df['sample_id'].unique():
            sample_data = collapsed_df[collapsed_df['sample_id'] == sample_id]
            if len(sample_data) == 0:
                continue

            n_variants = sample_data['is_variant'].sum()
            n_total = len(sample_data)
            if n_total == 0:
                continue

            mean_error_rate = error_model_df['error_rate'].mean()
            expected_variants = n_total * mean_error_rate

            if test_type == "poisson":
                p_value = self._poisson_test(n_variants, expected_variants)
            elif test_type == "binomial":
                p_value = self._binomial_test(n_variants, n_total, mean_error_rate)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            fold_change = n_variants / expected_variants if expected_variants > 0 else (float('inf') if n_variants > 0 else 1.0)
            
            call_data.append({
                'sample_id': sample_id,
                'allele_fraction': sample_data['allele_fraction'].iloc[0],
                'n_variants': n_variants, 'n_total': n_total,
                'variant_fraction': n_variants / n_total if n_total > 0 else 0,
                'expected_variants': expected_variants, 'fold_change': fold_change,
                'p_value': p_value, 'mean_quality': sample_data['quality_score'].mean(),
                'mean_consensus': sample_data['consensus_agreement'].mean(),
                'test_type': test_type, 'config_hash': self.config.config_hash(),
            })

        if not call_data:
            return pd.DataFrame()

        df = pd.DataFrame(call_data)
        rejected, adjusted_p = self._benjamini_hochberg_correction(df['p_value'].values, alpha)
        df['p_adjusted'] = adjusted_p
        df['significant'] = rejected
        df['alpha'] = alpha
        df['fdr_method'] = fdr_method
        return df

    def _poisson_test(self, observed: int, expected: float) -> float:
        """Two-tailed Poisson test."""
        if expected <= 0:
            return 1.0
        left_tail = stats.poisson.cdf(observed, expected)
        right_tail = stats.poisson.sf(observed - 1, expected)
        p_value = 2 * min(left_tail, right_tail)
        return min(p_value, 1.0)

    def _binomial_test(self, successes: int, trials: int, p: float) -> float:
        """Two-tailed binomial test."""
        if trials == 0:
            return 1.0
        if p <= 0:
            return 0.0 if successes > 0 else 1.0
        if p >= 1:
            return 0.0 if successes < trials else 1.0
        return stats.binomtest(successes, trials, p, alternative='two-sided').pvalue

    def _benjamini_hochberg_correction(self, p_values: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Benjamini-Hochberg FDR correction."""
        p_values = np.asarray(p_values, dtype=float)
        m = len(p_values)
        if m == 0:
            return np.array([], dtype=bool), np.array([])
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        k_values = np.arange(m) + 1
        thresholds = k_values / m * alpha
        valid_k = sorted_p <= thresholds
        
        if np.any(valid_k):
            max_k = np.max(np.where(valid_k)[0])
            rejected = np.zeros(m, dtype=bool)
            rejected[sorted_indices[:max_k + 1]] = True
        else:
            rejected = np.zeros(m, dtype=bool)
            
        adjusted_p = sorted_p * m / k_values
        adjusted_p = np.minimum.accumulate(adjusted_p[::-1])[::-1]
        adjusted_p = np.clip(adjusted_p, 0, 1)
        
        final_adjusted = np.zeros_like(p_values)
        final_adjusted[sorted_indices] = adjusted_p
        
        return rejected, final_adjusted

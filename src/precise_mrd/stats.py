"""
Statistical testing module for MRD variant calling.

This module provides:
- Poisson and binomial hypothesis testing
- Benjamini-Hochberg FDR correction
- P-value calibration utilities
- Multiple testing correction
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass


@dataclass
class TestResult:
    """Container for statistical test results."""
    statistic: float
    pvalue: float
    alternative: str
    method: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class StatisticalTester:
    """Perform statistical hypothesis testing for variant calling."""
    
    def __init__(
        self,
        test_type: str = "poisson",
        alpha: float = 0.05,
        fdr_method: str = "benjamini_hochberg"
    ):
        """Initialize statistical tester."""
        self.test_type = test_type
        self.alpha = alpha
        self.fdr_method = fdr_method
        
        if test_type not in ["poisson", "binomial"]:
            raise ValueError("test_type must be 'poisson' or 'binomial'")
    
    def poisson_test(
        self,
        observed: int,
        expected: float,
        alternative: str = "greater"
    ) -> TestResult:
        """Perform Poisson test for observed vs expected counts."""
        
        if expected <= 0:
            return TestResult(
                statistic=observed,
                pvalue=1.0,
                alternative=alternative,
                method="poisson"
            )
        
        if alternative == "greater":
            # P(X >= observed)
            pvalue = 1 - stats.poisson.cdf(observed - 1, expected)
        elif alternative == "less":
            # P(X <= observed)
            pvalue = stats.poisson.cdf(observed, expected)
        elif alternative == "two-sided":
            # Two-sided test
            if observed <= expected:
                pvalue = 2 * stats.poisson.cdf(observed, expected)
            else:
                pvalue = 2 * (1 - stats.poisson.cdf(observed - 1, expected))
            pvalue = min(pvalue, 1.0)
        else:
            raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
        
        # Calculate effect size (log rate ratio for numerical stability)
        effect_size = (np.log(observed + 1e-8) - np.log(expected + 1e-8) 
                      if expected > 1e-12 else np.nan)
        
        return TestResult(
            statistic=observed,
            pvalue=pvalue,
            alternative=alternative,
            method="poisson",
            effect_size=effect_size
        )
    
    def binomial_test(
        self,
        successes: int,
        trials: int,
        expected_prob: float,
        alternative: str = "greater"
    ) -> TestResult:
        """Perform binomial test for observed success rate."""
        
        if trials <= 0:
            return TestResult(
                statistic=successes,
                pvalue=1.0,
                alternative=alternative,
                method="binomial"
            )
        
        # Use scipy.stats.binomtest for exact binomial test
        result = stats.binomtest(
            successes, 
            trials, 
            expected_prob, 
            alternative=alternative
        )
        
        # Calculate effect size (log odds ratio for numerical stability)
        observed_prob = successes / trials
        if expected_prob > 1e-12 and expected_prob < (1 - 1e-12):
            # Use log odds ratio for numerical stability
            observed_logit = np.log(observed_prob + 1e-12) - np.log(1 - observed_prob + 1e-12)
            expected_logit = np.log(expected_prob + 1e-12) - np.log(1 - expected_prob + 1e-12)
            odds_ratio = observed_logit - expected_logit
        else:
            odds_ratio = np.clip(observed_prob / (expected_prob + 1e-12), 1e-12, 1e12)
        
        return TestResult(
            statistic=result.statistic,
            pvalue=result.pvalue,
            alternative=alternative,
            method="binomial",
            effect_size=odds_ratio,
            confidence_interval=result.proportion_ci()
        )
    
    def test_variant_significance(
        self,
        observed_alt: int,
        total_depth: int,
        expected_error_rate: float,
        context: Optional[str] = None
    ) -> TestResult:
        """Test significance of observed variant count."""
        
        if self.test_type == "poisson":
            expected_count = expected_error_rate * total_depth
            return self.poisson_test(observed_alt, expected_count, alternative="greater")
        
        elif self.test_type == "binomial":
            return self.binomial_test(
                observed_alt, 
                total_depth, 
                expected_error_rate, 
                alternative="greater"
            )
        
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
    
    def multiple_testing_correction(
        self,
        pvalues: List[float],
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple testing correction to p-values."""
        
        if method is None:
            method = self.fdr_method
        
        # Convert method name for statsmodels
        if method == "benjamini_hochberg":
            sm_method = "fdr_bh"
        elif method == "bonferroni":
            sm_method = "bonferroni"
        elif method == "holm":
            sm_method = "holm"
        else:
            sm_method = method
        
        rejected, pvals_corrected, _, _ = multipletests(
            pvalues, 
            alpha=self.alpha, 
            method=sm_method
        )
        
        return rejected, pvals_corrected
    
    def test_multiple_variants(
        self,
        variant_data: pd.DataFrame,
        error_rates: Dict[str, float]
    ) -> pd.DataFrame:
        """Test multiple variants with FDR correction.
        
        Args:
            variant_data: DataFrame with columns 'alt_count', 'total_depth'
            error_rates: Dictionary mapping contexts to expected error rates
            
        Returns:
            DataFrame with test results and corrected p-values
            
        Raises:
            ValueError: If input data is invalid
        """
        if variant_data.empty:
            return pd.DataFrame()
        
        required_columns = ['alt_count', 'total_depth']
        missing_columns = [col for col in required_columns if col not in variant_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(variant_data['alt_count']):
            raise ValueError("alt_count column must be numeric")
        if not pd.api.types.is_numeric_dtype(variant_data['total_depth']):
            raise ValueError("total_depth column must be numeric")
        
        # Check for negative values
        if (variant_data['alt_count'] < 0).any():
            raise ValueError("alt_count cannot be negative")
        if (variant_data['total_depth'] <= 0).any():
            raise ValueError("total_depth must be positive")
        
        # Check consistency
        if (variant_data['alt_count'] > variant_data['total_depth']).any():
            raise ValueError("alt_count cannot exceed total_depth")
        
        results = []
        pvalues = []
        
        for _, row in variant_data.iterrows():
            try:
                # Get expected error rate for this variant
                context = row.get('context', 'default')
                expected_rate = error_rates.get(context, 1e-4)
                
                # Validate expected rate
                if not isinstance(expected_rate, (int, float)) or expected_rate < 0 or expected_rate > 1:
                    expected_rate = 1e-4  # Fallback to default
                
                # Perform test
                test_result = self.test_variant_significance(
                    observed_alt=int(row['alt_count']),
                    total_depth=int(row['total_depth']),
                    expected_error_rate=expected_rate,
                    context=context
                )
                
                results.append({
                    'site_key': row.get('site_key', ''),
                    'context': context,
                    'observed_alt': int(row['alt_count']),
                    'total_depth': int(row['total_depth']),
                    'expected_rate': expected_rate,
                    'test_statistic': test_result.statistic,
                    'pvalue': test_result.pvalue,
                    'effect_size': test_result.effect_size,
                    'method': test_result.method
                })
                
                pvalues.append(test_result.pvalue)
                
            except Exception as e:
                # Log error but continue with other variants
                import logging
                logging.warning(f"Failed to test variant at row {_}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Apply multiple testing correction
        try:
            rejected, qvalues = self.multiple_testing_correction(pvalues)
        except Exception as e:
            # Fallback: no correction
            import logging
            logging.warning(f"Multiple testing correction failed: {e}")
            rejected = [p < self.alpha for p in pvalues]
            qvalues = pvalues
        
        # Add corrected results
        results_df = pd.DataFrame(results)
        results_df['qvalue'] = qvalues
        results_df['significant'] = rejected
        results_df['significant_uncorrected'] = results_df['pvalue'] < self.alpha
        
        return results_df
    
    def calibrate_pvalues(
        self,
        null_pvalues: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Assess p-value calibration under null hypothesis."""
        
        if not null_pvalues:
            return {'ks_statistic': 1.0, 'ks_pvalue': 0.0, 'well_calibrated': False}
        
        # Test if p-values follow uniform distribution
        ks_stat, ks_pval = stats.kstest(null_pvalues, 'uniform')
        
        # Additional calibration metrics
        alpha_levels = [0.01, 0.05, 0.1, 0.2]
        observed_rates = [np.mean(np.array(null_pvalues) < alpha) for alpha in alpha_levels]
        
        calibration_metrics = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'well_calibrated': ks_pval > (1 - confidence_level),
            'alpha_levels': alpha_levels,
            'observed_rates': observed_rates,
            'expected_rates': alpha_levels,
            'calibration_errors': [obs - exp for obs, exp in zip(observed_rates, alpha_levels)]
        }
        
        return calibration_metrics
    
    def power_analysis(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """Perform power analysis for different effect sizes and sample sizes."""
        
        power_results = []
        
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                if self.test_type == "poisson":
                    # For Poisson test, effect size is rate ratio
                    null_rate = 1e-4
                    alt_rate = null_rate * effect_size
                    expected_null = null_rate * sample_size
                    expected_alt = alt_rate * sample_size
                    
                    # Calculate power using normal approximation
                    if expected_null > 5:  # Normal approximation valid
                        z_alpha = stats.norm.ppf(1 - alpha)
                        z_beta = (expected_alt - expected_null - z_alpha * np.sqrt(expected_null)) / np.sqrt(expected_alt)
                        power = stats.norm.cdf(z_beta)
                    else:
                        # Use exact calculation for small counts
                        critical_value = stats.poisson.ppf(1 - alpha, expected_null)
                        power = 1 - stats.poisson.cdf(critical_value, expected_alt)
                
                elif self.test_type == "binomial":
                    # For binomial test, effect size is odds ratio
                    null_prob = 1e-4
                    alt_prob = min(0.99, null_prob * effect_size)  # Cap at 99%
                    
                    # Use normal approximation for large samples
                    if sample_size * null_prob > 5 and sample_size * (1 - null_prob) > 5:
                        se_null = np.sqrt(null_prob * (1 - null_prob) / sample_size)
                        z_alpha = stats.norm.ppf(1 - alpha)
                        critical_prob = null_prob + z_alpha * se_null
                        
                        se_alt = np.sqrt(alt_prob * (1 - alt_prob) / sample_size)
                        z_beta = (alt_prob - critical_prob) / se_alt
                        power = stats.norm.cdf(z_beta)
                    else:
                        # Exact calculation
                        critical_value = stats.binom.ppf(1 - alpha, sample_size, null_prob)
                        power = 1 - stats.binom.cdf(critical_value, sample_size, alt_prob)
                
                power_results.append({
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'alpha': alpha,
                    'power': max(0, min(1, power)),  # Clamp to [0, 1]
                    'test_type': self.test_type
                })
        
        return pd.DataFrame(power_results)
    
    def generate_test_summary(self, test_results: pd.DataFrame) -> Dict[str, any]:
        """Generate summary statistics from test results."""
        
        if test_results.empty:
            return {'n_tests': 0}
        
        summary = {
            'n_tests': len(test_results),
            'n_significant_uncorrected': test_results['significant_uncorrected'].sum(),
            'n_significant_corrected': test_results['significant'].sum(),
            'median_pvalue': test_results['pvalue'].median(),
            'median_qvalue': test_results['qvalue'].median(),
            'min_pvalue': test_results['pvalue'].min(),
            'max_pvalue': test_results['pvalue'].max(),
            'test_method': self.test_type,
            'fdr_method': self.fdr_method,
            'alpha': self.alpha
        }
        
        # Effect size statistics
        if 'effect_size' in test_results.columns:
            effect_sizes = test_results['effect_size'].replace([np.inf, -np.inf], np.nan).dropna()
            if not effect_sizes.empty:
                summary.update({
                    'median_effect_size': effect_sizes.median(),
                    'mean_effect_size': effect_sizes.mean(),
                    'max_effect_size': effect_sizes.max()
                })
        
        return summary
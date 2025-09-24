"""
Tests for statistical testing and p-value calibration.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from precise_mrd.stats import StatisticalTester, TestResult


class TestStatisticalTester:
    """Test statistical hypothesis testing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = StatisticalTester(
            test_type="poisson",
            alpha=0.05,
            fdr_method="benjamini_hochberg"
        )
    
    def test_poisson_test_greater(self):
        """Test Poisson test with greater alternative."""
        # Test case: observed=10, expected=5 (should be significant)
        result = self.tester.poisson_test(observed=10, expected=5.0, alternative="greater")
        
        assert isinstance(result, TestResult)
        assert result.method == "poisson"
        assert result.alternative == "greater"
        assert result.statistic == 10
        assert result.pvalue < 0.05  # Should be significant
        assert result.effect_size == 2.0  # 10/5
    
    def test_poisson_test_two_sided(self):
        """Test Poisson test with two-sided alternative."""
        result = self.tester.poisson_test(observed=10, expected=10.0, alternative="two-sided")
        
        assert result.alternative == "two-sided"
        assert result.pvalue > 0.05  # Should not be significant when observed = expected
    
    def test_poisson_test_edge_cases(self):
        """Test Poisson test edge cases."""
        # Zero expected
        result = self.tester.poisson_test(observed=5, expected=0.0)
        assert result.pvalue == 1.0
        
        # Zero observed
        result = self.tester.poisson_test(observed=0, expected=5.0)
        assert result.pvalue > 0.05
    
    def test_binomial_test_greater(self):
        """Test binomial test with greater alternative."""
        # Test case: 15 successes out of 100 trials, expected prob = 0.1
        result = self.tester.binomial_test(
            successes=15, 
            trials=100, 
            expected_prob=0.1, 
            alternative="greater"
        )
        
        assert isinstance(result, TestResult)
        assert result.method == "binomial"
        assert result.alternative == "greater"
        assert result.pvalue > 0.05  # 15/100 = 0.15 vs 0.1 should not be highly significant
    
    def test_binomial_test_two_sided(self):
        """Test binomial test with two-sided alternative."""
        result = self.tester.binomial_test(
            successes=10, 
            trials=100, 
            expected_prob=0.1, 
            alternative="two-sided"
        )
        
        assert result.alternative == "two-sided"
        assert hasattr(result, 'confidence_interval')
    
    def test_binomial_test_edge_cases(self):
        """Test binomial test edge cases."""
        # Zero trials
        result = self.tester.binomial_test(successes=0, trials=0, expected_prob=0.1)
        assert result.pvalue == 1.0
        
        # All successes
        result = self.tester.binomial_test(successes=10, trials=10, expected_prob=0.5)
        assert result.pvalue < 0.05  # Should be significant
    
    def test_test_variant_significance_poisson(self):
        """Test variant significance testing with Poisson."""
        self.tester.test_type = "poisson"
        
        result = self.tester.test_variant_significance(
            observed_alt=5,
            total_depth=1000,
            expected_error_rate=1e-3
        )
        
        # Expected count = 1e-3 * 1000 = 1, observed = 5
        assert result.pvalue < 0.05  # Should be significant
        assert result.effect_size > 1.0
    
    def test_test_variant_significance_binomial(self):
        """Test variant significance testing with binomial."""
        self.tester.test_type = "binomial"
        
        result = self.tester.test_variant_significance(
            observed_alt=5,
            total_depth=1000,
            expected_error_rate=1e-3
        )
        
        # 5/1000 = 0.005 vs expected 0.001
        assert result.method == "binomial"
        assert result.pvalue < 0.05  # Should be significant
    
    def test_multiple_testing_correction(self):
        """Test multiple testing correction."""
        # Generate some p-values
        pvalues = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8]
        
        rejected, corrected_pvals = self.tester.multiple_testing_correction(pvalues)
        
        assert len(rejected) == len(pvalues)
        assert len(corrected_pvals) == len(pvalues)
        assert all(corrected_pvals >= pvalues)  # Corrected p-values should be >= original
    
    def test_test_multiple_variants(self):
        """Test multiple variant testing with FDR correction."""
        # Create test data
        variant_data = pd.DataFrame({
            'site_key': ['site1', 'site2', 'site3', 'site4'],
            'context': ['ACG', 'CCG', 'ACG', 'TCG'],
            'alt_count': [5, 2, 10, 1],
            'total_depth': [1000, 500, 2000, 100]
        })
        
        error_rates = {
            'ACG': 1e-3,
            'CCG': 5e-4,
            'TCG': 2e-3
        }
        
        results_df = self.tester.test_multiple_variants(variant_data, error_rates)
        
        assert not results_df.empty
        assert 'pvalue' in results_df.columns
        assert 'qvalue' in results_df.columns
        assert 'significant' in results_df.columns
        assert 'significant_uncorrected' in results_df.columns
        assert len(results_df) == 4
    
    def test_pvalue_calibration_uniform(self):
        """Test p-value calibration with uniform distribution."""
        # Generate uniform p-values (well-calibrated null)
        np.random.seed(42)
        null_pvalues = np.random.uniform(0, 1, 1000)
        
        calibration = self.tester.calibrate_pvalues(null_pvalues.tolist())
        
        assert 'ks_statistic' in calibration
        assert 'ks_pvalue' in calibration
        assert 'well_calibrated' in calibration
        assert calibration['ks_pvalue'] > 0.05  # Should pass KS test for uniformity
        assert calibration['well_calibrated'] is True
    
    def test_pvalue_calibration_biased(self):
        """Test p-value calibration with biased distribution."""
        # Generate biased p-values (poorly calibrated)
        np.random.seed(42)
        biased_pvalues = np.random.beta(0.5, 2, 1000)  # Skewed toward 0
        
        calibration = self.tester.calibrate_pvalues(biased_pvalues.tolist())
        
        assert calibration['ks_pvalue'] < 0.05  # Should fail KS test
        assert calibration['well_calibrated'] is False
        
        # Check calibration errors at different alpha levels
        assert 'calibration_errors' in calibration
        assert len(calibration['calibration_errors']) > 0
    
    def test_power_analysis_poisson(self):
        """Test power analysis for Poisson test."""
        self.tester.test_type = "poisson"
        
        effect_sizes = [1.5, 2.0, 3.0, 5.0]
        sample_sizes = [100, 500, 1000]
        
        power_df = self.tester.power_analysis(effect_sizes, sample_sizes)
        
        assert not power_df.empty
        assert 'effect_size' in power_df.columns
        assert 'sample_size' in power_df.columns
        assert 'power' in power_df.columns
        assert len(power_df) == len(effect_sizes) * len(sample_sizes)
        
        # Power should increase with effect size and sample size
        assert power_df['power'].min() >= 0.0
        assert power_df['power'].max() <= 1.0
    
    def test_power_analysis_binomial(self):
        """Test power analysis for binomial test."""
        self.tester.test_type = "binomial"
        
        effect_sizes = [2.0, 5.0, 10.0]
        sample_sizes = [100, 500, 1000]
        
        power_df = self.tester.power_analysis(effect_sizes, sample_sizes)
        
        assert not power_df.empty
        assert all(power_df['test_type'] == 'binomial')
        
        # Check that power increases with effect size
        high_effect = power_df[power_df['effect_size'] == 10.0]['power'].mean()
        low_effect = power_df[power_df['effect_size'] == 2.0]['power'].mean()
        assert high_effect >= low_effect
    
    def test_generate_test_summary(self):
        """Test test summary generation."""
        # Create mock test results
        test_results = pd.DataFrame({
            'pvalue': [0.001, 0.01, 0.05, 0.1, 0.5],
            'qvalue': [0.005, 0.02, 0.08, 0.15, 0.5],
            'significant_uncorrected': [True, True, True, False, False],
            'significant': [True, True, False, False, False],
            'effect_size': [5.0, 3.0, 2.0, 1.5, 1.0]
        })
        
        summary = self.tester.generate_test_summary(test_results)
        
        assert summary['n_tests'] == 5
        assert summary['n_significant_uncorrected'] == 3
        assert summary['n_significant_corrected'] == 2
        assert summary['test_method'] == 'poisson'
        assert 'median_pvalue' in summary
        assert 'median_effect_size' in summary
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty p-value calibration
        calibration = self.tester.calibrate_pvalues([])
        assert 'ks_statistic' in calibration
        
        # Empty multiple testing
        empty_df = pd.DataFrame(columns=['site_key', 'context', 'alt_count', 'total_depth'])
        results = self.tester.test_multiple_variants(empty_df, {})
        assert results.empty
        
        # Empty test summary
        empty_results = pd.DataFrame()
        summary = self.tester.generate_test_summary(empty_results)
        assert summary['n_tests'] == 0
    
    def test_different_test_types(self):
        """Test initialization with different test types."""
        # Valid test types
        poisson_tester = StatisticalTester(test_type="poisson")
        assert poisson_tester.test_type == "poisson"
        
        binomial_tester = StatisticalTester(test_type="binomial")
        assert binomial_tester.test_type == "binomial"
        
        # Invalid test type
        with pytest.raises(ValueError):
            StatisticalTester(test_type="invalid_test")
    
    def test_fdr_method_validation(self):
        """Test FDR method handling."""
        # Test different FDR methods
        pvalues = [0.001, 0.01, 0.05, 0.1]
        
        # Benjamini-Hochberg
        rejected_bh, corrected_bh = self.tester.multiple_testing_correction(
            pvalues, method="benjamini_hochberg"
        )
        
        # Bonferroni
        rejected_bonf, corrected_bonf = self.tester.multiple_testing_correction(
            pvalues, method="bonferroni"
        )
        
        # Bonferroni should be more conservative
        assert corrected_bonf[0] >= corrected_bh[0]


class TestTestResult:
    """Test TestResult data structure."""
    
    def test_test_result_creation(self):
        """Test TestResult creation and attributes."""
        result = TestResult(
            statistic=10.0,
            pvalue=0.001,
            alternative="greater",
            method="poisson",
            effect_size=2.5
        )
        
        assert result.statistic == 10.0
        assert result.pvalue == 0.001
        assert result.alternative == "greater"
        assert result.method == "poisson"
        assert result.effect_size == 2.5
        assert result.confidence_interval is None
    
    def test_test_result_with_ci(self):
        """Test TestResult with confidence interval."""
        result = TestResult(
            statistic=0.15,
            pvalue=0.05,
            alternative="two-sided",
            method="binomial",
            confidence_interval=(0.1, 0.2)
        )
        
        assert result.confidence_interval == (0.1, 0.2)


if __name__ == '__main__':
    pytest.main([__file__])
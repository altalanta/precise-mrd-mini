"""
Property-based tests for numerical stability and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from scipy import stats

from precise_mrd.stats import StatisticalTester, TestResult
from precise_mrd.lod import LODEstimator


class TestNumericalStability:
    """Test numerical stability across extreme parameter ranges."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = StatisticalTester(test_type="poisson", alpha=0.05)
        self.lod_estimator = LODEstimator()
    
    @given(
        observed=st.integers(min_value=0, max_value=10000),
        expected=st.floats(min_value=1e-12, max_value=1e6, exclude_min=False)
    )
    @settings(max_examples=100, deadline=5000)
    def test_poisson_test_numerical_stability(self, observed, expected):
        """Test Poisson test with extreme parameters."""
        
        result = self.tester.poisson_test(observed, expected)
        
        # Basic sanity checks
        assert isinstance(result, TestResult)
        assert result.method == "poisson"
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.pvalue)
        assert not np.isnan(result.pvalue)
        
        # Effect size should be finite or NaN (not infinite)
        assert not np.isinf(result.effect_size) or np.isnan(result.effect_size)
    
    @given(
        successes=st.integers(min_value=0, max_value=1000),
        trials=st.integers(min_value=1, max_value=1000),
        expected_prob=st.floats(min_value=1e-12, max_value=1-1e-12)
    )
    @settings(max_examples=100, deadline=5000)
    def test_binomial_test_numerical_stability(self, successes, trials, expected_prob):
        """Test binomial test with extreme parameters."""
        
        assume(successes <= trials)  # Valid constraint
        
        result = self.tester.binomial_test(successes, trials, expected_prob)
        
        # Basic sanity checks
        assert isinstance(result, TestResult)
        assert result.method == "binomial"
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.pvalue)
        assert not np.isnan(result.pvalue)
        
        # Effect size should be finite
        assert not np.isinf(result.effect_size)
        assert not np.isnan(result.effect_size)
    
    def test_poisson_zero_expected(self):
        """Test Poisson test with zero expected value."""
        result = self.tester.poisson_test(observed=5, expected=0.0)
        assert result.pvalue == 1.0
        assert np.isnan(result.effect_size)
    
    def test_poisson_extreme_values(self):
        """Test Poisson test with extreme values."""
        # Very large values
        result = self.tester.poisson_test(observed=10000, expected=1e-10)
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.effect_size)
        
        # Very small expected
        result = self.tester.poisson_test(observed=1, expected=1e-15)
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.effect_size)
    
    def test_binomial_edge_cases(self):
        """Test binomial test edge cases."""
        # All successes
        result = self.tester.binomial_test(successes=100, trials=100, expected_prob=0.5)
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.effect_size)
        
        # No successes
        result = self.tester.binomial_test(successes=0, trials=100, expected_prob=0.5)
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.effect_size)
        
        # Extreme probabilities
        result = self.tester.binomial_test(successes=1, trials=100, expected_prob=1e-10)
        assert 0.0 <= result.pvalue <= 1.0
        assert not np.isinf(result.effect_size)


class TestLoDStability:
    """Test LoD estimation numerical stability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = LODEstimator(n_bootstrap=10)  # Small for testing
    
    def test_detection_probability_edge_cases(self):
        """Test detection probability calculation edge cases."""
        
        def mock_detector(af, depth):
            return np.random.random() < af * depth / 1000
        
        # Very small AF
        prob, ci = self.estimator.estimate_detection_probability(
            allele_fraction=1e-6,
            depth=1000,
            n_replicates=50,
            detection_function=mock_detector
        )
        
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= ci[0] <= ci[1] <= 1.0
        
        # Very large depth
        prob, ci = self.estimator.estimate_detection_probability(
            allele_fraction=0.001,
            depth=1000000,
            n_replicates=50,
            detection_function=mock_detector
        )
        
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= ci[0] <= ci[1] <= 1.0


class TestStatisticalInvariants:
    """Test statistical invariants and properties."""
    
    @given(
        n=st.integers(min_value=10, max_value=1000),
        p=st.floats(min_value=0.01, max_value=0.99)
    )
    @settings(max_examples=50)
    def test_binomial_variance_increases_with_n(self, n, p):
        """Test that binomial variance increases with n for fixed p."""
        
        # Generate binomial samples
        sample1 = np.random.binomial(n, p, size=100)
        sample2 = np.random.binomial(n * 2, p, size=100)
        
        # Theoretical variances
        var1_theory = n * p * (1 - p)
        var2_theory = (n * 2) * p * (1 - p)
        
        # Should satisfy variance relationship
        assert var2_theory > var1_theory
        
        # Empirical check (with some tolerance for random variation)
        var1_empirical = np.var(sample1)
        var2_empirical = np.var(sample2)
        
        # Allow some flexibility due to sampling variation
        assert var2_empirical >= 0.5 * var1_empirical  # Rough check
    
    def test_pvalue_bounds(self):
        """Test that p-values are always in [0, 1]."""
        tester = StatisticalTester()
        
        # Test many combinations
        for observed in [0, 1, 10, 100, 1000]:
            for expected in [0.001, 1.0, 10.0, 100.0]:
                result = tester.poisson_test(observed, expected)
                assert 0.0 <= result.pvalue <= 1.0
    
    def test_effect_size_consistency(self):
        """Test that effect sizes are calculated consistently."""
        tester = StatisticalTester()
        
        # When observed = expected, log effect size should be near 0
        result = tester.poisson_test(observed=100, expected=100.0)
        assert abs(result.effect_size) < 0.1  # log(1) = 0, allow for numerical precision
        
        # When observed > expected, effect size should be positive
        result = tester.poisson_test(observed=200, expected=100.0)
        assert result.effect_size > 0
        
        # When observed < expected, effect size should be negative
        result = tester.poisson_test(observed=50, expected=100.0)
        assert result.effect_size < 0


if __name__ == "__main__":
    pytest.main([__file__])
"""Test Type I error rate control in statistical tests."""

import numpy as np
import pytest
from scipy import stats

from precise_mrd.call import poisson_test, binomial_test


def test_poisson_test_type1_error_control():
    """Test that Poisson test controls Type I error rate."""
    
    alpha = 0.05
    n_simulations = 1000
    false_positives = 0
    
    rng = np.random.default_rng(12345)
    
    for _ in range(n_simulations):
        # Generate null data (Poisson with known rate)
        true_rate = 0.001
        n_trials = 10000
        observed = rng.poisson(true_rate * n_trials)
        expected = true_rate * n_trials
        
        # Test against true null hypothesis
        p_value = poisson_test(observed, expected)
        
        if p_value < alpha:
            false_positives += 1
    
    observed_type_i_rate = false_positives / n_simulations
    
    # Use binomial test to check if Type I error rate is as expected
    # Allow some tolerance due to simulation variance
    p_val_type_i = stats.binom_test(false_positives, n_simulations, alpha)
    
    # Should not reject the null hypothesis that error rate = alpha
    assert p_val_type_i > 0.01, (
        f"Type I error rate {observed_type_i_rate:.3f} significantly different from {alpha}\n"
        f"Observed: {false_positives}/{n_simulations} = {observed_type_i_rate:.3f}\n"
        f"Expected: {alpha:.3f} ± {np.sqrt(alpha * (1 - alpha) / n_simulations):.3f}"
    )


def test_binomial_test_type1_error_control():
    """Test that binomial test controls Type I error rate."""
    
    alpha = 0.05
    n_simulations = 1000
    false_positives = 0
    
    rng = np.random.default_rng(54321)
    
    for _ in range(n_simulations):
        # Generate null data (binomial with known probability)
        true_p = 0.01
        n_trials = 1000
        observed = rng.binomial(n_trials, true_p)
        
        # Test against true null hypothesis
        p_value = binomial_test(observed, n_trials, true_p)
        
        if p_value < alpha:
            false_positives += 1
    
    observed_type_i_rate = false_positives / n_simulations
    
    # Check if Type I error rate is as expected
    p_val_type_i = stats.binom_test(false_positives, n_simulations, alpha)
    
    assert p_val_type_i > 0.01, (
        f"Type I error rate {observed_type_i_rate:.3f} significantly different from {alpha}\n"
        f"Observed: {false_positives}/{n_simulations} = {observed_type_i_rate:.3f}\n"
        f"Expected: {alpha:.3f}"
    )


def test_poisson_test_edge_cases():
    """Test Poisson test edge cases."""
    
    # Zero expected should return 1.0
    assert poisson_test(0, 0) == 1.0
    assert poisson_test(1, 0) == 1.0
    
    # Negative expected should return 1.0
    assert poisson_test(5, -1) == 1.0
    
    # Very large observed vs small expected should give small p-value
    p_val = poisson_test(100, 1)
    assert p_val < 0.001


def test_binomial_test_edge_cases():
    """Test binomial test edge cases."""
    
    # Zero trials should return 1.0
    assert binomial_test(0, 0, 0.5) == 1.0
    
    # Zero probability
    assert binomial_test(0, 10, 0) == 1.0
    assert binomial_test(1, 10, 0) < 0.001  # Should be very significant
    
    # Probability = 1
    assert binomial_test(10, 10, 1.0) == 1.0
    assert binomial_test(9, 10, 1.0) < 0.001  # Should be very significant


def test_multiple_testing_scenario():
    """Test Type I error control in multiple testing scenario."""
    
    alpha = 0.05
    n_tests = 100
    n_simulations = 100  # Reduced for speed
    
    violation_count = 0
    
    rng = np.random.default_rng(98765)
    
    for sim in range(n_simulations):
        # Generate multiple null tests
        p_values = []
        
        for test_i in range(n_tests):
            # Generate null data for Poisson test
            true_rate = rng.uniform(0.0001, 0.01)  # Random true rate
            n_obs = int(rng.uniform(1000, 10000))  # Random sample size
            observed = rng.poisson(true_rate * n_obs)
            expected = true_rate * n_obs
            
            p_val = poisson_test(observed, expected)
            p_values.append(p_val)
        
        # Count false positives (should be ~alpha * n_tests)
        false_positives = sum(1 for p in p_values if p < alpha)
        
        # If we have too many false positives, it's a violation
        expected_fp = alpha * n_tests
        # Allow up to 2 standard deviations above expected
        threshold = expected_fp + 2 * np.sqrt(expected_fp * (1 - alpha))
        
        if false_positives > threshold:
            violation_count += 1
    
    # Should have few violations
    violation_rate = violation_count / n_simulations
    assert violation_rate < 0.1, (
        f"Too many Type I error violations: {violation_count}/{n_simulations} = {violation_rate:.3f}"
    )
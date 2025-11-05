"""Unit tests for the MRD calling and statistical testing module."""

import pytest
import numpy as np
from scipy import stats

from precise_mrd.call import poisson_test, binomial_test, benjamini_hochberg_correction

# Test cases for poisson_test
@pytest.mark.parametrize("observed, expected, expected_p_value", [
    (5, 1.0, 0.0076),  # Significantly higher
    (1, 5.0, 0.174),   # Significantly lower
    (5, 5.0, 1.0),     # Exactly as expected
    (0, 0.0, 1.0),     # Edge case: zero expectation
    (10, 0.1, 1.4e-14),# Extreme case
])
def test_poisson_test(observed, expected, expected_p_value):
    p_value = poisson_test(observed, expected)
    assert np.isclose(p_value, expected_p_value, atol=1e-4)

# Test cases for binomial_test
@pytest.mark.parametrize("successes, trials, p, expected_p_value", [
    (10, 100, 0.01, 0.0001), # Significantly higher
    (1, 100, 0.1, 0.0058),   # Significantly lower
    (10, 100, 0.1, 1.0),     # Exactly as expected
    (0, 0, 0.5, 1.0),       # Edge case: zero trials
    (0, 10, 0.0, 1.0),      # Edge case: p=0, no successes
    (1, 10, 0.0, 0.0),      # Edge case: p=0, with successes
    (10, 10, 1.0, 1.0),     # Edge case: p=1, all successes
    (9, 10, 1.0, 0.0),      # Edge case: p=1, with failures
])
def test_binomial_test(successes, trials, p, expected_p_value):
    p_value = binomial_test(successes, trials, p)
    assert np.isclose(p_value, expected_p_value, atol=1e-4)

# Test cases for benjamini_hochberg_correction
def test_benjamini_hochberg_correction_simple():
    p_values = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.1])
    alpha = 0.05
    rejected, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
    
    expected_rejected = np.array([True, True, True, True, False, False])
    expected_adjusted_p = np.array([0.006, 0.015, 0.02, 0.045, 0.06, 0.1])
    
    assert np.array_equal(rejected, expected_rejected)
    assert np.allclose(adjusted_p, expected_adjusted_p)

def test_benjamini_hochberg_correction_empty():
    p_values = np.array([])
    alpha = 0.05
    rejected, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
    assert len(rejected) == 0
    assert len(adjusted_p) == 0

def test_benjamini_hochberg_correction_no_rejections():
    p_values = np.array([0.1, 0.2, 0.3])
    alpha = 0.05
    rejected, _ = benjamini_hochberg_correction(p_values, alpha)
    assert not np.any(rejected)


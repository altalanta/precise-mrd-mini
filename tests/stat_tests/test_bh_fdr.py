"""Test Benjamini-Hochberg FDR correction properties."""

import numpy as np
import pytest

from precise_mrd.call import benjamini_hochberg_correction


def test_bh_monotonicity():
    """Test that BH adjusted p-values are monotonic."""
    
    # Create test p-values
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9])
    alpha = 0.1
    
    rejected, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
    
    # Adjusted p-values should be monotonic (non-decreasing)
    for i in range(len(adjusted_p) - 1):
        assert adjusted_p[i] <= adjusted_p[i + 1], (
            f"Adjusted p-values not monotonic: {adjusted_p[i]:.6f} > {adjusted_p[i + 1]:.6f} "
            f"at indices {i}, {i + 1}"
        )
    
    # Adjusted p-values should be >= original p-values
    assert np.all(adjusted_p >= p_values), "Adjusted p-values should be >= original p-values"


def test_bh_idempotence():
    """Test that applying BH twice gives same result."""
    
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.2])
    alpha = 0.05
    
    # Apply BH once
    rejected1, adjusted1 = benjamini_hochberg_correction(p_values, alpha)
    
    # Apply BH to already adjusted p-values
    rejected2, adjusted2 = benjamini_hochberg_correction(adjusted1, alpha)
    
    # Results should be identical (within numerical precision)
    np.testing.assert_array_equal(rejected1, rejected2)
    np.testing.assert_allclose(adjusted1, adjusted2, rtol=1e-10)


def test_bh_known_case():
    """Test BH correction on a known small case."""
    
    # Known example from Benjamini & Hochberg (1995)
    p_values = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.02, 0.025, 0.05, 0.3, 0.5])
    alpha = 0.05
    
    rejected, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
    
    # Check that some hypotheses are rejected
    assert np.sum(rejected) > 0, "Should reject some hypotheses in this known case"
    
    # The most significant p-values should be rejected
    assert rejected[0] == True, "Most significant p-value should be rejected"
    
    # Adjusted p-values should be reasonable
    assert adjusted_p[0] < adjusted_p[-1], "First adjusted p-value should be < last"
    
    # No adjusted p-value should exceed 1
    assert np.all(adjusted_p <= 1.0), "Adjusted p-values should not exceed 1"


def test_bh_fdr_control_simulation():
    """Test FDR control through simulation."""
    
    alpha = 0.1
    n_tests = 50
    n_nulls = 40  # 80% null hypotheses
    n_simulations = 100  # Reduced for speed
    
    fdr_violations = 0
    total_discoveries = 0
    total_false_discoveries = 0
    
    rng = np.random.default_rng(54321)
    
    for _ in range(n_simulations):
        # Generate mixture of null and alternative p-values
        p_values = np.zeros(n_tests)
        
        # Null p-values (uniform on [0,1])
        p_values[:n_nulls] = rng.uniform(0, 1, n_nulls)
        
        # Alternative p-values (enriched for small values)
        p_values[n_nulls:] = rng.beta(0.5, 3, n_tests - n_nulls)
        
        # Apply BH correction
        rejected, _ = benjamini_hochberg_correction(p_values, alpha)
        
        # Count discoveries and false discoveries
        discoveries = np.sum(rejected)
        false_discoveries = np.sum(rejected[:n_nulls])  # Rejections among nulls
        
        total_discoveries += discoveries
        total_false_discoveries += false_discoveries
        
        # Check FDR for this simulation
        if discoveries > 0:
            fdr = false_discoveries / discoveries
            if fdr > alpha:
                fdr_violations += 1
    
    # Overall FDR should be controlled
    if total_discoveries > 0:
        overall_fdr = total_false_discoveries / total_discoveries
        assert overall_fdr <= alpha + 0.05, (  # Allow some tolerance
            f"Overall FDR {overall_fdr:.3f} exceeds {alpha + 0.05}"
        )
    
    # Violation rate should be reasonable
    violation_rate = fdr_violations / n_simulations
    assert violation_rate < 0.3, (  # Allow higher tolerance due to simulation variance
        f"FDR violation rate {violation_rate:.3f} too high"
    )


def test_bh_edge_cases():
    """Test BH correction edge cases."""
    
    # Empty array
    rejected, adjusted = benjamini_hochberg_correction(np.array([]), 0.05)
    assert len(rejected) == 0
    assert len(adjusted) == 0
    
    # Single p-value
    rejected, adjusted = benjamini_hochberg_correction(np.array([0.01]), 0.05)
    assert len(rejected) == 1
    assert len(adjusted) == 1
    assert rejected[0] == True
    assert adjusted[0] == 0.01
    
    # All p-values = 1
    p_vals = np.ones(5)
    rejected, adjusted = benjamini_hochberg_correction(p_vals, 0.05)
    assert np.all(rejected == False)
    assert np.all(adjusted == 1.0)
    
    # All p-values = 0 (should all be rejected)
    p_vals = np.zeros(5)
    rejected, adjusted = benjamini_hochberg_correction(p_vals, 0.05)
    assert np.all(rejected == True)
    assert np.all(adjusted == 0.0)


def test_bh_alpha_zero():
    """Test BH with alpha = 0 (should reject nothing)."""
    
    p_values = np.array([0.001, 0.01, 0.05, 0.1])
    rejected, adjusted = benjamini_hochberg_correction(p_values, alpha=0.0)
    
    # Should reject nothing when alpha = 0
    assert np.all(rejected == False)


def test_bh_alpha_one():
    """Test BH with alpha = 1 (should reject all with p < 1)."""
    
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])
    rejected, adjusted = benjamini_hochberg_correction(p_values, alpha=1.0)
    
    # Should reject all with alpha = 1 (except p-values = 1)
    assert np.all(rejected == True)
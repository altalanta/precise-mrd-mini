"""Test LoB/LoD bootstrap estimation sanity checks."""

import numpy as np
import pytest

from precise_mrd.metrics import bootstrap_metric, roc_auc_score


def generate_synthetic_detection_data(
    n_samples: int, 
    true_af: float, 
    depth: int,
    background_rate: float = 1e-4,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic detection data for LoB/LoD testing."""
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    # True labels: 1 if variant should be detectable, 0 if background
    y_true = np.zeros(n_samples)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        if true_af > 0:
            # Simulate variant detection
            # Higher AF and depth increase detection probability
            detection_prob = min(0.95, true_af * depth / 1000)
            is_detected = rng.random() < detection_prob
            
            if is_detected:
                y_true[i] = 1
                # Score based on variant strength
                scores[i] = rng.uniform(true_af * 0.5, true_af * 1.5)
            else:
                # Missed variant (background level)
                scores[i] = rng.uniform(0, background_rate * 2)
        else:
            # Pure background/negative control
            y_true[i] = 0
            scores[i] = rng.uniform(0, background_rate * 2)
    
    return y_true, scores


def test_lob_estimation_negatives():
    """Test LoB estimation using negative controls."""
    
    rng = np.random.default_rng(12345)
    
    # Generate negative control data (no true variants)
    n_negatives = 1000
    y_true, scores = generate_synthetic_detection_data(
        n_negatives, true_af=0.0, depth=10000, rng=rng
    )
    
    # All should be true negatives
    assert np.all(y_true == 0), "Negative controls should have no true variants"
    
    # Estimate LoB as 95th percentile of scores
    lob = np.percentile(scores, 95)
    
    # LoB should be low for good negative controls
    assert lob < 0.001, f"LoB too high for negative controls: {lob}"
    
    # Most scores should be below LoB
    below_lob = np.sum(scores < lob)
    assert below_lob >= 0.9 * n_negatives, f"Too many negatives above LoB: {below_lob}/{n_negatives}"


def test_lod_greater_than_lob():
    """Test that LoD > LoB in synthetic data."""
    
    rng = np.random.default_rng(54321)
    
    # Generate negative controls for LoB
    n_negatives = 500
    y_neg, scores_neg = generate_synthetic_detection_data(
        n_negatives, true_af=0.0, depth=10000, rng=rng
    )
    lob = np.percentile(scores_neg, 95)
    
    # Test different allele fractions for LoD
    allele_fractions = [0.001, 0.005, 0.01, 0.05]
    lod_values = []
    
    for af in allele_fractions:
        # Generate low-positive data
        n_samples = 200
        y_true, scores = generate_synthetic_detection_data(
            n_samples, true_af=af, depth=5000, rng=rng
        )
        
        # Estimate detection rate
        detection_rate = np.mean(y_true)
        
        # LoD95 approximation: AF where 95% detection achieved
        if detection_rate >= 0.8:  # Rough threshold for LoD
            lod_values.append(af)
    
    if lod_values:
        min_lod = min(lod_values)
        # LoD should be greater than LoB
        assert min_lod > lob, f"LoD ({min_lod}) should be > LoB ({lob})"


def test_bootstrap_ci_coverage():
    """Test that bootstrap confidence intervals have reasonable coverage."""
    
    rng = np.random.default_rng(98765)
    
    # Generate data with known ROC AUC
    n_samples = 200
    
    # Create data with moderate separation (known AUC ≈ 0.7)
    y_true = np.concatenate([
        np.zeros(n_samples // 2),  # Negatives  
        np.ones(n_samples // 2)    # Positives
    ])
    
    # Scores with some overlap but clear separation
    scores = np.concatenate([
        rng.normal(0.3, 0.2, n_samples // 2),  # Negative scores
        rng.normal(0.7, 0.2, n_samples // 2)   # Positive scores  
    ])
    scores = np.clip(scores, 0, 1)
    
    # Bootstrap confidence interval
    n_bootstrap = 100  # Reduced for speed
    ci_result = bootstrap_metric(y_true, scores, roc_auc_score, n_bootstrap, rng)
    
    # Check CI properties
    assert ci_result["lower"] <= ci_result["mean"] <= ci_result["upper"], (
        "CI bounds should bracket the mean"
    )
    
    assert ci_result["std"] > 0, "Bootstrap standard error should be positive"
    
    # CI should be reasonable width (not too narrow or too wide)
    ci_width = ci_result["upper"] - ci_result["lower"]
    assert 0.01 < ci_width < 0.5, f"CI width {ci_width:.3f} seems unreasonable"
    
    # Mean should be reasonable for this synthetic data
    assert 0.5 < ci_result["mean"] < 0.9, f"Mean ROC AUC {ci_result['mean']:.3f} unexpected"


def test_bootstrap_reproducibility():
    """Test that bootstrap with same seed gives identical results."""
    
    # Generate test data
    rng = np.random.default_rng(11111)
    n_samples = 100
    y_true = rng.choice([0, 1], n_samples)
    scores = rng.uniform(0, 1, n_samples)
    
    # Run bootstrap twice with same seed
    rng1 = np.random.default_rng(22222)
    ci1 = bootstrap_metric(y_true, scores, roc_auc_score, 50, rng1)
    
    rng2 = np.random.default_rng(22222)  # Same seed
    ci2 = bootstrap_metric(y_true, scores, roc_auc_score, 50, rng2)
    
    # Results should be identical
    assert ci1["mean"] == ci2["mean"], "Bootstrap should be reproducible with same seed"
    assert ci1["lower"] == ci2["lower"], "Bootstrap CI lower should be reproducible"
    assert ci1["upper"] == ci2["upper"], "Bootstrap CI upper should be reproducible"
    assert ci1["std"] == ci2["std"], "Bootstrap std should be reproducible"


def test_lod_increases_with_depth():
    """Test that detection improves with sequencing depth."""
    
    rng = np.random.default_rng(33333)
    
    af = 0.005  # Fixed allele fraction
    depths = [1000, 5000, 10000, 20000]
    detection_rates = []
    
    for depth in depths:
        n_samples = 100
        y_true, scores = generate_synthetic_detection_data(
            n_samples, true_af=af, depth=depth, rng=rng
        )
        
        detection_rate = np.mean(y_true)
        detection_rates.append(detection_rate)
    
    # Detection rate should generally increase with depth
    # (Allow some noise due to simulation variance)
    for i in range(len(depths) - 1):
        # Later depths should have detection rate at least as good
        # Allow small decreases due to simulation noise
        assert detection_rates[i + 1] >= detection_rates[i] - 0.1, (
            f"Detection rate decreased with higher depth: "
            f"{depths[i]}→{depths[i+1]}: {detection_rates[i]:.3f}→{detection_rates[i+1]:.3f}"
        )


def test_bootstrap_edge_cases():
    """Test bootstrap behavior with edge cases."""
    
    rng = np.random.default_rng(44444)
    
    # All same class (should handle gracefully)
    y_true_all_zero = np.zeros(100)
    scores_all_zero = rng.uniform(0, 1, 100)
    
    ci_result = bootstrap_metric(y_true_all_zero, scores_all_zero, roc_auc_score, 20, rng)
    
    # Should return reasonable defaults for degenerate case
    assert 0 <= ci_result["mean"] <= 1, "Bootstrap mean should be in [0,1]"
    assert ci_result["lower"] <= ci_result["upper"], "CI bounds should be ordered"
    
    # Perfect separation case
    y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
    scores_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    ci_perfect = bootstrap_metric(y_true_perfect, scores_perfect, roc_auc_score, 20, rng)
    
    # Should get high AUC for perfect separation
    assert ci_perfect["mean"] > 0.9, f"Perfect separation should give high AUC: {ci_perfect['mean']:.3f}"
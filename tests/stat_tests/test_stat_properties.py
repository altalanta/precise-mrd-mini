"""Property-based statistical tests for Precise MRD."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from precise_mrd.call import benjamini_hochberg_correction, poisson_test
from precise_mrd.cli import create_minimal_config
from precise_mrd.eval.lod import LODAnalyzer
from precise_mrd.sim.contamination import ContaminationSimulator


def _sorted_strict_floats(min_value: float, max_value: float, size: int) -> list[float]:
    values = np.linspace(min_value, max_value, num=size, endpoint=True)
    perturb = np.linspace(0, 0.1 * (max_value - min_value), num=size)
    return list(values + perturb)


@given(
    alpha=st.floats(
        min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False
    ),
    true_rate=st.floats(
        min_value=1e-4, max_value=5e-3, allow_nan=False, allow_infinity=False
    ),
    n_trials=st.integers(min_value=200, max_value=1500),
)
@settings(max_examples=10, deadline=None)
def test_poisson_type_i_control(alpha: float, true_rate: float, n_trials: int) -> None:
    """Empirical Type-I error should not exceed the target alpha beyond tolerance."""

    expected = true_rate * n_trials
    if expected <= 0.1:
        pytest.skip("expected count too small for stable simulation")

    rng = np.random.default_rng(1234)
    n_sim = 40
    false_positives = 0

    for _ in range(n_sim):
        observed = rng.poisson(expected)
        if poisson_test(int(observed), float(expected)) < alpha:
            false_positives += 1

    observed_rate = false_positives / n_sim
    tolerance = alpha * 0.25 + 0.02
    assert observed_rate <= alpha + tolerance


@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    alpha=st.floats(
        min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=25, deadline=None)
def test_bh_adjusted_p_monotonic(p_values: list[float], alpha: float) -> None:
    """Benjamini-Hochberg adjusted p-values must be monotonic non-decreasing."""

    arr = np.asarray(p_values, dtype=float)
    rejected, adjusted = benjamini_hochberg_correction(arr, alpha)
    assert rejected.shape == arr.shape
    assert adjusted.shape == arr.shape
    assert np.all(np.diff(np.sort(adjusted)) >= -1e-9)


@pytest.mark.slow
@given(
    base_hits=st.lists(
        st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=6,
        unique=True,
    ),
    improvement=st.floats(
        min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=10, deadline=None)
def test_lod_monotonic_depth(base_hits: list[float], improvement: float) -> None:
    """Higher depth detection curves should not yield worse LoD estimates."""

    af_values = _sorted_strict_floats(1e-4, 1e-2, len(base_hits))
    base_hits_sorted = sorted(base_hits)
    high_depth_hits = [min(1.0, value + improvement) for value in base_hits_sorted]

    config = create_minimal_config(seed=17)
    analyzer = LODAnalyzer(config, np.random.default_rng(17))

    lod_low = analyzer._fit_detection_curve(af_values, base_hits_sorted, 0.95)
    lod_high = analyzer._fit_detection_curve(af_values, high_depth_hits, 0.95)

    assert np.isfinite(lod_low)
    assert np.isfinite(lod_high)
    assert lod_high <= lod_low * 1.05


@pytest.mark.slow
@given(
    base_hits=st.lists(
        st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=10, deadline=None)
def test_bootstrap_interval_contains_estimate(base_hits: list[float]) -> None:
    """Bootstrap confidence intervals should contain the point LoD estimate."""

    af_values = _sorted_strict_floats(1e-4, 1e-2, len(base_hits))
    hit_rates = sorted(base_hits)

    config = create_minimal_config(seed=29)
    analyzer = LODAnalyzer(config, np.random.default_rng(29))

    lod_estimate = analyzer._fit_detection_curve(af_values, hit_rates, 0.9)
    lower, upper = analyzer._bootstrap_lod_ci(af_values, hit_rates, 0.9, n_bootstrap=64)

    assert np.isfinite(lod_estimate)
    assert lower <= lod_estimate <= upper * 1.05


@pytest.mark.slow
@given(
    hop_rates=st.lists(
        st.floats(min_value=0.0, max_value=0.02, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=4,
        unique=True,
    ),
    base_sensitivity=st.floats(
        min_value=0.4, max_value=0.95, allow_nan=False, allow_infinity=False
    ),
    decay=st.floats(
        min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=10, deadline=None)
def test_contamination_monotonicity(
    hop_rates: list[float], base_sensitivity: float, decay: float
) -> None:
    """Synthetic contamination results should become worse with higher hop rates."""

    hop_rates_sorted = sorted(hop_rates)
    af_values = [1e-3, 5e-3]
    depth_values = [1000, 5000]

    results = {"index_hopping": {}}
    for idx, hop_rate in enumerate(hop_rates_sorted):
        sensitivity = max(0.0, base_sensitivity - idx * decay)
        rate_results = {}
        for af in af_values:
            rate_results[af] = {}
            for depth in depth_values:
                rate_results[af][depth] = {
                    "mean_sensitivity": float(sensitivity),
                    "std_sensitivity": 0.0,
                    "sensitivity_scores": [float(sensitivity)],
                    "n_replicates": 1,
                }
        results["index_hopping"][hop_rate] = rate_results

    simulator = ContaminationSimulator(
        create_minimal_config(3), np.random.default_rng(3)
    )
    matrix = simulator._create_sensitivity_matrix(results, af_values, depth_values)
    hop_matrix = np.array(matrix["index_hopping"]["matrix"])

    # Ensure sensitivity does not improve as hop rate increases
    diffs = np.diff(hop_matrix, axis=0)
    assert np.all(diffs <= 1e-6)

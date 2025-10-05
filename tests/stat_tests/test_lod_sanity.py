"""Statistical sanity tests for LoB/LoD/LoQ analytics.

These tests ensure that detection limit estimates are statistically sound
and maintain expected relationships (LoB < LoD < LoQ, monotonicity, etc.).
"""

import pytest
import numpy as np
from precise_mrd.config import load_config
from precise_mrd.eval.lod import LODAnalyzer
from precise_mrd.eval.stratified import StratifiedAnalyzer


class TestLODSanity:
    """Test statistical sanity of detection limit estimates."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config("configs/smoke.yaml")
    
    @pytest.fixture  
    def rng(self, config):
        """Create seeded random number generator."""
        return np.random.default_rng(config.seed)
    
    @pytest.fixture
    def lod_analyzer(self, config, rng):
        """Create LoD analyzer instance."""
        return LODAnalyzer(config, rng)
    
    def test_lob_basic_properties(self, lod_analyzer):
        """Test that LoB estimation has expected properties."""
        # Run LoB estimation with small sample for speed
        lob_results = lod_analyzer.estimate_lob(n_blank_runs=20)
        
        # Basic sanity checks
        assert 'lob_value' in lob_results
        assert 'blank_mean' in lob_results
        assert 'blank_std' in lob_results
        
        # LoB should be positive and finite
        assert lob_results['lob_value'] >= 0
        assert np.isfinite(lob_results['lob_value'])
        
        # LoB should be greater than blank mean (95th percentile property)
        assert lob_results['lob_value'] >= lob_results['blank_mean']
        
        # Standard deviation should be non-negative
        assert lob_results['blank_std'] >= 0
        
        # Blank measurements should be reasonable (not all zeros, not too high)
        blank_measurements = np.array(lob_results['blank_measurements'])
        assert len(blank_measurements) == 20
        assert blank_measurements.max() < 100  # Sanity check for reasonable values
    
    def test_lod_basic_properties(self, lod_analyzer):
        """Test that LoD estimation has expected properties."""
        # Run LoD estimation with reduced grid for speed
        lod_results = lod_analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[1000, 5000],
            n_replicates=10
        )
        
        # Basic structure checks
        assert 'depth_results' in lod_results
        assert 1000 in lod_results['depth_results']
        assert 5000 in lod_results['depth_results']
        
        for depth, results in lod_results['depth_results'].items():
            # LoD should be positive and finite
            assert results['lod_af'] > 0
            assert np.isfinite(results['lod_af'])
            
            # LoD should be within tested AF range
            assert 1e-3 <= results['lod_af'] <= 1e-2
            
            # Confidence intervals should be valid
            assert results['lod_ci_lower'] <= results['lod_af'] <= results['lod_ci_upper']
            assert results['lod_ci_lower'] > 0
            assert np.isfinite(results['lod_ci_upper'])
    
    def test_loq_basic_properties(self, lod_analyzer):
        """Test that LoQ estimation has expected properties."""
        # Run LoQ estimation with reduced parameters
        loq_results = lod_analyzer.estimate_loq(
            af_range=(1e-3, 1e-2),
            depth_values=[1000, 5000], 
            n_replicates=10,
            cv_threshold=0.20
        )
        
        # Basic structure checks
        assert 'depth_results' in loq_results
        
        for depth, results in loq_results['depth_results'].items():
            # LoQ should be positive and finite (if found)
            if results['loq_af_cv'] is not None:
                assert results['loq_af_cv'] > 0
                assert np.isfinite(results['loq_af_cv'])
                
                # LoQ should be within or above tested AF range
                assert results['loq_af_cv'] >= 1e-3
            
            # CV threshold should match input
            assert results['cv_threshold'] == 0.20
    
    def test_lob_lod_relationship(self, lod_analyzer):
        """Test that LoB < LoD relationship holds."""
        # Run both analyses with minimal parameters
        lob_results = lod_analyzer.estimate_lob(n_blank_runs=15)
        lod_results = lod_analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],  # Single depth for speed
            n_replicates=10
        )
        
        lob_value = lob_results['lob_value']
        lod_af = lod_results['depth_results'][5000]['lod_af']
        
        # Convert LoD AF to approximate detection count for comparison
        # Rough approximation: LoD_counts ≈ LoD_AF * depth * pipeline_efficiency
        pipeline_efficiency = 0.8  # Approximate efficiency
        lod_counts = lod_af * 5000 * pipeline_efficiency
        
        # LoB should be less than expected LoD detections
        # Allow some tolerance due to different units and approximations
        tolerance_factor = 2.0  # Allow 2x tolerance
        assert lob_value <= lod_counts * tolerance_factor, \
            f"LoB ({lob_value}) should be less than LoD detections (~{lod_counts:.1f})"
    
    def test_lod_depth_monotonicity(self, lod_analyzer):
        """Test that LoD decreases (improves) with increasing depth."""
        # Test with two depths
        lod_results = lod_analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[1000, 5000],
            n_replicates=10
        )
        
        lod_1k = lod_results['depth_results'][1000]['lod_af']
        lod_5k = lod_results['depth_results'][5000]['lod_af']
        
        # LoD should improve (decrease) with higher depth
        # Allow some tolerance due to statistical variation
        tolerance = 1.5  # Allow 50% tolerance
        assert lod_1k >= lod_5k / tolerance, \
            f"LoD should improve with depth: 1K LoD ({lod_1k:.2e}) >= 5K LoD ({lod_5k:.2e})"
    
    def test_lod_confidence_intervals(self, lod_analyzer):
        """Test that LoD confidence intervals are reasonable."""
        lod_results = lod_analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=15  # Larger sample for better CI estimation
        )
        
        results = lod_results['depth_results'][5000]
        lod_af = results['lod_af']
        ci_lower = results['lod_ci_lower']
        ci_upper = results['lod_ci_upper']
        
        # CIs should bracket the estimate
        assert ci_lower <= lod_af <= ci_upper
        
        # CI width should be reasonable (not too narrow or too wide)
        ci_width = ci_upper - ci_lower
        relative_width = ci_width / lod_af
        
        # Relative CI width should be between 10% and 200%
        assert 0.1 <= relative_width <= 2.0, \
            f"CI relative width ({relative_width:.1%}) seems unreasonable"
        
        # CIs should be positive
        assert ci_lower > 0
        assert ci_upper > 0
    
    def test_loq_cv_threshold_behavior(self, lod_analyzer):
        """Test that LoQ respects CV threshold."""
        # Test with different CV thresholds
        strict_loq = lod_analyzer.estimate_loq(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=10,
            cv_threshold=0.10  # Strict 10% CV
        )
        
        lenient_loq = lod_analyzer.estimate_loq(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=10,
            cv_threshold=0.30  # Lenient 30% CV
        )
        
        strict_result = strict_loq['depth_results'][5000]['loq_af_cv']
        lenient_result = lenient_loq['depth_results'][5000]['loq_af_cv']
        
        # Stricter threshold should give higher (worse) LoQ
        if strict_result is not None and lenient_result is not None:
            assert strict_result >= lenient_result, \
                f"Stricter CV threshold should give higher LoQ: {strict_result:.2e} >= {lenient_result:.2e}"
    
    def test_detection_limit_consistency(self, lod_analyzer):
        """Test overall consistency of detection limits."""
        # Run all three analyses
        lob_results = lod_analyzer.estimate_lob(n_blank_runs=15)
        lod_results = lod_analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=10
        )
        loq_results = lod_analyzer.estimate_loq(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=10,
            cv_threshold=0.20
        )
        
        # Extract values
        lod_af = lod_results['depth_results'][5000]['lod_af']
        loq_af = loq_results['depth_results'][5000]['loq_af_cv']
        
        # LoQ should be >= LoD (if both found)
        if loq_af is not None:
            tolerance = 0.8  # Allow 20% tolerance
            assert loq_af >= lod_af * tolerance, \
                f"LoQ ({loq_af:.2e}) should be >= LoD ({lod_af:.2e})"
        
        # All results should have consistent metadata
        assert lob_results['config_hash'] == lod_results['config_hash']
        assert lod_results['config_hash'] == loq_results['config_hash']


class TestStratifiedSanity:
    """Test statistical sanity of stratified analysis."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config("configs/smoke.yaml")
    
    @pytest.fixture
    def rng(self, config):
        """Create seeded random number generator."""
        return np.random.default_rng(config.seed)
    
    @pytest.fixture
    def stratified_analyzer(self, config, rng):
        """Create stratified analyzer instance."""
        return StratifiedAnalyzer(config, rng)
    
    def test_stratified_power_basic_properties(self, stratified_analyzer):
        """Test basic properties of stratified power analysis."""
        # Run with minimal parameters
        power_results = stratified_analyzer.analyze_stratified_power(
            af_values=[0.005, 0.01],
            depth_values=[1000, 5000],
            contexts=['CpG', 'CHG'],
            n_replicates=10
        )
        
        # Basic structure checks
        assert 'stratified_results' in power_results
        assert 'CpG' in power_results['stratified_results']
        assert 'CHG' in power_results['stratified_results']
        
        for context in ['CpG', 'CHG']:
            context_results = power_results['stratified_results'][context]
            
            for depth in [1000, 5000]:
                assert depth in context_results
                
                for af in [0.005, 0.01]:
                    assert af in context_results[depth]
                    
                    result = context_results[depth][af]
                    
                    # Detection rates should be between 0 and 1
                    assert 0 <= result['mean_detection_rate'] <= 1
                    assert result['std_detection_rate'] >= 0
                    
                    # Should have correct number of replicates
                    assert result['n_replicates'] == 10
    
    def test_calibration_basic_properties(self, stratified_analyzer):
        """Test basic properties of calibration analysis."""
        # Run with minimal parameters  
        calib_results = stratified_analyzer.analyze_calibration_by_bins(
            af_values=[0.005, 0.01],
            depth_values=[1000, 5000],
            n_bins=5,  # Fewer bins for speed
            n_replicates=15
        )
        
        # Basic structure checks
        assert 'calibration_data' in calib_results
        assert len(calib_results['calibration_data']) > 0
        
        for data_point in calib_results['calibration_data']:
            # ECE should be between 0 and 1
            assert 0 <= data_point['ece'] <= 1
            
            # Max calibration error should be >= ECE
            assert data_point['max_ce'] >= data_point['ece']
            
            # Should have correct number of bins
            assert len(data_point['bin_accuracies']) == 5
            assert len(data_point['bin_confidences']) == 5
            assert len(data_point['bin_counts']) == 5
            
            # Bin counts should be non-negative
            assert all(count >= 0 for count in data_point['bin_counts'])
    
    def test_power_af_monotonicity(self, stratified_analyzer):
        """Test that detection power increases with AF."""
        power_results = stratified_analyzer.analyze_stratified_power(
            af_values=[0.001, 0.01],  # Large AF difference
            depth_values=[5000],
            contexts=['CpG'],
            n_replicates=15
        )
        
        context_results = power_results['stratified_results']['CpG'][5000]
        
        low_af_power = context_results[0.001]['mean_detection_rate']
        high_af_power = context_results[0.01]['mean_detection_rate']
        
        # Higher AF should have higher detection power
        # Allow some tolerance due to statistical variation
        tolerance = 0.1  # 10% tolerance
        assert high_af_power >= low_af_power - tolerance, \
            f"Detection power should increase with AF: {high_af_power:.3f} >= {low_af_power:.3f}"
    
    def test_power_depth_monotonicity(self, stratified_analyzer):
        """Test that detection power increases with depth."""
        power_results = stratified_analyzer.analyze_stratified_power(
            af_values=[0.005],
            depth_values=[1000, 5000],
            contexts=['CpG'], 
            n_replicates=15
        )
        
        context_results = power_results['stratified_results']['CpG']
        
        low_depth_power = context_results[1000][0.005]['mean_detection_rate']
        high_depth_power = context_results[5000][0.005]['mean_detection_rate']
        
        # Higher depth should have higher detection power
        tolerance = 0.1  # 10% tolerance
        assert high_depth_power >= low_depth_power - tolerance, \
            f"Detection power should increase with depth: {high_depth_power:.3f} >= {low_depth_power:.3f}"


class TestDetectionLimitIntegration:
    """Integration tests for detection limit pipeline."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration.""" 
        return load_config("configs/smoke.yaml")
    
    @pytest.fixture
    def rng(self, config):
        """Create seeded random number generator."""
        return np.random.default_rng(config.seed)
    
    def test_full_detection_limit_pipeline(self, config, rng):
        """Test complete detection limit analysis pipeline."""
        analyzer = LODAnalyzer(config, rng)
        
        # Run all analyses with minimal parameters
        lob_results = analyzer.estimate_lob(n_blank_runs=10)
        lod_results = analyzer.estimate_lod(
            af_range=(1e-3, 1e-2),
            depth_values=[5000],
            n_replicates=8
        )
        loq_results = analyzer.estimate_loq(
            af_range=(1e-3, 1e-2), 
            depth_values=[5000],
            n_replicates=8,
            cv_threshold=0.20
        )
        
        # All analyses should complete without errors
        assert lob_results is not None
        assert lod_results is not None
        assert loq_results is not None
        
        # Should have consistent configuration hashes
        assert lob_results['config_hash'] == lod_results['config_hash']
        assert lod_results['config_hash'] == loq_results['config_hash']
        
        # Basic value sanity checks
        assert lob_results['lob_value'] >= 0
        assert lod_results['depth_results'][5000]['lod_af'] > 0
        
        # Pipeline should be deterministic
        # Run again with same seed
        analyzer2 = LODAnalyzer(config, np.random.default_rng(config.seed))
        lob_results2 = analyzer2.estimate_lob(n_blank_runs=10)
        
        # Should get identical results
        assert lob_results['lob_value'] == lob_results2['lob_value']
        assert lob_results['blank_mean'] == lob_results2['blank_mean']
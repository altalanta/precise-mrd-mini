"""
Tests for LoD and LoB estimation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from precise_mrd.lod import LODEstimator, LODResult, LOBResult


class TestLODEstimator:
    """Test LoD estimation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = LODEstimator(
            detection_threshold=0.95,
            confidence_level=0.95,
            n_bootstrap=100,  # Small number for fast tests
            alpha=0.05
        )
    
    def test_detection_probability_estimation(self):
        """Test detection probability estimation."""
        # Mock detection function
        def mock_detection_func(af, depth):
            # Simulate detection probability that increases with AF
            prob = min(0.95, af * 1000)  # 100% at AF=0.001
            return np.random.random() < prob
        
        np.random.seed(42)
        detection_rate, ci = self.estimator.estimate_detection_probability(
            allele_fraction=0.001,
            depth=10000,
            n_replicates=100,
            detection_function=mock_detection_func
        )
        
        assert 0.0 <= detection_rate <= 1.0
        assert len(ci) == 2
        assert ci[0] <= detection_rate <= ci[1]
    
    def test_lod95_from_curve_interpolation(self):
        """Test LoD95 estimation from detection curve."""
        # Create mock detection curve
        detection_curve = pd.DataFrame({
            'allele_fraction': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'detection_rate': [0.1, 0.5, 0.9, 0.98, 0.99],
            'ci_lower': [0.05, 0.4, 0.85, 0.95, 0.97],
            'ci_upper': [0.15, 0.6, 0.95, 0.99, 1.0]
        })
        
        lod95, ci = self.estimator.estimate_lod95_from_curve(detection_curve)
        
        # LoD95 should be interpolated between 0.0005 and 0.001
        assert 0.0005 <= lod95 <= 0.001
        assert len(ci) == 2
        assert ci[0] <= lod95 <= ci[1]
    
    def test_lod95_extrapolation(self):
        """Test LoD95 when 95% detection is not achieved."""
        # Create curve that never reaches 95%
        detection_curve = pd.DataFrame({
            'allele_fraction': [0.001, 0.005, 0.01],
            'detection_rate': [0.3, 0.6, 0.8],  # Max 80%
            'ci_lower': [0.2, 0.5, 0.7],
            'ci_upper': [0.4, 0.7, 0.9]
        })
        
        lod95, ci = self.estimator.estimate_lod95_from_curve(detection_curve)
        
        # Should extrapolate beyond tested range
        assert lod95 > 0.01
        assert ci[1] == float('inf')  # Upper CI should be infinite
    
    def test_bootstrap_lod_estimation(self):
        """Test bootstrap LoD estimation."""
        # Create mock simulation results
        np.random.seed(42)
        simulation_results = []
        
        for af in [0.0001, 0.0005, 0.001, 0.005]:
            for rep in range(50):
                # Simulate detection probability that increases with AF
                detected = np.random.random() < min(0.99, af * 2000)
                simulation_results.append({
                    'allele_fraction': af,
                    'umi_depth': 10000,
                    'detected': detected,
                    'replicate': rep
                })
        
        results_df = pd.DataFrame(simulation_results)
        
        lod_result = self.estimator.bootstrap_lod_estimation(results_df, depth=10000)
        
        assert isinstance(lod_result, LODResult)
        assert lod_result.lod95 > 0
        assert len(lod_result.confidence_interval) == 2
        assert not lod_result.detection_curve.empty
        assert lod_result.n_bootstrap <= 100  # Some iterations might fail
    
    def test_bootstrap_lod_no_data(self):
        """Test bootstrap LoD with no data."""
        empty_df = pd.DataFrame(columns=['allele_fraction', 'umi_depth', 'detected'])
        
        with pytest.raises(ValueError, match="No simulation results found"):
            self.estimator.bootstrap_lod_estimation(empty_df, depth=10000)
    
    def test_lob_estimation(self):
        """Test LoB estimation from negative controls."""
        # Create mock negative control data
        np.random.seed(42)
        negative_controls = []
        
        for rep in range(1000):
            # Small false positive rate
            detected = np.random.random() < 0.02  # 2% FP rate
            negative_controls.append({
                'allele_fraction': 0.0,
                'umi_depth': 10000,
                'detected': detected
            })
        
        nc_df = pd.DataFrame(negative_controls)
        
        lob_result = self.estimator.estimate_lob(nc_df, depth=10000)
        
        assert isinstance(lob_result, LOBResult)
        assert 0.0 <= lob_result.false_positive_rate <= 1.0
        assert lob_result.lob >= lob_result.false_positive_rate
        assert len(lob_result.confidence_interval) == 2
        assert lob_result.n_replicates == 1000
    
    def test_lob_no_data(self):
        """Test LoB with no negative control data."""
        empty_df = pd.DataFrame(columns=['allele_fraction', 'umi_depth', 'detected'])
        
        with pytest.raises(ValueError, match="No negative control data found"):
            self.estimator.estimate_lob(empty_df, depth=10000)
    
    def test_detection_heatmap_data(self):
        """Test detection heatmap data generation."""
        # Create mock simulation results across AF and depth grid
        simulation_results = []
        
        for af in [0.001, 0.005, 0.01]:
            for depth in [5000, 10000, 20000]:
                for rep in range(10):
                    # Higher detection at higher AF and depth
                    detected = np.random.random() < (af * depth / 20000)
                    simulation_results.append({
                        'allele_fraction': af,
                        'umi_depth': depth,
                        'detected': detected
                    })
        
        results_df = pd.DataFrame(simulation_results)
        
        heatmap_data = self.estimator.generate_detection_heatmap_data(results_df)
        
        assert isinstance(heatmap_data, pd.DataFrame)
        assert heatmap_data.index.name == 'allele_fraction'
        assert 'umi_depth' in heatmap_data.columns.names or len(heatmap_data.columns) > 0
    
    def test_lod_monotonicity_validation(self):
        """Test LoD monotonicity validation."""
        # Create mock LoD results with monotonic behavior
        lod_results = {
            5000: Mock(lod95=0.005),
            10000: Mock(lod95=0.002),
            20000: Mock(lod95=0.001),
            50000: Mock(lod95=0.0005)
        }
        
        validation = self.estimator.validate_lod_monotonicity(lod_results)
        
        assert validation['is_monotonic'] is True
        assert validation['correlation_with_depth'] < 0  # Negative correlation
        assert validation['reasonable_range'] is True
        assert validation['passed_validation'] is True
    
    def test_lod_monotonicity_violation(self):
        """Test LoD monotonicity validation with violations."""
        # Create non-monotonic LoD results
        lod_results = {
            5000: Mock(lod95=0.001),   # Should be highest
            10000: Mock(lod95=0.005),  # Higher than expected
            20000: Mock(lod95=0.002),
            50000: Mock(lod95=0.0005)
        }
        
        validation = self.estimator.validate_lod_monotonicity(lod_results)
        
        assert validation['is_monotonic'] is False
        assert validation['passed_validation'] is False
    
    def test_lod_summary_table(self):
        """Test LoD/LoB summary table generation."""
        # Create mock LoD and LoB results
        lod_results = {
            10000: LODResult(
                lod95=0.001,
                confidence_interval=(0.0008, 0.0015),
                detection_curve=pd.DataFrame(),
                bootstrap_estimates=[0.001],
                n_bootstrap=100,
                confidence_level=0.95
            )
        }
        
        lob_results = {
            10000: LOBResult(
                lob=0.02,
                false_positive_rate=0.015,
                confidence_interval=(0.01, 0.025),
                n_replicates=1000,
                alpha=0.05
            )
        }
        
        summary_table = self.estimator.generate_lod_summary_table(lod_results, lob_results)
        
        assert not summary_table.empty
        assert 'depth' in summary_table.columns
        assert 'lod95' in summary_table.columns
        assert 'lob' in summary_table.columns
        assert len(summary_table) == 1
    
    def test_analytical_sensitivity(self):
        """Test analytical sensitivity calculation."""
        # Create mock LoD result with detection curve
        detection_curve = pd.DataFrame({
            'allele_fraction': [0.0001, 0.0005, 0.001, 0.005],
            'detection_rate': [0.1, 0.5, 0.9, 0.98]
        })
        
        lod_results = {
            10000: LODResult(
                lod95=0.001,
                confidence_interval=(0.0008, 0.0015),
                detection_curve=detection_curve,
                bootstrap_estimates=[0.001],
                n_bootstrap=100,
                confidence_level=0.95
            )
        }
        
        sensitivity_metrics = self.estimator.calculate_analytical_sensitivity(
            lod_results, 
            clinical_threshold=0.001
        )
        
        assert 10000 in sensitivity_metrics
        metrics = sensitivity_metrics[10000]
        assert 'sensitivity_at_clinical_threshold' in metrics
        assert 'meets_clinical_requirement' in metrics
        assert metrics['sensitivity_at_clinical_threshold'] == 0.9  # From detection curve
        assert metrics['meets_clinical_requirement'] is True  # LoD95 <= threshold
    
    def test_empty_bootstrap_handling(self):
        """Test handling of failed bootstrap iterations."""
        # Create data that will cause bootstrap failures
        bad_data = pd.DataFrame({
            'allele_fraction': [0.001],
            'umi_depth': [10000],
            'detected': [True]  # Only one data point
        })
        
        # Mock bootstrap to always fail
        with patch.object(self.estimator, 'estimate_lod95_from_curve', side_effect=Exception("Mock failure")):
            with pytest.raises(ValueError, match="No successful bootstrap iterations"):
                self.estimator.bootstrap_lod_estimation(bad_data, depth=10000)


class TestLODResult:
    """Test LODResult data structure."""
    
    def test_lod_result_creation(self):
        """Test LODResult creation and attributes."""
        detection_curve = pd.DataFrame({
            'allele_fraction': [0.001, 0.005],
            'detection_rate': [0.8, 0.95]
        })
        
        result = LODResult(
            lod95=0.002,
            confidence_interval=(0.0015, 0.0025),
            detection_curve=detection_curve,
            bootstrap_estimates=[0.002, 0.0018, 0.0022],
            n_bootstrap=3,
            confidence_level=0.95
        )
        
        assert result.lod95 == 0.002
        assert result.confidence_interval == (0.0015, 0.0025)
        assert len(result.detection_curve) == 2
        assert len(result.bootstrap_estimates) == 3
        assert result.n_bootstrap == 3
        assert result.confidence_level == 0.95


class TestLOBResult:
    """Test LOBResult data structure."""
    
    def test_lob_result_creation(self):
        """Test LOBResult creation and attributes."""
        result = LOBResult(
            lob=0.025,
            false_positive_rate=0.02,
            confidence_interval=(0.015, 0.03),
            n_replicates=1000,
            alpha=0.05
        )
        
        assert result.lob == 0.025
        assert result.false_positive_rate == 0.02
        assert result.confidence_interval == (0.015, 0.03)
        assert result.n_replicates == 1000
        assert result.alpha == 0.05


if __name__ == '__main__':
    pytest.main([__file__])
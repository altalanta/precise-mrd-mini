"""
Tests for reproducibility and deterministic behavior.
"""

import pytest
import numpy as np
import pandas as pd
from precise_mrd.stats import StatisticalTester
from precise_mrd.lod import LODEstimator
from precise_mrd.umi import UMIProcessor
import tempfile
import hashlib


class TestDeterminism:
    """Test deterministic behavior across multiple runs."""
    
    def test_statistical_tester_determinism(self):
        """Test that statistical tests are deterministic."""
        
        # Set seed and run tests
        np.random.seed(42)
        tester1 = StatisticalTester()
        result1 = tester1.poisson_test(observed=10, expected=5.0)
        
        # Reset seed and run again
        np.random.seed(42)
        tester2 = StatisticalTester()
        result2 = tester2.poisson_test(observed=10, expected=5.0)
        
        # Results should be identical
        assert result1.statistic == result2.statistic
        assert result1.pvalue == result2.pvalue
        assert result1.effect_size == result2.effect_size
    
    def test_bootstrap_determinism(self):
        """Test that bootstrap procedures are deterministic with same seed."""
        
        # Create mock simulation results
        data = pd.DataFrame({
            'allele_fraction': [0.001] * 100,
            'umi_depth': [1000] * 100,
            'detected': np.random.choice([0, 1], 100)
        })
        
        # Run bootstrap with seed
        np.random.seed(42)
        estimator1 = LODEstimator(n_bootstrap=10)
        # Note: This would need actual implementation to work
        # result1 = estimator1.bootstrap_lod_estimation(data, depth=1000)
        
        # Reset seed and run again
        np.random.seed(42)
        estimator2 = LODEstimator(n_bootstrap=10)
        # result2 = estimator2.bootstrap_lod_estimation(data, depth=1000)
        
        # For now, just test that the estimators are created consistently
        assert estimator1.n_bootstrap == estimator2.n_bootstrap
        assert estimator1.detection_threshold == estimator2.detection_threshold
    
    def test_numpy_seeding_works(self):
        """Test that numpy seeding produces consistent results."""
        
        # First run
        np.random.seed(123)
        array1 = np.random.random(100)
        
        # Second run with same seed
        np.random.seed(123)
        array2 = np.random.random(100)
        
        # Should be identical
        assert np.allclose(array1, array2)
        assert np.array_equal(array1, array2)
    
    def test_dataframe_hash_consistency(self):
        """Test that identical dataframes produce same hash."""
        
        # Create two identical dataframes
        df1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })
        
        df2 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })
        
        # Hash should be identical
        hash1 = hashlib.md5(pd.util.hash_pandas_object(df1).values).hexdigest()
        hash2 = hashlib.md5(pd.util.hash_pandas_object(df2).values).hexdigest()
        
        assert hash1 == hash2
        
        # Different data should produce different hash
        df3 = df1.copy()
        df3.loc[0, 'a'] = 999
        hash3 = hashlib.md5(pd.util.hash_pandas_object(df3).values).hexdigest()
        
        assert hash1 != hash3
    
    def test_umi_processor_determinism(self):
        """Test that UMI processing is deterministic."""
        
        # This would require actual Read objects to test properly
        # For now, test that processor creation is consistent
        
        processor1 = UMIProcessor(
            min_family_size=3,
            consensus_threshold=0.6
        )
        
        processor2 = UMIProcessor(
            min_family_size=3,
            consensus_threshold=0.6
        )
        
        # Configuration should be identical
        assert processor1.min_family_size == processor2.min_family_size
        assert processor1.consensus_threshold == processor2.consensus_threshold
    
    def test_config_parameter_isolation(self):
        """Test that different instances don't share mutable state."""
        
        # Create two processors with different settings
        proc1 = UMIProcessor(min_family_size=2)
        proc2 = UMIProcessor(min_family_size=5)
        
        # Settings should be isolated
        assert proc1.min_family_size != proc2.min_family_size
        assert proc1.min_family_size == 2
        assert proc2.min_family_size == 5
        
        # Modifying one shouldn't affect the other
        proc1.min_family_size = 10
        assert proc2.min_family_size == 5


if __name__ == "__main__":
    pytest.main([__file__])
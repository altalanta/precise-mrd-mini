"""
Tests for UMI processing and consensus calling.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from precise_mrd.umi import UMIFamily, UMIProcessor
from precise_mrd.io import Read, TargetSite


class TestUMIFamily:
    """Test UMI family data structure."""
    
    def test_family_creation(self):
        """Test UMI family creation and basic properties."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 28, "-", 75),
            Read("chr1", 1000, "A", "A", "AAGGCCTT", 32, "+", 60)
        ]
        
        family = UMIFamily(umi="AAGGCCTT", reads=reads)
        
        assert family.umi == "AAGGCCTT"
        assert family.family_size == 3
        assert len(family.reads) == 3
        assert family.site_key == "chr1:1000:A"
    
    def test_strand_counts(self):
        """Test strand counting in UMI family."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 28, "+", 75),
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 32, "-", 60)
        ]
        
        family = UMIFamily(umi="AAGGCCTT", reads=reads)
        strand_counts = family.strand_counts
        
        assert strand_counts["+"] == 2
        assert strand_counts["-"] == 1
    
    def test_allele_counts(self):
        """Test allele counting in UMI family."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 28, "+", 75),
            Read("chr1", 1000, "A", "A", "AAGGCCTT", 32, "-", 60)
        ]
        
        family = UMIFamily(umi="AAGGCCTT", reads=reads)
        allele_counts = family.allele_counts
        
        assert allele_counts["T"] == 2
        assert allele_counts["A"] == 1


class TestUMIProcessor:
    """Test UMI processing logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = UMIProcessor(
            min_family_size=2,
            max_family_size=100,
            quality_threshold=20,
            consensus_threshold=0.6
        )
    
    def test_edit_distance(self):
        """Test UMI edit distance calculation."""
        assert self.processor.edit_distance("AAGGCCTT", "AAGGCCTT") == 0
        assert self.processor.edit_distance("AAGGCCTT", "AAGGCCTA") == 1
        assert self.processor.edit_distance("AAGGCCTT", "TTCCGGAA") == 8
        assert self.processor.edit_distance("AAGGCCTT", "AAGGCC") == 8  # Different length
    
    def test_consensus_calling_simple(self):
        """Test consensus calling with simple majority."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 28, "+", 75),
            Read("chr1", 1000, "A", "A", "AAGGCCTT", 32, "-", 60)
        ]
        
        consensus_allele, consensus_quality = self.processor.call_consensus(reads)
        
        assert consensus_allele == "T"  # Majority allele
        assert consensus_quality > 0
    
    def test_consensus_calling_quality_threshold(self):
        """Test consensus calling respects quality threshold."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 15, "+", 50),  # Below threshold
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 10, "+", 75),  # Below threshold
            Read("chr1", 1000, "A", "A", "AAGGCCTT", 32, "-", 60)   # Above threshold
        ]
        
        consensus_allele, consensus_quality = self.processor.call_consensus(reads)
        
        assert consensus_allele == "A"  # Only high-quality read
    
    def test_consensus_calling_insufficient_consensus(self):
        """Test consensus calling fails with insufficient agreement."""
        reads = [
            Read("chr1", 1000, "A", "T", "AAGGCCTT", 30, "+", 50),
            Read("chr1", 1000, "A", "C", "AAGGCCTT", 28, "+", 75),
            Read("chr1", 1000, "A", "G", "AAGGCCTT", 32, "-", 60)
        ]
        
        consensus_allele, consensus_quality = self.processor.call_consensus(reads)
        
        # Should fail due to insufficient consensus (no allele > 60%)
        assert consensus_allele is None
        assert consensus_quality is None
    
    def test_family_size_filtering(self):
        """Test family size filtering."""
        # Create families of different sizes
        families = []
        
        # Too small family (size 1)
        small_reads = [Read("chr1", 1000, "A", "T", "UMI1", 30, "+", 50)]
        small_family = UMIFamily(umi="UMI1", reads=small_reads)
        small_family.consensus_allele = "T"
        families.append(small_family)
        
        # Good size family (size 3)
        good_reads = [
            Read("chr1", 1000, "A", "T", "UMI2", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "UMI2", 28, "+", 75),
            Read("chr1", 1000, "A", "T", "UMI2", 32, "-", 60)
        ]
        good_family = UMIFamily(umi="UMI2", reads=good_reads)
        good_family.consensus_allele = "T"
        families.append(good_family)
        
        filtered_families = self.processor.filter_family_size(families)
        
        assert len(filtered_families) == 1
        assert filtered_families[0].umi == "UMI2"
    
    def test_oversized_family_sampling(self):
        """Test oversized family gets sampled down."""
        # Create oversized family
        reads = []
        for i in range(150):  # Exceeds max_family_size of 100
            reads.append(Read("chr1", 1000, "A", "T", "UMI_BIG", 30, "+", 50))
        
        family = UMIFamily(umi="UMI_BIG", reads=reads)
        family.consensus_allele = "T"
        
        with patch('numpy.random.choice') as mock_choice:
            mock_choice.return_value = reads[:100]  # Mock sampling
            filtered_families = self.processor.filter_family_size([family])
        
        assert len(filtered_families) == 1
        assert filtered_families[0].family_size == 100
    
    def test_process_reads_integration(self):
        """Test complete read processing pipeline."""
        reads = [
            # Family 1: UMI1, size 3, should pass
            Read("chr1", 1000, "A", "T", "UMI1", 30, "+", 50),
            Read("chr1", 1000, "A", "T", "UMI1", 28, "+", 75),
            Read("chr1", 1000, "A", "T", "UMI1", 32, "-", 60),
            
            # Family 2: UMI2, size 1, should be filtered out
            Read("chr1", 1000, "A", "C", "UMI2", 30, "+", 50),
            
            # Family 3: UMI3, different site, size 2, should pass
            Read("chr2", 2000, "G", "A", "UMI3", 30, "+", 50),
            Read("chr2", 2000, "G", "A", "UMI3", 28, "+", 75)
        ]
        
        families = self.processor.process_reads(reads)
        
        # Should have 2 families (UMI1 and UMI3)
        assert len(families) == 2
        
        # Check families have consensus
        consensus_families = [f for f in families if f.consensus_allele is not None]
        assert len(consensus_families) >= 1
    
    def test_consensus_counts_dataframe(self):
        """Test consensus counts DataFrame generation."""
        # Create mock families with consensus
        family1 = UMIFamily(umi="UMI1", reads=[])
        family1.consensus_allele = "T"
        family1.reads = [Read("chr1", 1000, "A", "T", "UMI1", 30, "+", 50)]
        
        family2 = UMIFamily(umi="UMI2", reads=[])
        family2.consensus_allele = "T"
        family2.reads = [Read("chr1", 1000, "A", "T", "UMI2", 30, "+", 50)]
        
        family3 = UMIFamily(umi="UMI3", reads=[])
        family3.consensus_allele = "A"
        family3.reads = [Read("chr1", 1000, "A", "A", "UMI3", 30, "+", 50)]
        
        families = [family1, family2, family3]
        
        consensus_df = self.processor.get_consensus_counts(families)
        
        assert not consensus_df.empty
        assert "consensus_count" in consensus_df.columns
        assert "site_key" in consensus_df.columns
        
        # Should have counts for T=2, A=1 at chr1:1000
        t_count = consensus_df[
            (consensus_df['allele'] == 'T') & 
            (consensus_df['site_key'] == 'chr1:1000:A')
        ]['consensus_count'].sum()
        assert t_count == 2
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        # Create families with mixed success
        family1 = UMIFamily(umi="UMI1", reads=[Mock(), Mock(), Mock()])  # size 3
        family1.consensus_allele = "T"
        
        family2 = UMIFamily(umi="UMI2", reads=[Mock()])  # size 1 
        family2.consensus_allele = None  # Failed consensus
        
        family3 = UMIFamily(umi="UMI3", reads=[Mock(), Mock()])  # size 2
        family3.consensus_allele = "A"
        
        families = [family1, family2, family3]
        
        metrics = self.processor.calculate_efficiency_metrics(families)
        
        assert metrics['total_families'] == 3
        assert metrics['consensus_families'] == 2
        assert metrics['consensus_rate'] == 2/3
        assert metrics['total_reads'] == 6  # 3 + 1 + 2
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        families = self.processor.process_reads([])
        assert families == []
        
        consensus_df = self.processor.get_consensus_counts([])
        assert consensus_df.empty
        
        metrics = self.processor.calculate_efficiency_metrics([])
        assert 'total_families' in metrics
        assert metrics['total_families'] == 0


if __name__ == '__main__':
    pytest.main([__file__])
"""
UMI processing module for consensus calling and family-size filtering.

This module provides:
- UMI family grouping with edit distance tolerance
- Consensus calling with quality weighting
- Family size thresholding and outlier handling
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from .io import Read


@dataclass
class UMIFamily:
    """Represents a group of reads sharing the same UMI."""
    umi: str
    reads: List[Read]
    consensus_allele: Optional[str] = None
    consensus_quality: Optional[float] = None
    family_size: int = 0
    
    def __post_init__(self):
        self.family_size = len(self.reads)
    
    @property
    def site_key(self) -> str:
        """Get the genomic site key for this family."""
        if self.reads:
            read = self.reads[0]
            return f"{read.chrom}:{read.pos}:{read.ref}"
        return ""
    
    @property
    def strand_counts(self) -> Dict[str, int]:
        """Count reads by strand."""
        return Counter(read.strand for read in self.reads)
    
    @property  
    def allele_counts(self) -> Dict[str, int]:
        """Count reads by allele."""
        return Counter(read.alt for read in self.reads)


class UMIProcessor:
    """Process UMI families for consensus calling."""
    
    def __init__(
        self,
        min_family_size: int = 3,
        max_family_size: int = 1000,
        max_edit_distance: int = 1,
        quality_threshold: int = 20,
        consensus_threshold: float = 0.6
    ):
        """Initialize UMI processor with thresholds."""
        self.min_family_size = min_family_size
        self.max_family_size = max_family_size  
        self.max_edit_distance = max_edit_distance
        self.quality_threshold = quality_threshold
        self.consensus_threshold = consensus_threshold
    
    def edit_distance(self, umi1: str, umi2: str) -> int:
        """Calculate edit distance between two UMI sequences."""
        if len(umi1) != len(umi2):
            return max(len(umi1), len(umi2))
        
        return sum(c1 != c2 for c1, c2 in zip(umi1, umi2))
    
    def group_umis_by_site(self, reads: List[Read]) -> Dict[str, List[Read]]:
        """Group reads by genomic site."""
        site_groups = defaultdict(list)
        for read in reads:
            site_key = f"{read.chrom}:{read.pos}:{read.ref}"
            site_groups[site_key].append(read)
        return dict(site_groups)
    
    def group_umis_with_distance(self, reads: List[Read]) -> List[List[Read]]:
        """Group reads by UMI sequence allowing for edit distance."""
        if not reads:
            return []
        
        # Simple grouping by exact UMI match for now
        # TODO: Implement edit distance clustering for production use
        umi_groups = defaultdict(list)
        for read in reads:
            umi_groups[read.umi].append(read)
        
        return list(umi_groups.values())
    
    def call_consensus(self, reads: List[Read]) -> Tuple[Optional[str], Optional[float]]:
        """Call consensus allele from family reads with quality weighting."""
        if not reads:
            return None, None
        
        # Filter low quality reads
        quality_reads = [r for r in reads if r.quality >= self.quality_threshold]
        if not quality_reads:
            return None, None
        
        # Count alleles weighted by quality
        allele_weights = defaultdict(float)
        total_weight = 0.0
        
        for read in quality_reads:
            weight = 10 ** (read.quality / 10.0)  # Convert Phred to probability weight
            allele_weights[read.alt] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None, None
        
        # Find consensus allele
        best_allele = max(allele_weights, key=allele_weights.get)
        consensus_fraction = allele_weights[best_allele] / total_weight
        
        # Require minimum consensus fraction
        if consensus_fraction < self.consensus_threshold:
            return None, None
        
        # Calculate average quality for consensus
        consensus_quality = np.mean([r.quality for r in quality_reads if r.alt == best_allele])
        
        return best_allele, consensus_quality
    
    def filter_family_size(self, families: List[UMIFamily]) -> List[UMIFamily]:
        """Filter families by size thresholds."""
        filtered = []
        
        for family in families:
            # Skip undersized families
            if family.family_size < self.min_family_size:
                continue
            
            # Cap oversized families (potential artifacts)
            if family.family_size > self.max_family_size:
                # Randomly sample down to max size
                sampled_reads = np.random.choice(
                    family.reads, 
                    size=self.max_family_size, 
                    replace=False
                ).tolist()
                family.reads = sampled_reads
                family.family_size = len(sampled_reads)
            
            filtered.append(family)
        
        return filtered
    
    def process_reads(self, reads: List[Read]) -> List[UMIFamily]:
        """Process reads into consensus UMI families."""
        if not reads:
            return []
        
        # Group by genomic site first
        site_groups = self.group_umis_by_site(reads)
        
        all_families = []
        
        for site_key, site_reads in site_groups.items():
            # Group by UMI within each site
            umi_groups = self.group_umis_with_distance(site_reads)
            
            for umi_reads in umi_groups:
                if not umi_reads:
                    continue
                
                # Create family
                family = UMIFamily(
                    umi=umi_reads[0].umi,
                    reads=umi_reads
                )
                
                # Call consensus
                consensus_allele, consensus_quality = self.call_consensus(umi_reads)
                family.consensus_allele = consensus_allele
                family.consensus_quality = consensus_quality
                
                all_families.append(family)
        
        # Filter by family size
        filtered_families = self.filter_family_size(all_families)
        
        return filtered_families
    
    def get_consensus_counts(self, families: List[UMIFamily]) -> pd.DataFrame:
        """Get consensus counts per site and allele."""
        data = []
        
        site_allele_counts = defaultdict(lambda: defaultdict(int))
        
        for family in families:
            if family.consensus_allele is None:
                continue
            
            site_key = family.site_key
            allele = family.consensus_allele
            site_allele_counts[site_key][allele] += 1
        
        for site_key, allele_counts in site_allele_counts.items():
            chrom, pos, ref = site_key.split(':')
            
            for allele, count in allele_counts.items():
                data.append({
                    'chrom': chrom,
                    'pos': int(pos),
                    'ref': ref,
                    'allele': allele,
                    'consensus_count': count,
                    'site_key': site_key
                })
        
        return pd.DataFrame(data)
    
    def get_family_size_distribution(self, families: List[UMIFamily]) -> Dict[int, int]:
        """Get distribution of family sizes."""
        return Counter(family.family_size for family in families)
    
    def calculate_efficiency_metrics(self, families: List[UMIFamily]) -> Dict[str, float]:
        """Calculate UMI processing efficiency metrics."""
        if not families:
            return {}
        
        total_families = len(families)
        consensus_families = sum(1 for f in families if f.consensus_allele is not None)
        total_reads = sum(f.family_size for f in families)
        
        size_dist = self.get_family_size_distribution(families)
        mean_family_size = np.mean(list(size_dist.keys()))
        
        return {
            'total_families': total_families,
            'consensus_families': consensus_families,
            'consensus_rate': consensus_families / total_families if total_families > 0 else 0.0,
            'total_reads': total_reads,
            'mean_family_size': mean_family_size,
            'families_below_threshold': sum(1 for f in families if f.family_size < self.min_family_size),
            'families_above_threshold': sum(1 for f in families if f.family_size > self.max_family_size)
        }
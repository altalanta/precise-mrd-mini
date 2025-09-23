"""
I/O module for synthetic read generation and target site definitions.

This module provides:
- Synthetic FASTQ/UMI read generation for simulation
- Target site definitions (VCF-like)
- Read data structures
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
from collections import namedtuple


@dataclass
class TargetSite:
    """Represents a genomic target site for MRD analysis."""
    chrom: str
    pos: int
    ref: str
    alt: str
    context: str  # trinucleotide context
    gene: Optional[str] = None
    
    @property
    def key(self) -> str:
        """Unique identifier for this site."""
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"
    
    def __hash__(self) -> int:
        return hash(self.key)


@dataclass 
class Read:
    """Represents a sequencing read with UMI."""
    chrom: str
    pos: int
    ref: str
    alt: str
    umi: str
    quality: int
    strand: str  # '+' or '-'
    read_position: int  # position within read (for end-repair artifacts)
    family_id: Optional[str] = None
    

ReadPair = namedtuple('ReadPair', ['read1', 'read2'])


class SyntheticReadGenerator:
    """Generates synthetic reads for simulation purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = np.random.RandomState(seed)
        
    def generate_umi(self, length: int = 12) -> str:
        """Generate random UMI sequence."""
        bases = 'ACGT'
        return ''.join(self.rng.choice(list(bases), size=length))
    
    def generate_quality_score(self, mean_quality: int = 30) -> int:
        """Generate Phred quality score."""
        # Use truncated normal distribution
        quality = self.rng.normal(mean_quality, 5)
        return max(10, min(40, int(quality)))
    
    def simulate_family_sizes(self, n_families: int, mean_size: float = 5.0) -> List[int]:
        """Simulate UMI family sizes using negative binomial distribution."""
        # Use negative binomial to model overdispersion in family sizes
        r = 2.0  # dispersion parameter
        p = r / (r + mean_size)
        sizes = self.rng.negative_binomial(r, p, size=n_families) + 1
        return sizes.tolist()
    
    def generate_reads_for_site(
        self,
        site: TargetSite,
        n_umi_families: int,
        allele_fraction: float = 0.0,
        contamination_rate: float = 0.0,
        mean_family_size: float = 5.0
    ) -> List[Read]:
        """Generate synthetic reads for a target site."""
        
        reads = []
        family_sizes = self.simulate_family_sizes(n_umi_families, mean_family_size)
        
        for family_idx, family_size in enumerate(family_sizes):
            umi = self.generate_umi()
            family_id = f"fam_{family_idx:06d}"
            
            # Determine if this family contains the variant
            has_variant = self.rng.random() < allele_fraction
            
            # Add contamination
            if contamination_rate > 0 and self.rng.random() < contamination_rate:
                has_variant = not has_variant  # flip variant status
            
            for read_idx in range(family_size):
                # Determine allele for this read
                if has_variant:
                    allele = site.alt if self.rng.random() < 0.9 else site.ref  # 90% concordance within family
                else:
                    allele = site.ref if self.rng.random() < 0.95 else site.alt  # 5% error rate
                
                read = Read(
                    chrom=site.chrom,
                    pos=site.pos,
                    ref=site.ref,
                    alt=allele,
                    umi=umi,
                    quality=self.generate_quality_score(),
                    strand=self.rng.choice(['+', '-']),
                    read_position=self.rng.randint(10, 150),  # position within 150bp read
                    family_id=family_id
                )
                reads.append(read)
        
        return reads
    
    def generate_target_sites(self, n_sites: int = 10) -> List[TargetSite]:
        """Generate a set of target sites for testing."""
        sites = []
        contexts = ['ACG', 'CCG', 'GCG', 'TCG', 'ACA', 'CCA', 'GCA', 'TCA']
        
        for i in range(n_sites):
            chrom = f"chr{self.rng.randint(1, 23)}"
            pos = self.rng.randint(1000000, 50000000)
            ref = self.rng.choice(['A', 'C', 'G', 'T'])
            alt = self.rng.choice([b for b in ['A', 'C', 'G', 'T'] if b != ref])
            context = self.rng.choice(contexts)
            
            site = TargetSite(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                context=context,
                gene=f"GENE_{i+1}"
            )
            sites.append(site)
        
        return sites


def load_target_sites(filepath: str) -> List[TargetSite]:
    """Load target sites from VCF-like file."""
    # Placeholder for VCF loading - would implement actual VCF parsing
    # For now, return empty list
    return []


def save_reads_to_fastq(reads: List[Read], filepath: str) -> None:
    """Save reads to FASTQ format (placeholder)."""
    # Would implement actual FASTQ writing
    pass


def reads_to_dataframe(reads: List[Read]) -> pd.DataFrame:
    """Convert read list to pandas DataFrame for analysis."""
    data = []
    for read in reads:
        data.append({
            'chrom': read.chrom,
            'pos': read.pos,
            'ref': read.ref,
            'alt': read.alt,
            'umi': read.umi,
            'quality': read.quality,
            'strand': read.strand,
            'read_position': read.read_position,
            'family_id': read.family_id,
            'site_key': f"{read.chrom}:{read.pos}:{read.ref}>{read.alt}"
        })
    
    return pd.DataFrame(data)
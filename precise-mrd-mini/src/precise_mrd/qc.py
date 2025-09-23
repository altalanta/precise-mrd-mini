"""
Quality control (QC) module for MRD analysis.

This module provides:
- Per-site QC metrics calculation
- Family size distribution analysis
- Depth and coverage metrics
- Clinical guardrail validation
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import Counter
import logging


@dataclass
class QCThresholds:
    """QC thresholds for clinical validation."""
    min_total_umi_families: int = 1000
    max_contamination_rate: float = 0.05
    min_depth_per_locus: int = 100
    min_family_size: int = 3
    max_family_size_outlier: int = 1000
    min_consensus_rate: float = 0.8


@dataclass
class SiteQCResult:
    """QC results for a single genomic site."""
    site_key: str
    total_families: int
    consensus_families: int
    consensus_rate: float
    mean_family_size: float
    depth_distribution: Dict[str, float]
    contamination_estimate: float
    passes_qc: bool
    failed_criteria: List[str]


class QCAnalyzer:
    """Analyze quality control metrics for MRD experiments."""
    
    def __init__(self, thresholds: Optional[QCThresholds] = None):
        """Initialize QC analyzer with thresholds."""
        self.thresholds = thresholds or QCThresholds()
        self.logger = logging.getLogger(__name__)
    
    def analyze_family_size_distribution(
        self,
        family_sizes: List[int]
    ) -> Dict[str, Union[float, int, Dict]]:
        """Analyze UMI family size distribution."""
        
        if not family_sizes:
            return {'error': 'No family sizes provided'}
        
        sizes_array = np.array(family_sizes)
        
        # Basic statistics
        stats_dict = {
            'count': len(family_sizes),
            'mean': float(np.mean(sizes_array)),
            'median': float(np.median(sizes_array)),
            'std': float(np.std(sizes_array)),
            'min': int(np.min(sizes_array)),
            'max': int(np.max(sizes_array)),
            'q25': float(np.percentile(sizes_array, 25)),
            'q75': float(np.percentile(sizes_array, 75))
        }
        
        # Distribution shape
        if len(sizes_array) > 1:
            stats_dict['skewness'] = float(stats.skew(sizes_array))
            stats_dict['kurtosis'] = float(stats.kurtosis(sizes_array))
        
        # Size distribution counts
        size_counts = Counter(family_sizes)
        stats_dict['size_distribution'] = dict(size_counts)
        
        # Identify potential issues
        issues = []
        if stats_dict['mean'] < 2:
            issues.append('low_mean_family_size')
        if stats_dict['max'] > 100:
            issues.append('extremely_large_families')
        if len(size_counts) == 1:
            issues.append('no_size_variation')
        
        stats_dict['potential_issues'] = issues
        
        return stats_dict
    
    def calculate_depth_metrics(
        self,
        consensus_data: pd.DataFrame
    ) -> Dict[str, Union[float, int]]:
        """Calculate depth and coverage metrics."""
        
        if consensus_data.empty:
            return {'error': 'No consensus data provided'}
        
        # Group by site
        site_depths = consensus_data.groupby('site_key')['consensus_count'].sum()
        
        metrics = {
            'n_sites': len(site_depths),
            'total_depth': int(site_depths.sum()),
            'mean_depth_per_site': float(site_depths.mean()),
            'median_depth_per_site': float(site_depths.median()),
            'min_depth_per_site': int(site_depths.min()),
            'max_depth_per_site': int(site_depths.max()),
            'std_depth_per_site': float(site_depths.std()),
            'sites_below_min_depth': int((site_depths < self.thresholds.min_depth_per_locus).sum()),
            'fraction_sites_below_min_depth': float((site_depths < self.thresholds.min_depth_per_locus).mean())
        }
        
        # Calculate uniformity metrics
        if len(site_depths) > 1:
            cv = site_depths.std() / site_depths.mean()  # Coefficient of variation
            metrics['depth_uniformity_cv'] = float(cv)
            
            # Calculate what fraction of sites are within 2-fold of median
            median_depth = site_depths.median()
            within_2fold = ((site_depths >= median_depth / 2) & 
                           (site_depths <= median_depth * 2)).mean()
            metrics['fraction_within_2fold_median'] = float(within_2fold)
        
        return metrics
    
    def estimate_contamination_metrics(
        self,
        sample_data: pd.DataFrame,
        reference_patterns: Optional[List[pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """Estimate contamination levels and patterns."""
        
        contamination_metrics = {
            'estimated_contamination_rate': 0.0,
            'cross_sample_evidence': 0.0,
            'unexpected_allele_fraction': 0.0
        }
        
        if sample_data.empty:
            return contamination_metrics
        
        # Simple contamination estimation based on allele frequency patterns
        total_observations = len(sample_data)
        
        # Look for unexpected low-frequency variants
        site_groups = sample_data.groupby('site_key')
        unexpected_variants = 0
        
        for site_key, site_data in site_groups:
            total_depth = site_data['consensus_count'].sum()
            
            # Check for variants at very low frequency that might indicate contamination
            variant_data = site_data[site_data['allele'] != site_data['ref'].iloc[0]]
            
            if not variant_data.empty:
                for _, variant in variant_data.iterrows():
                    vaf = variant['consensus_count'] / total_depth
                    
                    # Flag variants at 0.1-5% frequency as potential contamination
                    if 0.001 <= vaf <= 0.05:
                        unexpected_variants += 1
        
        contamination_metrics['unexpected_allele_fraction'] = (
            unexpected_variants / total_observations if total_observations > 0 else 0.0
        )
        
        # Overall contamination estimate (simplified)
        contamination_metrics['estimated_contamination_rate'] = min(
            contamination_metrics['unexpected_allele_fraction'],
            0.1  # Cap at 10%
        )
        
        return contamination_metrics
    
    def analyze_consensus_efficiency(
        self,
        families_data: List[Dict[str, any]]
    ) -> Dict[str, float]:
        """Analyze UMI consensus calling efficiency."""
        
        if not families_data:
            return {'error': 'No families data provided'}
        
        total_families = len(families_data)
        consensus_families = sum(1 for f in families_data if f.get('consensus_allele') is not None)
        
        # Family size statistics
        family_sizes = [f.get('family_size', 0) for f in families_data]
        below_threshold = sum(1 for size in family_sizes if size < self.thresholds.min_family_size)
        above_threshold = sum(1 for size in family_sizes if size > self.thresholds.max_family_size_outlier)
        
        efficiency_metrics = {
            'total_families': total_families,
            'consensus_families': consensus_families,
            'consensus_rate': consensus_families / total_families if total_families > 0 else 0.0,
            'families_below_min_size': below_threshold,
            'families_above_max_size': above_threshold,
            'fraction_below_min_size': below_threshold / total_families if total_families > 0 else 0.0,
            'fraction_above_max_size': above_threshold / total_families if total_families > 0 else 0.0,
            'effective_families': total_families - below_threshold - above_threshold,
            'effective_family_rate': ((total_families - below_threshold - above_threshold) / 
                                    total_families if total_families > 0 else 0.0)
        }
        
        return efficiency_metrics
    
    def validate_clinical_guardrails(
        self,
        qc_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Validate sample against clinical guardrails."""
        
        validation_results = {
            'sample_valid': True,
            'failed_criteria': [],
            'warnings': []
        }
        
        # Check minimum UMI families
        total_families = qc_metrics.get('total_families', 0)
        if total_families < self.thresholds.min_total_umi_families:
            validation_results['sample_valid'] = False
            validation_results['failed_criteria'].append(
                f"insufficient_total_families: {total_families} < {self.thresholds.min_total_umi_families}"
            )
        
        # Check contamination rate
        contamination_rate = qc_metrics.get('estimated_contamination_rate', 0.0)
        if contamination_rate > self.thresholds.max_contamination_rate:
            validation_results['sample_valid'] = False
            validation_results['failed_criteria'].append(
                f"high_contamination: {contamination_rate:.4f} > {self.thresholds.max_contamination_rate}"
            )
        
        # Check consensus rate
        consensus_rate = qc_metrics.get('consensus_rate', 0.0)
        if consensus_rate < self.thresholds.min_consensus_rate:
            validation_results['warnings'].append(
                f"low_consensus_rate: {consensus_rate:.4f} < {self.thresholds.min_consensus_rate}"
            )
        
        # Check depth per locus
        mean_depth = qc_metrics.get('mean_depth_per_site', 0)
        if mean_depth < self.thresholds.min_depth_per_locus:
            validation_results['warnings'].append(
                f"low_mean_depth: {mean_depth:.1f} < {self.thresholds.min_depth_per_locus}"
            )
        
        return validation_results
    
    def generate_site_qc_report(
        self,
        site_key: str,
        consensus_data: pd.DataFrame,
        families_data: List[Dict[str, any]]
    ) -> SiteQCResult:
        """Generate QC report for a single genomic site."""
        
        # Filter data for this site
        site_consensus = consensus_data[consensus_data['site_key'] == site_key]
        site_families = [f for f in families_data if f.get('site_key') == site_key]
        
        # Basic metrics
        total_families = len(site_families)
        consensus_families = sum(1 for f in site_families if f.get('consensus_allele') is not None)
        consensus_rate = consensus_families / total_families if total_families > 0 else 0.0
        
        # Family size metrics
        family_sizes = [f.get('family_size', 0) for f in site_families]
        mean_family_size = np.mean(family_sizes) if family_sizes else 0.0
        
        # Depth distribution
        if not site_consensus.empty:
            total_depth = site_consensus['consensus_count'].sum()
            depth_dist = {
                'total_depth': total_depth,
                'n_alleles': len(site_consensus),
                'max_allele_depth': site_consensus['consensus_count'].max()
            }
        else:
            depth_dist = {'total_depth': 0, 'n_alleles': 0, 'max_allele_depth': 0}
        
        # Contamination estimate (simplified)
        contamination_estimate = 0.0  # Would implement actual estimation
        
        # QC validation
        failed_criteria = []
        
        if total_families < self.thresholds.min_depth_per_locus:
            failed_criteria.append('insufficient_families')
        
        if consensus_rate < self.thresholds.min_consensus_rate:
            failed_criteria.append('low_consensus_rate')
        
        if mean_family_size < self.thresholds.min_family_size:
            failed_criteria.append('low_mean_family_size')
        
        passes_qc = len(failed_criteria) == 0
        
        return SiteQCResult(
            site_key=site_key,
            total_families=total_families,
            consensus_families=consensus_families,
            consensus_rate=consensus_rate,
            mean_family_size=mean_family_size,
            depth_distribution=depth_dist,
            contamination_estimate=contamination_estimate,
            passes_qc=passes_qc,
            failed_criteria=failed_criteria
        )
    
    def generate_comprehensive_qc_report(
        self,
        consensus_data: pd.DataFrame,
        families_data: List[Dict[str, any]],
        simulation_metadata: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Generate comprehensive QC report for entire experiment."""
        
        # Overall metrics
        family_sizes = [f.get('family_size', 0) for f in families_data]
        family_size_metrics = self.analyze_family_size_distribution(family_sizes)
        depth_metrics = self.calculate_depth_metrics(consensus_data)
        contamination_metrics = self.estimate_contamination_metrics(consensus_data)
        efficiency_metrics = self.analyze_consensus_efficiency(families_data)
        
        # Combine all metrics
        overall_metrics = {
            **family_size_metrics,
            **depth_metrics,
            **contamination_metrics,
            **efficiency_metrics
        }
        
        # Clinical validation
        clinical_validation = self.validate_clinical_guardrails(overall_metrics)
        
        # Per-site QC (sample a few sites for detailed analysis)
        unique_sites = consensus_data['site_key'].unique()
        site_qc_results = []
        
        for site_key in unique_sites[:10]:  # Limit to first 10 sites for performance
            site_qc = self.generate_site_qc_report(site_key, consensus_data, families_data)
            site_qc_results.append(site_qc.__dict__)
        
        # Summary statistics
        sites_passing_qc = sum(1 for s in site_qc_results if s['passes_qc'])
        
        qc_report = {
            'overall_metrics': overall_metrics,
            'clinical_validation': clinical_validation,
            'site_qc_summary': {
                'total_sites_analyzed': len(site_qc_results),
                'sites_passing_qc': sites_passing_qc,
                'site_pass_rate': sites_passing_qc / len(site_qc_results) if site_qc_results else 0.0
            },
            'detailed_site_qc': site_qc_results,
            'simulation_metadata': simulation_metadata or {}
        }
        
        return qc_report
    
    def export_qc_metrics(self, qc_report: Dict[str, any], filepath: str) -> None:
        """Export QC metrics to JSON file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_numpy_types(qc_report)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        self.logger.info(f"QC report exported to {filepath}")
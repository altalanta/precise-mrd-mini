"""
Quality filtering module for MRD variant calling.

This module provides:
- Strand bias testing and filtering
- End-repair artifact detection
- Quality score filtering
- Coverage depth filtering
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Container for filter results."""
    passed: bool
    reason: Optional[str] = None
    statistic: Optional[float] = None
    pvalue: Optional[float] = None


class QualityFilter:
    """Apply quality filters to variant calls."""
    
    def __init__(
        self,
        strand_bias_threshold: float = 0.01,
        end_repair_filter: bool = True,
        end_repair_distance: int = 10,
        end_repair_contexts: Optional[List[str]] = None,
        min_alt_count: int = 2,
        min_total_depth: int = 10,
        min_qual_score: int = 20
    ):
        """Initialize quality filter with thresholds."""
        self.strand_bias_threshold = strand_bias_threshold
        self.end_repair_filter = end_repair_filter
        self.end_repair_distance = end_repair_distance
        self.end_repair_contexts = end_repair_contexts or ['G>T', 'C>A']
        self.min_alt_count = min_alt_count
        self.min_total_depth = min_total_depth
        self.min_qual_score = min_qual_score
    
    def test_strand_bias(
        self,
        alt_forward: int,
        alt_reverse: int, 
        ref_forward: int,
        ref_reverse: int
    ) -> FilterResult:
        """Test for strand bias using Fisher's exact test."""
        
        # Create 2x2 contingency table
        # Rows: ref, alt
        # Cols: forward, reverse
        table = np.array([
            [ref_forward, ref_reverse],
            [alt_forward, alt_reverse]
        ])
        
        # Check if we have enough data
        if table.sum() < 4:
            return FilterResult(
                passed=True,
                reason="insufficient_data_for_strand_bias_test"
            )
        
        # Perform Fisher's exact test
        try:
            odds_ratio, pvalue = fisher_exact(table, alternative='two-sided')
            
            # Filter if significant strand bias
            strand_bias_detected = pvalue < self.strand_bias_threshold
            
            # Additional check for extreme imbalance
            total_alt = alt_forward + alt_reverse
            if total_alt > 0:
                alt_strand_ratio = alt_forward / total_alt
                # Flag extreme imbalance (>95% on one strand)
                extreme_imbalance = alt_strand_ratio > 0.95 or alt_strand_ratio < 0.05
                strand_bias_detected = strand_bias_detected and extreme_imbalance
            
            return FilterResult(
                passed=not strand_bias_detected,
                reason="strand_bias" if strand_bias_detected else None,
                statistic=odds_ratio,
                pvalue=pvalue
            )
            
        except Exception as e:
            # If test fails, be conservative and pass
            return FilterResult(
                passed=True,
                reason=f"strand_bias_test_failed: {str(e)}"
            )
    
    def test_end_repair_artifact(
        self,
        ref: str,
        alt: str,
        read_positions: List[int],
        read_length: int = 150
    ) -> FilterResult:
        """Test for end-repair artifacts."""
        
        if not self.end_repair_filter:
            return FilterResult(passed=True)
        
        mutation_type = f"{ref}>{alt}"
        
        # Check if this is a known end-repair context
        if mutation_type not in self.end_repair_contexts:
            return FilterResult(passed=True)
        
        # Check if variants are enriched near read ends
        near_ends = [
            pos <= self.end_repair_distance or 
            pos >= (read_length - self.end_repair_distance)
            for pos in read_positions
        ]
        
        n_near_ends = sum(near_ends)
        n_total = len(read_positions)
        
        if n_total == 0:
            return FilterResult(passed=True)
        
        # Calculate expected proportion near ends
        end_region_length = 2 * self.end_repair_distance
        expected_proportion = end_region_length / read_length
        
        # Test if observed proportion is significantly higher
        observed_proportion = n_near_ends / n_total
        
        # Use binomial test
        try:
            result = stats.binomtest(
                n_near_ends, 
                n_total, 
                expected_proportion, 
                alternative='greater'
            )
            
            # Filter if significantly enriched at ends
            is_end_repair_artifact = (
                result.pvalue < 0.05 and 
                observed_proportion > 2 * expected_proportion
            )
            
            return FilterResult(
                passed=not is_end_repair_artifact,
                reason="end_repair_artifact" if is_end_repair_artifact else None,
                statistic=observed_proportion,
                pvalue=result.pvalue
            )
            
        except Exception as e:
            return FilterResult(
                passed=True,
                reason=f"end_repair_test_failed: {str(e)}"
            )
    
    def filter_by_depth(self, alt_count: int, total_depth: int) -> FilterResult:
        """Filter variants by depth requirements."""
        
        if alt_count < self.min_alt_count:
            return FilterResult(
                passed=False,
                reason="insufficient_alt_count",
                statistic=alt_count
            )
        
        if total_depth < self.min_total_depth:
            return FilterResult(
                passed=False,
                reason="insufficient_total_depth",
                statistic=total_depth
            )
        
        return FilterResult(passed=True)
    
    def filter_by_quality(self, quality_scores: List[int]) -> FilterResult:
        """Filter variants by quality scores."""
        
        if not quality_scores:
            return FilterResult(
                passed=False,
                reason="no_quality_scores"
            )
        
        mean_quality = np.mean(quality_scores)
        
        if mean_quality < self.min_qual_score:
            return FilterResult(
                passed=False,
                reason="low_quality",
                statistic=mean_quality
            )
        
        return FilterResult(passed=True)
    
    def apply_all_filters(
        self,
        variant_data: Dict[str, any]
    ) -> Dict[str, FilterResult]:
        """Apply all filters to a variant."""
        
        filter_results = {}
        
        # Depth filter
        filter_results['depth'] = self.filter_by_depth(
            alt_count=variant_data.get('alt_count', 0),
            total_depth=variant_data.get('total_depth', 0)
        )
        
        # Quality filter
        if 'quality_scores' in variant_data:
            filter_results['quality'] = self.filter_by_quality(
                quality_scores=variant_data['quality_scores']
            )
        
        # Strand bias filter
        if all(key in variant_data for key in ['alt_forward', 'alt_reverse', 'ref_forward', 'ref_reverse']):
            filter_results['strand_bias'] = self.test_strand_bias(
                alt_forward=variant_data['alt_forward'],
                alt_reverse=variant_data['alt_reverse'],
                ref_forward=variant_data['ref_forward'],
                ref_reverse=variant_data['ref_reverse']
            )
        
        # End-repair artifact filter
        if all(key in variant_data for key in ['ref', 'alt', 'read_positions']):
            filter_results['end_repair'] = self.test_end_repair_artifact(
                ref=variant_data['ref'],
                alt=variant_data['alt'],
                read_positions=variant_data['read_positions']
            )
        
        return filter_results
    
    def summarize_filters(self, filter_results: Dict[str, FilterResult]) -> Dict[str, any]:
        """Summarize filter results."""
        
        passed_filters = [name for name, result in filter_results.items() if result.passed]
        failed_filters = [name for name, result in filter_results.items() if not result.passed]
        
        summary = {
            'all_filters_passed': len(failed_filters) == 0,
            'n_filters_applied': len(filter_results),
            'n_filters_passed': len(passed_filters),
            'n_filters_failed': len(failed_filters),
            'passed_filters': passed_filters,
            'failed_filters': failed_filters,
            'failure_reasons': [
                result.reason for result in filter_results.values() 
                if not result.passed and result.reason
            ]
        }
        
        return summary
    
    def filter_variant_dataframe(self, variants_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to a DataFrame of variants."""
        
        if variants_df.empty:
            return variants_df.copy()
        
        filtered_df = variants_df.copy()
        
        # Initialize filter columns
        filter_columns = ['depth_filter', 'quality_filter', 'strand_bias_filter', 'end_repair_filter']
        for col in filter_columns:
            filtered_df[col] = True
            filtered_df[f'{col}_reason'] = None
        
        # Apply filters row by row
        for idx, row in filtered_df.iterrows():
            variant_data = row.to_dict()
            filter_results = self.apply_all_filters(variant_data)
            
            # Update filter columns
            for filter_name, result in filter_results.items():
                col_name = f'{filter_name}_filter'
                reason_col = f'{col_name}_reason'
                
                if col_name in filtered_df.columns:
                    filtered_df.loc[idx, col_name] = result.passed
                    if not result.passed:
                        filtered_df.loc[idx, reason_col] = result.reason
        
        # Add overall pass/fail column
        filter_cols = [col for col in filtered_df.columns if col.endswith('_filter') and not col.endswith('_reason')]
        filtered_df['all_filters_passed'] = filtered_df[filter_cols].all(axis=1)
        
        return filtered_df
    
    def generate_filter_report(self, filtered_variants: pd.DataFrame) -> Dict[str, any]:
        """Generate summary report of filtering results."""
        
        if filtered_variants.empty:
            return {'total_variants': 0}
        
        total_variants = len(filtered_variants)
        
        report = {
            'total_variants': total_variants,
            'variants_passing_all_filters': filtered_variants['all_filters_passed'].sum(),
            'overall_pass_rate': filtered_variants['all_filters_passed'].mean(),
        }
        
        # Per-filter statistics
        filter_cols = [col for col in filtered_variants.columns if col.endswith('_filter') and not col.endswith('_reason')]
        
        for filter_col in filter_cols:
            if filter_col == 'all_filters_passed':
                continue
                
            filter_name = filter_col.replace('_filter', '')
            n_passed = filtered_variants[filter_col].sum()
            pass_rate = n_passed / total_variants
            
            report[f'{filter_name}_pass_rate'] = pass_rate
            report[f'{filter_name}_n_passed'] = n_passed
            report[f'{filter_name}_n_failed'] = total_variants - n_passed
        
        # Failure reason statistics
        reason_cols = [col for col in filtered_variants.columns if col.endswith('_reason')]
        failure_reasons = {}
        
        for reason_col in reason_cols:
            reasons = filtered_variants[reason_col].dropna()
            for reason in reasons:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        report['failure_reasons'] = failure_reasons
        
        return report
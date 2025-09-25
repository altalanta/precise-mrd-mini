"""
Limit of Detection (LoD) and Limit of Blank (LoB) estimation module.

This module provides:
- LoD95 estimation via bootstrap
- LoB calculation from negative controls
- Detection probability curves
- Performance validation
"""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import logging


@dataclass
class LODResult:
    """Container for LoD estimation results."""
    lod95: float
    confidence_interval: Tuple[float, float]
    detection_curve: pd.DataFrame
    bootstrap_estimates: List[float]
    n_bootstrap: int
    confidence_level: float
    multiple_testing_correction: str = "bonferroni"
    corrected_alpha: float = 0.05


@dataclass
class LOBResult:
    """Container for LoB estimation results."""
    lob: float
    false_positive_rate: float
    confidence_interval: Tuple[float, float]
    n_replicates: int
    alpha: float


class LODEstimator:
    """Estimate Limit of Detection and Limit of Blank for MRD assays."""
    
    def __init__(
        self,
        detection_threshold: float = 0.95,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ):
        """Initialize LoD estimator."""
        self.detection_threshold = detection_threshold
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
    
    def estimate_detection_probability(
        self,
        allele_fraction: float,
        depth: int,
        n_replicates: int,
        detection_function: Callable[[float, int], bool]
    ) -> Tuple[float, float]:
        """Estimate detection probability for given AF and depth."""
        
        detections = []
        for _ in range(n_replicates):
            detected = detection_function(allele_fraction, depth)
            detections.append(detected)
        
        detection_rate = np.mean(detections)
        
        # Calculate confidence interval using Wilson score
        n = len(detections)
        p = detection_rate
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        return detection_rate, (ci_lower, ci_upper)
    
    def estimate_lod95_from_curve(
        self,
        detection_curve: pd.DataFrame,
        interpolation_method: str = "linear"
    ) -> Tuple[float, Tuple[float, float]]:
        """Estimate LoD95 from detection probability curve."""
        
        # Sort by allele fraction
        curve_sorted = detection_curve.sort_values('allele_fraction')
        
        # Find LoD95 by interpolation
        af_values = curve_sorted['allele_fraction'].values
        detection_rates = curve_sorted['detection_rate'].values
        
        # Check if we achieve 95% detection
        if detection_rates.max() < self.detection_threshold:
            # LoD95 is beyond our tested range
            lod95 = af_values.max() * 2  # Extrapolate
            ci_lower = af_values.max()
            ci_upper = float('inf')
        else:
            # Interpolate to find exact LoD95
            lod95 = np.interp(self.detection_threshold, detection_rates, af_values)
            
            # Calculate confidence interval from curve CIs
            ci_lower_rates = curve_sorted['ci_lower'].values
            ci_upper_rates = curve_sorted['ci_upper'].values
            
            # Conservative CI estimation
            ci_lower = np.interp(self.detection_threshold, ci_upper_rates, af_values)
            ci_upper = np.interp(self.detection_threshold, ci_lower_rates, af_values)
        
        return lod95, (ci_lower, ci_upper)
    
    def bootstrap_lod_estimation(
        self,
        simulation_results: pd.DataFrame,
        depth: int,
        n_comparisons: int = 1,
        multiple_testing_method: str = "bonferroni"
    ) -> LODResult:
        """Bootstrap LoD estimation from simulation results with multiple testing correction."""
        
        # Apply multiple testing correction to alpha
        if multiple_testing_method == "bonferroni":
            corrected_alpha = self.alpha / n_comparisons
        elif multiple_testing_method == "holm":
            # Simplified Holm correction (full implementation would need p-values)
            corrected_alpha = self.alpha / n_comparisons  
        else:
            corrected_alpha = self.alpha
            
        # Filter results for specific depth
        depth_results = simulation_results[simulation_results['umi_depth'] == depth]
        
        if depth_results.empty:
            raise ValueError(f"No simulation results found for depth {depth}")
        
        bootstrap_lods = []
        
        self.logger.info(f"Running {self.n_bootstrap} bootstrap iterations for LoD estimation")
        
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap LoD"):
            # Resample with replacement
            bootstrap_sample = depth_results.sample(
                n=len(depth_results), 
                replace=True
            )
            
            # Calculate detection rates for this bootstrap sample
            detection_curve = bootstrap_sample.groupby('allele_fraction').agg({
                'detected': 'mean'
            }).reset_index()
            detection_curve.rename(columns={'detected': 'detection_rate'}, inplace=True)
            
            # Add mock confidence intervals for interpolation
            detection_curve['ci_lower'] = detection_curve['detection_rate'] - 0.05
            detection_curve['ci_upper'] = detection_curve['detection_rate'] + 0.05
            
            # Estimate LoD95 for this bootstrap sample
            try:
                lod95_boot, _ = self.estimate_lod95_from_curve(detection_curve)
                bootstrap_lods.append(lod95_boot)
            except Exception:
                # If estimation fails, skip this bootstrap iteration
                continue
        
        if not bootstrap_lods:
            raise ValueError("No successful bootstrap iterations")
        
        # Calculate final LoD95 and confidence interval
        lod95_estimate = np.median(bootstrap_lods)
        
        alpha_ci = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_lods, 100 * alpha_ci / 2)
        ci_upper = np.percentile(bootstrap_lods, 100 * (1 - alpha_ci / 2))
        
        # Generate final detection curve
        final_detection_curve = depth_results.groupby('allele_fraction').agg({
            'detected': ['mean', 'std', 'count']
        }).round(4)
        final_detection_curve.columns = ['detection_rate', 'detection_std', 'n_replicates']
        final_detection_curve = final_detection_curve.reset_index()
        
        # Add confidence intervals to curve
        final_detection_curve['ci_lower'] = np.maximum(
            0, 
            final_detection_curve['detection_rate'] - 1.96 * final_detection_curve['detection_std'] / np.sqrt(final_detection_curve['n_replicates'])
        )
        final_detection_curve['ci_upper'] = np.minimum(
            1,
            final_detection_curve['detection_rate'] + 1.96 * final_detection_curve['detection_std'] / np.sqrt(final_detection_curve['n_replicates'])
        )
        
        return LODResult(
            lod95=lod95_estimate,
            confidence_interval=(ci_lower, ci_upper),
            detection_curve=final_detection_curve,
            bootstrap_estimates=bootstrap_lods,
            n_bootstrap=len(bootstrap_lods),
            confidence_level=self.confidence_level,
            multiple_testing_correction=multiple_testing_method,
            corrected_alpha=corrected_alpha
        )
    
    def estimate_lob(
        self,
        negative_control_results: pd.DataFrame,
        depth: int
    ) -> LOBResult:
        """Estimate Limit of Blank from negative control data."""
        
        # Filter negative controls (AF = 0) for specific depth
        lob_data = negative_control_results[
            (negative_control_results['allele_fraction'] == 0.0) &
            (negative_control_results['umi_depth'] == depth)
        ]
        
        if lob_data.empty:
            raise ValueError(f"No negative control data found for depth {depth}")
        
        # Calculate false positive rate
        false_positive_rate = lob_data['detected'].mean()
        n_replicates = len(lob_data)
        
        # Use the upper confidence limit as LoB
        # This represents the false positive rate we expect 95% of the time
        z = stats.norm.ppf(1 - self.alpha / 2)
        se = np.sqrt(false_positive_rate * (1 - false_positive_rate) / n_replicates)
        
        ci_lower = max(0, false_positive_rate - z * se)
        ci_upper = min(1, false_positive_rate + z * se)
        
        # LoB is typically defined as the concentration where FP rate = alpha
        # For now, use the upper CI as a conservative estimate
        lob = ci_upper
        
        return LOBResult(
            lob=lob,
            false_positive_rate=false_positive_rate,
            confidence_interval=(ci_lower, ci_upper),
            n_replicates=n_replicates,
            alpha=self.alpha
        )
    
    def generate_detection_heatmap_data(
        self,
        simulation_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate data for detection probability heatmap."""
        
        heatmap_data = simulation_results.groupby(['allele_fraction', 'umi_depth']).agg({
            'detected': ['mean', 'std', 'count']
        }).round(4)
        
        heatmap_data.columns = ['detection_rate', 'detection_std', 'n_replicates']
        heatmap_data = heatmap_data.reset_index()
        
        # Pivot for heatmap format
        heatmap_pivot = heatmap_data.pivot(
            index='allele_fraction', 
            columns='umi_depth', 
            values='detection_rate'
        )
        
        return heatmap_pivot
    
    def validate_lod_monotonicity(
        self,
        lod_results: Dict[int, LODResult]
    ) -> Dict[str, any]:
        """Validate that LoD estimates follow expected monotonic behavior."""
        
        depths = sorted(lod_results.keys())
        lod_values = [lod_results[depth].lod95 for depth in depths]
        
        # Check if LoD decreases with increasing depth (monotonic)
        is_monotonic = all(
            lod_values[i] >= lod_values[i+1] 
            for i in range(len(lod_values)-1)
        )
        
        # Calculate correlation
        correlation = stats.pearsonr(depths, lod_values)[0]
        
        # Check for reasonable LoD values (not too extreme)
        reasonable_range = all(1e-6 <= lod <= 1.0 for lod in lod_values)
        
        validation_result = {
            'is_monotonic': is_monotonic,
            'correlation_with_depth': correlation,
            'reasonable_range': reasonable_range,
            'depths': depths,
            'lod_values': lod_values,
            'passed_validation': is_monotonic and reasonable_range and correlation < -0.5
        }
        
        return validation_result
    
    def generate_lod_summary_table(
        self,
        lod_results: Dict[int, LODResult],
        lob_results: Dict[int, LOBResult]
    ) -> pd.DataFrame:
        """Generate summary table of LoD and LoB results."""
        
        summary_data = []
        
        for depth in sorted(set(lod_results.keys()) | set(lob_results.keys())):
            row = {'depth': depth}
            
            if depth in lod_results:
                lod_result = lod_results[depth]
                row.update({
                    'lod95': lod_result.lod95,
                    'lod95_ci_lower': lod_result.confidence_interval[0],
                    'lod95_ci_upper': lod_result.confidence_interval[1],
                    'lod_n_bootstrap': lod_result.n_bootstrap
                })
            
            if depth in lob_results:
                lob_result = lob_results[depth]
                row.update({
                    'lob': lob_result.lob,
                    'false_positive_rate': lob_result.false_positive_rate,
                    'lob_ci_lower': lob_result.confidence_interval[0],
                    'lob_ci_upper': lob_result.confidence_interval[1],
                    'lob_n_replicates': lob_result.n_replicates
                })
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def calculate_analytical_sensitivity(
        self,
        lod_results: Dict[int, LODResult],
        clinical_threshold: float = 0.001  # 0.1% VAF
    ) -> Dict[str, any]:
        """Calculate analytical sensitivity metrics."""
        
        sensitivity_metrics = {}
        
        for depth, lod_result in lod_results.items():
            # Sensitivity at clinical threshold
            detection_curve = lod_result.detection_curve
            
            if clinical_threshold in detection_curve['allele_fraction'].values:
                sensitivity = detection_curve[
                    detection_curve['allele_fraction'] == clinical_threshold
                ]['detection_rate'].iloc[0]
            else:
                # Interpolate
                af_values = detection_curve['allele_fraction'].values
                det_rates = detection_curve['detection_rate'].values
                sensitivity = np.interp(clinical_threshold, af_values, det_rates)
            
            sensitivity_metrics[depth] = {
                'sensitivity_at_clinical_threshold': sensitivity,
                'clinical_threshold': clinical_threshold,
                'lod95': lod_result.lod95,
                'meets_clinical_requirement': lod_result.lod95 <= clinical_threshold
            }
        
        return sensitivity_metrics
"""
Simulation module for MRD spike-in experiments.

This module provides:
- Spike-in simulation across allele fractions and depths
- Contamination simulation
- Synthetic read generation for testing
- End-to-end pipeline simulation
"""

from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
from dataclasses import dataclass
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

from .io import SyntheticReadGenerator, TargetSite, reads_to_dataframe
from .umi import UMIProcessor
from .context import ContextAnalyzer
from .errors import ErrorModel
from .stats import StatisticalTester
from .filters import QualityFilter


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    allele_fractions: List[float]
    umi_depths: List[int]
    n_replicates: int
    n_bootstrap: int
    seed: int
    target_sites: Optional[List[TargetSite]] = None


@dataclass
class SimulationResult:
    """Container for simulation results."""
    config: SimulationConfig
    detection_matrix: pd.DataFrame
    statistical_results: pd.DataFrame
    filter_results: pd.DataFrame
    qc_metrics: Dict[str, any]
    runtime_metrics: Dict[str, float]


class Simulator:
    """Run MRD simulation experiments."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize simulator with configuration."""
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._create_default_config(**kwargs)
        
        # Initialize components
        self.read_generator = SyntheticReadGenerator(seed=self.config.seed)
        self.umi_processor = UMIProcessor(**kwargs)
        self.context_analyzer = ContextAnalyzer()
        self.error_model = ErrorModel(**kwargs)
        self.statistical_tester = StatisticalTester(**kwargs)
        self.quality_filter = QualityFilter(**kwargs)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> SimulationConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        sim_config = config_data.get('simulation', {})
        
        return SimulationConfig(
            allele_fractions=sim_config.get('allele_fractions', [0.01, 0.001]),
            umi_depths=sim_config.get('umi_depths', [5000, 10000]),
            n_replicates=sim_config.get('n_replicates', 100),
            n_bootstrap=sim_config.get('n_bootstrap', 100),
            seed=config_data.get('seed', 42)
        )
    
    def _create_default_config(self, **kwargs) -> SimulationConfig:
        """Create default simulation configuration."""
        return SimulationConfig(
            allele_fractions=[0.05, 0.01, 0.001, 0.0005, 0.0001],
            umi_depths=[5000, 10000, 20000, 50000],
            n_replicates=1000,
            n_bootstrap=1000,
            seed=42
        )
    
    def generate_target_sites(self, n_sites: int = 10) -> List[TargetSite]:
        """Generate target sites for simulation."""
        if self.config.target_sites:
            return self.config.target_sites
        
        return self.read_generator.generate_target_sites(n_sites)
    
    def simulate_single_condition(
        self,
        site: TargetSite,
        allele_fraction: float,
        umi_depth: int,
        replicate_id: int = 0,
        contamination_rate: float = 0.0
    ) -> Dict[str, any]:
        """Simulate a single experimental condition."""
        
        # Generate synthetic reads
        reads = self.read_generator.generate_reads_for_site(
            site=site,
            n_umi_families=umi_depth,
            allele_fraction=allele_fraction,
            contamination_rate=contamination_rate
        )
        
        # Convert to DataFrame for processing
        reads_df = reads_to_dataframe(reads)
        
        # Apply contamination if specified
        if contamination_rate > 0:
            reads_df = self.error_model.simulate_contamination_events(
                reads_df, contamination_rate
            )
        
        # Process UMI families
        families = self.umi_processor.process_reads(reads)
        consensus_df = self.umi_processor.get_consensus_counts(families)
        
        # Calculate metrics
        efficiency_metrics = self.umi_processor.calculate_efficiency_metrics(families)
        
        # Prepare data for statistical testing
        if not consensus_df.empty:
            # Group by site and allele
            site_summary = self._summarize_site_data(consensus_df, site)
            
            # Get expected error rate
            expected_rate = self.error_model.context_error_rates.get(
                site.context, {}).get(f"{site.ref}>{site.alt}", 1e-4
            )
            
            # Perform statistical test
            if site_summary['alt_count'] > 0:
                test_result = self.statistical_tester.test_variant_significance(
                    observed_alt=site_summary['alt_count'],
                    total_depth=site_summary['total_depth'],
                    expected_error_rate=expected_rate,
                    context=site.context
                )
                
                detected = test_result.pvalue < self.statistical_tester.alpha
            else:
                test_result = None
                detected = False
        else:
            site_summary = {'alt_count': 0, 'total_depth': 0}
            test_result = None
            detected = False
        
        result = {
            'site_key': site.key,
            'allele_fraction': allele_fraction,
            'umi_depth': umi_depth,
            'replicate_id': replicate_id,
            'contamination_rate': contamination_rate,
            'n_reads_generated': len(reads),
            'n_families_processed': len(families),
            'consensus_alt_count': site_summary['alt_count'],
            'consensus_total_depth': site_summary['total_depth'],
            'detected': detected,
            'pvalue': test_result.pvalue if test_result else 1.0,
            'effect_size': test_result.effect_size if test_result else 0.0,
            'efficiency_metrics': efficiency_metrics
        }
        
        return result
    
    def _summarize_site_data(self, consensus_df: pd.DataFrame, site: TargetSite) -> Dict[str, int]:
        """Summarize consensus data for a site."""
        site_data = consensus_df[
            (consensus_df['chrom'] == site.chrom) &
            (consensus_df['pos'] == site.pos) &
            (consensus_df['ref'] == site.ref)
        ]
        
        if site_data.empty:
            return {'alt_count': 0, 'total_depth': 0}
        
        alt_count = site_data[site_data['allele'] == site.alt]['consensus_count'].sum()
        total_depth = site_data['consensus_count'].sum()
        
        return {'alt_count': alt_count, 'total_depth': total_depth}
    
    def run_simulation_grid(
        self,
        target_sites: Optional[List[TargetSite]] = None,
        contamination_rate: float = 0.0
    ) -> SimulationResult:
        """Run complete simulation grid across all conditions."""
        
        if target_sites is None:
            target_sites = self.generate_target_sites(n_sites=5)
        
        self.logger.info(f"Starting simulation with {len(target_sites)} sites")
        self.logger.info(f"AFs: {self.config.allele_fractions}")
        self.logger.info(f"Depths: {self.config.umi_depths}")
        self.logger.info(f"Replicates: {self.config.n_replicates}")
        
        # Calculate total iterations for progress bar
        total_iterations = (
            len(target_sites) * 
            len(self.config.allele_fractions) * 
            len(self.config.umi_depths) * 
            self.config.n_replicates
        )
        
        results = []
        
        with tqdm(total=total_iterations, desc="Running simulation") as pbar:
            for site in target_sites:
                for af in self.config.allele_fractions:
                    for depth in self.config.umi_depths:
                        for rep in range(self.config.n_replicates):
                            result = self.simulate_single_condition(
                                site=site,
                                allele_fraction=af,
                                umi_depth=depth,
                                replicate_id=rep,
                                contamination_rate=contamination_rate
                            )
                            results.append(result)
                            pbar.update(1)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate detection matrix
        detection_matrix = self._calculate_detection_matrix(results_df)
        
        # Generate summary statistics
        statistical_results = self._generate_statistical_summary(results_df)
        
        # Apply quality filters
        filter_results = self._apply_quality_filters(results_df)
        
        # Calculate QC metrics
        qc_metrics = self._calculate_qc_metrics(results_df)
        
        # Runtime metrics
        runtime_metrics = self._calculate_runtime_metrics(results_df)
        
        return SimulationResult(
            config=self.config,
            detection_matrix=detection_matrix,
            statistical_results=statistical_results,
            filter_results=filter_results,
            qc_metrics=qc_metrics,
            runtime_metrics=runtime_metrics
        )
    
    def _calculate_detection_matrix(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate detection probability matrix."""
        detection_rates = results_df.groupby(['allele_fraction', 'umi_depth']).agg({
            'detected': ['mean', 'std', 'count']
        }).round(4)
        
        detection_rates.columns = ['detection_rate', 'detection_std', 'n_replicates']
        detection_rates = detection_rates.reset_index()
        
        return detection_rates
    
    def _generate_statistical_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical summary of results."""
        stats_summary = results_df.groupby(['allele_fraction', 'umi_depth']).agg({
            'pvalue': ['mean', 'median', 'std'],
            'effect_size': ['mean', 'median', 'std'],
            'consensus_alt_count': ['mean', 'std'],
            'consensus_total_depth': ['mean', 'std']
        }).round(6)
        
        # Flatten column names
        stats_summary.columns = ['_'.join(col).strip() for col in stats_summary.columns]
        stats_summary = stats_summary.reset_index()
        
        return stats_summary
    
    def _apply_quality_filters(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters and summarize results."""
        # For simulation, create mock filter data
        filter_results = results_df.copy()
        
        # Add mock filter columns
        filter_results['depth_filter'] = filter_results['consensus_total_depth'] >= 10
        filter_results['alt_count_filter'] = filter_results['consensus_alt_count'] >= 2
        filter_results['all_filters_passed'] = (
            filter_results['depth_filter'] & 
            filter_results['alt_count_filter']
        )
        
        # Summarize filter performance
        filter_summary = filter_results.groupby(['allele_fraction', 'umi_depth']).agg({
            'all_filters_passed': 'mean',
            'depth_filter': 'mean', 
            'alt_count_filter': 'mean'
        }).round(4)
        
        return filter_summary.reset_index()
    
    def _calculate_qc_metrics(self, results_df: pd.DataFrame) -> Dict[str, any]:
        """Calculate QC metrics from simulation results."""
        
        qc_metrics = {
            'total_simulations': len(results_df),
            'mean_families_per_depth': results_df.groupby('umi_depth')['n_families_processed'].mean().to_dict(),
            'mean_reads_per_family': (results_df['n_reads_generated'] / results_df['n_families_processed']).mean(),
            'overall_detection_rate': results_df['detected'].mean(),
            'detection_by_af': results_df.groupby('allele_fraction')['detected'].mean().to_dict(),
            'depth_efficiency': (results_df['consensus_total_depth'] / results_df['umi_depth']).mean()
        }
        
        return qc_metrics
    
    def _calculate_runtime_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate runtime performance metrics."""
        # Mock runtime metrics for simulation
        return {
            'total_runtime_seconds': 0.0,  # Would measure actual runtime
            'simulations_per_second': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def save_results(self, results: SimulationResult, output_dir: str) -> None:
        """Save simulation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detection matrix
        results.detection_matrix.to_csv(
            output_path / "detection_matrix.csv", index=False
        )
        
        # Save statistical results
        results.statistical_results.to_csv(
            output_path / "statistical_results.csv", index=False
        )
        
        # Save filter results
        results.filter_results.to_csv(
            output_path / "filter_results.csv", index=False
        )
        
        # Save QC metrics
        import json
        with open(output_path / "qc_metrics.json", 'w') as f:
            json.dump(results.qc_metrics, f, indent=2)
        
        # Save runtime metrics
        with open(output_path / "runtime_metrics.json", 'w') as f:
            json.dump(results.runtime_metrics, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")


def run_simulation_from_config(config_path: str, output_dir: str) -> SimulationResult:
    """Run simulation from configuration file."""
    simulator = Simulator(config_path=config_path)
    results = simulator.run_simulation_grid()
    simulator.save_results(results, output_dir)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simulate.py <config_path> <output_dir>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    run_simulation_from_config(config_path, output_dir)
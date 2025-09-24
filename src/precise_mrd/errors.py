"""
Error modeling and contamination simulation module.

This module provides:
- Context-specific error rate estimation
- Contamination modeling and simulation
- Background error characterization
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from .context import ContextAnalyzer


class ErrorModel:
    """Model sequencing errors and contamination for MRD analysis."""
    
    def __init__(self, base_error_rate: float = 1e-4, contamination_rate: float = 1e-3):
        """Initialize error model with baseline rates."""
        self.base_error_rate = base_error_rate
        self.contamination_rate = contamination_rate
        self.context_analyzer = ContextAnalyzer()
        self.context_error_rates = {}
        self.contamination_matrix = {}
        
    def estimate_background_errors(
        self,
        negative_control_data: pd.DataFrame,
        min_depth: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Estimate background error rates from negative controls."""
        
        # Filter sites with sufficient depth
        high_depth = negative_control_data[
            negative_control_data['consensus_count'] >= min_depth
        ]
        
        # Group by context and mutation type
        context_errors = defaultdict(lambda: defaultdict(list))
        
        for _, row in high_depth.iterrows():
            context = row.get('context', 'NNN')
            ref = row['ref']
            alt = row['allele']
            
            if alt != ref:  # Only count actual errors
                norm_context, norm_ref, norm_alt = self.context_analyzer.normalize_context(
                    context, ref, alt
                )
                mutation_key = f"{norm_ref}>{norm_alt}"
                
                # Calculate error rate for this observation
                error_rate = 1.0 / row['consensus_count']  # Assuming single error observation
                context_errors[norm_context][mutation_key].append(error_rate)
        
        # Calculate mean error rates per context
        context_rates = {}
        for context, mutations in context_errors.items():
            context_rates[context] = {}
            for mutation, rates in mutations.items():
                context_rates[context][mutation] = np.mean(rates)
        
        self.context_error_rates = context_rates
        return context_rates
    
    def model_contamination_matrix(
        self,
        sample_pairs: List[Tuple[str, str]],
        observed_contamination: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[Tuple[str, str], float]:
        """Model cross-sample contamination rates."""
        
        contamination_matrix = {}
        
        for sample1, sample2 in sample_pairs:
            if observed_contamination and (sample1, sample2) in observed_contamination:
                rate = observed_contamination[(sample1, sample2)]
            else:
                # Use baseline contamination rate with some variation
                rate = np.random.lognormal(
                    np.log(self.contamination_rate), 
                    0.5  # log-scale variation
                )
            
            contamination_matrix[(sample1, sample2)] = rate
        
        self.contamination_matrix = contamination_matrix
        return contamination_matrix
    
    def simulate_contamination_events(
        self,
        reads_data: pd.DataFrame,
        contamination_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """Simulate contamination events in read data."""
        
        if contamination_rate is None:
            contamination_rate = self.contamination_rate
        
        contaminated_data = reads_data.copy()
        n_reads = len(reads_data)
        n_contaminated = int(n_reads * contamination_rate)
        
        if n_contaminated == 0:
            return contaminated_data
        
        # Randomly select reads to contaminate
        contaminate_indices = np.random.choice(
            reads_data.index, 
            size=n_contaminated, 
            replace=False
        )
        
        for idx in contaminate_indices:
            # Flip the allele (simple contamination model)
            current_allele = contaminated_data.loc[idx, 'allele']
            ref_allele = contaminated_data.loc[idx, 'ref']
            
            if current_allele == ref_allele:
                # Introduce a random alternative allele
                possible_alts = [b for b in ['A', 'C', 'G', 'T'] if b != ref_allele]
                new_allele = np.random.choice(possible_alts)
            else:
                # Revert to reference (loss of signal)
                new_allele = ref_allele
            
            contaminated_data.loc[idx, 'allele'] = new_allele
            contaminated_data.loc[idx, 'is_contaminated'] = True
        
        # Mark non-contaminated reads
        non_contaminated = ~contaminated_data.index.isin(contaminate_indices)
        contaminated_data.loc[non_contaminated, 'is_contaminated'] = False
        
        return contaminated_data
    
    def estimate_contamination_level(
        self,
        sample_data: pd.DataFrame,
        reference_samples: List[pd.DataFrame]
    ) -> float:
        """Estimate contamination level in a sample."""
        
        # Simple contamination estimation based on unexpected allele frequencies
        total_reads = len(sample_data)
        
        if total_reads == 0:
            return 0.0
        
        # Count reads that match patterns from reference samples
        contamination_evidence = 0
        
        for ref_sample in reference_samples:
            # Get variant signatures from reference
            ref_variants = set(
                ref_sample.apply(
                    lambda x: f"{x['chrom']}:{x['pos']}:{x['ref']}>{x['allele']}", 
                    axis=1
                )
            )
            
            # Count matches in target sample
            sample_variants = set(
                sample_data.apply(
                    lambda x: f"{x['chrom']}:{x['pos']}:{x['ref']}>{x['allele']}", 
                    axis=1
                )
            )
            
            contamination_evidence += len(ref_variants.intersection(sample_variants))
        
        estimated_rate = contamination_evidence / total_reads
        return min(estimated_rate, 0.1)  # Cap at 10%
    
    def calculate_error_confidence_intervals(
        self,
        context_rates: Dict[str, Dict[str, float]],
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for error rates."""
        
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for context, mutations in context_rates.items():
            confidence_intervals[context] = {}
            
            for mutation, rate in mutations.items():
                # Use Wilson score interval for binomial proportions
                # Simplified calculation assuming some observation count
                n_obs = 100  # placeholder - would use actual observation count
                n_success = int(rate * n_obs)
                
                if n_obs > 0:
                    lower, upper = self._wilson_confidence_interval(
                        n_success, n_obs, alpha
                    )
                else:
                    lower, upper = 0.0, 1.0
                
                confidence_intervals[context][mutation] = (lower, upper)
        
        return confidence_intervals
    
    @staticmethod
    def _wilson_confidence_interval(
        successes: int, 
        trials: int, 
        alpha: float
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if trials == 0:
            return 0.0, 1.0
        
        z = stats.norm.ppf(1 - alpha/2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        lower = max(0, centre - margin)
        upper = min(1, centre + margin)
        
        return lower, upper
    
    def generate_error_profile_report(self) -> Dict[str, any]:
        """Generate comprehensive error profile report."""
        
        report = {
            'base_error_rate': self.base_error_rate,
            'contamination_rate': self.contamination_rate,
            'n_contexts': len(self.context_error_rates),
            'context_summary': {}
        }
        
        # Summarize per-context error rates
        for context, mutations in self.context_error_rates.items():
            total_rate = sum(mutations.values())
            max_rate = max(mutations.values()) if mutations else 0
            n_mutations = len(mutations)
            
            report['context_summary'][context] = {
                'total_error_rate': total_rate,
                'max_single_mutation_rate': max_rate,
                'n_mutation_types': n_mutations,
                'mutations': mutations
            }
        
        # Calculate overall statistics
        all_rates = [
            rate for mutations in self.context_error_rates.values() 
            for rate in mutations.values()
        ]
        
        if all_rates:
            report['overall_stats'] = {
                'mean_error_rate': np.mean(all_rates),
                'median_error_rate': np.median(all_rates),
                'std_error_rate': np.std(all_rates),
                'min_error_rate': np.min(all_rates),
                'max_error_rate': np.max(all_rates)
            }
        
        return report
    
    def export_error_model(self, filepath: str) -> None:
        """Export error model to file for reuse.
        
        Args:
            filepath: Path to output JSON file
            
        Raises:
            IOError: If file cannot be written
            ValueError: If filepath is invalid
        """
        import json
        from pathlib import Path
        
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        
        try:
            # Ensure parent directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'base_error_rate': float(self.base_error_rate),
                'contamination_rate': float(self.contamination_rate),
                'context_error_rates': self.context_error_rates,
                'contamination_matrix': {
                    f"{k[0]}->{k[1]}": float(v) for k, v in self.contamination_matrix.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
                
        except (IOError, OSError) as e:
            raise IOError(f"Failed to write error model to {filepath}: {e}") from e
    
    def load_error_model(self, filepath: str) -> None:
        """Load error model from file.
        
        Args:
            filepath: Path to input JSON file
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
            FileNotFoundError: If file does not exist
        """
        import json
        from pathlib import Path
        
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Error model file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in error model file {filepath}: {e}") from e
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read error model from {filepath}: {e}") from e
        
        # Validate required fields
        required_fields = ['base_error_rate', 'contamination_rate']
        for field in required_fields:
            if field not in model_data:
                raise ValueError(f"Missing required field '{field}' in error model file")
        
        try:
            self.base_error_rate = float(model_data['base_error_rate'])
            self.contamination_rate = float(model_data['contamination_rate'])
            self.context_error_rates = model_data.get('context_error_rates', {})
            
            # Reconstruct contamination matrix
            self.contamination_matrix = {}
            for key, value in model_data.get('contamination_matrix', {}).items():
                if '->' not in key:
                    continue  # Skip invalid keys
                try:
                    sample1, sample2 = key.split('->', 1)  # Split on first occurrence only
                    self.contamination_matrix[(sample1, sample2)] = float(value)
                except (ValueError, TypeError):
                    continue  # Skip invalid entries
                    
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format in error model file: {e}") from e
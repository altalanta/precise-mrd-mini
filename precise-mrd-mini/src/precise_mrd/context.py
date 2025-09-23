"""
Trinucleotide context analysis module.

This module provides:
- Context extraction from genomic coordinates
- Context-specific error rate modeling
- Background error estimation
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from .io import TargetSite


class ContextAnalyzer:
    """Analyze trinucleotide contexts and mutation patterns."""
    
    def __init__(self):
        """Initialize context analyzer."""
        self.context_error_rates = {}
        self.context_counts = defaultdict(int)
    
    @staticmethod
    def get_complement(base: str) -> str:
        """Get complement of a DNA base."""
        complements = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return complements.get(base, 'N')
    
    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get reverse complement of a DNA sequence."""
        return ''.join(ContextAnalyzer.get_complement(base) for base in reversed(sequence))
    
    @staticmethod
    def normalize_context(context: str, ref: str, alt: str) -> Tuple[str, str, str]:
        """Normalize trinucleotide context to canonical form (C or T as central base)."""
        if len(context) != 3:
            raise ValueError("Context must be exactly 3 bases")
        
        central_base = context[1]
        
        # If central base is A or G, use reverse complement
        if central_base in ['A', 'G']:
            norm_context = ContextAnalyzer.reverse_complement(context)
            norm_ref = ContextAnalyzer.get_complement(ref)
            norm_alt = ContextAnalyzer.get_complement(alt)
        else:
            norm_context = context
            norm_ref = ref  
            norm_alt = alt
        
        return norm_context, norm_ref, norm_alt
    
    @staticmethod
    def get_mutation_type(ref: str, alt: str) -> str:
        """Classify mutation type (transition vs transversion)."""
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        if (ref, alt) in transitions:
            return 'transition'
        else:
            return 'transversion'
    
    def extract_context_from_sequence(self, sequence: str, position: int) -> Optional[str]:
        """Extract trinucleotide context from sequence at given position."""
        if position < 1 or position >= len(sequence) - 1:
            return None
        
        return sequence[position-1:position+2].upper()
    
    def estimate_context_error_rates(
        self, 
        consensus_data: pd.DataFrame,
        negative_control_sites: Optional[List[TargetSite]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Estimate error rates per trinucleotide context."""
        
        # Group data by context and mutation type
        context_mutations = defaultdict(lambda: defaultdict(int))
        context_totals = defaultdict(int)
        
        for _, row in consensus_data.iterrows():
            # Extract context info (would need reference sequence in practice)
            context = getattr(row, 'context', 'NNN')  # placeholder
            ref = row['ref']
            alt = row['allele']
            count = row['consensus_count']
            
            # Skip if same as reference (no mutation)
            if alt == ref:
                context_totals[context] += count
                continue
            
            # Normalize context
            norm_context, norm_ref, norm_alt = self.normalize_context(context, ref, alt)
            mutation_key = f"{norm_ref}>{norm_alt}"
            
            context_mutations[norm_context][mutation_key] += count
            context_totals[norm_context] += count
        
        # Calculate error rates
        error_rates = {}
        for context, mutations in context_mutations.items():
            total = context_totals[context]
            if total == 0:
                continue
                
            context_rates = {}
            for mutation, count in mutations.items():
                context_rates[mutation] = count / total
            
            error_rates[context] = context_rates
        
        self.context_error_rates = error_rates
        return error_rates
    
    def get_expected_error_rate(self, context: str, ref: str, alt: str) -> float:
        """Get expected error rate for a specific context and mutation."""
        norm_context, norm_ref, norm_alt = self.normalize_context(context, ref, alt)
        mutation_key = f"{norm_ref}>{norm_alt}"
        
        if norm_context in self.context_error_rates:
            return self.context_error_rates[norm_context].get(mutation_key, 1e-6)
        
        # Default error rate if context not found
        return 1e-4
    
    def calculate_context_enrichment(
        self, 
        observed_mutations: pd.DataFrame,
        expected_rates: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """Calculate enrichment of mutations per context vs expected."""
        
        if expected_rates is None:
            expected_rates = self.context_error_rates
        
        enrichment_data = []
        
        # Group by context
        context_groups = observed_mutations.groupby('context')
        
        for context, group in context_groups:
            norm_context = self.normalize_context(context, 'C', 'T')[0]  # placeholder normalization
            
            observed_count = len(group)
            
            # Get expected rate for this context
            if norm_context in expected_rates:
                expected_rate = sum(expected_rates[norm_context].values())
            else:
                expected_rate = 1e-4
            
            # Calculate enrichment (would need total depth for proper calculation)
            total_depth = group['consensus_count'].sum() if 'consensus_count' in group.columns else 1000
            expected_count = expected_rate * total_depth
            
            enrichment = observed_count / expected_count if expected_count > 0 else 0
            
            enrichment_data.append({
                'context': context,
                'observed_mutations': observed_count,
                'expected_mutations': expected_count,
                'enrichment': enrichment,
                'total_depth': total_depth
            })
        
        return pd.DataFrame(enrichment_data)
    
    def get_context_specific_thresholds(
        self, 
        contexts: List[str],
        alpha: float = 0.05,
        depth_range: Tuple[int, int] = (1000, 50000)
    ) -> Dict[str, float]:
        """Calculate context-specific significance thresholds."""
        
        thresholds = {}
        
        for context in contexts:
            # Get baseline error rate for this context
            error_rate = sum(self.context_error_rates.get(context, {'default': 1e-4}).values())
            
            # Calculate threshold based on Poisson distribution
            # (simplified - would use more sophisticated approach in practice)
            mean_depth = np.mean(depth_range)
            expected_errors = error_rate * mean_depth
            
            # Use 3-sigma threshold
            threshold = expected_errors + 3 * np.sqrt(expected_errors)
            thresholds[context] = threshold
        
        return thresholds
    
    def classify_end_repair_artifacts(
        self, 
        mutations: pd.DataFrame,
        end_repair_distance: int = 10
    ) -> pd.DataFrame:
        """Classify potential end-repair artifacts."""
        
        # Common end-repair artifact patterns
        end_repair_patterns = ['G>T', 'C>A']
        
        mutations = mutations.copy()
        mutations['is_end_repair_artifact'] = False
        
        for _, row in mutations.iterrows():
            mutation_type = f"{row['ref']}>{row['allele']}"
            read_position = getattr(row, 'read_position', 75)  # placeholder
            
            # Check if near read end and matches end-repair pattern
            if (mutation_type in end_repair_patterns and 
                (read_position <= end_repair_distance or read_position >= 150 - end_repair_distance)):
                mutations.loc[mutations.index == row.name, 'is_end_repair_artifact'] = True
        
        return mutations
    
    def generate_context_summary(self) -> pd.DataFrame:
        """Generate summary statistics for all contexts."""
        
        summary_data = []
        
        for context, mutations in self.context_error_rates.items():
            total_rate = sum(mutations.values())
            n_mutation_types = len(mutations)
            
            # Calculate transition/transversion ratio
            transitions = 0
            transversions = 0
            
            for mutation in mutations.keys():
                ref, alt = mutation.split('>')
                if self.get_mutation_type(ref, alt) == 'transition':
                    transitions += mutations[mutation]
                else:
                    transversions += mutations[mutation]
            
            ti_tv_ratio = transitions / transversions if transversions > 0 else float('inf')
            
            summary_data.append({
                'context': context,
                'total_error_rate': total_rate,
                'n_mutation_types': n_mutation_types,
                'transition_rate': transitions,
                'transversion_rate': transversions,
                'ti_tv_ratio': ti_tv_ratio
            })
        
        return pd.DataFrame(summary_data)
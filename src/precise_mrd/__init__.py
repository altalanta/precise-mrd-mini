"""
Precise MRD: ctDNA/UMI toy MRD pipeline with UMI-aware error modeling and LoB/LoD estimation.

This package provides tools for:
- UMI consensus calling with family-size thresholds
- Trinucleotide context-aware error modeling
- Statistical hypothesis testing with FDR control
- LoB/LoD estimation via bootstrap
- Contamination simulation and detection
- Clinical QC metrics and guardrails
"""

__version__ = "0.1.1"
__author__ = "Precise MRD Team"

# Import main classes for convenience
from .io import SyntheticReadGenerator, TargetSite
from .umi import UMIFamily, UMIProcessor
from .context import ContextAnalyzer
from .errors import ErrorModel
from .stats import StatisticalTester, TestResult
from .filters import QualityFilter, FilterResult
from .simulate import Simulator
from .lod import LODEstimator
from .qc import QCAnalyzer
from .exceptions import PreciseMRDError, ValidationError, ProcessingError
from .logging_config import setup_logging, get_logger, time_it

__all__ = [
    "SyntheticReadGenerator",
    "TargetSite", 
    "UMIFamily",
    "UMIProcessor",
    "ContextAnalyzer",
    "ErrorModel",
    "StatisticalTester",
    "TestResult",
    "QualityFilter",
    "FilterResult",
    "Simulator",
    "LODEstimator",
    "QCAnalyzer",
    "PreciseMRDError",
    "ValidationError",
    "ProcessingError",
    "setup_logging",
    "get_logger",
    "time_it",
]
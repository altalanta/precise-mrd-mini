"""Precise MRD: ctDNA/UMI MRD simulator + caller with deterministic error modeling."""

from __future__ import annotations

__version__ = "0.1.0"

# Core simulation and calling functionality
from .simulate import simulate_reads
from .collapse import collapse_umis
from .call import call_mrd
from .error_model import fit_error_model
from .fastq import process_fastq_to_dataframe, detect_umi_format

# Configuration and I/O
from .config import PipelineConfig, load_config, dump_config
from .utils import PipelineIO

# Determinism and reproducibility
from .determinism_utils import set_global_seed, env_fingerprint

# Reporting
from .reporting import render_report, render_plots
from .metrics import roc_auc_score, average_precision

__all__ = [
    "__version__",
    # Core pipeline
    "simulate_reads",
    "collapse_umis",
    "call_mrd",
    "fit_error_model",
    "process_fastq_to_dataframe",
    "detect_umi_format",
    # Configuration
    "PipelineConfig",
    "load_config",
    "dump_config",
    "PipelineIO",
    # Determinism
    "set_global_seed",
    "env_fingerprint",
    # Reporting
    "render_report",
    "render_plots",
    "roc_auc_score",
    "average_precision",
]
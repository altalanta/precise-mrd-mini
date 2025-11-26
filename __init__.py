"""Precise MRD: ctDNA/UMI MRD simulator + caller with deterministic error modeling."""

from __future__ import annotations

__version__ = "0.1.0"

# Core simulation and calling functionality
from .cache import PipelineCache
from .call import predict_from_model, train_model
from .collapse import collapse_umis

# Configuration and I/O
from .config import PipelineConfig, dump_config, load_config

# Determinism and reproducibility
from .determinism_utils import env_fingerprint, set_global_seed
from .error_model import fit_error_model
from .fastq import detect_umi_format, process_fastq_to_dataframe
from .metrics import average_precision, roc_auc_score
from .performance import get_performance_report, reset_performance_monitor

# Reporting
from .reporting import render_plots, render_report
from .simulate import simulate_reads
from .utils import PipelineIO
from .validation import assert_hashes_stable, validate_artifacts

__all__ = [
    "__version__",
    # Core pipeline
    "simulate_reads",
    "collapse_umis",
    "train_model",
    "predict_from_model",
    "fit_error_model",
    "process_fastq_to_dataframe",
    "detect_umi_format",
    # Configuration
    "PipelineConfig",
    "load_config",
    "dump_config",
    "PipelineIO",
    "PipelineCache",
    # Determinism
    "set_global_seed",
    "env_fingerprint",
    "validate_artifacts",
    "assert_hashes_stable",
    # Reporting
    "render_report",
    "render_plots",
    "roc_auc_score",
    "average_precision",
    "get_performance_report",
    "reset_performance_monitor",
]

"""Pandera schemas for data validation in the Precise MRD pipeline."""

import pandera as pa
from pandera.typing import DataFrame, Series

class SimulatedReadsSchema(pa.DataFrameModel):
    """Schema for the output of the simulate_reads function."""
    sample_id: Series[int] = pa.Field(ge=0, description="Unique identifier for each simulated sample.")
    allele_fraction: Series[float] = pa.Field(ge=0, le=1, description="Target allele fraction for the simulation.")
    target_depth: Series[int] = pa.Field(ge=0, description="Target UMI depth for the simulation.")
    replicate: Series[int] = pa.Field(ge=0, description="Replicate number for a given condition.")
    n_families: Series[int] = pa.Field(ge=0, description="Number of UMI families simulated.")
    n_true_variants: Series[int] = pa.Field(ge=0, description="Number of true variant families simulated.")
    n_false_positives: Series[int] = pa.Field(ge=0, description="Number of false positive families simulated.")
    background_rate: Series[float] = pa.Field(ge=0, le=1, description="Background error rate used.")
    mean_family_size: Series[float] = pa.Field(ge=0, description="Mean family size for the sample.")
    mean_quality: Series[float] = pa.Field(ge=0, description="Mean quality score for the sample.")
    config_hash: Series[str] = pa.Field(description="Hash of the configuration used.")

class CollapsedUmisSchema(pa.DataFrameModel):
    """Schema for the output of the collapse_umis function."""
    sample_id: Series[int] = pa.Field(ge=0)
    family_id: Series[int] = pa.Field(ge=0)
    family_size: Series[int] = pa.Field(gt=0)
    quality_score: Series[float] = pa.Field(ge=0)
    consensus_agreement: Series[float] = pa.Field(ge=0, le=1)
    passes_quality: Series[bool]
    passes_consensus: Series[bool]
    is_variant: Series[bool]
    allele_fraction: Series[float] = pa.Field(ge=0, le=1)

class ErrorModelSchema(pa.DataFrameModel):
    """Schema for the output of the fit_error_model function."""
    trinucleotide_context: Series[str]
    error_rate: Series[float] = pa.Field(ge=0, le=1)
    ci_lower: Series[float] = pa.Field(ge=0, le=1)
    ci_upper: Series[float] = pa.Field(ge=0, le=1)
    n_observations: Series[int] = pa.Field(ge=0)
    config_hash: Series[str]

class StatisticalCallsSchema(pa.DataFrameModel):
    """Schema for MRD calls using statistical methods."""
    sample_id: Series[int] = pa.Field(ge=0)
    n_variants: Series[int] = pa.Field(ge=0)
    n_total: Series[int] = pa.Field(ge=0)
    p_value: Series[float] = pa.Field(ge=0, le=1)
    p_adjusted: Series[float] = pa.Field(ge=0, le=1)
    significant: Series[bool]

class MLCallsSchema(pa.DataFrameModel):
    """Schema for MRD calls using machine learning models."""
    sample_id: Series[int] = pa.Field(ge=0)
    family_id: Series[int] = pa.Field(ge=0)
    is_variant: Series[int] = pa.Field(isin=[0, 1])
    ml_probability: Series[float] = pa.Field(ge=0, le=1)
    ml_threshold: Series[float] = pa.Field(ge=0, le=1)
    calling_method: Series[str]

class DLCallsSchema(pa.DataFrameModel):
    """Schema for MRD calls using deep learning models."""
    sample_id: Series[int] = pa.Field(ge=0)
    family_id: Series[int] = pa.Field(ge=0)
    is_variant: Series[int] = pa.Field(isin=[0, 1])
    dl_probability: Series[float] = pa.Field(ge=0, le=1)
    dl_threshold: Series[float] = pa.Field(ge=0, le=1)
    calling_method: Series[str]


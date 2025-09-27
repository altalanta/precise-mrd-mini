"""Simulation stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .rng import RandomState
from .schemas import SIMULATED_READS_SCHEMA


@dataclass(slots=True)
class SimulationOutput:
    reads: pd.DataFrame


def _simulate_family_sizes(rng: RandomState, size: int, minimum: int, mean: int) -> np.ndarray:
    lam = max(mean - minimum, 1)
    draws = rng.generator.poisson(lam=lam, size=size)
    return draws + minimum


def simulate_reads(config: PipelineConfig, rng: RandomState) -> SimulationOutput:
    """Simulate per-read data for all variants and samples."""

    sim_cfg = config.simulation
    records: list[dict[str, object]] = []
    variant_ids = [f"VAR{i+1:02d}" for i in range(len(sim_cfg.allele_fractions))]

    case_counts = sim_cfg.replicates
    control_counts = sim_cfg.controls

    umi_ids = [f"UMI{i:03d}" for i in range(sim_cfg.umi_per_variant)]

    for depth in sim_cfg.umi_depths:
        family_sizes = _simulate_family_sizes(
            rng,
            size=sim_cfg.umi_per_variant,
            minimum=sim_cfg.umi_family_min,
            mean=sim_cfg.umi_family_mean,
        )

        for variant_id, truth_af in zip(variant_ids, sim_cfg.allele_fractions):
            for idx in range(case_counts):
                sample_id = f"case_{variant_id}_{idx:02d}"
                _append_sample(
                    records,
                    rng=rng,
                    sample_id=sample_id,
                    sample_type="case",
                    variant_id=variant_id,
                    allele_fraction=truth_af,
                    depth=depth,
                    umi_ids=umi_ids,
                    family_sizes=family_sizes,
                    truth_af=truth_af,
                )
        for idx in range(control_counts):
            sample_id = f"control_{idx:02d}"
            for variant_id in variant_ids:
                _append_sample(
                    records,
                    rng=rng,
                    sample_id=sample_id,
                    sample_type="control",
                    variant_id=variant_id,
                    allele_fraction=0.0,
                    depth=depth,
                    umi_ids=umi_ids,
                    family_sizes=family_sizes,
                    truth_af=0.0,
                )

    reads = pd.DataFrame.from_records(records)
    SIMULATED_READS_SCHEMA.validate(reads)
    return SimulationOutput(reads=reads)


def _append_sample(
    records: list[dict[str, object]],
    rng: RandomState,
    sample_id: str,
    sample_type: str,
    variant_id: str,
    allele_fraction: float,
    depth: int,
    umi_ids: Iterable[str],
    family_sizes: np.ndarray,
    truth_af: float,
) -> None:
    for umi_id, family_size in zip(umi_ids, family_sizes):
        alt_reads = int(rng.generator.binomial(family_size, truth_af))
        ref_reads = int(family_size - alt_reads)
        alleles = ["ALT"] * alt_reads + ["REF"] * ref_reads
        for read_index, allele in enumerate(alleles):
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "variant_id": variant_id,
                    "allele_fraction": allele_fraction,
                    "depth": depth,
                    "umi_id": umi_id,
                    "read_index": read_index,
                    "allele": allele,
                }
            )


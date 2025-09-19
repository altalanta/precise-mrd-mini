"""Simulate synthetic ctDNA reads with UMIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer
import yaml
from pydantic import BaseModel, Field, field_validator

NUCLEOTIDES = np.array(list("ACGT"))
app = typer.Typer(add_completion=False)


class SimulationConfig(BaseModel):
    """Configuration for the synthetic MRD simulation."""

    patient_id: str = Field(..., description="Patient identifier")
    n_variants: int = Field(..., gt=0)
    depth_mean: int = Field(..., gt=0)
    umi_family_geom_p: float = Field(..., gt=0, lt=1)
    base_error_rate: float = Field(..., ge=0, lt=0.1)
    vaf_true: List[float]
    panel_error_alpha_beta: Tuple[float, float]

    @field_validator("vaf_true")
    def _check_vafs(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("vaf_true must contain at least one entry")
        if any(v < 0 or v > 1 for v in value):
            raise ValueError("vaf_true entries should be proportions (0-1)")
        return value


def _choose_alt(ref: str, rng: np.random.Generator) -> str:
    options = [nt for nt in "ACGT" if nt != ref]
    return rng.choice(options)


def _introduce_errors(sequence: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(sequence.shape[0]) < epsilon
    if mask.any():
        replacements = rng.choice(NUCLEOTIDES, size=mask.sum())
        sequence[mask] = replacements
    return sequence


def _sample_family_sizes(n_reads: int, geom_p: float, rng: np.random.Generator) -> List[int]:
    sizes: List[int] = []
    remaining = n_reads
    while remaining > 0:
        draw = int(rng.geometric(geom_p))
        draw = max(1, min(draw, 6))  # keep families small for tiny datasets
        draw = min(draw, remaining)
        sizes.append(draw)
        remaining -= draw
    return sizes


def _simulate_variant_reads(
    patient_id: str,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    depth: int,
    vaf: float,
    epsilon: float,
    geom_p: float,
    rng: np.random.Generator,
) -> List[dict[str, object]]:
    reads: List[dict[str, object]] = []
    if depth <= 0:
        return reads

    if vaf <= 0:
        alt_fraction = 0.0
    else:
        conc = max(5.0, 80 * vaf)
        alt_fraction = float(rng.beta(vaf * conc + 1, (1 - vaf) * conc + 1))
    alt_fraction = min(max(alt_fraction, 0.0), 1.0)
    alt_reads = int(rng.binomial(depth, alt_fraction))
    alt_flags = np.array([True] * alt_reads + [False] * (depth - alt_reads))
    rng.shuffle(alt_flags)

    family_sizes = _sample_family_sizes(depth, geom_p, rng)
    idx = 0
    umi_counter = 0
    for family in family_sizes:
        family_flags = alt_flags[idx : idx + family]
        idx += family
        umi = "".join(rng.choice(list("ACGT"), size=8))
        # introduce occasional single-base UMI mutations to exercise clustering
        if rng.random() < 0.15:
            pos_mut = rng.integers(0, len(umi))
            umi = umi[:pos_mut] + rng.choice(list("ACGT")) + umi[pos_mut + 1 :]
        start = int(pos + rng.integers(-3, 4))
        for flag in family_flags:
            template = np.array(list(ref * 5 + alt + ref * 4))
            base_index = 5
            if flag:
                template[base_index] = alt
            sequence = _introduce_errors(template.copy(), epsilon, rng)
            reads.append(
                {
                    "patient_id": patient_id,
                    "chrom": chrom,
                    "pos": pos,
                    "umi": umi,
                    "start": start,
                    "sequence": "".join(sequence.tolist()),
                    "true_alt": bool(flag),
                    "family_id": f"{chrom}:{pos}:{umi_counter}",
                }
            )
        umi_counter += 1
    return reads


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Argument(help="YAML configuration describing the simulation")
    ] = Path("data/synthetic_config.yaml"),
    output_reads: Annotated[
        Path, typer.Option(help="Output parquet of simulated reads")
    ] = Path("simulated_reads/reads.parquet"),
    variants_csv: Annotated[
        Path, typer.Option(help="Ground truth variant table")
    ] = Path("data/ground_truth/variants.csv"),
    seed: Annotated[int, typer.Option(help="Random seed")] = 1234,
) -> None:
    """Simulate ctDNA reads with UMIs and sequencing errors."""
    config_data = yaml.safe_load(config_path.read_text())
    config = SimulationConfig(**config_data)
    rng = np.random.default_rng(seed)

    variants: List[dict[str, object]] = []
    read_records: List[dict[str, object]] = []

    for idx in range(config.n_variants):
        chrom = f"chr{1 + (idx % 2)}"
        pos = 100_000 + idx * 37
        ref = "C"
        alt = _choose_alt(ref, rng)
        depth = int(max(10, rng.poisson(config.depth_mean)))
        vaf = config.vaf_true[idx % len(config.vaf_true)]
        reads = _simulate_variant_reads(
            patient_id=config.patient_id,
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            depth=depth,
            vaf=vaf,
            epsilon=config.base_error_rate,
            geom_p=config.umi_family_geom_p,
            rng=rng,
        )
        read_records.extend(reads)
        variants.append(
            {
                "patient_id": config.patient_id,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "weight": 1.0,
                "expected_depth": depth,
                "vaf": vaf,
            }
        )

    variants_df = pd.DataFrame(variants)
    variants_csv.parent.mkdir(parents=True, exist_ok=True)
    variants_df.to_csv(variants_csv, index=False)

    reads_df = pd.DataFrame(read_records)
    output_reads.parent.mkdir(parents=True, exist_ok=True)
    reads_df.to_parquet(output_reads, index=False)

    metadata = {
        "config": config.model_dump(),
        "seed": seed,
        "n_reads": len(reads_df),
    }
    (output_reads.parent / "metadata.json").write_text(json.dumps(metadata, indent=2))
    typer.echo(
        f"Simulated {len(reads_df)} reads across {config.n_variants} variants for {config.patient_id}"
    )


def main() -> None:  # pragma: no cover - Typer entrpoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

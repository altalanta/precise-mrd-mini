"""Minimal MRD calling logic using beta-binomial error model."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Tuple

import numpy as np
import pandas as pd
import typer
from scipy import stats

from .error_model import p_alt

app = typer.Typer(add_completion=False)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    prev = 1.0
    for idx, rank in enumerate(order, start=1):
        bh_value = min(prev, p_values[rank] * n / idx)
        prev = bh_value
        ranked[rank] = bh_value
    return ranked


@app.command()
def run(
    collapsed_path: Annotated[Path, typer.Argument()] = Path("tmp/collapsed.parquet"),
    error_model_path: Annotated[Path, typer.Argument()] = Path("tmp/error_model.parquet"),
    output_path: Annotated[Path, typer.Option()] = Path("tmp/mrd_calls.csv"),
    summary_path: Annotated[Path, typer.Option()] = Path("tmp/mrd_summary.json"),
    seed: Annotated[int, typer.Option()] = 1234,
) -> None:
    """Compute MRD calls and summary statistics."""
    _ = seed
    collapsed = pd.read_parquet(collapsed_path)
    params = pd.read_parquet(error_model_path)
    merged = collapsed.merge(params, on=["chrom", "pos"], how="left")
    merged["alpha"] = merged["alpha"].fillna(params["alpha"].mean())
    merged["beta"] = merged["beta"].fillna(params["beta"].mean())

    p_vals = []
    z_scores = []
    fisher_components = []
    for _, row in merged.iterrows():
        p_value = p_alt(int(row.alt_count), int(row.total_count), float(row.alpha), float(row.beta))
        p_vals.append(p_value)
        exp_rate = row.alpha / (row.alpha + row.beta)
        direction = 1 if row.alt_count / max(row.total_count, 1) >= exp_rate else -1
        z = direction * stats.norm.isf(p_value / 2)
        z_scores.append(z)
        fisher_components.append(p_value)

    merged["p_value"] = p_vals
    merged["z_score"] = z_scores
    weights = np.sqrt(merged["total_count"].clip(lower=1))
    stouffer_z = float(np.sum(weights * merged["z_score"]) / np.sqrt(np.sum(weights ** 2)))
    stouffer_p = float(stats.norm.sf(stouffer_z))
    fisher_stat, fisher_p = stats.combine_pvalues(fisher_components, method="fisher")

    merged["q_value"] = benjamini_hochberg(merged["p_value"].to_numpy())
    merged["mrd_call"] = merged["q_value"] <= 0.05

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    summary = {
        "stouffer_z": stouffer_z,
        "stouffer_p": stouffer_p,
        "fisher_p": float(fisher_p),
        "fisher_stat": float(fisher_stat),
        "positives": int(merged["mrd_call"].sum()),
        "total_variants": int(len(merged)),
    }
    summary_path.write_text(pd.Series(summary).to_json())
    typer.echo(
        f"Called {summary['positives']} / {summary['total_variants']} variants (global z={stouffer_z:.2f})"
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

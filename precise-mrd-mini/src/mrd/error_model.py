"""Error model estimation for MRD pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Iterable, Tuple

import numpy as np
import pandas as pd
import typer
from scipy.stats import betabinom

app = typer.Typer(add_completion=False)


def method_of_moments_beta_binomial(fractions: np.ndarray) -> Tuple[float, float]:
    """Estimate beta parameters given observed variant allele fractions."""
    mean = float(np.mean(fractions))
    var = float(np.var(fractions, ddof=1)) if fractions.size > 1 else 0.0
    var = max(var, 1e-6)
    common = mean * (1 - mean) / var - 1
    if common <= 0:
        common = 1e3
    alpha = mean * common
    beta = (1 - mean) * common
    return float(alpha), float(beta)


def p_alt(obs_alt: int, depth: int, alpha: float, beta: float) -> float:
    """Tail probability of observing >= obs_alt counts under beta-binomial."""
    obs_alt = max(obs_alt, 0)
    depth = max(depth, 1)
    return float(betabinom(depth, alpha, beta).sf(obs_alt - 1))


def _simulate_panel(
    loci: pd.DataFrame,
    alpha: float,
    beta: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for _, row in loci.iterrows():
        depth = max(int(row.get("total_count", 0)), 1)
        simulated = betabinom(depth, alpha, beta).rvs(random_state=rng)
        records.append({"chrom": row.chrom, "pos": row.pos, "alt_count": simulated, "total_count": depth})
    return pd.DataFrame(records)


@app.command()
def run(
    collapsed_path: Annotated[Path, typer.Argument()] = Path("tmp/collapsed.parquet"),
    output_path: Annotated[Path, typer.Option()] = Path("tmp/error_model.parquet"),
    panel_alpha: Annotated[float, typer.Option(help="Prior alpha")] = 1.5,
    panel_beta: Annotated[float, typer.Option(help="Prior beta")] = 3000.0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 1234,
) -> None:
    """Estimate per-locus beta-binomial error parameters."""
    collapsed = pd.read_parquet(collapsed_path)
    if collapsed.empty:
        raise typer.BadParameter("Collapsed table is empty")

    fractions = collapsed["alt_count"] / collapsed["total_count"].clip(lower=1)
    if fractions.sum() == 0:
        panel = _simulate_panel(collapsed, panel_alpha, panel_beta, seed)
        fractions = panel["alt_count"] / panel["total_count"].clip(lower=1)

    alpha, beta = method_of_moments_beta_binomial(fractions.to_numpy())
    params = collapsed[["chrom", "pos"]].drop_duplicates().copy()
    params["alpha"] = alpha
    params["beta"] = beta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params.to_parquet(output_path, index=False)
    typer.echo(f"Estimated beta-binomial parameters alpha={alpha:.2f}, beta={beta:.2f}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

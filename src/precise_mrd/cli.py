"""Typer-based command-line interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from .call import call_mrd
from .collapse import collapse_umis
from .config import PipelineConfig, ConfigError, dump_config, load_config
from .error_model import fit_error_model
from .lineage import LineageWriter, lineage_record
from .reporting import render_plots, render_report
from .rng import choose_rng
from .simulate import simulate_reads
from .utils import ARTIFACT_FILENAMES, PipelineIO, as_json_ready, ensure_directory

app = typer.Typer(help="Deterministic MRD mini-pipeline")


def _prepare_output(out: Path, config: PipelineConfig, randomize: bool) -> Path:
    base = Path(out)
    run_dir = base / config.run_id
    if randomize:
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        run_dir = run_dir / stamp
    return ensure_directory(run_dir)


def _load_config(path: Path | None) -> PipelineConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def simulate(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(42, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    _ = threads  # threads currently unused but kept for interface parity
    cfg = _load_config(config)
    out_dir = _prepare_output(out, cfg, randomize)
    io = PipelineIO(out_dir)
    io.path("config").write_text(json.dumps(dump_config(cfg), indent=2), encoding="utf-8")

    rng = choose_rng(seed)
    result = simulate_reads(cfg, rng)
    path = io.write_parquet("simulate", result.reads)

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "simulate", {"seed": seed}))

    _emit(
        {
            "stage": "simulate",
            "output_dir": str(out_dir),
            "artifacts": {"reads": str(path)},
            "config": dump_config(cfg),
        }
    )


@app.command()
def collapse(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(42, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    _ = seed, threads, randomize
    cfg = _load_config(config)
    out_dir = _prepare_output(out, cfg, False)
    io = PipelineIO(out_dir)
    reads_path = io.path("simulate")
    if not reads_path.exists():
        typer.echo("simulated reads not found; run `mrd simulate` first", err=True)
        raise typer.Exit(code=2)

    reads = io.read_parquet("simulate")
    collapsed = collapse_umis(reads, cfg)
    path = io.write_parquet("collapse", collapsed)

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "collapse", {}))

    _emit({"stage": "collapse", "artifacts": {"collapsed": str(path)}})


@app.command("error-model")
def error_model(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(42, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    _ = seed, threads, randomize
    cfg = _load_config(config)
    io = PipelineIO(_prepare_output(out, cfg, False))

    collapsed_path = io.path("collapse")
    if not collapsed_path.exists():
        typer.echo("collapsed UMI table not found; run `mrd collapse` first", err=True)
        raise typer.Exit(code=2)

    collapsed = io.read_parquet("collapse")
    error_df = fit_error_model(collapsed, cfg)
    path = io.write_parquet("error_model", error_df)

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "error_model", {}))

    _emit({"stage": "error_model", "artifacts": {"error_model": str(path)}})


@app.command()
def call(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(42, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    _ = threads, randomize
    cfg = _load_config(config)
    out_dir = _prepare_output(out, cfg, False)
    io = PipelineIO(out_dir)

    collapsed = io.read_parquet("collapse")
    error_df = io.read_parquet("error_model")

    rng = choose_rng(seed)
    calls, metrics_payload, lod = call_mrd(collapsed, error_df, cfg, rng)
    call_path = io.write_parquet("call", calls)
    lod_path = io.write_parquet("lod_grid", lod)
    metrics_path = io.write_json("metrics", metrics_payload)

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "call", {"seed": seed}))

    _emit(
        {
            "stage": "call",
            "artifacts": {
                "calls": str(call_path),
                "lod_grid": str(lod_path),
                "metrics": str(metrics_path),
            },
            "metrics": metrics_payload,
        }
    )


@app.command()
def report(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(42, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    _ = seed, threads, randomize
    cfg = _load_config(config)
    out_dir = _prepare_output(out, cfg, False)
    io = PipelineIO(out_dir)

    metrics_path = io.path("metrics")
    lod_path = io.path("lod_grid")
    if not metrics_path.exists() or not lod_path.exists():
        typer.echo("metrics or LoD grid missing; run `mrd call` first", err=True)
        raise typer.Exit(code=2)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lod = pd.read_parquet(lod_path)
    calls = io.read_parquet("call")
    plot_artifacts = {}
    if cfg.report.include_plots:
        plot_artifacts = render_plots(calls, out_dir)
    md_path, html_path = render_report(
        metrics, lod, out_dir, cfg.report.template and Path(cfg.report.template)
    )

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "report", {}))

    artifacts = {"markdown": str(md_path), "html": str(html_path)}
    artifacts.update(plot_artifacts)
    _emit(
        {
            "stage": "report",
            "artifacts": artifacts,
        }
    )


@app.command()
def smoke(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    seed: int = typer.Option(7, "--seed", help="Global random seed"),
    out: Path = typer.Option(Path("artifacts"), "--out", help="Artifact directory"),
    threads: int = typer.Option(1, "--threads", help="Reserved for Snakemake integration"),
    randomize: bool = typer.Option(
        False, "--randomize", help="Append timestamp to output directory"
    ),
) -> None:
    from .determinism_utils import set_all_seeds
    import numpy as np

    # Set deterministic seeds for reproducibility
    set_all_seeds(seed)

    cfg = _load_config(config)
    out_dir = _prepare_output(out, cfg, randomize)
    io = PipelineIO(out_dir)
    io.path("config").write_text(json.dumps(dump_config(cfg), indent=2), encoding="utf-8")

    rng = choose_rng(seed)
    simulated = simulate_reads(cfg, rng)
    io.write_parquet("simulate", simulated.reads)

    collapsed = collapse_umis(simulated.reads, cfg)
    io.write_parquet("collapse", collapsed)

    error_df = fit_error_model(collapsed, cfg)
    io.write_parquet("error_model", error_df)

    calls, metrics_payload, lod = call_mrd(collapsed, error_df, cfg, rng)
    io.write_parquet("call", calls)
    io.write_parquet("lod_grid", lod)
    io.write_json("metrics", metrics_payload)

    # Save first 10 variant scores for determinism testing
    scores = 1.0 - calls["pvalue"].head(10).to_numpy()
    smoke_scores_path = out_dir / "smoke_scores.npy"
    np.save(smoke_scores_path, scores)

    # Save run context for reproducibility
    context = {
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat(),
        "config": dump_config(cfg),
        "parameters": {
            "threads": threads,
            "randomize": randomize,
        },
    }
    io.write_json("run_context", context)

    render_report(metrics_payload, lod, out_dir, cfg.report.template and Path(cfg.report.template))

    lineage = LineageWriter(io.path("lineage"))
    lineage.append(lineage_record(cfg.run_id, "smoke", {"seed": seed}))

    _emit(
        {
            "stage": "smoke",
            "output_dir": str(out_dir),
            "artifacts": {
                key: str(out_dir / filename)
                for key, filename in ARTIFACT_FILENAMES.items()
                if (out_dir / filename).exists()
            },
            "metrics": metrics_payload,
            "smoke_scores": str(smoke_scores_path),
            "run_context": str(io.path("run_context")),
        }
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()

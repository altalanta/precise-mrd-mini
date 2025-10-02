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
from . import __version__

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


@app.command("init-config")
def init_config(
    output: Path = typer.Option(
        Path("configs/custom.yaml"), 
        "--output", 
        help="Output path for configuration file"
    ),
    template: str = typer.Option(
        "default", 
        "--template", 
        help="Configuration template (default, small, large)"
    ),
) -> None:
    """Initialize a new configuration file from template."""
    from .config import create_default_config
    
    config = create_default_config(template)
    
    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(output, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(dump_config(config), f, default_flow_style=False, sort_keys=False)
    
    _emit({
        "stage": "init-config",
        "template": template,
        "output_file": str(output),
        "config": dump_config(config),
    })


@app.command()
def validate(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    results: Path = typer.Option(
        Path("artifacts"), "--results", help="Results directory to validate"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Validate pipeline results and configuration."""
    from .validation import validate_results, validate_config
    
    validation_results = {"config": None, "results": None, "overall": "UNKNOWN"}
    
    # Validate configuration if provided
    if config:
        try:
            cfg = _load_config(config)
            config_validation = validate_config(cfg)
            validation_results["config"] = config_validation
        except Exception as e:
            validation_results["config"] = {"status": "FAILED", "error": str(e)}
    
    # Validate results if directory exists
    if results.exists():
        try:
            results_validation = validate_results(results)
            validation_results["results"] = results_validation
        except Exception as e:
            validation_results["results"] = {"status": "FAILED", "error": str(e)}
    
    # Determine overall status
    statuses = [v.get("status", "UNKNOWN") for v in validation_results.values() if v and isinstance(v, dict)]
    if all(s == "PASSED" for s in statuses):
        validation_results["overall"] = "PASSED"
    elif any(s == "FAILED" for s in statuses):
        validation_results["overall"] = "FAILED"
    else:
        validation_results["overall"] = "WARNING"
    
    if json_output:
        _emit(validation_results)
    else:
        # Human-readable output
        status_color = {
            "PASSED": typer.colors.GREEN,
            "FAILED": typer.colors.RED,
            "WARNING": typer.colors.YELLOW,
            "UNKNOWN": typer.colors.BLUE,
        }
        
        typer.echo(f"Validation Results:")
        typer.echo(f"Overall: ", nl=False)
        typer.secho(
            validation_results["overall"], 
            fg=status_color[validation_results["overall"]]
        )
        
        for component, result in validation_results.items():
            if component != "overall" and result:
                typer.echo(f"  {component}: ", nl=False)
                typer.secho(result["status"], fg=status_color[result["status"]])
                if "error" in result:
                    typer.echo(f"    Error: {result['error']}")


@app.command()
def benchmark(
    config: Path = typer.Option(None, "--config", help="Pipeline configuration YAML"),
    n_runs: int = typer.Option(3, "--n-runs", help="Number of benchmark runs"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Run performance benchmarks."""
    import time
    from statistics import mean, stdev
    
    cfg = _load_config(config)
    
    # Ensure we use a small configuration for benchmarking
    cfg.simulation.n_replicates = min(cfg.simulation.n_replicates, 100)
    cfg.simulation.n_bootstrap = min(cfg.simulation.n_bootstrap, 100)
    
    times = {"simulate": [], "collapse": [], "error_model": [], "call": [], "total": []}
    
    for run in range(n_runs):
        if not json_output:
            typer.echo(f"Benchmark run {run + 1}/{n_runs}...")
        
        total_start = time.time()
        
        # Simulate
        start = time.time()
        rng = choose_rng(42 + run)
        simulated = simulate_reads(cfg, rng)
        times["simulate"].append(time.time() - start)
        
        # Collapse
        start = time.time()
        collapsed = collapse_umis(simulated.reads, cfg)
        times["collapse"].append(time.time() - start)
        
        # Error model
        start = time.time()
        error_df = fit_error_model(collapsed, cfg)
        times["error_model"].append(time.time() - start)
        
        # Call
        start = time.time()
        call_mrd(collapsed, error_df, cfg, rng)
        times["call"].append(time.time() - start)
        
        times["total"].append(time.time() - total_start)
    
    # Calculate statistics
    results = {}
    for stage, stage_times in times.items():
        results[stage] = {
            "mean": mean(stage_times),
            "std": stdev(stage_times) if len(stage_times) > 1 else 0.0,
            "min": min(stage_times),
            "max": max(stage_times),
            "runs": stage_times,
        }
    
    benchmark_summary = {
        "stage": "benchmark",
        "n_runs": n_runs,
        "config_size": "small" if cfg.simulation.n_replicates <= 100 else "full",
        "timing": results,
        "passed_60s_budget": results["total"]["max"] < 60.0,
    }
    
    if json_output:
        _emit(benchmark_summary)
    else:
        typer.echo(f"\nBenchmark Results ({n_runs} runs):")
        typer.echo(f"Configuration: {benchmark_summary['config_size']}")
        for stage, stats in results.items():
            typer.echo(f"  {stage:12s}: {stats['mean']:.2f}±{stats['std']:.2f}s "
                      f"(min: {stats['min']:.2f}s, max: {stats['max']:.2f}s)")
        
        if benchmark_summary["passed_60s_budget"]:
            typer.secho("✓ Passed 60s performance budget", fg=typer.colors.GREEN)
        else:
            typer.secho("✗ Failed 60s performance budget", fg=typer.colors.RED)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"precise-mrd {__version__}")
        raise typer.Exit()


def deterministic_callback(value: bool):
    """Print determinism information and exit."""
    if value:
        from .determinism_utils import get_git_sha
        import platform
        
        info = {
            "version": __version__,
            "git_sha": get_git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "deterministic_flags": {
                "PYTHONHASHSEED": "0 (recommended)",
                "numpy_random_state": "controlled via --seed",
                "pytorch_deterministic": "set if PyTorch available",
            }
        }
        _emit(info)
        raise typer.Exit()


# Add global options to the main app
@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=version_callback, help="Show version and exit"
    ),
    deterministic: bool = typer.Option(
        False, "--deterministic", callback=deterministic_callback, 
        help="Show determinism configuration and exit"
    ),
):
    """Precise MRD: ctDNA/UMI MRD simulator + caller with deterministic error modeling."""
    pass


def main() -> None:  # pragma: no cover - CLI entry point
    """Main CLI entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

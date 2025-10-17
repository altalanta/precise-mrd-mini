"""Command-line interface for the Precise MRD pipeline."""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
from importlib import resources

from .call import call_mrd
from .collapse import collapse_umis
from .config import PipelineConfig, load_config
from .determinism_utils import env_fingerprint, set_global_seed, write_manifest
from .error_model import fit_error_model
from .eval.lod import LODAnalyzer
from .eval.stratified import run_stratified_analysis
from .metrics import calculate_metrics
from .reporting import render_report
from .sim.contamination import run_contamination_stress_test
from .simulate import simulate_reads
from .utils import PipelineIO
from .validation import assert_hashes_stable, validate_artifacts
from .performance import get_performance_report, reset_performance_monitor
from .statistical_validation import CrossValidator, StatisticalTester, RobustnessAnalyzer

DEFAULT_CONFIG_NAME = "smoke.yaml"
REPORTS_DIR = Path("reports")
DATA_ROOT = Path("data")


@dataclass(slots=True)
class CLIContext:
    """Shared CLI configuration."""

    seed: int
    config_override: Optional[Path]


def _load_pipeline_config(config_option: Optional[Path], seed: int) -> PipelineConfig:
    """Load a pipeline configuration, falling back to the packaged smoke config."""
    with ExitStack() as stack:
        if config_option:
            config_path = Path(config_option)
            if not config_path.exists():
                raise click.ClickException(f"Configuration file not found: {config_path}")
        else:
            resource = resources.files("precise_mrd.assets.configs") / DEFAULT_CONFIG_NAME
            config_path = stack.enter_context(resources.as_file(resource))

        try:
            config = load_config(config_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            if config_option:
                raise click.ClickException(f"Failed to load configuration {config_path}: {exc}") from exc
            config = create_minimal_config(seed)

    config.seed = seed
    return config


def _json_ready(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert numpy/scalar values to native Python types for JSON output."""
    def _convert(obj: Any) -> Any:
        try:
            import numpy as np  # noqa: WPS433 (import inside helper)
        except ModuleNotFoundError:  # pragma: no cover - numpy always present in runtime deps
            np = None

        if isinstance(obj, dict):
            return {key: _convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_convert(item) for item in obj]
        if np is not None and isinstance(obj, (np.floating, float)):
            return float(obj)
        if np is not None and isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    return _convert(payload)


def _run_smoke_pipeline(config: PipelineConfig, seed: int, output_dir: Path) -> Dict[str, Path]:
    """Execute the smoke pipeline and persist canonical artifacts."""
    rng = set_global_seed(seed, deterministic_ops=True)
    stage_dir = PipelineIO.ensure_dir(output_dir)
    reports_dir = PipelineIO.ensure_dir(REPORTS_DIR)

    reads_path = stage_dir / "simulated_reads.parquet"
    collapsed_path = stage_dir / "collapsed_umis.parquet"
    error_model_path = stage_dir / "error_model.parquet"
    calls_path = stage_dir / "mrd_calls.parquet"
    metrics_path = reports_dir / "metrics.json"
    context_path = reports_dir / "run_context.json"
    report_path = reports_dir / "auto_report.html"
    manifest_path = reports_dir / "hash_manifest.txt"

    reads_df = simulate_reads(config, rng, output_path=str(reads_path), use_cache=True, chunked_processing=True)
    collapsed_df = collapse_umis(reads_df, config, rng, output_path=str(collapsed_path))
    error_model_df = fit_error_model(
        collapsed_df,
        config,
        rng,
        output_path=str(error_model_path),
        use_advanced_stats=False,
    )
    calls_df = call_mrd(
        collapsed_df,
        error_model_df,
        config,
        rng,
        output_path=str(calls_path),
        use_ml_calling=False,
    )

    metrics = calculate_metrics(
        calls_df,
        rng,
        n_bootstrap=config.simulation.n_bootstrap if config.simulation else 100,
        config=config,
        use_advanced_ci=False,
        run_validation=False,
    )
    metrics = _json_ready(metrics)

    metrics["schema_version"] = "1.0.0"

    run_context = {
        "schema_version": "1.0.0",
        "seed": seed,
        "timestamp": datetime(2024, 10, 3, 12, 0, 0).isoformat(),
        "config_hash": config.config_hash(),
        "cli_args": {
            "command": "smoke",
            "seed": seed,
            "config_run_id": config.run_id,
            "output_dir": str(stage_dir),
        },
        **env_fingerprint(),
    }

    PipelineIO.save_json(metrics, metrics_path)
    PipelineIO.save_json(run_context, context_path)

    render_report(calls_df, metrics, config.to_dict(), run_context, str(report_path))

    write_manifest(
        [
            metrics_path,
            context_path,
            report_path,
        ],
        out_manifest=manifest_path,
    )

    return {
        "reads": reads_path,
        "collapsed": collapsed_path,
        "error_model": error_model_path,
        "calls": calls_path,
        "metrics": metrics_path,
        "run_context": context_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def _ensure_reports_dir() -> Path:
    """Guarantee the reports directory exists."""
    return PipelineIO.ensure_dir(REPORTS_DIR)


@click.group()
@click.option("--seed", default=7, show_default=True, type=int, help="Seed for deterministic runs.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Path to a pipeline configuration file. Defaults to the packaged smoke config.",
)
@click.pass_context
def main(ctx: click.Context, seed: int, config_path: Optional[Path]) -> None:
    """Precise MRD: deterministic MRD analytics with hardened artifact contracts."""
    ctx.obj = CLIContext(seed=seed, config_override=config_path)


@main.command("smoke")
@click.option(
    "--out-dir",
    default=DATA_ROOT / "smoke",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for intermediate smoke artifacts.",
)
@click.pass_obj
def smoke_cmd(ctx: CLIContext, out_dir: Path) -> None:
    """Run the fast deterministic smoke pipeline."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir)
    validate_artifacts(REPORTS_DIR)

    click.echo(
        json.dumps(
            {
                "stage": "smoke",
                "seed": ctx.seed,
                "config": config.to_dict(),
                "artifacts": {key: str(path) for key, path in artifacts.items()},
            },
            indent=2,
        )
    )


@main.command("determinism")
@click.option(
    "--out-dir",
    default=DATA_ROOT / "determinism",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Base directory for determinism check intermediates.",
)
@click.pass_obj
def determinism_cmd(ctx: CLIContext, out_dir: Path) -> None:
    """Run the smoke pipeline twice and assert identical artifacts."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    _ensure_reports_dir()

    first_artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir / "run1")
    validate_artifacts(REPORTS_DIR)
    manifest_path = Path(first_artifacts["manifest"])
    snapshot_manifest = manifest_path.with_name("hash_manifest_run1.txt")
    shutil.copy2(manifest_path, snapshot_manifest)

    second_artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir / "run2")
    validate_artifacts(REPORTS_DIR)

    assert_hashes_stable(snapshot_manifest, Path(second_artifacts["manifest"]))

    click.echo(
        json.dumps(
            {
                "stage": "determinism",
                "seed": ctx.seed,
                "artifacts": {
                    "first_manifest": str(snapshot_manifest),
                    "second_manifest": str(second_artifacts["manifest"]),
                },
                "status": "hashes-identical",
            },
            indent=2,
        )
    )


def _run_lod_analysis(ctx: CLIContext) -> Tuple[LODAnalyzer, Dict[str, Path]]:
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    rng = set_global_seed(ctx.seed, deterministic_ops=True)
    analyzer = LODAnalyzer(config, rng)
    output = _ensure_reports_dir()
    return analyzer, {"reports": output}


@main.command("eval-lob")
@click.option("--n-blank", default=50, show_default=True, type=int, help="Blank replicates.")
@click.pass_obj
def eval_lob_cmd(ctx: CLIContext, n_blank: int) -> None:
    """Estimate the Limit of Blank and persist lob.json."""
    analyzer, artifacts = _run_lod_analysis(ctx)
    analyzer.estimate_lob(n_blank_runs=n_blank)
    analyzer.generate_reports(str(artifacts["reports"]))
    validate_artifacts(REPORTS_DIR)
    click.echo(
        json.dumps(
            {
                "stage": "eval-lob",
                "lob_json": str(REPORTS_DIR / "lob.json"),
                "blank_runs": n_blank,
            },
            indent=2,
        )
    )


@main.command("eval-lod")
@click.option("--replicates", default=25, show_default=True, type=int, help="Replicates per AF/depth.")
@click.pass_obj
def eval_lod_cmd(ctx: CLIContext, replicates: int) -> None:
    """Estimate the Limit of Detection across configured depths."""
    analyzer, artifacts = _run_lod_analysis(ctx)
    analyzer.estimate_lob(n_blank_runs=max(10, replicates // 2))
    analyzer.estimate_lod(n_replicates=replicates)
    analyzer.generate_reports(str(artifacts["reports"]))
    validate_artifacts(REPORTS_DIR)
    click.echo(
        json.dumps(
            {
                "stage": "eval-lod",
                "lod_table": str(REPORTS_DIR / "lod_table.csv"),
            },
            indent=2,
        )
    )


@main.command("eval-loq")
@click.option("--replicates", default=25, show_default=True, type=int, help="Replicates per AF/depth.")
@click.pass_obj
def eval_loq_cmd(ctx: CLIContext, replicates: int) -> None:
    """Estimate the Limit of Quantification and persist loq_table.csv."""
    analyzer, artifacts = _run_lod_analysis(ctx)
    analyzer.estimate_lob(n_blank_runs=max(10, replicates // 2))
    analyzer.estimate_loq(n_replicates=replicates)
    analyzer.generate_reports(str(artifacts["reports"]))
    validate_artifacts(REPORTS_DIR)
    click.echo(
        json.dumps(
            {
                "stage": "eval-loq",
                "loq_table": str(REPORTS_DIR / "loq_table.csv"),
            },
            indent=2,
        )
    )


@main.command("eval-contamination")
@click.pass_obj
def eval_contamination_cmd(ctx: CLIContext) -> None:
    """Run contamination and index hopping stress tests."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    rng = set_global_seed(ctx.seed, deterministic_ops=True)
    _ensure_reports_dir()
    results = run_contamination_stress_test(config, rng, output_dir=str(REPORTS_DIR))
    validate_artifacts(REPORTS_DIR)
    click.echo(
        json.dumps(
            {
                "stage": "eval-contamination",
                "contamination_summary": str(REPORTS_DIR / "contam_sensitivity.json"),
                "heatmap": str(REPORTS_DIR / "contam_heatmap.png"),
                "contexts": list(results.keys()),
            },
            indent=2,
        )
    )


@main.command("eval-stratified")
@click.pass_obj
def eval_stratified_cmd(ctx: CLIContext) -> None:
    """Conduct stratified power and calibration analysis."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    rng = set_global_seed(ctx.seed, deterministic_ops=True)
    _ensure_reports_dir()
    power_results, calibration_results = run_stratified_analysis(config, rng, output_dir=str(REPORTS_DIR))
    validate_artifacts(REPORTS_DIR)
    click.echo(
        json.dumps(
            {
                "stage": "eval-stratified",
                "power_results": str(REPORTS_DIR / "power_by_stratum.json"),
                "calibration_csv": str(REPORTS_DIR / "calibration_by_bin.csv"),
                "contexts": power_results.get("contexts", []),
                "depths": power_results.get("depth_values", []),
            },
            indent=2,
        )
    )


def create_minimal_config(seed: int) -> PipelineConfig:
    """Create minimal configuration for smoke tests when none is provided."""
    from .config import LODConfig, SimulationConfig, StatsConfig, UMIConfig

    return PipelineConfig(
        run_id="smoke_test",
        seed=seed,
        simulation=SimulationConfig(
            allele_fractions=[0.01, 0.001, 0.0001],
            umi_depths=[1000, 5000],
            n_replicates=10,
            n_bootstrap=100,
        ),
        umi=UMIConfig(
            min_family_size=3,
            max_family_size=1000,
            quality_threshold=20,
            consensus_threshold=0.6,
        ),
        stats=StatsConfig(
            test_type="poisson",
            alpha=0.05,
            fdr_method="benjamini_hochberg",
        ),
        lod=LODConfig(
            detection_threshold=0.95,
            confidence_level=0.95,
        ),
    )


@main.command("performance")
@click.option("--reset", is_flag=True, help="Reset performance monitoring data")
@click.pass_obj
def performance_cmd(ctx: CLIContext, reset: bool) -> None:
    """Show performance monitoring statistics."""
    if reset:
        reset_performance_monitor()
        click.echo("Performance monitoring data reset.")
        return

    report = get_performance_report()

    click.echo("Performance Report:")
    click.echo("=" * 50)

    # Timing statistics
    if report["timing_statistics"]:
        click.echo("\nTiming Statistics:")
        for func_name, stats in report["timing_statistics"].items():
            click.echo(f"  {func_name}:")
            click.echo(f"    Calls: {stats['calls']}")
            click.echo(f"    Total: {stats['total_time']:.3f}s")
            click.echo(f"    Average: {stats['avg_time']:.3f}s")
            click.echo(f"    Min: {stats['min_time']:.3f}s")
            click.echo(f"    Max: {stats['max_time']:.3f}s")

    # Memory usage
    click.echo(f"\nPeak Memory Usage: {report['peak_memory_mb']:.1f} MB")
    click.echo(f"Functions Tracked: {report['total_functions_tracked']}")

    click.echo(json.dumps(report, indent=2, default=str))


@main.command("validate-model")
@click.option("--data-path", type=click.Path(path_type=Path), help="Path to processed data for validation")
@click.option("--k-folds", default=5, type=int, help="Number of cross-validation folds")
@click.option("--scoring", default="roc_auc", type=str, help="Scoring metric for validation")
@click.pass_obj
def validate_model_cmd(ctx: CLIContext, data_path: Optional[Path], k_folds: int, scoring: str) -> None:
    """Run comprehensive model validation and statistical testing."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    rng = set_global_seed(ctx.seed, deterministic_ops=True)

    if data_path is None:
        # Use default smoke data
        data_path = DATA_ROOT / "smoke" / "smoke"

    # Load data for validation
    try:
        calls_df = pd.read_parquet(data_path / "mrd_calls.parquet")
        collapsed_df = pd.read_parquet(data_path / "collapsed_umis.parquet")
    except FileNotFoundError:
        click.echo(f"Error: Required data files not found in {data_path}")
        return

    click.echo(f"🔬 Running model validation on {len(calls_df)} samples...")

    # Cross-validation
    cv = CrossValidator(config)
    if 'ml_probability' in calls_df.columns:
        X = calls_df[['family_size', 'quality_score', 'consensus_agreement']].values
        y = calls_df['is_variant'].values

        def simple_model_func(X_train, y_train):
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=config.seed)
            model.fit(X_train, y_train)
            return model

        cv_results = cv.k_fold_cross_validation(X, y, simple_model_func, k_folds=k_folds, scoring=scoring)
        click.echo(f"  Cross-validation {scoring}: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")

    # Robustness analysis
    robustness = RobustnessAnalyzer(config)
    robustness_results = robustness.bootstrap_robustness(calls_df, n_bootstrap=100)
    click.echo(f"  Robustness analysis: {len(robustness_results['robustness_statistics'])} metrics evaluated")

    # Statistical testing
    tester = StatisticalTester(config)
    # Example: Multiple testing correction on p-values
    if 'p_value' in calls_df.columns:
        p_values = calls_df['p_value'].values
        correction_results = tester.multiple_testing_correction(p_values, method='benjamini_hochberg')
        click.echo(f"  Multiple testing: {correction_results['n_rejected']}/{correction_results['n_tests']} tests rejected")

    # Save validation results
    validation_results = {
        'cross_validation': cv_results if 'cv_results' in locals() else None,
        'robustness_analysis': robustness_results,
        'multiple_testing': correction_results if 'correction_results' in locals() else None,
        'parameters': {
            'k_folds': k_folds,
            'scoring': scoring,
            'n_samples': len(calls_df)
        }
    }

    output_path = REPORTS_DIR / "model_validation.json"
    PipelineIO.save_json(validation_results, str(output_path))
    click.echo(f"📊 Validation results saved to {output_path}")

    click.echo(json.dumps({
        'stage': 'validate-model',
        'validation_file': str(output_path),
        'samples_validated': len(calls_df)
    }, indent=2))


def cli() -> None:  # pragma: no cover - convenience shim
    """Entry point compatible with legacy console_scripts."""
    main(standalone_mode=True)


if __name__ == "__main__":  # pragma: no cover
    cli()

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
from .cache import PipelineCache
from .config import ConfigValidator, PredefinedTemplates, PipelineConfig, ConfigVersionManager, dump_config

DEFAULT_CONFIG_NAME = "smoke.yaml"
REPORTS_DIR = Path("reports")
DATA_ROOT = Path("data")


@dataclass(slots=True)
class CLIContext:
    """Shared CLI configuration."""

    seed: int
    config_override: Optional[Path]
    ml_model_type: str


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


def _run_smoke_pipeline(config: PipelineConfig, seed: int, output_dir: Path, use_parallel: bool = False, n_partitions: int = None, use_cache: bool = True, cache_dir: Path = None, use_ml_calling: bool = False, ml_model_type: str = 'ensemble') -> Dict[str, Path]:
    """Execute the smoke pipeline and persist canonical artifacts."""
    rng = set_global_seed(seed, deterministic_ops=True)
    stage_dir = PipelineIO.ensure_dir(output_dir)
    reports_dir = PipelineIO.ensure_dir(REPORTS_DIR)

    # Initialize cache if requested
    cache = PipelineCache(cache_dir, enabled=use_cache) if use_cache and cache_dir else None

    reads_path = stage_dir / "simulated_reads.parquet"
    collapsed_path = stage_dir / "collapsed_umis.parquet"
    error_model_path = stage_dir / "error_model.parquet"
    calls_path = stage_dir / "mrd_calls.parquet"
    metrics_path = reports_dir / "metrics.json"
    context_path = reports_dir / "run_context.json"
    report_path = reports_dir / "auto_report.html"
    manifest_path = reports_dir / "hash_manifest.txt"

    # Auto-tune configuration based on data characteristics if enabled
    if use_cache:
        # Check if we have cached data characteristics for auto-tuning
        cache_key = f"data_characteristics_{config.config_hash()}"
        cached_characteristics = cache.get("data_analysis", config, (cache_key,)) if cache else None

        if cached_characteristics:
            click.echo("🔧 Auto-tuning configuration based on cached data characteristics...")
            config = config.adapt_to_data(cached_characteristics)
        else:
            click.echo("📊 Analyzing data characteristics for potential auto-tuning...")

    reads_df = simulate_reads(config, rng, output_path=str(reads_path))
    collapsed_df = collapse_umis(reads_df, config, rng, output_path=str(collapsed_path), use_parallel=use_parallel, n_partitions=n_partitions)
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
        use_ml_calling=use_ml_calling,
        ml_model_type=ml_model_type,
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
        "parallel_processing": {
            "enabled": use_parallel,
            "partitions": n_partitions,
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
@click.option(
    "--ml-model",
    "ml_model_type",
    type=click.Choice(['ensemble', 'xgboost', 'lightgbm', 'gbm']),
    default='ensemble',
    show_default=True,
    help="Type of ML model to use for variant calling.",
)
@click.pass_context
def main(ctx: click.Context, seed: int, config_path: Optional[Path], ml_model_type: str) -> None:
    """Precise MRD: deterministic MRD analytics with hardened artifact contracts."""
    ctx.obj = CLIContext(seed=seed, config_override=config_path, ml_model_type=ml_model_type)


@main.command("smoke")
@click.option(
    "--out-dir",
    default=DATA_ROOT / "smoke",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for intermediate smoke artifacts.",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Use parallel processing for UMI collapse operations.",
)
@click.option(
    "--n-partitions",
    default=None,
    type=int,
    help="Number of partitions for parallel processing (auto-detected if not specified).",
)
@click.option(
    "--cache-dir",
    default=DATA_ROOT / "cache",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for caching intermediate results.",
)
@click.option(
    "--use-cache/--no-cache",
    default=True,
    help="Enable or disable caching of intermediate results.",
)
@click.option(
    "--ml-calling",
    is_flag=True,
    help="Use machine learning-based variant calling instead of statistical tests.",
)
@click.pass_obj
def smoke_cmd(ctx: CLIContext, out_dir: Path, parallel: bool, n_partitions: int, cache_dir: Path, use_cache: bool, ml_calling: bool) -> None:
    """Run the fast deterministic smoke pipeline."""
    config = _load_pipeline_config(ctx.config_override, ctx.seed)
    artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir, use_parallel=parallel, n_partitions=n_partitions, use_cache=use_cache, cache_dir=cache_dir, use_ml_calling=ml_calling, ml_model_type=ctx.ml_model_type)
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

    cache_dir = out_dir / "cache"
    first_artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir / "run1", use_parallel=True, n_partitions=2, use_cache=True, cache_dir=cache_dir, use_ml_calling=True, ml_model_type=ctx.ml_model_type)
    validate_artifacts(REPORTS_DIR)
    manifest_path = Path(first_artifacts["manifest"])
    snapshot_manifest = manifest_path.with_name("hash_manifest_run1.txt")
    shutil.copy2(manifest_path, snapshot_manifest)

    second_artifacts = _run_smoke_pipeline(config, ctx.seed, out_dir / "run2", use_parallel=True, n_partitions=2, use_cache=True, cache_dir=cache_dir, use_ml_calling=True, ml_model_type=ctx.ml_model_type)
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


@main.command("cache-info")
@click.option(
    "--cache-dir",
    default=DATA_ROOT / "cache",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory containing cached results.",
)
def cache_info_cmd(cache_dir: Path) -> None:
    """Show information about cached results."""
    cache = PipelineCache(cache_dir, enabled=True)

    if not cache.enabled:
        click.echo("Caching is disabled.")
        return

    cache.cleanup_expired()

    cache_files = list(cache_dir.glob("*.pkl"))
    metadata_file = cache_dir / "cache_metadata.json"

    click.echo("Cache Information:")
    click.echo("=" * 50)
    click.echo(f"Cache directory: {cache_dir}")
    click.echo(f"Cache enabled: {cache.enabled}")
    click.echo(f"TTL: {cache.ttl_seconds} seconds")

    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            click.echo(f"Active cache entries: {len(metadata)}")
            click.echo(f"Cache files: {len(cache_files)}")

            if metadata:
                # Show some stats about cache entries
                func_names = [entry.get('func_name', 'unknown') for entry in metadata.values()]
                from collections import Counter
                func_counts = Counter(func_names)
                click.echo("Functions cached:")
                for func, count in func_counts.most_common():
                    click.echo(f"  {func}: {count} entries")
        except json.JSONDecodeError:
            click.echo("Cache metadata is corrupted.")
    else:
        click.echo("No cache metadata found.")


@main.command("cache-clear")
@click.option(
    "--cache-dir",
    default=DATA_ROOT / "cache",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory containing cached results.",
)
@click.option("--force", is_flag=True, help="Clear cache without confirmation.")
def cache_clear_cmd(cache_dir: Path, force: bool) -> None:
    """Clear cached results."""
    cache = PipelineCache(cache_dir, enabled=True)

    if not cache.enabled:
        click.echo("Caching is disabled.")
        return

    if not force:
        click.confirm(f"This will clear all cached results in {cache_dir}. Continue?", abort=True)

    cache.clear()
    click.echo(f"Cleared cache in {cache_dir}")


@main.command("config-validate")
@click.argument("config_path", type=click.Path(path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation results.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save validation report to file.")
def config_validate_cmd(config_path: Path, verbose: bool, output: Path) -> None:
    """Validate a pipeline configuration file."""
    try:
        config = load_config(config_path)

        # Run comprehensive validation
        validation_result = ConfigValidator.validate_config(config)

        if validation_result['is_valid']:
            click.echo(f"✅ Configuration '{config.run_id}' is valid!")
        else:
            click.echo(f"❌ Configuration '{config.run_id}' has issues:")
            for issue in validation_result['issues']:
                click.echo(f"  • {issue}")

        if validation_result['warnings']:
            click.echo(f"⚠️  Warnings:")
            for warning in validation_result['warnings']:
                click.echo(f"  • {warning}")

        if validation_result['suggestions']:
            click.echo(f"💡 Suggestions:")
            for suggestion in validation_result['suggestions']:
                click.echo(f"  • {suggestion}")

        if verbose or validation_result['warnings'] or validation_result['suggestions']:
            click.echo(f"\n📊 Configuration Summary:")
            click.echo(f"  • Estimated runtime: {validation_result['estimated_runtime_minutes']:.1f} minutes")
            click.echo(f"  • Configuration hash: {validation_result['config_hash']}")
            click.echo(f"  • Version: {config.config_version}")

        if output:
            PipelineIO.save_json(validation_result, str(output))
            click.echo(f"\n📄 Validation report saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}")
        raise click.ClickException(f"Configuration validation failed: {e}")


@main.command("config-templates")
@click.option("--list", "-l", is_flag=True, help="List available configuration templates.")
@click.option("--create", type=str, help="Create a new configuration from template.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path for generated configuration.")
def config_templates_cmd(list: bool, create: str, output: Path) -> None:
    """Manage configuration templates."""
    if list:
        click.echo("📋 Available Configuration Templates:")
        click.echo("=" * 50)

        templates = [
            PredefinedTemplates.get_smoke_test_template(),
            PredefinedTemplates.get_production_template()
        ]

        for template in templates:
            click.echo(f"\n🏷️  {template['template_name']}")
            click.echo(f"   Description: {template['description']}")
            click.echo(f"   Tags: {', '.join(template['tags'])}")
            click.echo(f"   Version: {template['version']}")

    elif create:
        # Find the requested template
        template = None
        if create == "smoke_test":
            template = PredefinedTemplates.get_smoke_test_template()
        elif create == "production":
            template = PredefinedTemplates.get_production_template()
        else:
            click.echo(f"❌ Unknown template: {create}")
            click.echo("Available templates: smoke_test, production")
            return

        if template:
            # Generate configuration from template
            config = PipelineConfig.from_template(template, run_id=f"from_{create}_template")

            if output:
                dump_config(config, output)
                click.echo(f"✅ Configuration created from '{create}' template and saved to: {output}")
            else:
                # Show configuration preview
                click.echo(f"📋 Configuration Preview from '{create}' template:")
                click.echo("=" * 50)
                click.echo(f"Run ID: {config.run_id}")
                click.echo(f"Description: {config.description}")
                click.echo(f"Version: {config.config_version}")
                click.echo(f"Tags: {', '.join(config.tags)}")

                if config.simulation:
                    click.echo("\nSimulation:")
                    click.echo(f"  • Allele fractions: {config.simulation.allele_fractions}")
                    click.echo(f"  • UMI depths: {config.simulation.umi_depths}")
                    click.echo(f"  • Replicates: {config.simulation.n_replicates}")
                    click.echo(f"  • Bootstrap: {config.simulation.n_bootstrap}")

                click.echo("\nUMI Processing:")
                click.echo(f"  • Min family size: {config.umi.min_family_size}")
                click.echo(f"  • Quality threshold: {config.umi.quality_threshold}")
                click.echo(f"  • Consensus threshold: {config.umi.consensus_threshold}")

                click.echo("\nStatistical Testing:")
                click.echo(f"  • Test type: {config.stats.test_type}")
                click.echo(f"  • Alpha: {config.stats.alpha}")
                click.echo(f"  • FDR method: {config.stats.fdr_method}")

                click.echo("\n💡 To save this configuration, use: --output <filename>.yaml")
    else:
        click.echo("❌ Please specify --list or --create <template_name>")
        click.echo("Use --help for more information.")


@main.command("config-adapt")
@click.argument("config_path", type=click.Path(path_type=Path))
@click.option("--data-stats", type=click.Path(path_type=Path), help="JSON file with data characteristics for adaptation.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path for adapted configuration.")
@click.option("--dry-run", is_flag=True, help="Show what would be adapted without saving.")
def config_adapt_cmd(config_path: Path, data_stats: Path, output: Path, dry_run: bool) -> None:
    """Adapt configuration based on data characteristics."""
    try:
        # Load original configuration
        original_config = load_config(config_path)

        if not data_stats:
            click.echo("❌ Please provide --data-stats file with data characteristics.")
            return

        # Load data characteristics
        with open(data_stats, 'r') as f:
            import json
            data_characteristics = json.load(f)

        click.echo(f"🔧 Adapting configuration '{original_config.run_id}' based on data characteristics...")

        # Adapt configuration
        adapted_config = original_config.adapt_to_data(data_characteristics)

        if dry_run:
            click.echo("\n📋 Adaptation Preview:")
            click.echo(f"Original: {original_config.run_id}")
            click.echo(f"Adapted: {adapted_config.run_id}")

            if original_config.simulation and adapted_config.simulation:
                click.echo("\nSimulation changes:")
                if original_config.simulation.allele_fractions != adapted_config.simulation.allele_fractions:
                    click.echo(f"  • Allele fractions: {original_config.simulation.allele_fractions} → {adapted_config.simulation.allele_fractions}")
                if original_config.simulation.umi_depths != adapted_config.simulation.umi_depths:
                    click.echo(f"  • UMI depths: {original_config.simulation.umi_depths} → {adapted_config.simulation.umi_depths}")

            if original_config.umi.quality_threshold != adapted_config.umi.quality_threshold:
                click.echo(f"  • Quality threshold: {original_config.umi.quality_threshold} → {adapted_config.umi.quality_threshold}")

            click.echo("\n💡 To apply these changes, run without --dry-run and specify --output")
        else:
            if output:
                dump_config(adapted_config, output)
                click.echo(f"✅ Adapted configuration saved to: {output}")
            else:
                click.echo("❌ Please specify --output to save the adapted configuration.")

    except Exception as e:
        click.echo(f"❌ Error adapting configuration: {e}")
        raise click.ClickException(f"Configuration adaptation failed: {e}")


@main.command("config-merge")
@click.argument("config1", type=click.Path(path_type=Path))
@click.argument("config2", type=click.Path(path_type=Path))
@click.option("--strategy", type=click.Choice(['override', 'inherit']), default='override',
              help="Merge strategy to use.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path for merged configuration.")
def config_merge_cmd(config1: Path, config2: Path, strategy: str, output: Path) -> None:
    """Merge two configuration files."""
    try:
        config_a = load_config(config1)
        config_b = load_config(config2)

        # Check compatibility
        compatibility_issues = config_a.validate_compatibility(config_b)
        if compatibility_issues:
            click.echo("⚠️  Compatibility issues detected:")
            for issue in compatibility_issues:
                click.echo(f"  • {issue}")

        # Perform merge
        merged_config = config_a.merge_with(config_b, strategy)

        click.echo(f"🔄 Merged configurations using '{strategy}' strategy:")
        click.echo(f"  • Base: {config_a.run_id}")
        click.echo(f"  • Override: {config_b.run_id}")
        click.echo(f"  • Result: {merged_config.run_id}")

        if output:
            dump_config(merged_config, output)
            click.echo(f"✅ Merged configuration saved to: {output}")
        else:
            click.echo("❌ Please specify --output to save the merged configuration.")

    except Exception as e:
        click.echo(f"❌ Error merging configurations: {e}")
        raise click.ClickException(f"Configuration merge failed: {e}")


@main.command("config-migrate")
@click.argument("config_path", type=click.Path(path_type=Path))
@click.option("--target-version", type=str, help="Target configuration version.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path for migrated configuration.")
@click.option("--dry-run", is_flag=True, help="Show migration info without applying changes.")
def config_migrate_cmd(config_path: Path, target_version: str, output: Path, dry_run: bool) -> None:
    """Migrate configuration to a different version."""
    try:
        # Load current configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        current_version = config_data.get('config_version', '1.0.0')
        target_version = target_version or ConfigVersionManager.get_latest_version()

        click.echo(f"🔄 Configuration Migration:")
        click.echo(f"  Current version: {current_version}")
        click.echo(f"  Target version: {target_version}")

        # Get migration information
        migration_info = ConfigVersionManager.get_migration_info(current_version, target_version)

        if migration_info['status'] == 'no_migration_needed':
            click.echo("✅ No migration needed - configuration is already at target version.")
            return
        elif migration_info['status'] == 'migration_not_found':
            click.echo(f"❌ No migration path found from {current_version} to {target_version}")
            return

        click.echo("\n📋 Migration Changes:")
        for change in migration_info['changes']:
            click.echo(f"  • {change}")

        if dry_run:
            click.echo("\n💡 This is a dry run - no changes will be made.")
            click.echo(f"💡 To apply migration, run without --dry-run and specify --output")
            return

        # Perform migration
        migrated_data = ConfigVersionManager.migrate_config(config_data, target_version)

        if output:
            with open(output, 'w') as f:
                yaml.safe_dump(migrated_data, f, default_flow_style=False, sort_keys=False)
            click.echo(f"✅ Migrated configuration saved to: {output}")
        else:
            click.echo("❌ Please specify --output to save the migrated configuration.")

    except Exception as e:
        click.echo(f"❌ Error migrating configuration: {e}")
        raise click.ClickException(f"Configuration migration failed: {e}")


@main.command("config-version")
@click.option("--current", is_flag=True, help="Show current configuration version requirements.")
@click.option("--migrate", type=str, help="Show migration path to specified version.")
def config_version_cmd(current: bool, migrate: str) -> None:
    """Show configuration version information."""
    if current:
        latest_version = ConfigVersionManager.get_latest_version()
        click.echo("📋 Configuration Version Information:")
        click.echo("=" * 50)
        click.echo(f"Latest version: {latest_version}")
        click.echo(f"Supported versions: 1.0.0, {latest_version}")

        click.echo("\n🔄 Migration Paths:")
        for from_version, migration_info in ConfigVersionManager.MIGRATION_PATHS.items():
            click.echo(f"  {from_version} → {migration_info['target']}")
            for change in migration_info['changes'][:3]:  # Show first 3 changes
                click.echo(f"    • {change}")
            if len(migration_info['changes']) > 3:
                click.echo(f"    • ... and {len(migration_info['changes']) - 3} more changes")

    elif migrate:
        migration_info = ConfigVersionManager.get_migration_info("1.0.0", migrate)
        if migration_info['status'] == 'migration_not_found':
            click.echo(f"❌ No migration path found to version {migrate}")
        else:
            click.echo(f"📋 Migration to version {migrate}:")
            for change in migration_info['changes']:
                click.echo(f"  • {change}")

    else:
        click.echo("❌ Please specify --current or --migrate <version>")
        click.echo("Use --help for more information.")


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



@main.command("ml-performance")
@click.option("--reset", is_flag=True, help="Reset ML performance monitoring data")
def ml_performance_cmd(reset: bool) -> None:
    """Show ML model performance metrics."""
    if reset:
        from .performance import reset_ml_performance_tracker
        reset_ml_performance_tracker()
        click.echo("ML performance monitoring data reset.")
        return

    from .performance import get_ml_performance_tracker
    tracker = get_ml_performance_tracker()
    ml_report = tracker.get_ml_report()
    comparison = tracker.compare_models()

    click.echo("ML Performance Report:")
    click.echo("=" * 50)

    if ml_report["n_models_tracked"] == 0:
        click.echo("No ML models have been trained yet.")
        return

    # Model metrics
    if ml_report["model_metrics"]:
        click.echo("
Model Performance:")
        for model_name, metrics in ml_report["model_metrics"].items():
            click.echo(f"  {model_name}:")
            click.echo(f"    ROC AUC: {metrics.get(\"roc_auc\", \"N/A\")}")
            click.echo(f"    Test ROC AUC: {metrics.get(\"test_roc_auc\", \"N/A\")}")
            click.echo(f"    Features: {metrics.get(\"n_features\", \"N/A\")}")

    # Feature importance
    if ml_report["feature_importance"]:
        click.echo("
Top Features (Ensemble):")
        ensemble_importance = ml_report["feature_importance"].get("ensemble_model", {})
        if ensemble_importance:
            sorted_features = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(sorted_features[:5]):
                click.echo(f"  {i+1}. {feat}: {imp:.3f}")

    # Training times
    if ml_report["training_times"]:
        click.echo("
Training Times:")
        for model_name, time_taken in ml_report["training_times"].items():
            click.echo(f"  {model_name}: {time_taken:.2f}s")

    # Model comparison
    if comparison.get("best_model"):
        click.echo(f"
🏆 Best Model: {comparison[\"best_model\"]} (AUC: {comparison[\"best_metric\"]:.3f})")

        click.echo("
Model Ranking:")
        for i, (model, auc) in enumerate(comparison["model_ranking"]):
            click.echo(f"  {i+1}. {model}: {auc:.3f}")

    click.echo(f"
📊 Models Tracked: {ml_report[\"n_models_tracked\"]}")


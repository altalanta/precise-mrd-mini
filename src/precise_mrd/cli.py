"""
Command-line interface for precise-mrd pipeline.

This module provides:
- CLI commands for running simulations
- Report generation commands
- Configuration management
- Utility commands
"""

import click
import yaml
import logging
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .simulate import Simulator, run_simulation_from_config
from .report import ReportGenerator
from .profiling import run_performance_benchmark
from .hashing import HashingManager
from .lod import LODEstimator
from .qc import QCAnalyzer
from .exceptions import PreciseMRDError, ConfigurationError
from .config_validator import validate_config_file
from .logging_config import setup_logging as setup_advanced_logging


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--seed', default=42, help='Global random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx: click.Context, seed: int, verbose: bool) -> None:
    """Precise MRD: ctDNA/UMI toy MRD pipeline."""
    ctx.ensure_object(dict)
    ctx.obj['seed'] = seed
    ctx.obj['verbose'] = verbose
    
    # Set global seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    setup_logging(verbose)
    if verbose:
        logging.info(f"Global random seed set to {seed}")


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              required=True, help='Configuration file path to validate')
@click.pass_context
def validate_config(ctx: click.Context, config: str) -> None:
    """Validate configuration file for errors and warnings."""
    
    try:
        config_path = Path(config)
        click.echo(f"Validating configuration: {config_path}")
        
        is_valid, errors, warnings = validate_config_file(config_path)
        
        if errors:
            click.echo("❌ Configuration validation failed:")
            for error in errors:
                click.echo(f"  • {error}", err=True)
        
        if warnings:
            click.echo("⚠️  Configuration warnings:")
            for warning in warnings:
                click.echo(f"  • {warning}")
        
        if is_valid:
            if warnings:
                click.echo("✅ Configuration is valid (with warnings)")
            else:
                click.echo("✅ Configuration is valid")
        else:
            raise click.ClickException("Configuration validation failed")
            
    except (ConfigurationError, FileNotFoundError) as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        raise click.ClickException(str(e))
    except Exception as e:
        click.echo(f"❌ Unexpected error during validation: {e}", err=True)
        raise click.ClickException(f"Validation failed: {e}")


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              default='configs/default.yaml', help='Configuration file path')
@click.option('--output', '-o', type=click.Path(), 
              default='results/runs', help='Output directory')
@click.option('--run-id', '-r', type=str, help='Run identifier')
@click.option('--validate/--no-validate', default=True, 
              help='Validate configuration before running simulation')
@click.pass_context
def simulate(ctx: click.Context, config: str, output: str, run_id: Optional[str], validate: bool) -> None:
    """Run MRD simulation with specified configuration."""
    
    if run_id is None:
        run_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(output) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Starting simulation: {run_id}")
    click.echo(f"Config: {config}")
    click.echo(f"Output: {output_dir}")
    
    try:
        # Validate configuration if requested
        if validate:
            click.echo("Validating configuration...")
            config_path = Path(config)
            is_valid, errors, warnings = validate_config_file(config_path)
            
            if errors:
                click.echo("❌ Configuration validation failed:")
                for error in errors:
                    click.echo(f"  • {error}", err=True)
                raise click.ClickException("Invalid configuration - fix errors before running simulation")
            
            if warnings:
                click.echo("⚠️  Configuration warnings:")
                for warning in warnings:
                    click.echo(f"  • {warning}")
                
            click.echo("✅ Configuration is valid")
        
        # Run simulation
        results = run_simulation_from_config(config, str(output_dir))
        
        # Create lockfile
        hashing_manager = HashingManager(str(output_dir / "lockfile.json"))
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        lockfile_data = hashing_manager.create_lockfile(
            run_id=run_id,
            config=config_data,
            input_files=[config],
            results_summary={
                "total_simulations": len(results.detection_matrix) if hasattr(results, 'detection_matrix') else 0,
                "success": True
            }
        )
        hashing_manager.save_lockfile(lockfile_data)
        
        # Update latest symlink
        latest_link = Path(output) / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_id)
        
        click.echo(f"✅ Simulation completed successfully!")
        click.echo(f"📁 Results: {output_dir}")
        click.echo(f"🔒 Lockfile: {output_dir / 'lockfile.json'}")
        
    except Exception as e:
        click.echo(f"❌ Simulation failed: {str(e)}", err=True)
        raise click.ClickException(f"Simulation failed: {str(e)}")


@main.command()
@click.option('--results', '-r', type=click.Path(exists=True), 
              default='results/latest', help='Results directory')
@click.option('--output', '-o', type=click.Path(), 
              default='reports/html', help='Report output directory')
@click.option('--template', '-t', type=str, 
              default='template.html', help='Report template name')
@click.pass_context
def report(ctx: click.Context, results: str, output: str, template: str) -> None:
    """Generate HTML report from simulation results."""
    
    results_path = Path(results)
    if not results_path.exists():
        raise click.ClickException(f"Results directory not found: {results_path}")
    
    click.echo(f"Generating report from: {results_path}")
    
    try:
        # Load results data
        results_data = {}
        
        # Load detection matrix
        detection_file = results_path / "detection_matrix.csv"
        if detection_file.exists():
            import pandas as pd
            results_data['detection_matrix'] = pd.read_csv(detection_file)
        
        # Load QC metrics
        qc_file = results_path / "qc_metrics.json"
        if qc_file.exists():
            import json
            with open(qc_file, 'r') as f:
                results_data['qc_metrics'] = json.load(f)
        
        # Load lockfile for config
        lockfile_path = results_path / "lockfile.json"
        config_data = {}
        if lockfile_path.exists():
            import json
            with open(lockfile_path, 'r') as f:
                lockfile_data = json.load(f)
                config_data = lockfile_data.get('config', {})
        
        # Generate report
        reporter = ReportGenerator(output_dir=output)
        run_id = results_path.name
        
        html_path = reporter.generate_html_report(
            results_data=results_data,
            config_data=config_data,
            run_id=run_id,
            template_name=template
        )
        
        click.echo(f"✅ Report generated successfully!")
        click.echo(f"📄 HTML Report: {html_path}")
        
    except Exception as e:
        click.echo(f"❌ Report generation failed: {str(e)}", err=True)
        raise click.ClickException(f"Report generation failed: {str(e)}")


@main.command()
@click.option('--output', '-o', type=click.Path(), 
              default='results/performance_benchmark.json', help='Output file')
@click.pass_context
def benchmark(ctx: click.Context, output: str) -> None:
    """Run performance benchmark."""
    
    click.echo("Running performance benchmark...")
    
    try:
        results = run_performance_benchmark()
        
        if 'error' in results:
            raise click.ClickException(results['error'])
        
        click.echo(f"✅ Benchmark completed!")
        click.echo(f"📊 Report: {results.get('report_path', output)}")
        
        # Print summary
        if 'benchmark_results' in results:
            click.echo("\nBenchmark Summary:")
            for size, metrics in results['benchmark_results'].items():
                click.echo(f"  {size:,} reads: {metrics['reads_per_second']:.0f} reads/sec")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {str(e)}", err=True)
        raise click.ClickException(f"Benchmark failed: {str(e)}")


@main.command()
@click.option('--config', '-c', type=click.Path(), 
              default='configs/default.yaml', help='Configuration template')
@click.option('--output', '-o', type=click.Path(), 
              default='configs/custom.yaml', help='Output configuration file')
@click.pass_context
def init_config(ctx: click.Context, config: str, output: str) -> None:
    """Initialize a new configuration file."""
    
    output_path = Path(output)
    
    if output_path.exists():
        if not click.confirm(f"Configuration file {output_path} already exists. Overwrite?"):
            click.echo("Configuration initialization cancelled.")
            return
    
    try:
        # Copy template or create default
        if Path(config).exists():
            import shutil
            shutil.copy2(config, output_path)
            click.echo(f"✅ Configuration copied from {config}")
        else:
            # Create default config
            default_config = {
                'run_id': 'custom_run',
                'seed': 42,
                'simulation': {
                    'allele_fractions': [0.01, 0.001],
                    'umi_depths': [5000, 10000],
                    'n_replicates': 100,
                    'n_bootstrap': 100
                },
                'umi': {
                    'min_family_size': 3,
                    'max_family_size': 1000,
                    'quality_threshold': 20
                },
                'stats': {
                    'test_type': 'poisson',
                    'alpha': 0.05,
                    'fdr_method': 'benjamini_hochberg'
                }
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(default_config, f, indent=2, default_flow_style=False)
            
            click.echo(f"✅ Default configuration created")
        
        click.echo(f"📝 Configuration file: {output_path}")
        click.echo("Edit the configuration file to customize your analysis.")
        
    except Exception as e:
        click.echo(f"❌ Configuration initialization failed: {str(e)}", err=True)
        raise click.ClickException(f"Configuration initialization failed: {str(e)}")


@main.command()
@click.option('--results', '-r', type=click.Path(exists=True), 
              default='results/latest', help='Results directory')
@click.pass_context
def validate(ctx: click.Context, results: str) -> None:
    """Validate simulation results and check reproducibility."""
    
    results_path = Path(results)
    click.echo(f"Validating results: {results_path}")
    
    try:
        # Load lockfile
        lockfile_path = results_path / "lockfile.json"
        if not lockfile_path.exists():
            raise click.ClickException("Lockfile not found - cannot validate reproducibility")
        
        hashing_manager = HashingManager()
        lockfile_data = hashing_manager.load_lockfile(str(lockfile_path))
        
        # Verify git status
        current_git = hashing_manager.get_git_info()
        lockfile_git = lockfile_data.get('git', {})
        
        click.echo(f"📋 Run ID: {lockfile_data.get('run_id', 'unknown')}")
        click.echo(f"⏰ Created: {lockfile_data.get('creation_timestamp', 'unknown')}")
        click.echo(f"🌿 Git commit: {lockfile_git.get('commit_hash', 'unknown')[:8]}")
        
        # Check for issues
        issues = []
        
        if lockfile_git.get('has_uncommitted_changes'):
            issues.append("Uncommitted changes were present during run")
        
        if current_git.get('commit_hash') != lockfile_git.get('commit_hash'):
            issues.append("Current git commit differs from run")
        
        # Load and validate results files
        required_files = ['detection_matrix.csv', 'qc_metrics.json']
        missing_files = []
        
        for filename in required_files:
            if not (results_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            issues.append(f"Missing result files: {', '.join(missing_files)}")
        
        # Report validation status
        if issues:
            click.echo("\n⚠️  Validation Issues:")
            for issue in issues:
                click.echo(f"  • {issue}")
        else:
            click.echo("\n✅ Validation passed - results appear reproducible")
        
        # Display summary statistics
        qc_file = results_path / "qc_metrics.json"
        if qc_file.exists():
            import json
            with open(qc_file, 'r') as f:
                qc_data = json.load(f)
            
            click.echo(f"\n📊 Summary Statistics:")
            if 'total_simulations' in qc_data:
                click.echo(f"  • Total simulations: {qc_data['total_simulations']:,}")
            if 'overall_detection_rate' in qc_data:
                click.echo(f"  • Overall detection rate: {qc_data['overall_detection_rate']:.1%}")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        raise click.ClickException(f"Validation failed: {str(e)}")


@main.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    
    from . import __version__
    
    click.echo(f"Precise MRD v{__version__}")
    click.echo("ctDNA/UMI toy MRD pipeline with UMI-aware error modeling")
    
    # Show component status
    try:
        import numpy
        import pandas
        import scipy
        import matplotlib
        
        click.echo(f"\nDependencies:")
        click.echo(f"  • NumPy: {numpy.__version__}")
        click.echo(f"  • Pandas: {pandas.__version__}")
        click.echo(f"  • SciPy: {scipy.__version__}")
        click.echo(f"  • Matplotlib: {matplotlib.__version__}")
        
    except ImportError as e:
        click.echo(f"⚠️  Missing dependency: {e}")


if __name__ == '__main__':
    main()
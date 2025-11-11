"""Service layer for running the Precise MRD pipeline."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .call import call_mrd
from .collapse import collapse_umis
from .config import PipelineConfig, load_config
from .determinism_utils import set_global_seed, write_manifest
from .error_model import fit_error_model
from .exceptions import DataProcessingError
from .logging_config import get_logger
from .metrics import calculate_metrics
from .reporting import render_report
from .simulate import simulate_reads
from .utils import PipelineIO, get_package_versions

if TYPE_CHECKING:
    from .api import JobManager, PipelineConfigRequest

log = get_logger(__name__)


class PipelineService:
    """Encapsulates the core logic for running the MRD pipeline."""

    def __init__(self):
        self.results_dir = Path("api_results")
        self.results_dir.mkdir(exist_ok=True)

    def run(
        self,
        job_id: str,
        config_request: "PipelineConfigRequest",
        job_manager: "JobManager",
    ):
        """
        Run the MRD pipeline for a job. This is the main business logic.
        """
        job_log = log.bind(job_id=job_id, run_id=config_request.run_id, seed=config_request.seed)
        try:
            job_log.info("Starting pipeline job")
            job_manager.update_job_status(job_id, 'running', 10.0)

            if config_request.config_override:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write(config_request.config_override)
                    config_path = f.name
                config = load_config(config_path)
                Path(config_path).unlink()
            else:
                config = PipelineConfig(
                    run_id=config_request.run_id,
                    seed=config_request.seed,
                    simulation=None, umi=None, stats=None, lod=None,
                )

            job_manager.update_job_status(job_id, 'running', 20.0)
            job_log.info("Configuration loaded", config_hash=config.config_hash())

            job_dir = self.results_dir / job_id
            job_dir.mkdir(exist_ok=True)
            reports_dir = job_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            data_dir = job_dir / "data"
            data_dir.mkdir(exist_ok=True)

            rng = set_global_seed(config_request.seed, deterministic_ops=True)

            job_manager.update_job_status(job_id, 'running', 30.0)
            job_log.info("Starting stage: simulate_reads")
            reads_path = data_dir / "simulated_reads.parquet"
            reads_df = simulate_reads(config, rng, output_path=str(reads_path))

            job_manager.update_job_status(job_id, 'running', 50.0)
            job_log.info("Finished stage: simulate_reads. Starting stage: collapse_umis")
            collapsed_path = data_dir / "collapsed_umis.parquet"
            collapsed_df = collapse_umis(
                reads_df, config, rng,
                output_path=str(collapsed_path),
                use_parallel=config_request.use_parallel
            )

            job_manager.update_job_status(job_id, 'running', 70.0)
            job_log.info("Finished stage: collapse_umis. Starting stage: fit_error_model")
            error_model_path = data_dir / "error_model.parquet"
            error_model_df = fit_error_model(
                collapsed_df, config, rng,
                output_path=str(error_model_path)
            )

            job_manager.update_job_status(job_id, 'running', 80.0)
            job_log.info("Finished stage: fit_error_model. Starting stage: call_mrd")
            calls_path = data_dir / "mrd_calls.parquet"
            calls_df = call_mrd(
                collapsed_df, error_model_df, config, rng,
                output_path=str(calls_path),
                use_ml_calling=config_request.use_ml_calling,
                ml_model_type=config_request.ml_model_type,
                use_deep_learning=config_request.use_deep_learning,
                dl_model_type=config_request.dl_model_type
            )

            job_manager.update_job_status(job_id, 'running', 90.0)
            job_log.info("Finished stage: call_mrd. Starting final reporting stage.")
            metrics = calculate_metrics(calls_df, rng)
            run_context = {
                "schema_version": "2.0.0", "job_id": job_id, "run_id": config_request.run_id,
                "seed": config_request.seed, "timestamp": datetime.now().isoformat(),
                "config_hash": config.config_hash(), "api_version": "2.0.0",
                "processing_options": {
                    "parallel": config_request.use_parallel,
                    "ml_calling": config_request.use_ml_calling, "ml_model_type": config_request.ml_model_type,
                    "deep_learning": config_request.use_deep_learning, "dl_model_type": config_request.dl_model_type
                },
                "environment": {
                    "package_versions": get_package_versions()
                }
            }

            metrics_path = reports_dir / "metrics.json"
            context_path = reports_dir / "run_context.json"
            report_path = reports_dir / "auto_report.html"
            PipelineIO.save_json(metrics, str(metrics_path))
            PipelineIO.save_json(run_context, str(context_path))
            render_report(calls_df, metrics, config.to_dict(), run_context, str(report_path))

            manifest_path = reports_dir / "hash_manifest.json"
            artifact_paths = [str(p) for p in [
                reads_path, collapsed_path, error_model_path, calls_path,
                metrics_path, context_path, report_path
            ]]
            write_manifest(artifact_paths, out_manifest=str(manifest_path))

            job_manager.update_job_status(job_id, 'completed', 100.0)
            results = {
                'job_id': job_id, 'run_id': config_request.run_id, 'config_hash': config.config_hash(),
                'status': 'completed', 'metrics': metrics, 'run_context': run_context,
                'artifacts': {
                    'reads': str(reads_path), 'collapsed': str(collapsed_path), 'error_model': str(error_model_path),
                    'calls': str(calls_path), 'metrics': str(metrics_path), 'run_context': str(context_path),
                    'report': str(report_path), 'manifest': str(manifest_path)
                }
            }
            job_manager.set_job_results(job_id, results)
            job_log.info("Pipeline job completed successfully.")

        except Exception as e:
            job_log.error("Pipeline job failed", error=str(e), exc_info=True)
            job_manager.update_job_status(job_id, 'failed', error=str(e))
        except Exception as e:
            job_log.error("An unexpected error occurred in the pipeline job", error=str(e), exc_info=True)
            job_manager.update_job_status(job_id, 'failed', error="An unexpected server error occurred.")
            # We wrap the original exception in a DataProcessingError to standardize
            raise DataProcessingError(f"Pipeline job {job_id} failed: {e}") from e


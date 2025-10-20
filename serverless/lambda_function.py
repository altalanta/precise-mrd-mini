"""AWS Lambda function for serverless MRD pipeline processing."""

import json
import uuid
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import zipfile
import io

from precise_mrd.config import PipelineConfig, load_config
from precise_mrd.simulate import simulate_reads
from precise_mrd.collapse import collapse_umis
from precise_mrd.call import call_mrd
from precise_mrd.error_model import fit_error_model
from precise_mrd.metrics import calculate_metrics
from precise_mrd.determinism_utils import set_global_seed


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """AWS Lambda handler for MRD pipeline processing.

    Args:
        event: Lambda event containing pipeline configuration
        context: Lambda context object

    Returns:
        Dictionary with processing results
    """
    try:
        # Parse input configuration
        config_data = event.get('config', {})
        run_id = config_data.get('run_id', f'lambda_{uuid.uuid4().hex[:8]}')
        seed = config_data.get('seed', 7)

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create configuration
            config = PipelineConfig(
                run_id=run_id,
                seed=seed,
                simulation=None,  # Use defaults for Lambda
                umi=None,
                stats=None,
                lod=None,
            )

            # Set up output directories
            data_dir = temp_path / "data"
            reports_dir = temp_path / "reports"
            data_dir.mkdir()
            reports_dir.mkdir()

            # Run pipeline
            rng = set_global_seed(seed, deterministic_ops=True)

            # Simulate reads
            reads_path = data_dir / "simulated_reads.parquet"
            reads_df = simulate_reads(config, rng, output_path=str(reads_path))

            # Collapse UMIs
            collapsed_path = data_dir / "collapsed_umis.parquet"
            collapsed_df = collapse_umis(reads_df, config, rng, output_path=str(collapsed_path))

            # Fit error model
            error_model_path = data_dir / "error_model.parquet"
            error_model_df = fit_error_model(collapsed_df, config, rng, output_path=str(error_model_path))

            # Call variants
            calls_path = data_dir / "mrd_calls.parquet"
            calls_df = call_mrd(
                collapsed_df, error_model_df, config, rng,
                output_path=str(calls_path),
                use_ml_calling=config_data.get('use_ml_calling', False),
                ml_model_type=config_data.get('ml_model_type', 'ensemble'),
                use_deep_learning=config_data.get('use_deep_learning', False),
                dl_model_type=config_data.get('dl_model_type', 'cnn_lstm')
            )

            # Calculate metrics
            metrics = calculate_metrics(calls_df, rng)

            # Create results summary
            results = {
                'job_id': run_id,
                'status': 'completed',
                'metrics': {
                    'n_samples': len(reads_df),
                    'n_variants': len(calls_df[calls_df['is_variant']]),
                    'variant_rate': len(calls_df[calls_df['is_variant']]) / len(calls_df),
                    'mean_allele_fraction': calls_df['allele_fraction'].mean() if 'allele_fraction' in calls_df.columns else 0
                },
                'processing_time_seconds': context.get_remaining_time_in_millis() / 1000 if hasattr(context, 'get_remaining_time_in_millis') else None,
                'memory_used_mb': context.memory_limit_in_mb if hasattr(context, 'memory_limit_in_mb') else None
            }

            return {
                'statusCode': 200,
                'body': json.dumps(results, default=str),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'job_id': run_id if 'run_id' in locals() else 'unknown'
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }


def create_deployment_package() -> bytes:
    """Create deployment package for AWS Lambda."""
    # This would package the precise_mrd module and dependencies
    # For production, you'd use a proper packaging tool like aws-sam-cli

    # Create a simple zip file with the function
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add the lambda function
        zip_file.writestr('lambda_function.py', __import__('__main__').__file__)

        # Add a simple requirements.txt
        requirements = """
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.2
"""
        zip_file.writestr('requirements.txt', requirements)

    buffer.seek(0)
    return buffer.read()


# Example usage for testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'config': {
            'run_id': 'test_lambda',
            'seed': 42,
            'use_ml_calling': True,
            'ml_model_type': 'ensemble'
        }
    }

    # Mock context
    class MockContext:
        def __init__(self):
            self.function_name = 'test-function'
            self.memory_limit_in_mb = 512
            self.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
            self.aws_request_id = str(uuid.uuid4())

    result = lambda_handler(test_event, MockContext())
    print("Lambda function test result:")
    print(json.dumps(json.loads(result['body']), indent=2))

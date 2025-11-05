"""AWS Lambda function for serverless MRD pipeline processing."""

import json
import os
import boto3
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

batch_client = boto3.client("batch")

def lambda_handler(event, context):
    """
    AWS Lambda function to trigger an AWS Batch job for the Precise MRD pipeline.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        body = json.loads(event.get("body", "{}"))
        
        # Extract parameters from the request body
        run_id = body.get("run_id", f"api-run-{context.aws_request_id}")
        seed = body.get("seed", 7)
        config_override = body.get("config_override") # This would be the full YAML content as a string
        
        # Construct the command for the Batch job
        command = ["precise-mrd", "smoke", "--seed", str(seed)]
        if config_override:
            # The container entrypoint would need to handle receiving this config
            # For simplicity, we assume a mechanism exists to pass this config
            # A more robust solution might use S3 for config files
            pass

        job_queue = os.environ["JOB_QUEUE_ARN"]
        job_definition = os.environ["JOB_DEFINITION_ARN"]
        
        response = batch_client.submit_job(
            jobName=run_id.replace("_", "-"),
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides={
                "command": command
            }
        )
        
        job_id = response["jobId"]
        logger.info(f"Submitted job {job_id} to AWS Batch.")
        
        return {
            "statusCode": 202,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Pipeline job submitted successfully.",
                "jobId": job_id,
                "run_id": run_id
            }),
        }

    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Failed to submit pipeline job."}),
        }











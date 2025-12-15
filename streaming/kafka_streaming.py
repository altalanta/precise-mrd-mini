"""Real-time streaming pipeline for continuous MRD processing using Apache Kafka."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

# Kafka imports with fallbacks
try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from ..call import call_mrd
from ..collapse import collapse_umis
from ..config import PipelineConfig
from ..determinism_utils import set_global_seed
from ..error_model import fit_error_model
from ..metrics import calculate_metrics
from ..simulate import simulate_reads


class StreamingMRDPipeline:
    """Real-time streaming MRD pipeline using Apache Kafka."""

    def __init__(
        self,
        kafka_bootstrap_servers: list[str] = None,
        input_topic: str = "mrd-input",
        output_topic: str = "mrd-output",
        config: PipelineConfig | None = None,
    ):
        """Initialize streaming pipeline.

        Args:
            kafka_bootstrap_servers: List of Kafka bootstrap servers
            input_topic: Kafka topic for input data
            output_topic: Kafka topic for output results
            config: Pipeline configuration
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or ["localhost:9092"]
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.config = config

        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required for streaming pipeline")

        # Initialize Kafka producer and consumer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k else None,
        )

        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=f"mrd-pipeline-{uuid.uuid4().hex[:8]}",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def process_streaming_data(self):
        """Process data from Kafka stream."""
        self.logger.info(
            f"Starting streaming pipeline. Listening to topic: {self.input_topic}",
        )

        try:
            for message in self.consumer:
                try:
                    # Parse input data
                    input_data = message.value
                    self.logger.info(f"Received message with key: {message.key}")

                    # Process the data
                    result = await self._process_single_sample(input_data)

                    # Send result to output topic
                    self.producer.send(self.output_topic, key=message.key, value=result)

                    self.logger.info(
                        f"Processed and sent result for key: {message.key}",
                    )

                except Exception as e:
                    self.logger.error(f"Error processing message {message.key}: {e}")

                    # Send error to a dead letter queue or error topic
                    error_result = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "input_key": str(message.key) if message.key else None,
                    }

                    self.producer.send(
                        f"{self.output_topic}-errors",
                        key=message.key,
                        value=error_result,
                    )

        except KeyboardInterrupt:
            self.logger.info("Streaming pipeline stopped by user")
        except Exception as e:
            self.logger.error(f"Streaming pipeline error: {e}")
        finally:
            self.producer.close()
            self.consumer.close()

    async def _process_single_sample(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a single sample from the stream.

        Args:
            input_data: Input data containing sample information

        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()

        try:
            # Extract or generate configuration for this sample
            if "config" in input_data:
                # Use provided configuration
                config_dict = input_data["config"]
                config = PipelineConfig(**config_dict)
            else:
                # Use default configuration
                config = PipelineConfig(
                    run_id=f"stream_{uuid.uuid4().hex[:8]}",
                    seed=input_data.get("seed", 7),
                    simulation=None,
                    umi=None,
                    stats=None,
                    lod=None,
                )

            # Extract sample data
            sample_data = input_data.get("sample_data", {})

            # Generate or use provided reads
            if "reads" in sample_data:
                # Use provided reads data
                reads_df = pd.DataFrame(sample_data["reads"])
            else:
                # Generate synthetic reads for this sample
                rng = set_global_seed(config.seed, deterministic_ops=True)
                reads_df = simulate_reads(config, rng)

            # Process through pipeline
            rng = set_global_seed(config.seed, deterministic_ops=True)

            # Collapse UMIs
            collapsed_df = collapse_umis(reads_df, config, rng)

            # Fit error model
            error_model_df = fit_error_model(collapsed_df, config, rng)

            # Call variants
            calls_df = call_mrd(
                collapsed_df,
                error_model_df,
                config,
                rng,
                use_ml_calling=input_data.get("use_ml_calling", False),
                ml_model_type=input_data.get("ml_model_type", "ensemble"),
                use_deep_learning=input_data.get("use_deep_learning", False),
                dl_model_type=input_data.get("dl_model_type", "cnn_lstm"),
            )

            # Calculate metrics
            calculate_metrics(calls_df, rng)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = {
                "status": "completed",
                "processing_time_seconds": processing_time,
                "sample_id": input_data.get("sample_id", "unknown"),
                "run_id": config.run_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "n_samples": len(reads_df),
                    "n_variants_detected": len(calls_df[calls_df["is_variant"]]),
                    "variant_detection_rate": len(calls_df[calls_df["is_variant"]])
                    / len(calls_df)
                    if len(calls_df) > 0
                    else 0,
                    "mean_allele_fraction": calls_df["allele_fraction"].mean()
                    if "allele_fraction" in calls_df.columns
                    else 0,
                },
                "config_summary": {
                    "use_ml_calling": input_data.get("use_ml_calling", False),
                    "use_deep_learning": input_data.get("use_deep_learning", False),
                    "ml_model_type": input_data.get("ml_model_type", "ensemble"),
                    "dl_model_type": input_data.get("dl_model_type", "cnn_lstm"),
                },
            }

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "status": "error",
                "error": str(e),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "sample_id": input_data.get("sample_id", "unknown"),
            }

    def send_sample_for_processing(
        self,
        sample_id: str,
        sample_data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ):
        """Send a sample for asynchronous processing.

        Args:
            sample_id: Unique identifier for the sample
            sample_data: Sample data and metadata
            config: Optional pipeline configuration overrides
        """
        message = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "sample_data": sample_data,
            "config": config or {},
        }

        # Send to input topic
        future = self.producer.send(self.input_topic, key=sample_id, value=message)

        # Ensure message is sent
        try:
            future.get(timeout=10)
            self.logger.info(f"Sample {sample_id} sent for processing")
        except KafkaError as e:
            self.logger.error(f"Failed to send sample {sample_id}: {e}")
            raise

    def get_processing_results(
        self,
        sample_id: str,
        timeout_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Get processing results for a specific sample.

        Args:
            sample_id: Sample identifier
            timeout_seconds: Maximum time to wait for results

        Returns:
            Processing results or None if timeout
        """
        # Create a consumer for this specific sample
        result_consumer = KafkaConsumer(
            self.output_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id=f"mrd-result-{sample_id}-{uuid.uuid4().hex[:8]}",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=timeout_seconds * 1000,
        )

        try:
            for message in result_consumer:
                if message.key == sample_id:
                    result_consumer.close()
                    return message.value
        except Exception:
            pass
        finally:
            result_consumer.close()

        return None

    def close(self):
        """Close Kafka connections."""
        self.producer.close()
        self.consumer.close()


class BatchStreamingPipeline:
    """Batch processing with streaming output for high-throughput scenarios."""

    def __init__(
        self,
        kafka_bootstrap_servers: list[str] = None,
        batch_size: int = 100,
        output_topic: str = "mrd-batch-output",
    ):
        """Initialize batch streaming pipeline.

        Args:
            kafka_bootstrap_servers: List of Kafka bootstrap servers
            batch_size: Number of samples to process in each batch
            output_topic: Kafka topic for batch results
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or ["localhost:9092"]
        self.batch_size = batch_size
        self.output_topic = output_topic

        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required for batch streaming pipeline")

        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        self.logger = logging.getLogger(__name__)

    async def process_batch(
        self,
        samples: list[dict[str, Any]],
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Process a batch of samples.

        Args:
            samples: List of sample data
            config: Pipeline configuration

        Returns:
            Dictionary with batch processing results
        """
        start_time = datetime.now()
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        self.logger.info(f"Processing batch {batch_id} with {len(samples)} samples")

        results = {
            "batch_id": batch_id,
            "status": "processing",
            "sample_results": [],
            "batch_metrics": {},
            "start_time": start_time.isoformat(),
            "end_time": None,
            "total_processing_time": None,
        }

        try:
            rng = set_global_seed(config.seed, deterministic_ops=True)

            # Process samples in parallel or sequentially
            # For simplicity, processing sequentially here
            # In production, you'd use multiprocessing or async processing

            for i, sample_data in enumerate(samples):
                sample_start = datetime.now()

                try:
                    # Generate reads for this sample
                    sample_config = config
                    if "custom_config" in sample_data:
                        # Merge custom config with base config
                        sample_config = self._merge_configs(
                            config,
                            sample_data["custom_config"],
                        )

                    reads_df = simulate_reads(sample_config, rng)

                    # Process through pipeline
                    collapsed_df = collapse_umis(reads_df, sample_config, rng)
                    error_model_df = fit_error_model(collapsed_df, sample_config, rng)

                    calls_df = call_mrd(
                        collapsed_df,
                        error_model_df,
                        sample_config,
                        rng,
                        use_ml_calling=sample_data.get("use_ml_calling", False),
                        ml_model_type=sample_data.get("ml_model_type", "ensemble"),
                        use_deep_learning=sample_data.get("use_deep_learning", False),
                        dl_model_type=sample_data.get("dl_model_type", "cnn_lstm"),
                    )

                    sample_processing_time = (
                        datetime.now() - sample_start
                    ).total_seconds()

                    sample_result = {
                        "sample_id": sample_data.get("sample_id", f"sample_{i}"),
                        "status": "completed",
                        "processing_time_seconds": sample_processing_time,
                        "n_variants_detected": len(calls_df[calls_df["is_variant"]]),
                        "variant_rate": len(calls_df[calls_df["is_variant"]])
                        / len(calls_df)
                        if len(calls_df) > 0
                        else 0,
                    }

                    results["sample_results"].append(sample_result)

                except Exception as e:
                    sample_processing_time = (
                        datetime.now() - sample_start
                    ).total_seconds()

                    sample_result = {
                        "sample_id": sample_data.get("sample_id", f"sample_{i}"),
                        "status": "error",
                        "error": str(e),
                        "processing_time_seconds": sample_processing_time,
                    }

                    results["sample_results"].append(sample_result)

            # Calculate batch metrics
            completed_samples = [
                r for r in results["sample_results"] if r["status"] == "completed"
            ]
            total_samples = len(results["sample_results"])

            if completed_samples:
                results["batch_metrics"] = {
                    "total_samples": total_samples,
                    "successful_samples": len(completed_samples),
                    "failed_samples": total_samples - len(completed_samples),
                    "success_rate": len(completed_samples) / total_samples,
                    "mean_processing_time": sum(
                        r["processing_time_seconds"] for r in completed_samples
                    )
                    / len(completed_samples),
                    "total_variants_detected": sum(
                        r["n_variants_detected"] for r in completed_samples
                    ),
                    "mean_variant_rate": sum(
                        r["variant_rate"] for r in completed_samples
                    )
                    / len(completed_samples),
                }

            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["total_processing_time"] = (end_time - start_time).total_seconds()
            results["status"] = "completed"

            # Send results to Kafka
            self.producer.send(self.output_topic, key=batch_id, value=results)

            self.logger.info(
                f"Batch {batch_id} completed in {results['total_processing_time']:.2f}s",
            )

            return results

        except Exception as e:
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["total_processing_time"] = (end_time - start_time).total_seconds()
            results["status"] = "error"
            results["error"] = str(e)

            # Send error results to Kafka
            self.producer.send(self.output_topic, key=batch_id, value=results)

            raise

    def _merge_configs(
        self,
        base_config: PipelineConfig,
        custom_config: dict[str, Any],
    ) -> PipelineConfig:
        """Merge base configuration with custom overrides."""
        # Simplified config merging - in production, you'd want more sophisticated merging
        merged_dict = base_config.to_dict()
        merged_dict.update(custom_config)
        return PipelineConfig(**merged_dict)

    def close(self):
        """Close Kafka connections."""
        self.producer.close()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example: Run streaming pipeline
    pipeline = StreamingMRDPipeline(
        kafka_bootstrap_servers=["localhost:9092"],
        input_topic="mrd-input",
        output_topic="mrd-output",
    )

    # Run the streaming pipeline
    try:
        asyncio.run(pipeline.process_streaming_data())
    except KeyboardInterrupt:
        print("Streaming pipeline stopped")
    finally:
        pipeline.close()

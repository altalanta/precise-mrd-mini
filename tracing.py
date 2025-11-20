"""
OpenTelemetry configuration for distributed tracing.

This module provides a centralized setup for OpenTelemetry, enabling
end-to-end tracing of requests as they flow from the FastAPI API
through the Celery task queue and into the pipeline execution.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor

from .logging_config import get_logger

log = get_logger(__name__)

def setup_telemetry():
    """
    Configures OpenTelemetry for the application.
    
    Initializes a tracer provider, a console span exporter for local debugging,
    and automatically instruments FastAPI and Celery.
    """
    
    # Create a resource to identify the service name
    resource = Resource(attributes={
        "service.name": "precise-mrd-pipeline"
    })

    # Set up a tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Use a console exporter for now to print traces to the console
    # In production, this would be replaced with an OTLP exporter (e.g., to Jaeger, Zipkin)
    console_exporter = ConsoleSpanExporter()
    
    # Use a BatchSpanProcessor to send spans in batches
    span_processor = BatchSpanProcessor(console_exporter)
    
    tracer_provider.add_span_processor(span_processor)
    
    # Set the global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    log.info("OpenTelemetry configured with ConsoleSpanExporter.")

    # Instrument FastAPI
    # Note: The `instrument_app` call will be done in the API module
    
    # Instrument Celery
    CeleryInstrumentor().instrument()
    log.info("Celery instrumentation complete.")

def instrument_fastapi_app(app):
    """
    Applies FastAPI instrumentation.
    
    This is separated to allow the app instance to be passed in from the main API file.
    """
    FastAPIInstrumentor.instrument_app(app)
    log.info("FastAPI instrumentation complete.")

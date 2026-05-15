"""
OpenTelemetry Tracing — Distributed observability for the agentic pipeline.

Captures spans for each agent step: intent extraction, vector retrieval,
reranking, scoring, booking, and LLM inference.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("observability.tracing")

_tracer = None
_initialized = False


def init_tracing(service_name: str = "ai-service-orchestrator"):
    """Initialize OpenTelemetry tracing."""
    global _tracer, _initialized
    if _initialized:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Console exporter for development
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        # OTLP exporter for production
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
                logger.info(f"OTLP exporter configured: {otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available.")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        _initialized = True
        logger.info(f"OpenTelemetry tracing initialized: {service_name}")

    except ImportError:
        logger.warning("OpenTelemetry not installed. Tracing disabled.")
        _tracer = None


def get_tracer():
    """Get the global tracer."""
    if not _initialized:
        init_tracing()
    return _tracer


@contextmanager
def trace_span(name: str, attributes: Optional[dict] = None):
    """
    Context manager for creating a traced span.

    Usage:
        with trace_span("intent.extract", {"user_message": msg}) as span:
            result = await intent_agent.extract(msg)
            span.set_attribute("confidence", result["confidence"])
    """
    tracer = get_tracer()
    if tracer:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)
            yield span
    else:
        yield NullSpan()


class NullSpan:
    """No-op span when tracing is disabled."""
    def set_attribute(self, key, value):
        pass
    def add_event(self, name, attributes=None):
        pass
    def set_status(self, status):
        pass

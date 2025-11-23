"""
Metrics collection and monitoring system for RAG components.
Implements resource monitoring, distributed tracing, and Prometheus metrics.
"""

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional, Protocol

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server

OTEL_ENABLED = os.getenv("RAG_tracing_enabled", "false").lower() == "true"
trace: Any = None
TracerProvider: Any = None
BatchSpanProcessor: Any = None
try:
    import opentelemetry.trace as trace  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
except Exception:  # pragma: no cover
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
import logging

log = logging.getLogger("rag.monitoring.metrics")
JaegerExporter: Any = None
try:
    from opentelemetry.exporter.jaeger.thrift import (
        JaegerExporter,
    )  # type: ignore[reportMissingImports]
except ImportError:
    try:
        from opentelemetry.exporter.jaeger import (
            JaegerExporter,
        )  # type: ignore[reportMissingImports]
    except ImportError:
        log.warning("JaegerExporter not available. Jaeger tracing will be disabled.")

# Prometheus metrics
REQUESTS = Counter("rag_requests_total", "Total RAG requests", ["component"])
ERRORS = Counter("rag_errors_total", "Total errors", ["component", "error_type"])
LATENCY = Histogram("rag_latency_seconds", "Request latency", ["component"])
MEMORY = Gauge("rag_memory_usage_bytes", "Memory usage in bytes")
CPU = Gauge("rag_cpu_usage_percent", "CPU usage percentage")
IO_READ = Counter("rag_io_read_bytes", "Total bytes read from disk")
IO_WRITE = Counter("rag_io_write_bytes", "Total bytes written to disk")


# Resource monitoring
@dataclass
class ResourceStats:
    cpu_percent: float
    memory_used: int
    io_read_bytes: int
    io_write_bytes: int


class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._process = psutil.Process()

    def start(self):
        """Start resource monitoring in background thread"""

        def _monitor():
            while not self._stop_event.is_set():
                stats = self._collect_stats()
                self._update_metrics(stats)
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop resource monitoring"""
        if self._thread:
            self._stop_event.set()
            self._thread.join()

    def _collect_stats(self) -> ResourceStats:
        """Collect current resource statistics"""
        io_counters = self._process.io_counters()
        return ResourceStats(
            cpu_percent=self._process.cpu_percent(),
            memory_used=self._process.memory_info().rss,
            io_read_bytes=io_counters.read_bytes,
            io_write_bytes=io_counters.write_bytes,
        )

    def _update_metrics(self, stats: ResourceStats):
        """Update Prometheus metrics with current stats"""
        CPU.set(stats.cpu_percent)
        MEMORY.set(stats.memory_used)
        IO_READ.inc(stats.io_read_bytes)
        IO_WRITE.inc(stats.io_write_bytes)


# Distributed tracing setup
# Distributed tracing setup
class SpanLike(Protocol):
    def set_status(self, status: object) -> None: ...

    def set_attribute(self, name: str, value: object) -> None: ...


def setup_tracing(service_name: str):
    """Configure OpenTelemetry tracing with Jaeger exporter"""
    if not OTEL_ENABLED or trace is None or TracerProvider is None:
        # Tracing disabled or dependencies missing
        from contextlib import contextmanager

        class _DummySpan:
            def set_status(self, status: object) -> None:
                return None

            def set_attribute(self, name: str, value: object) -> None:
                return None

        class _DummyTracer:
            def start_as_current_span(self, *a: object, **k: object):
                @contextmanager
                def _cm() -> Generator[SpanLike, None, None]:
                    yield _DummySpan()

                return _cm()

        return _DummyTracer()

    tracer_provider = TracerProvider()
    if JaegerExporter is not None:
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        except Exception as e:
            log.warning(f"Could not initialize JaegerExporter: {e}")
    else:
        log.info("JaegerExporter not available. Skipping Jaeger tracing setup.")
    trace.set_tracer_provider(tracer_provider)
    return trace.get_tracer(service_name)


# Performance monitoring contexts
@contextmanager
def monitor_operation(
    component: str, operation: str
) -> Generator[SpanLike, None, None]:
    """Context manager for monitoring operation latency and creating trace spans"""
    tracer = (
        trace.get_tracer(__name__)
        if (trace and OTEL_ENABLED)
        else setup_tracing("rag-service")
    )

    with tracer.start_as_current_span(
        f"{component}.{operation}",
        attributes={"component": component, "operation": operation},
    ) as span:
        start_time = time.time()
        try:
            yield span
            REQUESTS.labels(component=component).inc()
        except Exception as e:
            ERRORS.labels(component=component, error_type=type(e).__name__).inc()
            if trace and OTEL_ENABLED:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
            else:
                span.set_status("error")
            raise
        finally:
            duration = time.time() - start_time
            LATENCY.labels(component=component).observe(duration)


# Initialize monitoring
def init_monitoring(
    prometheus_port: Optional[int] = 8000, service_name: str = "rag-service"
):
    """Initialize all monitoring systems"""
    # Start Prometheus metrics server (only if port provided)
    if prometheus_port is not None:
        start_http_server(prometheus_port)

    # Setup tracing
    setup_tracing(service_name)

    # Start resource monitoring
    monitor = ResourceMonitor()
    monitor.start()

    return monitor

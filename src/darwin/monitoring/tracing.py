"""
Darwin Distributed Tracing System

This module provides comprehensive distributed tracing capabilities for the Darwin platform,
including request flow tracking, dependency mapping, span management, trace correlation,
and integration with OpenTelemetry and Logfire.

Features:
- Distributed tracing across services and components
- Automatic span creation and management
- Request correlation and dependency tracking
- Custom instrumentation for genetic algorithms
- Integration with OpenTelemetry standards
- Trace sampling and filtering
- Performance analysis through traces
- Service dependency mapping
"""

import asyncio
import functools
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Span status enumeration."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SpanKind(Enum):
    """Span kind enumeration."""

    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


@dataclass
class SpanAttribute:
    """Individual span attribute."""

    key: str
    value: Union[str, int, float, bool]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"key": self.key, "value": self.value}


@dataclass
class SpanEvent:
    """Span event with timestamp and attributes."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """Individual trace span."""

    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    service_name: str = "darwin-platform"

    # Span data
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    def __post_init__(self):
        """Initialize span after creation."""
        if not self.span_id:
            self.span_id = self._generate_span_id()
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()

    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        return str(uuid.uuid4())[:16]

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        return str(uuid.uuid4()).replace("-", "")

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def set_tag(self, key: str, value: str):
        """Set span tag."""
        self.tags[key] = value

    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to span."""
        event = SpanEvent(name=name, attributes=attributes or {})
        self.events.append(event)

    def set_error(self, error: Exception):
        """Set span error information."""
        self.status = SpanStatus.ERROR
        self.error_message = str(error)
        self.error_type = type(error).__name__

        # Capture stack trace
        import traceback

        self.stack_trace = traceback.format_exc()

    def finish(self):
        """Finish the span."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def is_finished(self) -> bool:
        """Check if span is finished."""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary format."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "kind": self.kind.value,
            "service_name": self.service_name,
            "attributes": self.attributes,
            "events": [event.to_dict() for event in self.events],
            "tags": self.tags,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "has_error": self.status == SpanStatus.ERROR,
        }

    def to_jaeger_format(self) -> Dict[str, Any]:
        """Convert span to Jaeger format."""
        return {
            "traceID": self.trace_id,
            "spanID": self.span_id,
            "parentSpanID": self.parent_span_id,
            "operationName": self.operation_name,
            "startTime": int(self.start_time.timestamp() * 1000000),  # microseconds
            "duration": int(self.duration_ms * 1000)
            if self.duration_ms
            else 0,  # microseconds
            "tags": [{"key": k, "value": v} for k, v in self.tags.items()],
            "process": {"serviceName": self.service_name, "tags": []},
            "logs": [
                {
                    "timestamp": int(event.timestamp.timestamp() * 1000000),
                    "fields": [
                        {"key": k, "value": v} for k, v in event.attributes.items()
                    ],
                }
                for event in self.events
            ],
        }


@dataclass
class Trace:
    """Complete trace with multiple spans."""

    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    service_names: List[str] = field(default_factory=list)

    def add_span(self, span: Span):
        """Add span to trace."""
        if span.trace_id != self.trace_id:
            span.trace_id = self.trace_id

        self.spans.append(span)

        # Update trace metadata
        if span.service_name not in self.service_names:
            self.service_names.append(span.service_name)

        if self.start_time is None or span.start_time < self.start_time:
            self.start_time = span.start_time

        if span.end_time:
            if self.end_time is None or span.end_time > self.end_time:
                self.end_time = span.end_time

        self._calculate_duration()

    def _calculate_duration(self):
        """Calculate total trace duration."""
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def get_root_span(self) -> Optional[Span]:
        """Get the root span of the trace."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None

    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_spans_by_service(self, service_name: str) -> List[Span]:
        """Get spans by service name."""
        return [span for span in self.spans if span.service_name == service_name]

    def has_errors(self) -> bool:
        """Check if trace has any errors."""
        return any(span.status == SpanStatus.ERROR for span in self.spans)

    def get_error_spans(self) -> List[Span]:
        """Get all error spans in trace."""
        return [span for span in self.spans if span.status == SpanStatus.ERROR]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary format."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "service_names": self.service_names,
            "span_count": len(self.spans),
            "has_errors": self.has_errors(),
            "error_count": len(self.get_error_spans()),
            "spans": [span.to_dict() for span in self.spans],
        }


class TraceContext:
    """Thread-local trace context management."""

    def __init__(self):
        """Initialize trace context."""
        self._context = {}
        self._span_stack = []
        self._lock = Lock()

    def set_current_span(self, span: Span):
        """Set current active span."""
        with self._lock:
            self._context["current_span"] = span
            if span not in self._span_stack:
                self._span_stack.append(span)

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._context.get("current_span")

    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        current_span = self.get_current_span()
        return current_span.trace_id if current_span else None

    def pop_span(self) -> Optional[Span]:
        """Pop span from stack."""
        with self._lock:
            if self._span_stack:
                span = self._span_stack.pop()
                # Set current span to parent if available
                if self._span_stack:
                    self._context["current_span"] = self._span_stack[-1]
                else:
                    self._context.pop("current_span", None)
                return span
        return None

    def clear(self):
        """Clear trace context."""
        with self._lock:
            self._context.clear()
            self._span_stack.clear()


class SpanManager:
    """Span lifecycle management."""

    def __init__(self, tracer):
        """Initialize span manager."""
        self.tracer = tracer
        self.active_spans: Dict[str, Span] = {}
        self._lock = Lock()

    def create_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ) -> Span:
        """Create a new span."""
        # Determine parent and trace ID
        parent_span_id = None
        trace_id = str(uuid.uuid4()).replace("-", "")

        if parent_span:
            parent_span_id = parent_span.span_id
            trace_id = parent_span.trace_id
        else:
            # Check for current span in context
            current_span = self.tracer.context.get_current_span()
            if current_span:
                parent_span_id = current_span.span_id
                trace_id = current_span.trace_id

        span = Span(
            span_id="",  # Will be generated
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            kind=kind,
            service_name=self.tracer.service_name,
        )

        # Set attributes and tags
        if attributes:
            span.attributes.update(attributes)
        if tags:
            span.tags.update(tags)

        # Store active span
        with self._lock:
            self.active_spans[span.span_id] = span

        return span

    def finish_span(self, span: Span):
        """Finish a span."""
        span.finish()

        # Remove from active spans
        with self._lock:
            self.active_spans.pop(span.span_id, None)

        # Send to tracer
        self.tracer.record_span(span)

    def get_active_span_count(self) -> int:
        """Get number of active spans."""
        return len(self.active_spans)

    def get_active_spans(self) -> List[Span]:
        """Get all active spans."""
        return list(self.active_spans.values())


class TracingManager:
    """Main distributed tracing manager."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize tracing manager.

        Args:
            config: Tracing configuration
        """
        self.config = config or {}
        self.service_name = self.config.get("service_name", "darwin-platform")
        self.enable_distributed_tracing = self.config.get(
            "enable_distributed_tracing", True
        )
        self.trace_optimization_runs = self.config.get("trace_optimization_runs", True)
        self.trace_api_requests = self.config.get("trace_api_requests", True)
        self.trace_database_queries = self.config.get("trace_database_queries", True)
        self.sampling_rate = self.config.get("sampling_rate", 1.0)
        self.max_traces = self.config.get("max_traces", 1000)

        self.context = TraceContext()
        self.span_manager = SpanManager(self)
        self.traces: Dict[str, Trace] = {}
        self.completed_traces: List[Trace] = []
        self._lock = Lock()

        # Integration with external systems
        self.logfire_manager = None

        logger.info(f"Tracing manager initialized for service: {self.service_name}")

    def set_logfire_manager(self, logfire_manager):
        """Set Logfire manager for integration."""
        self.logfire_manager = logfire_manager

    def should_sample_trace(self) -> bool:
        """Determine if trace should be sampled."""
        import random

        return random.random() < self.sampling_rate

    @contextmanager
    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ):
        """Context manager for creating spans."""
        if not self.enable_distributed_tracing or not self.should_sample_trace():
            yield None
            return

        span = self.span_manager.create_span(
            operation_name=operation_name,
            parent_span=parent_span,
            kind=kind,
            attributes=attributes,
            tags=tags,
        )

        # Set as current span
        self.context.set_current_span(span)

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            # Pop span from context
            self.context.pop_span()

            # Finish span
            self.span_manager.finish_span(span)

    @asynccontextmanager
    async def start_async_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ):
        """Async context manager for creating spans."""
        if not self.enable_distributed_tracing or not self.should_sample_trace():
            yield None
            return

        span = self.span_manager.create_span(
            operation_name=operation_name,
            parent_span=parent_span,
            kind=kind,
            attributes=attributes,
            tags=tags,
        )

        # Set as current span
        self.context.set_current_span(span)

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            # Pop span from context
            self.context.pop_span()

            # Finish span
            self.span_manager.finish_span(span)

    def trace_function(
        self,
        operation_name: str = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ):
        """Decorator for tracing functions."""

        def decorator(func):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.start_async_span(
                        operation_name=span_name,
                        kind=kind,
                        attributes=attributes,
                        tags=tags,
                    ) as span:
                        if span:
                            span.set_attribute("function.name", func.__name__)
                            span.set_attribute("function.module", func.__module__)
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.start_span(
                        operation_name=span_name,
                        kind=kind,
                        attributes=attributes,
                        tags=tags,
                    ) as span:
                        if span:
                            span.set_attribute("function.name", func.__name__)
                            span.set_attribute("function.module", func.__module__)
                        return func(*args, **kwargs)

                return sync_wrapper

        return decorator

    def record_span(self, span: Span):
        """Record a completed span."""
        with self._lock:
            # Add span to trace
            if span.trace_id not in self.traces:
                self.traces[span.trace_id] = Trace(trace_id=span.trace_id)

            self.traces[span.trace_id].add_span(span)

            # Check if trace is complete (no active spans for this trace)
            trace = self.traces[span.trace_id]
            active_spans_for_trace = [
                s
                for s in self.span_manager.get_active_spans()
                if s.trace_id == span.trace_id
            ]

            if not active_spans_for_trace:
                # Trace is complete
                self.completed_traces.append(trace)
                del self.traces[span.trace_id]

                # Trim completed traces if needed
                if len(self.completed_traces) > self.max_traces:
                    self.completed_traces = self.completed_traces[-self.max_traces :]

        # Send to Logfire if configured
        if self.logfire_manager:
            self._send_span_to_logfire(span)

    def _send_span_to_logfire(self, span: Span):
        """Send span to Logfire for logging."""
        try:
            span_data = span.to_dict()

            if span.status == SpanStatus.ERROR:
                self.logfire_manager.log_error(
                    error=Exception(span.error_message or "Unknown error"),
                    context={
                        "span_id": span.span_id,
                        "trace_id": span.trace_id,
                        "operation_name": span.operation_name,
                        "duration_ms": span.duration_ms,
                        "attributes": span.attributes,
                    },
                    severity="error",
                )
            else:
                self.logfire_manager.log_performance_metric(
                    metric_name="span_duration",
                    value=span.duration_ms or 0,
                    unit="milliseconds",
                    tags={
                        "span_id": span.span_id,
                        "trace_id": span.trace_id,
                        "operation_name": span.operation_name,
                        "service_name": span.service_name,
                        "span_kind": span.kind.value,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to send span to Logfire: {e}")

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        # Check active traces
        if trace_id in self.traces:
            return self.traces[trace_id]

        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace

        return None

    def get_traces_by_service(self, service_name: str) -> List[Trace]:
        """Get traces by service name."""
        matching_traces = []

        # Check active traces
        for trace in self.traces.values():
            if service_name in trace.service_names:
                matching_traces.append(trace)

        # Check completed traces
        for trace in self.completed_traces:
            if service_name in trace.service_names:
                matching_traces.append(trace)

        return matching_traces

    def get_recent_traces(self, limit: int = 50) -> List[Trace]:
        """Get recent traces."""
        return self.completed_traces[-limit:]

    def get_error_traces(self, limit: int = 50) -> List[Trace]:
        """Get traces with errors."""
        error_traces = [trace for trace in self.completed_traces if trace.has_errors()]
        return error_traces[-limit:]

    def get_slow_traces(
        self, threshold_ms: float = 1000, limit: int = 50
    ) -> List[Trace]:
        """Get slow traces above threshold."""
        slow_traces = [
            trace
            for trace in self.completed_traces
            if trace.duration_ms and trace.duration_ms > threshold_ms
        ]
        return sorted(slow_traces, key=lambda t: t.duration_ms or 0, reverse=True)[
            :limit
        ]

    def get_tracing_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        total_traces = len(self.traces) + len(self.completed_traces)
        error_traces = len(self.get_error_traces(1000))

        # Calculate average duration
        completed_with_duration = [t for t in self.completed_traces if t.duration_ms]
        avg_duration = 0
        if completed_with_duration:
            avg_duration = sum(t.duration_ms for t in completed_with_duration) / len(
                completed_with_duration
            )

        return {
            "total_traces": total_traces,
            "active_traces": len(self.traces),
            "completed_traces": len(self.completed_traces),
            "error_traces": error_traces,
            "error_rate": (error_traces / len(self.completed_traces))
            if self.completed_traces
            else 0,
            "active_spans": self.span_manager.get_active_span_count(),
            "average_duration_ms": avg_duration,
            "sampling_rate": self.sampling_rate,
            "service_name": self.service_name,
        }

    def add_tracing_middleware(self, app):
        """Add tracing middleware to FastAPI app."""
        if not self.trace_api_requests:
            return

        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware

        class TracingMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, tracing_manager):
                super().__init__(app)
                self.tracing_manager = tracing_manager

            async def dispatch(self, request: Request, call_next):
                operation_name = f"HTTP {request.method} {request.url.path}"

                async with self.tracing_manager.start_async_span(
                    operation_name=operation_name,
                    kind=SpanKind.SERVER,
                    attributes={
                        "http.method": request.method,
                        "http.url": str(request.url),
                        "http.scheme": request.url.scheme,
                        "http.host": request.url.hostname,
                        "http.target": request.url.path,
                        "user_agent": request.headers.get("user-agent", ""),
                    },
                    tags={"component": "http", "span.kind": "server"},
                ) as span:
                    try:
                        response = await call_next(request)

                        if span:
                            span.set_attribute("http.status_code", response.status_code)
                            span.set_tag("http.status_code", str(response.status_code))

                            if response.status_code >= 400:
                                span.status = SpanStatus.ERROR
                                span.set_attribute("error", True)

                        return response

                    except Exception as e:
                        if span:
                            span.set_error(e)
                        raise

        app.add_middleware(TracingMiddleware, tracing_manager=self)

        # Add tracing endpoints
        @app.get("/tracing")
        async def get_tracing_stats():
            """Get tracing statistics."""
            return self.get_tracing_statistics()

        @app.get("/tracing/traces")
        async def get_traces(limit: int = 50):
            """Get recent traces."""
            traces = self.get_recent_traces(limit)
            return {
                "traces": [trace.to_dict() for trace in traces],
                "count": len(traces),
            }

        @app.get("/tracing/traces/{trace_id}")
        async def get_trace(trace_id: str):
            """Get specific trace."""
            trace = self.get_trace(trace_id)
            if not trace:
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Trace not found")
            return trace.to_dict()

        @app.get("/tracing/errors")
        async def get_error_traces(limit: int = 50):
            """Get traces with errors."""
            error_traces = self.get_error_traces(limit)
            return {
                "traces": [trace.to_dict() for trace in error_traces],
                "count": len(error_traces),
            }

        @app.get("/tracing/slow")
        async def get_slow_traces(threshold_ms: float = 1000, limit: int = 50):
            """Get slow traces."""
            slow_traces = self.get_slow_traces(threshold_ms, limit)
            return {
                "traces": [trace.to_dict() for trace in slow_traces],
                "count": len(slow_traces),
                "threshold_ms": threshold_ms,
            }

        logger.info("Tracing middleware and endpoints added to FastAPI application")

    def cleanup_old_traces(self, hours: int = 24):
        """Clean up old traces."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            self.completed_traces = [
                trace
                for trace in self.completed_traces
                if trace.start_time and trace.start_time > cutoff_time
            ]

        logger.info(f"Cleaned up traces older than {hours} hours")

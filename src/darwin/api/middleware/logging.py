"""
Darwin API LogFire Middleware

This module implements comprehensive logging middleware for the Darwin FastAPI application
using LogFire for structured logging, request/response tracking, and performance monitoring.
"""

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import logfire
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LogFireMiddleware(BaseHTTPMiddleware):
    """
    LogFire middleware for comprehensive API request/response logging.

    Features:
    - Request/response logging with correlation IDs
    - Performance metrics collection
    - Error tracking and reporting
    - Structured logging format
    - Request size and response time tracking
    """

    def __init__(
        self,
        app,
        exclude_paths: Optional[list] = None,
        track_performance_metrics: bool = True,
        correlation_id_header: str = "X-Request-ID",
        max_body_size: int = 10000,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        ]
        self.track_performance_metrics = track_performance_metrics
        self.correlation_id_header = correlation_id_header
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request with comprehensive logging."""

        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Generate correlation ID
        correlation_id = request.headers.get(self.correlation_id_header) or str(
            uuid.uuid4()
        )

        # Start timing
        start_time = time.time()

        # Log request start
        await self._log_request_start(request, correlation_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log successful response
            await self._log_response_success(
                request, response, correlation_id, process_time
            )

            # Add correlation ID to response headers
            response.headers[self.correlation_id_header] = correlation_id
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as exc:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time

            # Log exception
            await self._log_response_error(request, exc, correlation_id, process_time)

            # Re-raise the exception
            raise exc

    async def _log_request_start(self, request: Request, correlation_id: str):
        """Log request start with comprehensive details."""

        # Extract client information
        client_host = (
            getattr(request.client, "host", "unknown") if request.client else "unknown"
        )
        client_port = getattr(request.client, "port", 0) if request.client else 0

        # Extract user agent and other headers
        user_agent = request.headers.get("user-agent", "unknown")
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "0")

        # Get query parameters
        query_params = dict(request.query_params) if request.query_params else {}

        # Log request body for non-GET requests (with size limit)
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"] and content_type:
            try:
                if (
                    "application/json" in content_type
                    and int(content_length or 0) <= self.max_body_size
                ):
                    # Note: This is a simplified approach. In production, you might want to
                    # implement proper body reading without consuming the stream
                    pass  # Body reading would be implemented here with proper stream handling
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")

        logfire.info(
            "API request started",
            correlation_id=correlation_id,
            request_method=request.method,
            request_url=str(request.url),
            request_path=request.url.path,
            request_query_params=query_params,
            client_host=client_host,
            client_port=client_port,
            user_agent=user_agent,
            content_type=content_type,
            content_length=content_length,
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def _log_response_success(
        self,
        request: Request,
        response: Response,
        correlation_id: str,
        process_time: float,
    ):
        """Log successful response with performance metrics."""

        # Extract response information
        status_code = response.status_code
        response_headers = dict(response.headers)

        # Performance categorization
        performance_category = self._categorize_performance(process_time)

        # Log the successful response
        logfire.info(
            "API request completed successfully",
            correlation_id=correlation_id,
            request_method=request.method,
            request_path=request.url.path,
            response_status_code=status_code,
            process_time_seconds=process_time,
            process_time_ms=process_time * 1000,
            performance_category=performance_category,
            response_headers=response_headers,
            timestamp=datetime.now(UTC).isoformat(),
        )

        # Track performance metrics if enabled
        if self.track_performance_metrics:
            self._track_performance_metrics(
                request.method, request.url.path, status_code, process_time
            )

    async def _log_response_error(
        self,
        request: Request,
        exception: Exception,
        correlation_id: str,
        process_time: float,
    ):
        """Log error response with exception details."""

        # Extract exception information
        exception_type = type(exception).__name__
        exception_message = str(exception)

        # Log the error
        logfire.error(
            "API request failed with exception",
            correlation_id=correlation_id,
            request_method=request.method,
            request_path=request.url.path,
            exception_type=exception_type,
            exception_message=exception_message,
            process_time_seconds=process_time,
            process_time_ms=process_time * 1000,
            timestamp=datetime.now(UTC).isoformat(),
            stack_trace=logfire.format_exception(exception),
        )

        # Track error metrics
        if self.track_performance_metrics:
            self._track_error_metrics(request.method, request.url.path, exception_type)

    def _categorize_performance(self, process_time: float) -> str:
        """Categorize request performance."""
        if process_time < 0.1:
            return "excellent"
        elif process_time < 0.5:
            return "good"
        elif process_time < 1.0:
            return "acceptable"
        elif process_time < 5.0:
            return "slow"
        else:
            return "very_slow"

    def _track_performance_metrics(
        self, method: str, path: str, status_code: int, process_time: float
    ):
        """Track performance metrics with LogFire."""

        # Log performance metrics
        logfire.info(
            "API performance metrics",
            metric_type="performance",
            method=method,
            path=path,
            status_code=status_code,
            response_time_seconds=process_time,
            response_time_ms=process_time * 1000,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _track_error_metrics(self, method: str, path: str, exception_type: str):
        """Track error metrics with LogFire."""

        # Log error metrics
        logfire.info(
            "API error metrics",
            metric_type="error",
            method=method,
            path=path,
            exception_type=exception_type,
            timestamp=datetime.now(UTC).isoformat(),
        )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request context information to all log messages.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add request context to logging context."""

        # Extract request information
        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Set context for LogFire
        with logfire.span("http_request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
            span.set_attribute("correlation_id", correlation_id)

            response = await call_next(request)

            span.set_attribute("http.status_code", response.status_code)

            return response


def setup_logfire_middleware(app: FastAPI):
    """
    Setup LogFire middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
    """

    # Add request context middleware first
    app.add_middleware(RequestContextMiddleware)

    # Add LogFire middleware
    app.add_middleware(
        LogFireMiddleware,
        exclude_paths=["/api/v1/health", "/docs", "/redoc", "/openapi.json"],
        track_performance_metrics=True,
        correlation_id_header="X-Request-ID",
        max_body_size=10000,
    )

    logger.info("LogFire middleware configured successfully")


def log_optimization_event(
    event_type: str,
    optimizer_id: str,
    details: Dict[str, Any],
    correlation_id: Optional[str] = None,
):
    """
    Log optimization-specific events with structured data.

    Args:
        event_type: Type of optimization event
        optimizer_id: Optimizer identifier
        details: Event details
        correlation_id: Optional request correlation ID
    """

    logfire.info(
        f"Optimization event: {event_type}",
        event_type=event_type,
        optimizer_id=optimizer_id,
        correlation_id=correlation_id,
        timestamp=datetime.now(UTC).isoformat(),
        **details,
    )


def log_database_operation(
    operation: str,
    table: str,
    duration_ms: float,
    success: bool,
    error: Optional[str] = None,
):
    """
    Log database operations with performance metrics.

    Args:
        operation: Database operation type (SELECT, INSERT, UPDATE, DELETE)
        table: Table or collection name
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        error: Error message if operation failed
    """

    log_data = {
        "database_operation": operation,
        "table": table,
        "duration_ms": duration_ms,
        "success": success,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if error:
        log_data["error"] = error
        logfire.error("Database operation failed", **log_data)
    else:
        logfire.info("Database operation completed", **log_data)


class PerformanceTracker:
    """
    Helper class for tracking performance metrics across the application.
    """

    def __init__(self):
        self.metrics = {}

    def track_operation(self, operation_name: str, duration: float):
        """Track operation performance."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }

        self.metrics[operation_name]["count"] += 1
        self.metrics[operation_name]["total_time"] += duration
        self.metrics[operation_name]["min_time"] = min(
            self.metrics[operation_name]["min_time"], duration
        )
        self.metrics[operation_name]["max_time"] = max(
            self.metrics[operation_name]["max_time"], duration
        )

        # Log performance metric
        logfire.info(
            "Performance metric recorded",
            operation=operation_name,
            duration_seconds=duration,
            avg_duration=self.metrics[operation_name]["total_time"]
            / self.metrics[operation_name]["count"],
            operation_count=self.metrics[operation_name]["count"],
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()


# Global performance tracker instance
performance_tracker = PerformanceTracker()

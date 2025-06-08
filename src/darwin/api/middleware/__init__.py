"""
Darwin API Middleware Package

This package contains middleware components for the Darwin FastAPI application,
including logging, authentication, rate limiting, and request processing middleware.
"""

from .logging import (
    LogFireMiddleware,
    PerformanceTracker,
    RequestContextMiddleware,
    log_database_operation,
    log_optimization_event,
    performance_tracker,
    setup_logfire_middleware,
)

__all__ = [
    "LogFireMiddleware",
    "RequestContextMiddleware",
    "setup_logfire_middleware",
    "log_optimization_event",
    "log_database_operation",
    "PerformanceTracker",
    "performance_tracker",
]

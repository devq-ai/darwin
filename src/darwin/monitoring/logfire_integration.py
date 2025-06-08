"""
Darwin Logfire Integration Module

This module provides comprehensive Logfire integration for the Darwin genetic algorithm
platform, including automatic instrumentation, custom logging, metrics collection,
distributed tracing, and performance monitoring.

Features:
- Automatic FastAPI instrumentation with Logfire
- Custom span creation and management
- Structured logging with contextual information
- Performance metrics and timing
- Error tracking and exception handling
- Resource utilization monitoring
- Optimization workflow tracing
- Custom dashboard integration
"""

import asyncio
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import logfire
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LogfireManager:
    """Main Logfire integration manager for Darwin platform."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Logfire manager with configuration.
        
        Args:
            config: Logfire configuration dictionary
        """
        self.config = config or {}
        self.service_name = self.config.get("service_name", "darwin-platform")
        self.service_version = self.config.get("service_version", "1.0.0")
        self.environment = self.config.get("environment", "development")
        self.send_to_logfire = self.config.get("send_to_logfire", True)
        self.trace_sample_rate = self.config.get("trace_sample_rate", 1.0)
        self.log_level = self.config.get("log_level", "INFO")
        
        self.is_configured = False
        self.spans = {}
        self.metrics = {}
        
        self._configure_logfire()

    def _configure_logfire(self):
        """Configure Logfire with Darwin-specific settings."""
        try:
            # Configure Logfire
            logfire.configure(
                service_name=self.service_name,
                service_version=self.service_version,
                environment=self.environment,
                send_to_logfire=self.send_to_logfire,
                trace_sample_rate=self.trace_sample_rate,
            )
            
            # Set up custom logging
            logging.basicConfig(level=getattr(logging, self.log_level.upper()))
            
            self.is_configured = True
            logger.info(f"Logfire configured for service: {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to configure Logfire: {e}")
            self.is_configured = False

    def configure_for_fastapi(self, app: FastAPI):
        """
        Configure Logfire for FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        if not self.is_configured:
            logger.warning("Logfire not configured, skipping FastAPI integration")
            return

        try:
            # Instrument FastAPI with Logfire
            logfire.instrument_fastapi(app)
            
            # Add custom middleware for Darwin-specific monitoring
            app.add_middleware(DarwinMonitoringMiddleware, logfire_manager=self)
            
            logger.info("FastAPI instrumented with Logfire")
            
        except Exception as e:
            logger.error(f"Failed to configure Logfire for FastAPI: {e}")

    @contextmanager
    def span(self, name: str, **attributes):
        """
        Create a custom span with Darwin context.
        
        Args:
            name: Span name
            **attributes: Additional span attributes
        """
        span_id = str(uuid4())
        start_time = time.time()
        
        try:
            with logfire.span(name, **attributes) as span:
                self.spans[span_id] = {
                    "name": name,
                    "start_time": start_time,
                    "attributes": attributes,
                    "span": span
                }
                
                yield span
                
        except Exception as e:
            logger.error(f"Error in span '{name}': {e}")
            if span_id in self.spans:
                self.spans[span_id]["span"].set_attribute("error", True)
                self.spans[span_id]["span"].set_attribute("error_message", str(e))
            raise
        finally:
            if span_id in self.spans:
                duration = time.time() - start_time
                self.spans[span_id]["duration"] = duration
                del self.spans[span_id]

    def log_optimization_start(self, optimizer_id: str, config: Dict[str, Any]):
        """
        Log the start of an optimization run.
        
        Args:
            optimizer_id: Unique optimizer identifier
            config: Optimization configuration
        """
        logfire.info(
            "Optimization started",
            optimizer_id=optimizer_id,
            population_size=config.get("population_size"),
            generations=config.get("generations"),
            mutation_rate=config.get("mutation_rate"),
            crossover_rate=config.get("crossover_rate"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def log_optimization_progress(
        self, 
        optimizer_id: str, 
        generation: int, 
        best_fitness: float,
        average_fitness: float,
        diversity: float = None
    ):
        """
        Log optimization progress.
        
        Args:
            optimizer_id: Unique optimizer identifier
            generation: Current generation number
            best_fitness: Best fitness value
            average_fitness: Average population fitness
            diversity: Population diversity metric
        """
        logfire.info(
            "Optimization progress",
            optimizer_id=optimizer_id,
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            diversity=diversity,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def log_optimization_complete(
        self, 
        optimizer_id: str, 
        final_fitness: float,
        total_generations: int,
        duration: float,
        success: bool = True
    ):
        """
        Log optimization completion.
        
        Args:
            optimizer_id: Unique optimizer identifier
            final_fitness: Final best fitness value
            total_generations: Total generations completed
            duration: Optimization duration in seconds
            success: Whether optimization completed successfully
        """
        logfire.info(
            "Optimization completed",
            optimizer_id=optimizer_id,
            final_fitness=final_fitness,
            total_generations=total_generations,
            duration=duration,
            success=success,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def log_api_request(
        self, 
        method: str, 
        path: str, 
        status_code: int,
        duration: float,
        user_id: str = None
    ):
        """
        Log API request details.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            user_id: Optional user identifier
        """
        logfire.info(
            "API request",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def log_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        severity: str = "error"
    ):
        """
        Log error with context information.
        
        Args:
            error: Exception object
            context: Additional context information
            severity: Error severity level
        """
        error_context = context or {}
        error_context.update({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        if severity == "critical":
            logfire.error("Critical error occurred", **error_context)
        elif severity == "warning":
            logfire.warn("Warning occurred", **error_context)
        else:
            logfire.error("Error occurred", **error_context)

    def log_performance_metric(
        self, 
        metric_name: str, 
        value: float,
        unit: str = None,
        tags: Dict[str, str] = None
    ):
        """
        Log performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Optional unit of measurement
            tags: Optional tags for categorization
        """
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if tags:
            metric_data.update(tags)
        
        logfire.info("Performance metric", **metric_data)

    def log_system_health(self, health_status: Dict[str, Any]):
        """
        Log system health status.
        
        Args:
            health_status: Dictionary containing health check results
        """
        logfire.info(
            "System health check",
            overall_status=health_status.get("status"),
            checks=health_status.get("checks", {}),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def log_websocket_event(
        self, 
        event_type: str, 
        connection_id: str,
        message_type: str = None,
        data_size: int = None
    ):
        """
        Log WebSocket events.
        
        Args:
            event_type: Type of WebSocket event
            connection_id: WebSocket connection identifier
            message_type: Optional message type
            data_size: Optional data size in bytes
        """
        logfire.info(
            "WebSocket event",
            event_type=event_type,
            connection_id=connection_id,
            message_type=message_type,
            data_size=data_size,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def create_dashboard_link(self) -> str:
        """
        Create a link to the Logfire dashboard for this service.
        
        Returns:
            URL to the service dashboard
        """
        base_url = "https://logfire.pydantic.dev"
        return f"{base_url}/services/{self.service_name}"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "is_configured": self.is_configured,
            "active_spans": len(self.spans),
            "dashboard_url": self.create_dashboard_link()
        }


class DarwinMonitoringMiddleware(BaseHTTPMiddleware):
    """Custom middleware for Darwin-specific monitoring."""

    def __init__(self, app, logfire_manager: LogfireManager):
        super().__init__(app)
        self.logfire_manager = logfire_manager

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add Darwin-specific monitoring.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        start_time = time.time()
        request_id = str(uuid4())
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        try:
            with self.logfire_manager.span(
                "darwin_api_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else None
            ):
                response = await call_next(request)
                
                duration = time.time() - start_time
                
                # Log request details
                self.logfire_manager.log_api_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                    user_id=getattr(request.state, "user_id", None)
                )
                
                # Add monitoring headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                
                return response
                
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logfire_manager.log_error(
                error=e,
                context={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": duration
                }
            )
            
            raise


def configure_logfire_monitoring(
    app: FastAPI = None,
    config: Dict[str, Any] = None
) -> LogfireManager:
    """
    Configure Logfire monitoring for Darwin platform.
    
    Args:
        app: Optional FastAPI application to instrument
        config: Optional configuration dictionary
        
    Returns:
        Configured LogfireManager instance
    """
    logfire_manager = LogfireManager(config)
    
    if app:
        logfire_manager.configure_for_fastapi(app)
    
    return logfire_manager


# Context manager for optimization monitoring
@contextmanager
def monitor_optimization(
    logfire_manager: LogfireManager,
    optimizer_id: str,
    config: Dict[str, Any]
):
    """
    Context manager for monitoring optimization runs.
    
    Args:
        logfire_manager: LogfireManager instance
        optimizer_id: Unique optimizer identifier
        config: Optimization configuration
    """
    start_time = time.time()
    
    try:
        logfire_manager.log_optimization_start(optimizer_id, config)
        
        with logfire_manager.span(
            "optimization_run",
            optimizer_id=optimizer_id,
            **config
        ):
            yield
            
        duration = time.time() - start_time
        logfire_manager.log_optimization_complete(
            optimizer_id=optimizer_id,
            final_fitness=0.0,  # This would be set by the calling code
            total_generations=config.get("generations", 0),
            duration=duration,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logfire_manager.log_error(
            error=e,
            context={
                "optimizer_id": optimizer_id,
                "duration": duration,
                "config": config
            },
            severity="error"
        )
        
        logfire_manager.log_optimization_complete(
            optimizer_id=optimizer_id,
            final_fitness=0.0,
            total_generations=0,
            duration=duration,
            success=False
        )
        
        raise


# Global logfire manager instance
_global_logfire_manager = None


def get_global_logfire_manager() -> LogfireManager:
    """Get the global Logfire manager instance."""
    global _global_logfire_manager
    if _global_logfire_manager is None:
        _global_logfire_manager = LogfireManager()
    return _global_logfire_manager


def set_global_logfire_manager(manager: LogfireManager):
    """Set the global Logfire manager instance."""
    global _global_logfire_manager
    _global_logfire_manager = manager
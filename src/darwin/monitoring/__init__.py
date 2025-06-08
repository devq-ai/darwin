"""
Darwin Monitoring & Observability Package

This package provides comprehensive monitoring and observability capabilities for the Darwin
genetic algorithm optimization platform, including Logfire integration, performance tracking,
health monitoring, metrics collection, and alerting systems.

Features:
- Real-time performance monitoring with Logfire
- System health checks and diagnostics
- Custom metrics collection and aggregation
- Distributed tracing for optimization workflows
- Resource utilization monitoring
- Alert management and notification systems
- Dashboard integration for monitoring visualization
- Export capabilities for monitoring data
"""

from .alerts import AlertManager, AlertRule, AlertSeverity, NotificationChannel
from .health import HealthChecker, HealthStatus, SystemHealth
from .logfire_integration import LogfireManager, configure_logfire_monitoring
from .metrics import MetricsCollector, MetricsAggregator, CustomMetric
from .performance import PerformanceMonitor, PerformanceTracker, BenchmarkSuite
from .tracing import TracingManager, SpanManager, TraceContext
from .utils import MonitoringUtils, TimeSeriesData, ThresholdManager

__version__ = "1.0.0"
__author__ = "DevQ.ai"

__all__ = [
    # Core monitoring
    "LogfireManager",
    "configure_logfire_monitoring",
    # Health monitoring
    "HealthChecker",
    "HealthStatus", 
    "SystemHealth",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceTracker",
    "BenchmarkSuite",
    # Metrics collection
    "MetricsCollector",
    "MetricsAggregator",
    "CustomMetric",
    # Distributed tracing
    "TracingManager",
    "SpanManager",
    "TraceContext",
    # Alert management
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "NotificationChannel",
    # Utilities
    "MonitoringUtils",
    "TimeSeriesData",
    "ThresholdManager",
]

# Default monitoring configuration
DEFAULT_MONITORING_CONFIG = {
    "logfire": {
        "service_name": "darwin-platform",
        "service_version": "1.0.0",
        "environment": "development",
        "send_to_logfire": True,
        "trace_sample_rate": 1.0,
        "log_level": "INFO",
    },
    "metrics": {
        "collection_interval": 30,  # seconds
        "retention_period": 7,      # days
        "aggregation_window": 300,  # seconds
        "export_format": "prometheus",
    },
    "health_checks": {
        "check_interval": 60,       # seconds
        "timeout": 10,              # seconds
        "critical_services": [
            "database",
            "api",
            "mcp_server",
            "websocket"
        ],
    },
    "performance": {
        "enable_profiling": True,
        "memory_threshold": 0.8,    # 80% memory usage
        "cpu_threshold": 0.9,       # 90% CPU usage
        "response_time_threshold": 2.0,  # seconds
    },
    "alerts": {
        "enable_notifications": True,
        "notification_channels": ["email", "slack"],
        "alert_cooldown": 300,      # seconds
        "escalation_timeout": 900,  # seconds
    },
    "tracing": {
        "enable_distributed_tracing": True,
        "trace_optimization_runs": True,
        "trace_api_requests": True,
        "trace_database_queries": True,
    }
}


def get_version():
    """Get the current version of the monitoring package."""
    return __version__


def create_monitoring_system(config: dict = None):
    """
    Factory function to create a configured monitoring system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured monitoring system components
    """
    monitoring_config = DEFAULT_MONITORING_CONFIG.copy()
    if config:
        monitoring_config.update(config)
    
    # Initialize core components
    logfire_manager = LogfireManager(config=monitoring_config["logfire"])
    health_checker = HealthChecker(config=monitoring_config["health_checks"])
    metrics_collector = MetricsCollector(config=monitoring_config["metrics"])
    performance_monitor = PerformanceMonitor(config=monitoring_config["performance"])
    alert_manager = AlertManager(config=monitoring_config["alerts"])
    tracing_manager = TracingManager(config=monitoring_config["tracing"])
    
    return {
        "logfire": logfire_manager,
        "health": health_checker,
        "metrics": metrics_collector,
        "performance": performance_monitor,
        "alerts": alert_manager,
        "tracing": tracing_manager,
        "config": monitoring_config
    }


def setup_monitoring_middleware(app, config: dict = None):
    """
    Set up monitoring middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Optional monitoring configuration
    """
    monitoring_system = create_monitoring_system(config)
    
    # Configure Logfire for the application
    monitoring_system["logfire"].configure_for_fastapi(app)
    
    # Add health check endpoints
    monitoring_system["health"].add_health_endpoints(app)
    
    # Add metrics collection middleware
    monitoring_system["metrics"].add_metrics_middleware(app)
    
    # Add performance monitoring
    monitoring_system["performance"].add_performance_middleware(app)
    
    # Add tracing middleware
    monitoring_system["tracing"].add_tracing_middleware(app)
    
    return monitoring_system


# Global monitoring instance (initialized when needed)
_global_monitoring_system = None


def get_global_monitoring_system():
    """Get the global monitoring system instance."""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = create_monitoring_system()
    return _global_monitoring_system


def configure_global_monitoring(config: dict):
    """Configure the global monitoring system with custom settings."""
    global _global_monitoring_system
    _global_monitoring_system = create_monitoring_system(config)


# Monitoring constants
class MonitoringConstants:
    """Constants for monitoring and observability."""
    
    # Service names
    SERVICE_API = "darwin-api"
    SERVICE_DASHBOARD = "darwin-dashboard"
    SERVICE_MCP = "darwin-mcp"
    SERVICE_OPTIMIZER = "darwin-optimizer"
    SERVICE_DATABASE = "darwin-database"
    
    # Metric names
    METRIC_REQUEST_COUNT = "darwin_requests_total"
    METRIC_REQUEST_DURATION = "darwin_request_duration_seconds"
    METRIC_OPTIMIZATION_COUNT = "darwin_optimizations_total"
    METRIC_OPTIMIZATION_DURATION = "darwin_optimization_duration_seconds"
    METRIC_MEMORY_USAGE = "darwin_memory_usage_bytes"
    METRIC_CPU_USAGE = "darwin_cpu_usage_percent"
    METRIC_ACTIVE_CONNECTIONS = "darwin_active_connections"
    
    # Span names
    SPAN_OPTIMIZATION_RUN = "optimization_run"
    SPAN_API_REQUEST = "api_request"
    SPAN_DATABASE_QUERY = "database_query"
    SPAN_WEBSOCKET_MESSAGE = "websocket_message"
    
    # Health check names
    HEALTH_DATABASE = "database"
    HEALTH_API = "api"
    HEALTH_MCP_SERVER = "mcp_server"
    HEALTH_WEBSOCKET = "websocket"
    HEALTH_MEMORY = "memory"
    HEALTH_CPU = "cpu"
    HEALTH_DISK = "disk"
    
    # Alert types
    ALERT_HIGH_CPU = "high_cpu_usage"
    ALERT_HIGH_MEMORY = "high_memory_usage"
    ALERT_API_ERROR_RATE = "api_error_rate"
    ALERT_OPTIMIZATION_FAILURE = "optimization_failure"
    ALERT_DATABASE_CONNECTION = "database_connection_error"
    ALERT_SERVICE_DOWN = "service_down"
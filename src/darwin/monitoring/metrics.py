"""
Darwin Metrics Collection and Aggregation System

This module provides comprehensive metrics collection, aggregation, and export capabilities
for the Darwin platform, including Prometheus-style metrics, custom metrics, time-series
data handling, and performance monitoring.

Features:
- Counter, Gauge, Histogram, and Summary metrics
- Custom metric types with labels and tags
- Time-series data collection and storage
- Metrics aggregation and rollup
- Prometheus-compatible export format
- Real-time metrics streaming
- Memory-efficient storage with TTL
- Integration with Logfire for metrics logging
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported by the system."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with timestamp and labels."""

    value: Union[int, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class CustomMetric:
    """Custom metric definition with metadata."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    help_text: str = ""

    # Storage for metric values
    values: List[MetricValue] = field(default_factory=list)
    current_value: Union[int, float] = 0

    # Configuration
    max_values: int = 1000
    ttl_seconds: int = 3600  # 1 hour default

    def __post_init__(self):
        """Initialize metric after creation."""
        self._lock = Lock()
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at

    def add_value(self, value: Union[int, float], labels: Dict[str, str] = None):
        """Add a new value to the metric."""
        with self._lock:
            metric_value = MetricValue(value=value, labels=labels or {})
            self.values.append(metric_value)
            self.last_updated = metric_value.timestamp

            # Update current value based on metric type
            if self.metric_type == MetricType.COUNTER:
                self.current_value += value
            elif self.metric_type == MetricType.GAUGE:
                self.current_value = value
            elif self.metric_type in [
                MetricType.HISTOGRAM,
                MetricType.SUMMARY,
                MetricType.TIMER,
            ]:
                # For these types, current_value represents the latest value
                self.current_value = value

            # Cleanup old values
            self._cleanup_old_values()

    def _cleanup_old_values(self):
        """Remove old values based on TTL and max_values."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)

        # Remove values older than TTL
        self.values = [v for v in self.values if v.timestamp > cutoff_time]

        # Keep only the most recent max_values
        if len(self.values) > self.max_values:
            self.values = self.values[-self.max_values :]

    def get_current_value(self) -> Union[int, float]:
        """Get the current value of the metric."""
        return self.current_value

    def get_values_in_range(
        self, start_time: datetime, end_time: datetime
    ) -> List[MetricValue]:
        """Get metric values within a time range."""
        return [v for v in self.values if start_time <= v.timestamp <= end_time]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of metric values."""
        if not self.values:
            return {}

        values = [v.value for v in self.values]

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "current": self.current_value,
            "last_updated": self.last_updated.isoformat(),
        }

        if len(values) > 1:
            stats["median"] = statistics.median(values)
            stats["stdev"] = statistics.stdev(values)

        # Add percentiles for histogram/summary metrics
        if self.metric_type in [
            MetricType.HISTOGRAM,
            MetricType.SUMMARY,
            MetricType.TIMER,
        ]:
            sorted_values = sorted(values)
            stats["p50"] = statistics.median(sorted_values)
            stats["p95"] = self._percentile(sorted_values, 0.95)
            stats["p99"] = self._percentile(sorted_values, 0.99)

        return stats

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        k = (len(values) - 1) * percentile
        f = int(k)
        c = k - f

        if f == len(values) - 1:
            return values[f]

        return values[f] * (1 - c) + values[f + 1] * c

    def to_prometheus_format(self) -> str:
        """Convert metric to Prometheus format."""
        lines = []

        # Add help text
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")

        # Add type
        prom_type = self.metric_type.value
        if prom_type == "timer":
            prom_type = "histogram"
        lines.append(f"# TYPE {self.name} {prom_type}")

        # Add current value
        if self.labels:
            label_str = ",".join([f'{k}="{v}"' for k, v in self.labels])
            lines.append(f"{self.name}{{{label_str}}} {self.current_value}")
        else:
            lines.append(f"{self.name} {self.current_value}")

        return "\n".join(lines)


class MetricsCollector:
    """Main metrics collection system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config or {}
        self.collection_interval = self.config.get("collection_interval", 30)
        self.retention_period = (
            self.config.get("retention_period", 7) * 24 * 3600
        )  # days to seconds
        self.export_format = self.config.get("export_format", "prometheus")

        self.metrics: Dict[str, CustomMetric] = {}
        self._lock = Lock()
        self.collectors: List[Callable] = []
        self.last_collection_time = datetime.now(timezone.utc)

        # Built-in system metrics
        self._register_system_metrics()

        logger.info("Metrics collector initialized")

    def _register_system_metrics(self):
        """Register built-in system metrics."""
        system_metrics = [
            CustomMetric(
                name="darwin_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total number of API requests",
                labels=["method", "endpoint", "status"],
                help_text="Total number of HTTP requests processed",
            ),
            CustomMetric(
                name="darwin_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="API request duration in seconds",
                labels=["method", "endpoint"],
                help_text="Time spent processing HTTP requests",
                unit="seconds",
            ),
            CustomMetric(
                name="darwin_optimization_count",
                metric_type=MetricType.COUNTER,
                description="Total number of optimization runs",
                labels=["algorithm", "status"],
                help_text="Total number of genetic algorithm optimization runs",
            ),
            CustomMetric(
                name="darwin_optimization_duration_seconds",
                metric_type=MetricType.TIMER,
                description="Optimization run duration in seconds",
                labels=["algorithm"],
                help_text="Time spent running genetic algorithm optimizations",
                unit="seconds",
            ),
            CustomMetric(
                name="darwin_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="Current memory usage in bytes",
                help_text="Current memory usage of the Darwin platform",
            ),
            CustomMetric(
                name="darwin_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="Current CPU usage percentage",
                help_text="Current CPU usage of the Darwin platform",
            ),
            CustomMetric(
                name="darwin_active_connections",
                metric_type=MetricType.GAUGE,
                description="Number of active connections",
                labels=["type"],
                help_text="Number of active database/websocket connections",
            ),
            CustomMetric(
                name="darwin_population_fitness",
                metric_type=MetricType.HISTOGRAM,
                description="Population fitness distribution",
                labels=["algorithm", "generation"],
                help_text="Distribution of fitness values in genetic algorithm populations",
            ),
            CustomMetric(
                name="darwin_convergence_rate",
                metric_type=MetricType.GAUGE,
                description="Algorithm convergence rate",
                labels=["algorithm"],
                help_text="Rate of convergence for genetic algorithm optimization",
            ),
        ]

        for metric in system_metrics:
            self.register_metric(metric)

    def register_metric(self, metric: CustomMetric):
        """Register a new metric."""
        with self._lock:
            self.metrics[metric.name] = metric
            logger.info(
                f"Registered metric: {metric.name} ({metric.metric_type.value})"
            )

    def create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: List[str] = None,
        help_text: str = "",
        unit: str = "",
    ) -> CustomMetric:
        """
        Create and register a new metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: List of label names
            help_text: Help text for Prometheus export
            unit: Unit of measurement

        Returns:
            Created CustomMetric instance
        """
        metric = CustomMetric(
            name=name,
            metric_type=metric_type,
            description=description,
            labels=labels or [],
            help_text=help_text,
            unit=unit,
        )

        self.register_metric(metric)
        return metric

    def increment_counter(
        self, name: str, value: Union[int, float] = 1, labels: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        if name in self.metrics:
            self.metrics[name].add_value(value, labels)
        else:
            logger.warning(f"Counter metric '{name}' not found")

    def set_gauge(
        self, name: str, value: Union[int, float], labels: Dict[str, str] = None
    ):
        """Set a gauge metric value."""
        if name in self.metrics:
            self.metrics[name].add_value(value, labels)
        else:
            logger.warning(f"Gauge metric '{name}' not found")

    def observe_histogram(
        self, name: str, value: Union[int, float], labels: Dict[str, str] = None
    ):
        """Observe a value for histogram metric."""
        if name in self.metrics:
            self.metrics[name].add_value(value, labels)
        else:
            logger.warning(f"Histogram metric '{name}' not found")

    def time_function(self, metric_name: str, labels: Dict[str, str] = None):
        """Decorator to time function execution."""

        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels)

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def register_collector(self, collector_func: Callable):
        """Register a custom metrics collector function."""
        self.collectors.append(collector_func)
        logger.info(f"Registered metrics collector: {collector_func.__name__}")

    async def collect_system_metrics(self):
        """Collect system metrics (CPU, memory, etc.)."""
        try:
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available for system metrics collection")
                return

            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.set_gauge("darwin_cpu_usage_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("darwin_memory_usage_bytes", memory.used)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.set_gauge("darwin_disk_usage_bytes", disk.used)

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def run_collectors(self):
        """Run all registered metric collectors."""
        # Collect system metrics
        await self.collect_system_metrics()

        # Run custom collectors
        for collector in self.collectors:
            try:
                if asyncio.iscoroutinefunction(collector):
                    await collector()
                else:
                    collector()
            except Exception as e:
                logger.error(f"Metrics collector failed: {e}")

    def get_metric(self, name: str) -> Optional[CustomMetric]:
        """Get a metric by name."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, CustomMetric]:
        """Get all registered metrics."""
        return self.metrics.copy()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}

        for name, metric in self.metrics.items():
            summary[name] = {
                "type": metric.metric_type.value,
                "description": metric.description,
                "current_value": metric.get_current_value(),
                "statistics": metric.get_statistics(),
                "last_updated": metric.last_updated.isoformat(),
            }

        return summary

    def export_prometheus_format(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []

        for metric in self.metrics.values():
            lines.append(metric.to_prometheus_format())
            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def export_json_format(self) -> str:
        """Export all metrics in JSON format."""
        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
        }

        for name, metric in self.metrics.items():
            export_data["metrics"][name] = {
                "type": metric.metric_type.value,
                "description": metric.description,
                "unit": metric.unit,
                "current_value": metric.get_current_value(),
                "statistics": metric.get_statistics(),
                "values": [v.to_dict() for v in metric.values[-10:]],  # Last 10 values
            }

        return json.dumps(export_data, indent=2)

    def add_metrics_middleware(self, app):
        """Add metrics collection middleware to FastAPI app."""
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware

        class MetricsMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, metrics_collector):
                super().__init__(app)
                self.metrics_collector = metrics_collector

            async def dispatch(self, request: Request, call_next):
                start_time = time.time()

                response = await call_next(request)

                duration = time.time() - start_time

                # Record request metrics
                labels = {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(response.status_code),
                }

                self.metrics_collector.increment_counter(
                    "darwin_requests_total", 1, labels
                )
                self.metrics_collector.observe_histogram(
                    "darwin_request_duration_seconds",
                    duration,
                    {"method": request.method, "endpoint": request.url.path},
                )

                return response

        app.add_middleware(MetricsMiddleware, metrics_collector=self)

        # Add metrics endpoints
        @app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint."""
            if self.export_format == "json":
                return {"metrics": self.get_metrics_summary()}
            else:
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(
                    self.export_prometheus_format(), media_type="text/plain"
                )

        @app.get("/metrics/json")
        async def get_metrics_json():
            """JSON metrics endpoint."""
            return {"metrics": self.get_metrics_summary()}

        @app.get("/metrics/prometheus")
        async def get_metrics_prometheus():
            """Prometheus format metrics endpoint."""
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse(
                self.export_prometheus_format(), media_type="text/plain"
            )

        logger.info("Metrics middleware and endpoints added to FastAPI application")

    async def start_periodic_collection(self):
        """Start periodic metrics collection."""
        logger.info(
            f"Starting periodic metrics collection every {self.collection_interval} seconds"
        )

        while True:
            try:
                await self.run_collectors()
                self.last_collection_time = datetime.now(timezone.utc)

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Periodic metrics collection failed: {e}")
                await asyncio.sleep(self.collection_interval)


class MetricsAggregator:
    """Metrics aggregation and rollup system."""

    def __init__(
        self, metrics_collector: MetricsCollector, config: Dict[str, Any] = None
    ):
        """
        Initialize metrics aggregator.

        Args:
            metrics_collector: MetricsCollector instance
            config: Aggregation configuration
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.aggregation_window = self.config.get(
            "aggregation_window", 300
        )  # 5 minutes
        self.rollup_intervals = self.config.get(
            "rollup_intervals", [300, 3600, 86400]
        )  # 5m, 1h, 1d

        self.aggregated_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = Lock()

        logger.info("Metrics aggregator initialized")

    async def aggregate_metrics(self):
        """Aggregate metrics over the configured window."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=self.aggregation_window)

        aggregated = {}

        for name, metric in self.metrics_collector.get_all_metrics().items():
            values_in_window = metric.get_values_in_range(start_time, end_time)

            if values_in_window:
                values = [v.value for v in values_in_window]

                aggregated[name] = {
                    "timestamp": end_time.isoformat(),
                    "window_seconds": self.aggregation_window,
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "sum": sum(values)
                    if metric.metric_type == MetricType.COUNTER
                    else None,
                }

        # Store aggregated data
        with self._lock:
            for name, data in aggregated.items():
                self.aggregated_data[name].append(data)

                # Keep only recent aggregations
                cutoff = len(self.aggregated_data[name]) - 100
                if cutoff > 0:
                    self.aggregated_data[name] = self.aggregated_data[name][cutoff:]

        return aggregated

    def get_aggregated_data(
        self, metric_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get aggregated data for a specific metric."""
        with self._lock:
            return self.aggregated_data.get(metric_name, [])[-limit:]

    async def start_periodic_aggregation(self):
        """Start periodic metrics aggregation."""
        logger.info(
            f"Starting periodic metrics aggregation every {self.aggregation_window} seconds"
        )

        while True:
            try:
                await self.aggregate_metrics()
                await asyncio.sleep(self.aggregation_window)

            except Exception as e:
                logger.error(f"Periodic metrics aggregation failed: {e}")
                await asyncio.sleep(self.aggregation_window)

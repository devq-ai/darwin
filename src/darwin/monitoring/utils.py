"""
Darwin Monitoring Utilities

This module provides utility functions and classes for the Darwin monitoring system,
including time-series data management, threshold calculations, data visualization
helpers, and common monitoring operations.

Features:
- Time-series data structures and operations
- Threshold management and alerting logic
- Data aggregation and statistical calculations
- Monitoring data visualization helpers
- Configuration management utilities
- Performance optimization helpers
- Data export and import utilities
- Common monitoring patterns and decorators
"""

import asyncio
import hashlib
import json
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of data aggregation."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "stdev"


class TimeWindow(Enum):
    """Predefined time windows for data aggregation."""

    MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    HOUR = 3600
    SIX_HOURS = 21600
    DAY = 86400
    WEEK = 604800


@dataclass
class DataPoint:
    """Individual data point with timestamp and value."""

    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """Create DataPoint from dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            timestamp=timestamp,
            value=data["value"],
            labels=data.get("labels", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TimeSeriesData:
    """Time-series data container with operations."""

    name: str
    data_points: deque = field(default_factory=deque)
    max_points: int = 10000
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def __post_init__(self):
        """Initialize after creation."""
        self._lock = Lock()
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at

    def add_point(
        self,
        value: Union[int, float],
        timestamp: datetime = None,
        labels: Dict[str, str] = None,
    ):
        """Add a new data point."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        point = DataPoint(timestamp=timestamp, value=value, labels=labels or {})

        with self._lock:
            self.data_points.append(point)
            self.last_updated = timestamp

            # Trim old points if necessary
            while len(self.data_points) > self.max_points:
                self.data_points.popleft()

    def get_points_in_range(
        self, start_time: datetime, end_time: datetime
    ) -> List[DataPoint]:
        """Get data points within time range."""
        with self._lock:
            return [
                point
                for point in self.data_points
                if start_time <= point.timestamp <= end_time
            ]

    def get_recent_points(self, duration_seconds: int) -> List[DataPoint]:
        """Get data points from recent time period."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=duration_seconds)
        return self.get_points_in_range(start_time, end_time)

    def aggregate(
        self, aggregation_type: AggregationType, window_seconds: int = None
    ) -> Union[float, int]:
        """Aggregate data points using specified method."""
        points = (
            self.data_points
            if window_seconds is None
            else self.get_recent_points(window_seconds)
        )

        if not points:
            return 0

        values = [point.value for point in points]

        if aggregation_type == AggregationType.SUM:
            return sum(values)
        elif aggregation_type == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif aggregation_type == AggregationType.MIN:
            return min(values)
        elif aggregation_type == AggregationType.MAX:
            return max(values)
        elif aggregation_type == AggregationType.COUNT:
            return len(values)
        elif aggregation_type == AggregationType.MEDIAN:
            return statistics.median(values)
        elif aggregation_type == AggregationType.PERCENTILE_95:
            return self._percentile(values, 0.95)
        elif aggregation_type == AggregationType.PERCENTILE_99:
            return self._percentile(values, 0.99)
        elif aggregation_type == AggregationType.STANDARD_DEVIATION:
            return statistics.stdev(values) if len(values) > 1 else 0
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f

        if f == len(sorted_values) - 1:
            return sorted_values[f]

        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c

    def get_statistics(self, window_seconds: int = None) -> Dict[str, Any]:
        """Get comprehensive statistics for the time series."""
        points = (
            self.data_points
            if window_seconds is None
            else self.get_recent_points(window_seconds)
        )

        if not points:
            return {"count": 0}

        values = [point.value for point in points]

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "sum": sum(values),
            "latest": values[-1],
            "oldest": values[0],
            "range": max(values) - min(values),
            "first_timestamp": points[0].timestamp.isoformat(),
            "last_timestamp": points[-1].timestamp.isoformat(),
        }

        if len(values) > 1:
            stats["median"] = statistics.median(values)
            stats["stdev"] = statistics.stdev(values)
            stats["variance"] = statistics.variance(values)
            stats["p95"] = self._percentile(values, 0.95)
            stats["p99"] = self._percentile(values, 0.99)

        return stats

    def detect_anomalies(
        self, threshold_stdev: float = 2.0, window_seconds: int = 3600
    ) -> List[DataPoint]:
        """Detect anomalous data points using standard deviation."""
        points = self.get_recent_points(window_seconds)

        if len(points) < 10:  # Need minimum points for meaningful analysis
            return []

        values = [point.value for point in points]
        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values) if len(values) > 1 else 0

        if stdev_val == 0:
            return []

        anomalies = []
        for point in points:
            z_score = abs(point.value - mean_val) / stdev_val
            if z_score > threshold_stdev:
                anomalies.append(point)

        return anomalies

    def calculate_trend(self, window_seconds: int = 3600) -> float:
        """Calculate trend using linear regression slope."""
        points = self.get_recent_points(window_seconds)

        if len(points) < 2:
            return 0.0

        # Convert timestamps to numeric values (seconds since first point)
        first_timestamp = points[0].timestamp.timestamp()
        x_values = [point.timestamp.timestamp() - first_timestamp for point in points]
        y_values = [point.value for point in points]

        # Calculate linear regression slope
        n = len(points)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def resample(
        self,
        interval_seconds: int,
        aggregation_type: AggregationType = AggregationType.AVERAGE,
    ) -> List[DataPoint]:
        """Resample data to regular intervals."""
        if not self.data_points:
            return []

        # Determine time range
        first_point = self.data_points[0]
        last_point = self.data_points[-1]

        start_time = first_point.timestamp
        end_time = last_point.timestamp

        resampled_points = []
        current_time = start_time

        while current_time <= end_time:
            window_start = current_time
            window_end = current_time + timedelta(seconds=interval_seconds)

            # Get points in this window
            window_points = [
                point
                for point in self.data_points
                if window_start <= point.timestamp < window_end
            ]

            if window_points:
                values = [point.value for point in window_points]

                # Apply aggregation
                if aggregation_type == AggregationType.AVERAGE:
                    aggregated_value = statistics.mean(values)
                elif aggregation_type == AggregationType.SUM:
                    aggregated_value = sum(values)
                elif aggregation_type == AggregationType.MAX:
                    aggregated_value = max(values)
                elif aggregation_type == AggregationType.MIN:
                    aggregated_value = min(values)
                elif aggregation_type == AggregationType.COUNT:
                    aggregated_value = len(values)
                else:
                    aggregated_value = statistics.mean(values)

                resampled_points.append(
                    DataPoint(timestamp=current_time, value=aggregated_value)
                )

            current_time = window_end

        return resampled_points

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        with self._lock:
            return {
                "name": self.name,
                "unit": self.unit,
                "description": self.description,
                "labels": self.labels,
                "data_points": [point.to_dict() for point in self.data_points],
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "point_count": len(self.data_points),
            }


class ThresholdManager:
    """Threshold management for alerts and monitoring."""

    def __init__(self):
        """Initialize threshold manager."""
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def set_threshold(
        self,
        metric_name: str,
        warning_threshold: Union[int, float] = None,
        critical_threshold: Union[int, float] = None,
        comparison: str = "gt",  # gt, lt, eq, ne
        duration_seconds: int = 300,
        enabled: bool = True,
    ):
        """Set threshold for a metric."""
        with self._lock:
            self.thresholds[metric_name] = {
                "warning_threshold": warning_threshold,
                "critical_threshold": critical_threshold,
                "comparison": comparison,
                "duration_seconds": duration_seconds,
                "enabled": enabled,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_triggered": None,
                "trigger_count": 0,
            }

    def check_threshold(
        self, metric_name: str, value: Union[int, float]
    ) -> Dict[str, Any]:
        """Check if value exceeds thresholds."""
        if metric_name not in self.thresholds:
            return {"status": "no_threshold", "level": None}

        threshold_config = self.thresholds[metric_name]

        if not threshold_config["enabled"]:
            return {"status": "disabled", "level": None}

        comparison = threshold_config["comparison"]
        warning_threshold = threshold_config["warning_threshold"]
        critical_threshold = threshold_config["critical_threshold"]

        # Check critical threshold first
        if critical_threshold is not None:
            if self._compare_value(value, critical_threshold, comparison):
                self._update_threshold_trigger(metric_name, "critical")
                return {
                    "status": "threshold_exceeded",
                    "level": "critical",
                    "threshold": critical_threshold,
                    "value": value,
                    "comparison": comparison,
                }

        # Check warning threshold
        if warning_threshold is not None:
            if self._compare_value(value, warning_threshold, comparison):
                self._update_threshold_trigger(metric_name, "warning")
                return {
                    "status": "threshold_exceeded",
                    "level": "warning",
                    "threshold": warning_threshold,
                    "value": value,
                    "comparison": comparison,
                }

        return {"status": "ok", "level": None}

    def _compare_value(
        self, value: Union[int, float], threshold: Union[int, float], comparison: str
    ) -> bool:
        """Compare value against threshold using specified comparison."""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lte":
            return value <= threshold
        elif comparison == "eq":
            return value == threshold
        elif comparison == "ne":
            return value != threshold
        else:
            logger.warning(f"Unknown comparison operator: {comparison}")
            return False

    def _update_threshold_trigger(self, metric_name: str, level: str):
        """Update threshold trigger information."""
        with self._lock:
            threshold_config = self.thresholds[metric_name]
            threshold_config["last_triggered"] = datetime.now(timezone.utc).isoformat()
            threshold_config["last_level"] = level
            threshold_config["trigger_count"] += 1

    def get_threshold(self, metric_name: str) -> Dict[str, Any]:
        """Get threshold configuration for metric."""
        return self.thresholds.get(metric_name, {})

    def get_all_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get all threshold configurations."""
        return self.thresholds.copy()

    def remove_threshold(self, metric_name: str):
        """Remove threshold for metric."""
        with self._lock:
            self.thresholds.pop(metric_name, None)

    def disable_threshold(self, metric_name: str):
        """Disable threshold for metric."""
        if metric_name in self.thresholds:
            with self._lock:
                self.thresholds[metric_name]["enabled"] = False

    def enable_threshold(self, metric_name: str):
        """Enable threshold for metric."""
        if metric_name in self.thresholds:
            with self._lock:
                self.thresholds[metric_name]["enabled"] = True


class MonitoringUtils:
    """Utility functions for monitoring operations."""

    @staticmethod
    def calculate_sla_uptime(up_points: int, total_points: int) -> float:
        """Calculate SLA uptime percentage."""
        if total_points == 0:
            return 100.0
        return (up_points / total_points) * 100.0

    @staticmethod
    def calculate_error_rate(error_count: int, total_count: int) -> float:
        """Calculate error rate percentage."""
        if total_count == 0:
            return 0.0
        return (error_count / total_count) * 100.0

    @staticmethod
    def calculate_apdex_score(
        satisfied_count: int, tolerating_count: int, total_count: int
    ) -> float:
        """Calculate Apdex score (Application Performance Index)."""
        if total_count == 0:
            return 1.0
        return (satisfied_count + (tolerating_count * 0.5)) / total_count

    @staticmethod
    def bytes_to_human_readable(bytes_value: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"

    @staticmethod
    def seconds_to_human_readable(seconds: float) -> str:
        """Convert seconds to human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.2f} ms"
        elif seconds < 60:
            return f"{seconds:.2f} s"
        elif seconds < 3600:
            return f"{seconds / 60:.2f} min"
        elif seconds < 86400:
            return f"{seconds / 3600:.2f} hours"
        else:
            return f"{seconds / 86400:.2f} days"

    @staticmethod
    def calculate_moving_average(values: List[float], window_size: int) -> List[float]:
        """Calculate moving average of values."""
        if not values or window_size <= 0:
            return []

        moving_averages = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window_values = values[start_idx : i + 1]
            moving_averages.append(sum(window_values) / len(window_values))

        return moving_averages

    @staticmethod
    def calculate_exponential_moving_average(
        values: List[float], alpha: float = 0.1
    ) -> List[float]:
        """Calculate exponential moving average."""
        if not values:
            return []

        ema_values = [values[0]]
        for i in range(1, len(values)):
            ema = alpha * values[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)

        return ema_values

    @staticmethod
    def detect_outliers_iqr(values: List[float], factor: float = 1.5) -> List[int]:
        """Detect outliers using Interquartile Range method."""
        if len(values) < 4:
            return []

        sorted_values = sorted(values)
        n = len(sorted_values)

        q1_idx = n // 4
        q3_idx = (3 * n) // 4

        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outlier_indices = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outlier_indices.append(i)

        return outlier_indices

    @staticmethod
    def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x2 - sum_x * sum_x
        denominator_y = n * sum_y2 - sum_y * sum_y

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        correlation = numerator / math.sqrt(denominator_x * denominator_y)
        return correlation

    @staticmethod
    def generate_metric_id(metric_name: str, labels: Dict[str, str] = None) -> str:
        """Generate unique metric ID from name and labels."""
        if labels:
            label_str = "&".join([f"{k}={v}" for k, v in sorted(labels.items())])
            metric_str = f"{metric_name}?{label_str}"
        else:
            metric_str = metric_name

        return hashlib.md5(metric_str.encode()).hexdigest()[:12]

    @staticmethod
    def parse_duration(duration_str: str) -> int:
        """Parse duration string to seconds."""
        unit_multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

        duration_str = duration_str.lower().strip()

        if duration_str.isdigit():
            return int(duration_str)

        unit = duration_str[-1]
        if unit in unit_multipliers:
            try:
                value = float(duration_str[:-1])
                return int(value * unit_multipliers[unit])
            except ValueError:
                pass

        raise ValueError(f"Invalid duration format: {duration_str}")

    @staticmethod
    def format_metric_value(value: Union[int, float], unit: str = "") -> str:
        """Format metric value for display."""
        if unit in ["bytes", "memory"]:
            return MonitoringUtils.bytes_to_human_readable(value)
        elif unit in ["seconds", "duration", "time"]:
            return MonitoringUtils.seconds_to_human_readable(value)
        elif unit == "percent":
            return f"{value:.2f}%"
        elif isinstance(value, float):
            return f"{value:.3f}"
        else:
            return str(value)

    @staticmethod
    def create_time_buckets(
        start_time: datetime, end_time: datetime, bucket_size_seconds: int
    ) -> List[datetime]:
        """Create time buckets for aggregation."""
        buckets = []
        current_time = start_time

        while current_time < end_time:
            buckets.append(current_time)
            current_time += timedelta(seconds=bucket_size_seconds)

        return buckets

    @staticmethod
    def interpolate_missing_points(
        data_points: List[DataPoint], interval_seconds: int
    ) -> List[DataPoint]:
        """Interpolate missing data points in time series."""
        if len(data_points) < 2:
            return data_points

        result = []

        for i in range(len(data_points) - 1):
            current_point = data_points[i]
            next_point = data_points[i + 1]

            result.append(current_point)

            # Calculate time gap
            time_gap = (next_point.timestamp - current_point.timestamp).total_seconds()

            if time_gap > interval_seconds * 1.5:  # Missing points detected
                num_missing = int(time_gap // interval_seconds) - 1

                for j in range(1, num_missing + 1):
                    interpolated_time = current_point.timestamp + timedelta(
                        seconds=interval_seconds * j
                    )
                    interpolated_value = current_point.value + (
                        next_point.value - current_point.value
                    ) * (j / (num_missing + 1))

                    result.append(
                        DataPoint(
                            timestamp=interpolated_time,
                            value=interpolated_value,
                            metadata={"interpolated": True},
                        )
                    )

        # Add the last point
        if data_points:
            result.append(data_points[-1])

        return result


def monitoring_timer(operation_name: str = None):
    """Decorator to time function execution for monitoring."""

    def decorator(func):
        operation = operation_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.info(f"Operation '{operation}' completed in {duration:.3f}s")

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.info(f"Operation '{operation}' completed in {duration:.3f}s")

            return sync_wrapper

    return decorator


def rate_limiter(max_calls: int, window_seconds: int):
    """Decorator for rate limiting function calls."""
    call_times = deque()
    lock = Lock()

    def decorator(func):
        def wrapper(*args, **kwargs):
            now = time.time()

            with lock:
                # Remove old calls outside the window
                while call_times and call_times[0] <= now - window_seconds:
                    call_times.popleft()

                # Check if we're within the rate limit
                if len(call_times) >= max_calls:
                    raise Exception(
                        f"Rate limit exceeded: {max_calls} calls per {window_seconds} seconds"
                    )

                call_times.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


class DataExporter:
    """Export monitoring data to various formats."""

    @staticmethod
    def to_csv(time_series: TimeSeriesData) -> str:
        """Export time series to CSV format."""
        lines = ["timestamp,value"]

        for point in time_series.data_points:
            lines.append(f"{point.timestamp.isoformat()},{point.value}")

        return "\n".join(lines)

    @staticmethod
    def to_json(time_series: TimeSeriesData, pretty: bool = True) -> str:
        """Export time series to JSON format."""
        data = time_series.to_dict()
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)

    @staticmethod
    def to_prometheus(time_series: TimeSeriesData) -> str:
        """Export time series to Prometheus format."""
        lines = []

        if time_series.description:
            lines.append(f"# HELP {time_series.name} {time_series.description}")

        lines.append(f"# TYPE {time_series.name} gauge")

        for point in time_series.data_points:
            timestamp_ms = int(point.timestamp.timestamp() * 1000)

            if point.labels:
                label_str = ",".join([f'{k}="{v}"' for k, v in point.labels.items()])
                lines.append(
                    f"{time_series.name}{{{label_str}}} {point.value} {timestamp_ms}"
                )
            else:
                lines.append(f"{time_series.name} {point.value} {timestamp_ms}")

        return "\n".join(lines)


class DataImporter:
    """Import monitoring data from various formats."""

    @staticmethod
    def from_csv(csv_data: str, name: str) -> TimeSeriesData:
        """Import time series from CSV format."""
        lines = csv_data.strip().split("\n")
        time_series = TimeSeriesData(name=name)

        # Skip header if present
        start_idx = 1 if lines[0].lower().startswith(("timestamp", "time")) else 0

        for line in lines[start_idx:]:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    timestamp = datetime.fromisoformat(parts[0])
                    value = float(parts[1])
                    time_series.add_point(value, timestamp)
                except (ValueError, IndexError):
                    continue

        return time_series

    @staticmethod
    def from_json(json_data: str) -> TimeSeriesData:
        """Import time series from JSON format."""
        data = json.loads(json_data)
        time_series = TimeSeriesData(
            name=data["name"],
            unit=data.get("unit", ""),
            description=data.get("description", ""),
            labels=data.get("labels", {}),
        )

        for point_data in data.get("data_points", []):
            point = DataPoint.from_dict(point_data)
            time_series.data_points.append(point)

        return time_series

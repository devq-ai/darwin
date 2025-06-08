"""
Comprehensive Test Suite for Darwin Monitoring System

This module provides thorough testing of all monitoring components including
health checks, metrics collection, alert management, performance monitoring,
distributed tracing, and utility functions.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from darwin.monitoring import (
    AggregationType,
    AlertManager,
    AlertRule,
    AlertSeverity,
    BenchmarkSuite,
    HealthChecker,
    HealthStatus,
    MetricsCollector,
    MetricType,
    MonitoringUtils,
    PerformanceMonitor,
    SpanStatus,
    SystemHealth,
    ThresholdManager,
    TimeSeriesData,
    TracingManager,
    create_monitoring_system,
    setup_monitoring_middleware,
)


class TestHealthChecker:
    """Test health checking functionality."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance for testing."""
        config = {
            "check_interval": 60,
            "timeout": 10,
            "critical_services": ["database", "api"],
        }
        return HealthChecker(config)

    @pytest.mark.asyncio
    async def test_database_health_check_success(self, health_checker):
        """Test successful database health check."""
        # Mock session factory
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = 1

        session_factory = AsyncMock(return_value=mock_session)
        health_checker.register_db_session_factory(session_factory)

        result = await health_checker.check_database_health()

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.critical is True
        assert "Database connection successful" in result.message

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, health_checker):
        """Test failed database health check."""
        # Mock session factory that raises exception
        session_factory = AsyncMock(side_effect=Exception("Connection failed"))
        health_checker.register_db_session_factory(session_factory)

        result = await health_checker.check_database_health()

        assert result.name == "database"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.critical is True
        assert "Connection failed" in result.message

    @pytest.mark.asyncio
    async def test_memory_health_check(self, health_checker):
        """Test memory health check."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 75.0
            mock_memory.return_value.used = 8000000000
            mock_memory.return_value.available = 2000000000
            mock_memory.return_value.total = 10000000000

            result = await health_checker.check_memory_health()

            assert result.name == "memory"
            assert result.status == HealthStatus.HEALTHY
            assert result.details["used_percent"] == 75.0

    @pytest.mark.asyncio
    async def test_memory_health_check_high_usage(self, health_checker):
        """Test memory health check with high usage."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 95.0
            mock_memory.return_value.used = 9500000000
            mock_memory.return_value.available = 500000000
            mock_memory.return_value.total = 10000000000

            result = await health_checker.check_memory_health()

            assert result.name == "memory"
            assert result.status == HealthStatus.UNHEALTHY
            assert "High memory usage" in result.message

    @pytest.mark.asyncio
    async def test_run_all_checks(self, health_checker):
        """Test running all health checks."""
        # Mock successful database check
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()

        session_factory = AsyncMock(return_value=mock_session)
        health_checker.register_db_session_factory(session_factory)

        with patch("psutil.virtual_memory") as mock_memory, patch(
            "psutil.cpu_percent"
        ) as mock_cpu, patch("psutil.disk_usage") as mock_disk:
            mock_memory.return_value.percent = 50.0
            mock_cpu.return_value = 30.0
            mock_disk.return_value.used = 5000000000
            mock_disk.return_value.total = 10000000000

            system_health = await health_checker.run_all_checks()

            assert isinstance(system_health, SystemHealth)
            assert system_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            assert len(system_health.checks) > 0

    def test_register_custom_check(self, health_checker):
        """Test registering custom health check."""

        async def custom_check():
            return health_checker.HealthCheck(
                name="custom",
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
            )

        health_checker.register_custom_check("custom", custom_check)

        assert "custom" in health_checker.custom_checks


class TestMetricsCollector:
    """Test metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance for testing."""
        config = {
            "collection_interval": 30,
            "retention_period": 7,
            "export_format": "prometheus",
        }
        return MetricsCollector(config)

    def test_create_counter_metric(self, metrics_collector):
        """Test creating counter metric."""
        metric = metrics_collector.create_metric(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            description="Test counter metric",
        )

        assert metric.name == "test_counter"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.description == "Test counter metric"
        assert "test_counter" in metrics_collector.metrics

    def test_increment_counter(self, metrics_collector):
        """Test incrementing counter metric."""
        metrics_collector.create_metric(
            "requests_total", MetricType.COUNTER, "Total requests"
        )

        metrics_collector.increment_counter("requests_total", 5)
        metrics_collector.increment_counter("requests_total", 3)

        metric = metrics_collector.get_metric("requests_total")
        assert metric.get_current_value() == 8

    def test_set_gauge(self, metrics_collector):
        """Test setting gauge metric."""
        metrics_collector.create_metric(
            "memory_usage", MetricType.GAUGE, "Memory usage in bytes"
        )

        metrics_collector.set_gauge("memory_usage", 1024)
        metrics_collector.set_gauge("memory_usage", 2048)

        metric = metrics_collector.get_metric("memory_usage")
        assert metric.get_current_value() == 2048

    def test_observe_histogram(self, metrics_collector):
        """Test observing histogram metric."""
        metrics_collector.create_metric(
            "request_duration", MetricType.HISTOGRAM, "Request duration in seconds"
        )

        metrics_collector.observe_histogram("request_duration", 0.5)
        metrics_collector.observe_histogram("request_duration", 1.0)
        metrics_collector.observe_histogram("request_duration", 0.8)

        metric = metrics_collector.get_metric("request_duration")
        stats = metric.get_statistics()

        assert stats["count"] == 3
        assert stats["min"] == 0.5
        assert stats["max"] == 1.0
        assert stats["mean"] == pytest.approx(0.77, rel=1e-1)

    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, metrics_collector):
        """Test collecting system metrics."""
        with patch("psutil.cpu_percent") as mock_cpu, patch(
            "psutil.virtual_memory"
        ) as mock_memory:
            mock_cpu.return_value = 45.0
            mock_memory.return_value.used = 4000000000

            await metrics_collector.collect_system_metrics()

            cpu_metric = metrics_collector.get_metric("darwin_cpu_usage_percent")
            memory_metric = metrics_collector.get_metric("darwin_memory_usage_bytes")

            assert cpu_metric.get_current_value() == 45.0
            assert memory_metric.get_current_value() == 4000000000

    def test_export_prometheus_format(self, metrics_collector):
        """Test exporting metrics in Prometheus format."""
        metrics_collector.create_metric(
            "test_metric", MetricType.GAUGE, "Test metric for export"
        )

        metrics_collector.set_gauge("test_metric", 42)

        prometheus_output = metrics_collector.export_prometheus_format()

        assert "test_metric 42" in prometheus_output
        assert "# TYPE test_metric gauge" in prometheus_output

    def test_time_function_decorator(self, metrics_collector):
        """Test timing function decorator."""
        metrics_collector.create_metric(
            "function_duration", MetricType.HISTOGRAM, "Function execution time"
        )

        @metrics_collector.time_function("function_duration")
        def test_function():
            time.sleep(0.1)
            return "result"

        result = test_function()

        assert result == "result"
        metric = metrics_collector.get_metric("function_duration")
        assert metric.get_current_value() >= 0.1


class TestAlertManager:
    """Test alert management functionality."""

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance for testing."""
        config = {
            "evaluation_interval": 60,
            "enable_notifications": False,  # Disable for testing
            "notifications": {},
        }
        return AlertManager(config)

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        metric = Mock()
        metric.get_current_value.return_value = 95.0
        collector.get_metric.return_value = metric
        return collector

    def test_create_alert_rule(self, alert_manager):
        """Test creating alert rule."""
        rule = alert_manager.create_rule(
            name="high_cpu",
            description="CPU usage too high",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.HIGH,
        )

        assert rule.name == "high_cpu"
        assert rule.threshold == 80.0
        assert rule.severity == AlertSeverity.HIGH
        assert rule.id in alert_manager.rules

    def test_alert_rule_evaluation(self, alert_manager):
        """Test alert rule evaluation logic."""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_name="test_metric",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.MEDIUM,
            duration=0,  # No duration for immediate triggering
        )

        # Test condition evaluation
        assert rule.evaluate_condition(60.0) is True
        assert rule.evaluate_condition(40.0) is False
        assert rule.evaluate_condition(50.0) is False  # Equal, not greater

    def test_alert_rule_conditions(self):
        """Test different alert rule conditions."""
        rule = AlertRule(
            name="test",
            description="Test",
            metric_name="test",
            condition="lt",
            threshold=10.0,
            severity=AlertSeverity.LOW,
        )

        assert rule.evaluate_condition(5.0) is True
        assert rule.evaluate_condition(15.0) is False

        rule.condition = "eq"
        assert rule.evaluate_condition(10.0) is True
        assert rule.evaluate_condition(10.1) is False

    @pytest.mark.asyncio
    async def test_trigger_alert(self, alert_manager, mock_metrics_collector):
        """Test triggering an alert."""
        alert_manager.set_metrics_collector(mock_metrics_collector)

        rule = alert_manager.create_rule(
            name="test_alert",
            description="Test alert",
            metric_name="test_metric",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.CRITICAL,
            duration=0,
        )

        await alert_manager.evaluate_rules()

        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0
        assert active_alerts[0].rule_name == "test_alert"
        assert active_alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        # Create a mock alert
        from darwin.monitoring.alerts import Alert, AlertStatus

        alert = Alert(
            id="test_alert_id",
            rule_name="test_rule",
            metric_name="test_metric",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message="Test alert",
            current_value=95.0,
            threshold=80.0,
        )

        alert_manager.active_alerts[alert.id] = alert

        success = await alert_manager.acknowledge_alert(
            alert.id, "test_user", "Acknowledged"
        )

        assert success is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        from darwin.monitoring.alerts import Alert, AlertStatus

        alert = Alert(
            id="test_alert_id",
            rule_name="test_rule",
            metric_name="test_metric",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message="Test alert",
            current_value=95.0,
            threshold=80.0,
        )

        alert_manager.active_alerts[alert.id] = alert

        success = await alert_manager.resolve_alert(
            alert.id, "test_user", "Issue fixed"
        )

        assert success is True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolution_note == "Issue fixed"
        assert alert.id not in alert_manager.active_alerts


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance for testing."""
        config = {
            "enable_profiling": True,
            "memory_threshold": 0.8,
            "cpu_threshold": 0.9,
        }
        return PerformanceMonitor(config)

    def test_performance_tracker_timer(self, performance_monitor):
        """Test performance timing functionality."""
        tracker = performance_monitor.tracker

        timer_id = tracker.start_timer("test_operation")
        time.sleep(0.1)
        duration = tracker.stop_timer(timer_id)

        assert duration >= 0.1
        assert timer_id not in tracker.active_timers

    def test_performance_tracker_context_manager(self, performance_monitor):
        """Test performance tracking context manager."""
        tracker = performance_monitor.tracker

        with tracker.time_operation("test_operation"):
            time.sleep(0.05)

        # Check that metric was recorded
        assert len(tracker.metrics_history) > 0
        metric = tracker.metrics_history[-1]
        assert metric.name == "test_operation_duration"
        assert metric.value >= 0.05

    def test_performance_tracker_statistics(self, performance_monitor):
        """Test performance metric statistics."""
        tracker = performance_monitor.tracker

        # Record some metrics
        for i in range(10):
            tracker.record_metric("test_metric", i * 0.1, "seconds")

        stats = tracker.get_metric_statistics("test_metric", hours=1)

        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 0.9
        assert stats["mean"] == pytest.approx(0.45, rel=1e-1)

    def test_performance_profile_operation(self, performance_monitor):
        """Test profiling operation context manager."""

        def test_function():
            # Simulate some work
            total = 0
            for i in range(1000):
                total += i * i
            return total

        with performance_monitor.profile_operation("test_function"):
            result = test_function()

        assert result == sum(i * i for i in range(1000))
        assert len(performance_monitor.profiles) > 0

        profile = performance_monitor.profiles[-1]
        assert profile.name == "test_function"
        assert profile.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_benchmark_suite(self):
        """Test benchmark suite functionality."""
        benchmark_suite = BenchmarkSuite()

        async def test_benchmark():
            # Simulate work
            await asyncio.sleep(0.01)
            return 100  # operations count

        result = await benchmark_suite.run_benchmark("test_benchmark", test_benchmark)

        assert result.test_name == "test_benchmark"
        assert result.success is True
        assert result.duration_seconds >= 0.01
        assert result.operations_per_second > 0

    def test_benchmark_statistics(self):
        """Test benchmark statistics calculation."""
        benchmark_suite = BenchmarkSuite()

        # Add some mock results
        from darwin.monitoring.performance import BenchmarkResult

        for i in range(5):
            result = BenchmarkResult(
                test_name="test_benchmark",
                duration_seconds=0.1 + i * 0.01,
                operations_per_second=100 - i * 5,
                memory_peak_mb=10 + i,
                cpu_usage_percent=30 + i * 2,
                success=True,
            )
            benchmark_suite.results.append(result)

        stats = benchmark_suite.get_benchmark_statistics("test_benchmark")

        assert stats["total_runs"] == 5
        assert stats["successful_runs"] == 5
        assert stats["success_rate"] == 1.0
        assert stats["duration"]["min"] == 0.1
        assert stats["duration"]["max"] == 0.14


class TestTracingManager:
    """Test distributed tracing functionality."""

    @pytest.fixture
    def tracing_manager(self):
        """Create tracing manager instance for testing."""
        config = {
            "service_name": "test-service",
            "enable_distributed_tracing": True,
            "sampling_rate": 1.0,
        }
        return TracingManager(config)

    def test_create_span(self, tracing_manager):
        """Test creating a span."""
        with tracing_manager.start_span("test_operation") as span:
            assert span is not None
            assert span.operation_name == "test_operation"
            assert span.service_name == "test-service"
            assert span.start_time is not None

        # Span should be finished after context exit
        assert span.is_finished()
        assert span.duration_ms > 0

    def test_span_with_attributes(self, tracing_manager):
        """Test span with attributes and tags."""
        attributes = {"user_id": "123", "request_id": "abc"}
        tags = {"component": "api", "version": "1.0"}

        with tracing_manager.start_span(
            "api_request", attributes=attributes, tags=tags
        ) as span:
            assert span.attributes == attributes
            assert span.tags == tags

    def test_span_error_handling(self, tracing_manager):
        """Test span error handling."""
        try:
            with tracing_manager.start_span("error_operation") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Test error"
        assert span.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_async_span(self, tracing_manager):
        """Test async span creation."""
        async with tracing_manager.start_async_span("async_operation") as span:
            assert span is not None
            assert span.operation_name == "async_operation"
            await asyncio.sleep(0.01)

        assert span.is_finished()
        assert span.duration_ms >= 10  # At least 10ms

    def test_trace_function_decorator(self, tracing_manager):
        """Test function tracing decorator."""

        @tracing_manager.trace_function("decorated_function")
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)

        assert result == 5
        assert len(tracing_manager.completed_traces) > 0

    @pytest.mark.asyncio
    async def test_async_trace_function_decorator(self, tracing_manager):
        """Test async function tracing decorator."""

        @tracing_manager.trace_function("async_decorated_function")
        async def async_test_function(x, y):
            await asyncio.sleep(0.01)
            return x * y

        result = await async_test_function(3, 4)

        assert result == 12
        assert len(tracing_manager.completed_traces) > 0

    def test_trace_collection(self, tracing_manager):
        """Test trace collection and statistics."""
        # Create multiple spans to form traces
        with tracing_manager.start_span("operation_1") as span1:
            span1.set_attribute("step", 1)

            with tracing_manager.start_span("operation_2") as span2:
                span2.set_attribute("step", 2)

        stats = tracing_manager.get_tracing_statistics()

        assert stats["service_name"] == "test-service"
        assert stats["completed_traces"] > 0
        assert stats["sampling_rate"] == 1.0


class TestMonitoringUtils:
    """Test monitoring utility functions."""

    def test_time_series_data(self):
        """Test time-series data functionality."""
        ts = TimeSeriesData("test_metric", unit="seconds")

        # Add some data points
        now = datetime.now(timezone.utc)
        for i in range(10):
            ts.add_point(i * 0.1, now + timedelta(seconds=i))

        assert len(ts.data_points) == 10

        # Test aggregation
        avg = ts.aggregate(AggregationType.AVERAGE)
        assert avg == pytest.approx(0.45, rel=1e-1)

        max_val = ts.aggregate(AggregationType.MAX)
        assert max_val == 0.9

    def test_time_series_statistics(self):
        """Test time-series statistics calculation."""
        ts = TimeSeriesData("test_metric")

        # Add data points with some variation
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # Last value is outlier
        for i, value in enumerate(values):
            ts.add_point(value, datetime.now(timezone.utc) + timedelta(seconds=i))

        stats = ts.get_statistics()

        assert stats["count"] == 6
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["mean"] == pytest.approx(19.17, rel=1e-1)

    def test_anomaly_detection(self):
        """Test anomaly detection in time series."""
        ts = TimeSeriesData("test_metric")

        # Add normal data points
        normal_values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2]
        for i, value in enumerate(normal_values):
            ts.add_point(value, datetime.now(timezone.utc) + timedelta(seconds=i))

        # Add anomalous point
        ts.add_point(50.0, datetime.now(timezone.utc) + timedelta(seconds=6))

        anomalies = ts.detect_anomalies(threshold_stdev=2.0)

        assert len(anomalies) == 1
        assert anomalies[0].value == 50.0

    def test_trend_calculation(self):
        """Test trend calculation for time series."""
        ts = TimeSeriesData("increasing_metric")

        # Add increasing values
        for i in range(10):
            ts.add_point(i * 2.0, datetime.now(timezone.utc) + timedelta(seconds=i))

        trend = ts.calculate_trend()

        assert trend > 0  # Should detect increasing trend

    def test_threshold_manager(self):
        """Test threshold management functionality."""
        threshold_manager = ThresholdManager()

        # Set threshold
        threshold_manager.set_threshold(
            "cpu_usage",
            warning_threshold=80.0,
            critical_threshold=95.0,
            comparison="gt",
        )

        # Test threshold checks
        result = threshold_manager.check_threshold("cpu_usage", 75.0)
        assert result["status"] == "ok"

        result = threshold_manager.check_threshold("cpu_usage", 85.0)
        assert result["status"] == "threshold_exceeded"
        assert result["level"] == "warning"

        result = threshold_manager.check_threshold("cpu_usage", 98.0)
        assert result["status"] == "threshold_exceeded"
        assert result["level"] == "critical"

    def test_monitoring_utils_functions(self):
        """Test utility functions."""
        # Test byte conversion
        assert MonitoringUtils.bytes_to_human_readable(1024) == "1.00 KB"
        assert MonitoringUtils.bytes_to_human_readable(1048576) == "1.00 MB"

        # Test time conversion
        assert MonitoringUtils.seconds_to_human_readable(0.5) == "500.00 ms"
        assert MonitoringUtils.seconds_to_human_readable(65) == "1.08 min"

        # Test SLA calculation
        uptime = MonitoringUtils.calculate_sla_uptime(99, 100)
        assert uptime == 99.0

        # Test error rate calculation
        error_rate = MonitoringUtils.calculate_error_rate(5, 100)
        assert error_rate == 5.0

    def test_moving_average_calculation(self):
        """Test moving average calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        moving_avg = MonitoringUtils.calculate_moving_average(values, 3)

        assert len(moving_avg) == 5
        assert moving_avg[0] == 1.0  # First value
        assert moving_avg[1] == 1.5  # (1+2)/2
        assert moving_avg[2] == 2.0  # (1+2+3)/3
        assert moving_avg[3] == 3.0  # (2+3+4)/3
        assert moving_avg[4] == 4.0  # (3+4+5)/3

    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier

        outlier_indices = MonitoringUtils.detect_outliers_iqr(values)

        assert len(outlier_indices) == 1
        assert outlier_indices[0] == 9  # Index of value 100


class TestMonitoringIntegration:
    """Test monitoring system integration."""

    @pytest.mark.asyncio
    async def test_create_monitoring_system(self):
        """Test creating complete monitoring system."""
        config = {
            "logfire": {"service_name": "test-service", "send_to_logfire": False},
            "health_checks": {"check_interval": 60},
            "metrics": {"collection_interval": 30},
        }

        monitoring_system = create_monitoring_system(config)

        assert "logfire" in monitoring_system
        assert "health" in monitoring_system
        assert "metrics" in monitoring_system
        assert "alerts" in monitoring_system
        assert "performance" in monitoring_system
        assert "tracing" in monitoring_system

    @pytest.mark.asyncio
    async def test_monitoring_middleware_setup(self):
        """Test setting up monitoring middleware."""
        from fastapi import FastAPI

        app = FastAPI()

        config = {
            "logfire": {"send_to_logfire": False},
            "health_checks": {"check_interval": 60},
        }

        monitoring_system = setup_monitoring_middleware(app, config)

        assert monitoring_system is not None
        assert "logfire" in monitoring_system

        # Check that routes were added
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/metrics" in routes


class TestDataExportImport:
    """Test data export and import functionality."""

    def test_csv_export_import(self):
        """Test CSV export and import."""
        from darwin.monitoring.utils import DataExporter, DataImporter

        # Create time series
        ts = TimeSeriesData("test_metric")
        now = datetime.now(timezone.utc)

        for i in range(5):
            ts.add_point(i * 10, now + timedelta(seconds=i))

        # Export to CSV
        csv_data = DataExporter.to_csv(ts)

        assert "timestamp,value" in csv_data
        assert "0" in csv_data
        assert "40" in csv_data

        # Import from CSV
        imported_ts = DataImporter.from_csv(csv_data, "imported_metric")

        assert imported_ts.name == "imported_metric"
        assert len(imported_ts.data_points) == 5

    def test_json_export_import(self):
        """Test JSON export and import."""

        # Create time

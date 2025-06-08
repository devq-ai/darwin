"""
Darwin Performance Monitoring System

This module provides comprehensive performance monitoring capabilities for the Darwin platform,
including profiling, benchmarking, resource tracking, performance analysis, and optimization
recommendations.

Features:
- Real-time performance monitoring and profiling
- Benchmark suite for genetic algorithm performance
- Resource utilization tracking (CPU, memory, I/O)
- Performance analysis and bottleneck detection
- Optimization recommendations
- Performance regression detection
- Integration with metrics and alerting systems
- Custom performance test framework
"""

import asyncio
import cProfile
import io
import logging
import os
import pstats
import statistics
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    name: str
    value: Union[int, float]
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class BenchmarkResult:
    """Benchmark test result."""

    test_name: str
    duration_seconds: float
    operations_per_second: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.duration_seconds,
            "operations_per_second": self.operations_per_second,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerformanceProfile:
    """Performance profiling result."""

    name: str
    duration_seconds: float
    function_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    top_functions: List[Dict[str, Any]]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "function_stats": self.function_stats,
            "memory_stats": self.memory_stats,
            "top_functions": self.top_functions,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceTracker:
    """Real-time performance tracking system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance tracker.

        Args:
            config: Performance tracking configuration
        """
        self.config = config or {}
        self.tracking_interval = self.config.get("tracking_interval", 10)  # seconds
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.enable_memory_tracking = self.config.get("enable_memory_tracking", True)

        self.metrics_history: List[PerformanceMetric] = []
        self.active_timers: Dict[str, float] = {}
        self._lock = Lock()

        # Initialize memory tracking if enabled
        if self.enable_memory_tracking:
            tracemalloc.start()

        logger.info("Performance tracker initialized")

    def start_timer(self, name: str) -> str:
        """Start a performance timer."""
        timer_id = f"{name}_{int(time.time() * 1000)}"
        self.active_timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str) -> Optional[float]:
        """Stop a performance timer and return duration."""
        if timer_id not in self.active_timers:
            logger.warning(f"Timer not found: {timer_id}")
            return None

        duration = time.time() - self.active_timers[timer_id]
        del self.active_timers[timer_id]
        return duration

    @contextmanager
    def time_operation(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = None

        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            start_memory = current

        try:
            yield
        finally:
            duration = time.time() - start_time

            # Calculate memory usage
            memory_delta = 0
            if self.enable_memory_tracking and start_memory is not None:
                current, peak = tracemalloc.get_traced_memory()
                memory_delta = current - start_memory

            # Record performance metric
            self.record_metric(
                name=f"{operation_name}_duration",
                value=duration,
                unit="seconds",
                context={
                    **(context or {}),
                    "memory_delta_bytes": memory_delta,
                    "operation": operation_name,
                },
            )

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        unit: str,
        context: Dict[str, Any] = None,
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name, value=value, unit=unit, context=context or {}
        )

        with self._lock:
            self.metrics_history.append(metric)

            # Trim history if needed
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size :]

    def get_metric_statistics(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a specific metric over time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        relevant_metrics = [
            m
            for m in self.metrics_history
            if m.name == metric_name and m.timestamp > cutoff_time
        ]

        if not relevant_metrics:
            return {}

        values = [m.value for m in relevant_metrics]

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "latest": values[-1],
            "timestamp_range": {
                "start": relevant_metrics[0].timestamp.isoformat(),
                "end": relevant_metrics[-1].timestamp.isoformat(),
            },
        }

        if len(values) > 1:
            stats["median"] = statistics.median(values)
            stats["stdev"] = statistics.stdev(values)

        return stats

    def get_system_performance(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    "error": "psutil not available",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0]

            # Memory information
            memory = psutil.virtual_memory()

            # Disk information
            disk = psutil.disk_usage("/")

            # Network information
            network = psutil.net_io_counters()

            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "load_average": load_avg,
                },
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "usage_percent": memory.percent,
                    "process_rss_bytes": process_memory.rss,
                    "process_vms_bytes": process_memory.vms,
                },
                "disk": {
                    "total_bytes": disk.total,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                    "usage_percent": (disk.used / disk.total) * 100,
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get system performance metrics: {e}")
            return {}

    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect potential performance issues."""
        issues = []

        try:
            system_perf = self.get_system_performance()

            # Check CPU usage
            cpu_usage = system_perf.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > 90:
                issues.append(
                    {
                        "type": "high_cpu",
                        "severity": "critical" if cpu_usage > 95 else "warning",
                        "message": f"High CPU usage: {cpu_usage:.1f}%",
                        "value": cpu_usage,
                    }
                )

            # Check memory usage
            memory_usage = system_perf.get("memory", {}).get("usage_percent", 0)
            if memory_usage > 85:
                issues.append(
                    {
                        "type": "high_memory",
                        "severity": "critical" if memory_usage > 95 else "warning",
                        "message": f"High memory usage: {memory_usage:.1f}%",
                        "value": memory_usage,
                    }
                )

            # Check disk usage
            disk_usage = system_perf.get("disk", {}).get("usage_percent", 0)
            if disk_usage > 85:
                issues.append(
                    {
                        "type": "high_disk",
                        "severity": "critical" if disk_usage > 95 else "warning",
                        "message": f"High disk usage: {disk_usage:.1f}%",
                        "value": disk_usage,
                    }
                )

            # Check for slow operations
            slow_operations = self._detect_slow_operations()
            issues.extend(slow_operations)

        except Exception as e:
            logger.error(f"Failed to detect performance issues: {e}")

        return issues

    def _detect_slow_operations(self) -> List[Dict[str, Any]]:
        """Detect slow operations from metrics history."""
        issues = []

        # Define thresholds for different operation types
        thresholds = {
            "api_request_duration": 2.0,  # seconds
            "database_query_duration": 1.0,  # seconds
            "optimization_step_duration": 5.0,  # seconds
        }

        for operation, threshold in thresholds.items():
            stats = self.get_metric_statistics(operation, hours=1)

            if stats and stats.get("mean", 0) > threshold:
                issues.append(
                    {
                        "type": "slow_operation",
                        "severity": "warning",
                        "message": f"Slow {operation}: avg {stats['mean']:.2f}s (threshold: {threshold}s)",
                        "operation": operation,
                        "average_duration": stats["mean"],
                        "threshold": threshold,
                    }
                )

        return issues


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance monitor.

        Args:
            config: Performance monitoring configuration
        """
        self.config = config or {}
        self.enable_profiling = self.config.get("enable_profiling", True)
        self.memory_threshold = self.config.get("memory_threshold", 0.8)
        self.cpu_threshold = self.config.get("cpu_threshold", 0.9)
        self.response_time_threshold = self.config.get("response_time_threshold", 2.0)

        self.tracker = PerformanceTracker(self.config.get("tracking", {}))
        self.profiles: List[PerformanceProfile] = []
        self._lock = Lock()

        logger.info("Performance monitor initialized")

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        if not self.enable_profiling:
            yield
            return

        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Start memory tracking
        if tracemalloc.is_tracing():
            tracemalloc.start()

        start_time = time.time()

        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()
            duration = time.time() - start_time

            # Analyze profile
            profile_result = self._analyze_profile(profiler, operation_name, duration)

            with self._lock:
                self.profiles.append(profile_result)

                # Keep only recent profiles
                if len(self.profiles) > 100:
                    self.profiles = self.profiles[-100:]

    def _analyze_profile(
        self, profiler: cProfile.Profile, operation_name: str, duration: float
    ) -> PerformanceProfile:
        """Analyze profiling results."""
        # Get profiling statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats("cumulative")

        # Get top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
            filename, line_number, function_name = func_info
            top_functions.append(
                {
                    "function": function_name,
                    "filename": filename,
                    "line_number": line_number,
                    "call_count": cc,
                    "total_time": tt,
                    "cumulative_time": ct,
                }
            )

        # Get memory statistics
        memory_stats = {}
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            memory_stats = {
                "current_bytes": current,
                "peak_bytes": peak,
                "peak_mb": peak / 1024 / 1024,
            }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            top_functions, memory_stats, duration
        )

        return PerformanceProfile(
            name=operation_name,
            duration_seconds=duration,
            function_stats={
                "total_functions": len(stats.stats),
                "total_calls": sum(
                    cc for cc, nc, tt, ct, callers in stats.stats.values()
                ),
            },
            memory_stats=memory_stats,
            top_functions=top_functions,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        top_functions: List[Dict[str, Any]],
        memory_stats: Dict[str, Any],
        duration: float,
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check for slow functions
        slow_functions = [f for f in top_functions if f.get("cumulative_time", 0) > 0.1]
        if slow_functions:
            recommendations.append(
                f"Consider optimizing slow functions: {', '.join([f['function'] for f in slow_functions[:3]])}"
            )

        # Check for high memory usage
        peak_mb = memory_stats.get("peak_mb", 0)
        if peak_mb > 100:
            recommendations.append(
                f"High memory usage detected ({peak_mb:.1f}MB) - consider memory optimization"
            )

        # Check for overall slow performance
        if duration > self.response_time_threshold:
            recommendations.append(
                f"Operation took {duration:.2f}s - consider performance optimization"
            )

        # Check for excessive function calls
        total_calls = sum(f.get("call_count", 0) for f in top_functions)
        if total_calls > 10000:
            recommendations.append(
                "High number of function calls detected - consider algorithm optimization"
            )

        return recommendations

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        recent_profiles = self.profiles[-10:] if self.profiles else []

        summary = {
            "total_profiles": len(self.profiles),
            "recent_profiles": len(recent_profiles),
            "system_performance": self.tracker.get_system_performance(),
            "performance_issues": self.tracker.detect_performance_issues(),
            "average_response_time": 0,
            "memory_usage_trend": [],
            "recommendations": [],
        }

        if recent_profiles:
            durations = [p.duration_seconds for p in recent_profiles]
            summary["average_response_time"] = statistics.mean(durations)

            # Collect all recommendations
            all_recommendations = []
            for profile in recent_profiles:
                all_recommendations.extend(profile.recommendations)

            # Get unique recommendations
            summary["recommendations"] = list(set(all_recommendations))

        return summary

    def add_performance_middleware(self, app):
        """Add performance monitoring middleware to FastAPI app."""
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware

        class PerformanceMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, performance_monitor):
                super().__init__(app)
                self.performance_monitor = performance_monitor

            async def dispatch(self, request: Request, call_next):
                # Start performance tracking
                operation_name = (
                    f"{request.method}_{request.url.path.replace('/', '_')}"
                )

                with self.performance_monitor.tracker.time_operation(
                    operation_name,
                    context={
                        "method": request.method,
                        "path": request.url.path,
                        "client_ip": request.client.host if request.client else None,
                    },
                ):
                    with self.performance_monitor.profile_operation(operation_name):
                        response = await call_next(request)

                return response

        app.add_middleware(PerformanceMiddleware, performance_monitor=self)

        # Add performance endpoints
        @app.get("/performance")
        async def get_performance():
            """Get performance summary."""
            return self.get_performance_summary()

        @app.get("/performance/profiles")
        async def get_performance_profiles(limit: int = 10):
            """Get recent performance profiles."""
            recent_profiles = self.profiles[-limit:] if self.profiles else []
            return {
                "profiles": [profile.to_dict() for profile in recent_profiles],
                "count": len(recent_profiles),
            }

        @app.get("/performance/metrics")
        async def get_performance_metrics():
            """Get performance metrics."""
            return {
                "system": self.tracker.get_system_performance(),
                "issues": self.tracker.detect_performance_issues(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(
            "Performance monitoring middleware and endpoints added to FastAPI application"
        )


class BenchmarkSuite:
    """Performance benchmark suite for Darwin genetic algorithms."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or {}
        self.results: List[BenchmarkResult] = []
        self._lock = Lock()

        logger.info("Benchmark suite initialized")

    async def run_benchmark(
        self, test_name: str, test_func: Callable, *args, **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.info(f"Running benchmark: {test_name}")

        # Start monitoring
        start_time = time.time()
        start_memory = None
        start_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            start_memory = current

        success = True
        error_message = None
        operations_count = 0

        try:
            # Run the test
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(*args, **kwargs)
            else:
                result = test_func(*args, **kwargs)

            # Extract operations count if returned
            if isinstance(result, dict) and "operations" in result:
                operations_count = result["operations"]
            elif isinstance(result, (int, float)):
                operations_count = result
            else:
                operations_count = 1  # Default

        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Benchmark {test_name} failed: {e}")

        # Calculate metrics
        duration = time.time() - start_time
        end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
        cpu_usage = (start_cpu + end_cpu) / 2

        memory_peak_mb = 0
        if tracemalloc.is_tracing() and start_memory is not None:
            current, peak = tracemalloc.get_traced_memory()
            memory_peak_mb = peak / 1024 / 1024

        operations_per_second = operations_count / duration if duration > 0 else 0

        # Create result
        result = BenchmarkResult(
            test_name=test_name,
            duration_seconds=duration,
            operations_per_second=operations_per_second,
            memory_peak_mb=memory_peak_mb,
            cpu_usage_percent=cpu_usage,
            success=success,
            error_message=error_message,
            metadata={
                "operations_count": operations_count,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
            },
        )

        with self._lock:
            self.results.append(result)

        logger.info(
            f"Benchmark {test_name} completed: {duration:.3f}s, {operations_per_second:.1f} ops/s"
        )
        return result

    def get_benchmark_results(self, test_name: str = None) -> List[BenchmarkResult]:
        """Get benchmark results, optionally filtered by test name."""
        if test_name:
            return [r for r in self.results if r.test_name == test_name]
        return self.results.copy()

    def get_benchmark_statistics(self, test_name: str) -> Dict[str, Any]:
        """Get statistics for a specific benchmark test."""
        test_results = self.get_benchmark_results(test_name)

        if not test_results:
            return {}

        successful_results = [r for r in test_results if r.success]

        if not successful_results:
            return {"error": "No successful test runs"}

        durations = [r.duration_seconds for r in successful_results]
        ops_per_sec = [r.operations_per_second for r in successful_results]
        memory_usage = [r.memory_peak_mb for r in successful_results]

        stats = {
            "test_name": test_name,
            "total_runs": len(test_results),
            "successful_runs": len(successful_results),
            "success_rate": len(successful_results) / len(test_results),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
            },
            "operations_per_second": {
                "min": min(ops_per_sec),
                "max": max(ops_per_sec),
                "mean": statistics.mean(ops_per_sec),
                "median": statistics.median(ops_per_sec),
            },
            "memory_peak_mb": {
                "min": min(memory_usage),
                "max": max(memory_usage),
                "mean": statistics.mean(memory_usage),
                "median": statistics.median(memory_usage),
            },
        }

        if len(durations) > 1:
            stats["duration"]["stdev"] = statistics.stdev(durations)
            stats["operations_per_second"]["stdev"] = statistics.stdev(ops_per_sec)
            stats["memory_peak_mb"]["stdev"] = statistics.stdev(memory_usage)

        return stats

    async def run_genetic_algorithm_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run standard genetic algorithm benchmarks."""
        benchmarks = {}

        # Simple fitness evaluation benchmark
        async def fitness_evaluation_benchmark():
            operations = 0
            for _ in range(1000):
                # Simulate fitness evaluation
                x = [i * 0.1 for i in range(100)]
                fitness = sum(xi**2 for xi in x)
                operations += 1
            return operations

        benchmarks["fitness_evaluation"] = await self.run_benchmark(
            "fitness_evaluation", fitness_evaluation_benchmark
        )

        # Population generation benchmark
        async def population_generation_benchmark():
            import random

            population_size = 100
            chromosome_length = 50

            population = []
            for _ in range(population_size):
                chromosome = [random.random() for _ in range(chromosome_length)]
                population.append(chromosome)

            return population_size

        benchmarks["population_generation"] = await self.run_benchmark(
            "population_generation", population_generation_benchmark
        )

        # Selection benchmark
        async def selection_benchmark():
            import random

            population_size = 100
            operations = 0

            # Simulate tournament selection
            for _ in range(population_size):
                tournament_size = 5
                selected = random.sample(range(population_size), tournament_size)
                winner = max(selected)  # Simplified selection
                operations += 1

            return operations

        benchmarks["selection"] = await self.run_benchmark(
            "selection", selection_benchmark
        )

        return benchmarks

    def compare_benchmarks(
        self, test_name: str, baseline_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """Compare current benchmark results with baseline."""
        recent_results = self.get_benchmark_results(test_name)

        if not recent_results:
            return {"error": "No recent results to compare"}

        latest_result = recent_results[-1]

        comparison = {
            "test_name": test_name,
            "baseline": baseline_result.to_dict(),
            "latest": latest_result.to_dict(),
            "performance_change": {
                "duration_change_percent": (
                    (latest_result.duration_seconds - baseline_result.duration_seconds)
                    / baseline_result.duration_seconds
                )
                * 100,
                "ops_per_sec_change_percent": (
                    (
                        latest_result.operations_per_second
                        - baseline_result.operations_per_second
                    )
                    / baseline_result.operations_per_second
                )
                * 100
                if baseline_result.operations_per_second > 0
                else 0,
                "memory_change_percent": (
                    (latest_result.memory_peak_mb - baseline_result.memory_peak_mb)
                    / baseline_result.memory_peak_mb
                )
                * 100
                if baseline_result.memory_peak_mb > 0
                else 0,
            },
            "regression_detected": False,
        }

        # Detect regression
        duration_regression = (
            comparison["performance_change"]["duration_change_percent"] > 20
        )  # 20% slower
        ops_regression = (
            comparison["performance_change"]["ops_per_sec_change_percent"] < -20
        )  # 20% fewer ops/sec
        memory_regression = (
            comparison["performance_change"]["memory_change_percent"] > 50
        )  # 50% more memory

        comparison["regression_detected"] = (
            duration_regression or ops_regression or memory_regression
        )

        if comparison["regression_detected"]:
            comparison["regression_details"] = {
                "duration_regression": duration_regression,
                "operations_regression": ops_regression,
                "memory_regression": memory_regression,
            }

        return comparison

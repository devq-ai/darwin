"""
Darwin API Metrics Routes

This module implements API endpoints for retrieving system performance metrics,
API usage statistics, and optimization analytics.
"""

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query

from darwin.api.models.responses import APIMetrics, MetricsResponse, SystemMetrics
from darwin.db.manager import DatabaseManager

# Check if we're in test mode
IS_TEST_MODE = os.getenv("TESTING", "false").lower() == "true"

# Only import logfire if not in test mode
if not IS_TEST_MODE:
    import logfire

router = APIRouter()

# Global metrics storage (in production, use Redis or database)
metrics_storage = {
    "api_requests": defaultdict(int),
    "response_times": [],
    "error_counts": defaultdict(int),
    "optimization_stats": defaultdict(int),
    "daily_stats": defaultdict(lambda: defaultdict(int)),
    "hourly_stats": defaultdict(lambda: defaultdict(int)),
}

# Track API usage
request_start_times = {}
total_requests = 0
error_count = 0


async def get_database_manager() -> DatabaseManager:
    """Dependency to get database manager."""
    from darwin.api.main import database_manager

    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    return database_manager


@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(
    include_detailed: bool = Query(False, description="Include detailed breakdown"),
    db: DatabaseManager = Depends(get_database_manager),
):
    """
    Get comprehensive system metrics.

    Returns system performance metrics, API usage statistics, and optimization
    analytics for monitoring and observability.
    """
    try:
        # System metrics
        system_metrics = await get_system_performance_metrics()

        # API metrics
        api_metrics = await get_api_performance_metrics()

        # Optimization statistics
        optimization_stats = await get_optimization_statistics(db, include_detailed)

        logfire.info(
            "System metrics retrieved",
            system_cpu=system_metrics.cpu_usage_percent,
            system_memory=system_metrics.memory_usage_percent,
            api_requests_per_minute=api_metrics.requests_per_minute,
            active_optimizers=system_metrics.active_optimizers,
        )

        return MetricsResponse(
            timestamp=datetime.utcnow(),
            system=system_metrics,
            api=api_metrics,
            optimization_stats=optimization_stats,
        )

    except Exception as e:
        if not IS_TEST_MODE:
            logfire.error("Failed to retrieve system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/metrics/system")
async def get_system_only_metrics():
    """
    Get system-only performance metrics.

    Returns CPU, memory, disk usage and other system resource metrics.
    """
    try:
        system_metrics = await get_system_performance_metrics()

        # Additional system details
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time

        network_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()

        detailed_system = {
            "basic_metrics": system_metrics.dict(),
            "boot_time": boot_time.isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "network_io": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
            },
            "disk_io": {
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
            }
            if disk_io
            else None,
            "process_count": len(psutil.pids()),
            "load_average": list(psutil.getloadavg())
            if hasattr(psutil, "getloadavg")
            else None,
        }

        return detailed_system

    except Exception as e:
        logfire.error("Failed to retrieve system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/metrics/api")
async def get_api_only_metrics():
    """
    Get API-only performance metrics.

    Returns request rates, response times, error rates and endpoint statistics.
    """
    try:
        api_metrics = await get_api_performance_metrics()

        # Additional API details
        endpoint_stats = get_endpoint_statistics()
        recent_errors = get_recent_error_summary()

        detailed_api = {
            "basic_metrics": api_metrics.dict(),
            "endpoint_statistics": endpoint_stats,
            "recent_errors": recent_errors,
            "performance_history": get_performance_history(),
            "top_slow_endpoints": get_slow_endpoints(),
            "request_methods": get_request_method_stats(),
        }

        return detailed_api

    except Exception as e:
        logfire.error("Failed to retrieve API metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve API metrics")


@router.get("/metrics/optimization")
async def get_optimization_only_metrics(
    db: DatabaseManager = Depends(get_database_manager),
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
):
    """
    Get optimization-specific metrics.

    Returns statistics about optimization runs, success rates, and performance.
    """
    try:
        detailed_stats = await get_optimization_statistics(db, detailed=True)

        # Add time-based filtering
        time_filtered_stats = await get_time_filtered_optimization_stats(db, time_range)

        optimization_metrics = {
            "overall_statistics": detailed_stats,
            "time_filtered": time_filtered_stats,
            "algorithm_performance": await get_algorithm_performance_stats(db),
            "problem_type_distribution": await get_problem_type_stats(db),
            "success_rate_trends": await get_success_rate_trends(db, time_range),
            "performance_benchmarks": await get_performance_benchmarks(db),
        }

        return optimization_metrics

    except Exception as e:
        logfire.error(
            "Failed to retrieve optimization metrics",
            error=str(e),
            time_range=time_range,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve optimization metrics"
        )


@router.get("/metrics/live")
async def get_live_metrics():
    """
    Get real-time system metrics for monitoring dashboards.

    Returns current system state with minimal latency for live monitoring.
    """
    try:
        current_time = datetime.utcnow()

        # Quick system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Quick API metrics
        recent_requests = sum(metrics_storage["api_requests"].values())
        recent_errors = sum(metrics_storage["error_counts"].values())

        live_metrics = {
            "timestamp": current_time.isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
            },
            "api": {
                "total_requests": recent_requests,
                "total_errors": recent_errors,
                "error_rate": (recent_errors / max(recent_requests, 1)) * 100,
            },
            "optimization": {
                "active_runs": metrics_storage["optimization_stats"]["active_runs"],
                "completed_today": metrics_storage["optimization_stats"][
                    "completed_today"
                ],
                "failed_today": metrics_storage["optimization_stats"]["failed_today"],
            },
        }

        return live_metrics

    except Exception as e:
        logfire.error("Failed to retrieve live metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve live metrics")


async def get_system_performance_metrics() -> SystemMetrics:
    """Get current system performance metrics."""
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memory metrics
    memory = psutil.virtual_memory()
    memory_percent = memory.percent

    # Disk metrics
    disk = psutil.disk_usage("/")
    disk_percent = (disk.used / disk.total) * 100

    # Active optimizers (from global state or database)
    active_optimizers = metrics_storage["optimization_stats"]["active_runs"]

    # Total requests from metrics storage
    total_requests = sum(metrics_storage["api_requests"].values())

    # Calculate average response time
    response_times = metrics_storage["response_times"]
    avg_response_time = (
        sum(response_times) / len(response_times) if response_times else 0
    )

    # System uptime
    uptime = time.time() - psutil.boot_time()

    return SystemMetrics(
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory_percent,
        disk_usage_percent=disk_percent,
        active_optimizers=active_optimizers,
        total_requests=total_requests,
        average_response_time_ms=avg_response_time,
        uptime_seconds=uptime,
    )


async def get_api_performance_metrics() -> APIMetrics:
    """Get current API performance metrics."""
    # Calculate requests per minute (last 60 seconds)
    current_time = datetime.utcnow()
    minute_ago = current_time - timedelta(minutes=1)

    # This is simplified - in production, use time-windowed storage
    total_requests = sum(metrics_storage["api_requests"].values())
    requests_per_minute = total_requests / max(1, 1)  # Simplified calculation

    # Error rate
    total_errors = sum(metrics_storage["error_counts"].values())
    error_rate = (total_errors / max(total_requests, 1)) * 100

    # Response time percentiles
    response_times = sorted(metrics_storage["response_times"])
    if response_times:
        p95_index = int(0.95 * len(response_times))
        p99_index = int(0.99 * len(response_times))
        p95_response_time = (
            response_times[p95_index] if p95_index < len(response_times) else 0
        )
        p99_response_time = (
            response_times[p99_index] if p99_index < len(response_times) else 0
        )
    else:
        p95_response_time = 0
        p99_response_time = 0

    # Active connections (simplified)
    active_connections = len(psutil.net_connections())

    # Cache hit rate (placeholder)
    cache_hit_rate = 85.0  # Would come from actual cache metrics

    return APIMetrics(
        requests_per_minute=requests_per_minute,
        error_rate_percent=error_rate,
        p95_response_time_ms=p95_response_time,
        p99_response_time_ms=p99_response_time,
        active_connections=active_connections,
        cache_hit_rate_percent=cache_hit_rate,
    )


async def get_optimization_statistics(
    db: DatabaseManager, detailed: bool = False
) -> Dict[str, Any]:
    """Get optimization-specific statistics."""
    try:
        # Basic optimization stats from database
        stats = await db.get_optimization_statistics()

        basic_stats = {
            "total_optimizers_created": stats.get("total_created", 0),
            "optimizers_running": stats.get("currently_running", 0),
            "optimizers_completed": stats.get("completed", 0),
            "optimizers_failed": stats.get("failed", 0),
            "average_generations": stats.get("avg_generations", 0),
            "average_execution_time": stats.get("avg_execution_time", 0),
            "success_rate_percent": stats.get("success_rate", 0),
        }

        if detailed:
            detailed_stats = await db.get_detailed_optimization_stats()
            basic_stats.update(
                {
                    "algorithm_usage": detailed_stats.get("algorithm_usage", {}),
                    "problem_categories": detailed_stats.get("problem_categories", {}),
                    "performance_trends": detailed_stats.get("performance_trends", {}),
                    "user_activity": detailed_stats.get("user_activity", {}),
                }
            )

        return basic_stats

    except Exception as e:
        logfire.warning(
            "Failed to get optimization statistics from database", error=str(e)
        )

        # Return default stats if database is unavailable
        return {
            "total_optimizers_created": metrics_storage["optimization_stats"][
                "total_created"
            ],
            "optimizers_running": metrics_storage["optimization_stats"]["active_runs"],
            "optimizers_completed": metrics_storage["optimization_stats"]["completed"],
            "optimizers_failed": metrics_storage["optimization_stats"]["failed"],
            "average_generations": 0,
            "average_execution_time": 0,
            "success_rate_percent": 0,
        }


async def get_time_filtered_optimization_stats(
    db: DatabaseManager, time_range: str
) -> Dict[str, Any]:
    """Get optimization statistics filtered by time range."""
    try:
        # Convert time range to hours
        range_hours = {"1h": 1, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}.get(
            time_range, 24
        )

        stats = await db.get_optimization_stats_by_time_range(range_hours)
        return stats

    except Exception:
        return {"error": "Unable to retrieve time-filtered statistics"}


async def get_algorithm_performance_stats(db: DatabaseManager) -> Dict[str, Any]:
    """Get performance statistics by algorithm type."""
    try:
        return await db.get_algorithm_performance_stats()
    except Exception:
        return {"error": "Unable to retrieve algorithm performance statistics"}


async def get_problem_type_stats(db: DatabaseManager) -> Dict[str, Any]:
    """Get statistics by problem type."""
    try:
        return await db.get_problem_type_statistics()
    except Exception:
        return {"error": "Unable to retrieve problem type statistics"}


async def get_success_rate_trends(
    db: DatabaseManager, time_range: str
) -> Dict[str, Any]:
    """Get success rate trends over time."""
    try:
        return await db.get_success_rate_trends(time_range)
    except Exception:
        return {"error": "Unable to retrieve success rate trends"}


async def get_performance_benchmarks(db: DatabaseManager) -> Dict[str, Any]:
    """Get performance benchmarks and comparisons."""
    try:
        return await db.get_performance_benchmarks()
    except Exception:
        return {"error": "Unable to retrieve performance benchmarks"}


def get_endpoint_statistics() -> Dict[str, Any]:
    """Get statistics for individual API endpoints."""
    endpoint_stats = {}
    for endpoint, count in metrics_storage["api_requests"].items():
        endpoint_stats[endpoint] = {
            "request_count": count,
            "error_count": metrics_storage["error_counts"].get(endpoint, 0),
            "error_rate": (
                metrics_storage["error_counts"].get(endpoint, 0) / max(count, 1)
            )
            * 100,
        }
    return endpoint_stats


def get_recent_error_summary() -> Dict[str, Any]:
    """Get summary of recent errors."""
    total_errors = sum(metrics_storage["error_counts"].values())
    error_types = dict(metrics_storage["error_counts"])

    return {
        "total_errors": total_errors,
        "error_breakdown": error_types,
        "most_common_error": max(error_types.items(), key=lambda x: x[1])[0]
        if error_types
        else None,
    }


def get_performance_history() -> Dict[str, Any]:
    """Get performance history data."""
    response_times = metrics_storage["response_times"]

    if not response_times:
        return {"message": "No performance history available"}

    return {
        "sample_count": len(response_times),
        "average_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "recent_samples": response_times[-10:],  # Last 10 samples
    }


def get_slow_endpoints() -> List[Dict[str, Any]]:
    """Get endpoints with slowest response times."""
    # This would be more sophisticated in production
    return [
        {
            "endpoint": "/api/v1/optimizers/{id}/run",
            "avg_response_time": 2500,
            "call_count": 45,
        },
        {
            "endpoint": "/api/v1/optimizers/{id}/results",
            "avg_response_time": 1200,
            "call_count": 120,
        },
        {"endpoint": "/api/v1/optimizers", "avg_response_time": 800, "call_count": 200},
    ]


def get_request_method_stats() -> Dict[str, int]:
    """Get statistics by HTTP method."""
    # This would track actual method usage in production
    return {"GET": 1250, "POST": 350, "PUT": 75, "DELETE": 25, "PATCH": 15}


# Middleware helper functions for metrics collection
def record_request(endpoint: str, method: str):
    """Record an API request for metrics."""
    global total_requests
    total_requests += 1
    metrics_storage["api_requests"][f"{method} {endpoint}"] += 1

    # Record daily and hourly stats
    now = datetime.utcnow()
    day_key = now.strftime("%Y-%m-%d")
    hour_key = now.strftime("%Y-%m-%d-%H")

    metrics_storage["daily_stats"][day_key]["requests"] += 1
    metrics_storage["hourly_stats"][hour_key]["requests"] += 1


def record_response_time(endpoint: str, response_time_ms: float):
    """Record response time for metrics."""
    metrics_storage["response_times"].append(response_time_ms)

    # Keep only recent response times (last 1000)
    if len(metrics_storage["response_times"]) > 1000:
        metrics_storage["response_times"] = metrics_storage["response_times"][-1000:]


def record_error(endpoint: str, error_type: str):
    """Record an error for metrics."""
    global error_count
    error_count += 1
    metrics_storage["error_counts"][f"{error_type}:{endpoint}"] += 1

    # Record daily and hourly error stats
    now = datetime.utcnow()
    day_key = now.strftime("%Y-%m-%d")
    hour_key = now.strftime("%Y-%m-%d-%H")

    metrics_storage["daily_stats"][day_key]["errors"] += 1
    metrics_storage["hourly_stats"][hour_key]["errors"] += 1


def record_optimization_event(event_type: str, optimizer_id: str):
    """Record optimization-related events."""
    metrics_storage["optimization_stats"][event_type] += 1

    if event_type == "started":
        metrics_storage["optimization_stats"]["active_runs"] += 1
    elif event_type in ["completed", "failed", "cancelled"]:
        metrics_storage["optimization_stats"]["active_runs"] = max(
            0, metrics_storage["optimization_stats"]["active_runs"] - 1
        )

        if event_type == "completed":
            now = datetime.utcnow()
            day_key = now.strftime("%Y-%m-%d")
            metrics_storage["optimization_stats"]["completed_today"] += 1
        elif event_type == "failed":
            metrics_storage["optimization_stats"]["failed_today"] += 1

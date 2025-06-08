"""
Darwin API Health Check Routes

This module implements health check endpoints for monitoring API and system health.
Provides comprehensive health status including database, external services, and system metrics.
"""

import os
import shutil
import time
from datetime import UTC, datetime
from typing import Any, Dict

try:
    import psutil
except ImportError:
    psutil = None

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

try:
    from darwin.api.models.responses import HealthCheck, HealthResponse, HealthStatus
except ImportError:
    # Create mock response models for testing
    from enum import Enum

    from pydantic import BaseModel

    class HealthStatus(str, Enum):
        HEALTHY = "healthy"
        UNHEALTHY = "unhealthy"
        DEGRADED = "degraded"

    class HealthCheck(BaseModel):
        service: str
        status: HealthStatus
        response_time_ms: float = 0.0
        details: Dict[str, Any] = {}

    class HealthResponse(BaseModel):
        status: HealthStatus
        timestamp: str
        version: str = "1.0.0"
        checks: Dict[str, HealthCheck] = {}


try:
    from darwin.db.manager import DatabaseManager
except ImportError:
    # Mock database manager for testing
    class DatabaseManager:
        def __init__(self):
            self.connected = True

        async def health_check(self):
            return True


# Check if we're in test mode
IS_TEST_MODE = os.getenv("TESTING", "false").lower() == "true"

# Only import logfire if not in test mode
if not IS_TEST_MODE:
    try:
        import logfire
    except ImportError:
        logfire = None
else:
    logfire = None

router = APIRouter()

# Store startup time for uptime calculation
startup_time = datetime.now(UTC)


async def get_database_manager() -> DatabaseManager:
    """Dependency to get database manager."""
    return DatabaseManager()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: DatabaseManager = Depends(get_database_manager)):
    """
    Comprehensive health check endpoint.

    Checks the health of all system components including:
    - Database connectivity
    - System resources (CPU, memory, disk)
    - External service dependencies
    - API performance metrics
    """
    start_time = time.time()

    try:
        # Perform individual health checks
        checks = []
        overall_status = HealthStatus.HEALTHY

        # Database health check
        db_check = await check_database_health(db)
        checks.append(db_check)
        if db_check.status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

        # Redis health check
        redis_check = await check_redis_health()
        checks.append(redis_check)
        if (
            redis_check.status != HealthStatus.HEALTHY
            and overall_status == HealthStatus.HEALTHY
        ):
            overall_status = HealthStatus.DEGRADED

        # System resource checks
        cpu_check = check_cpu_health()
        checks.append(cpu_check)
        if cpu_check.status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

        memory_check = check_memory_health()
        checks.append(memory_check)
        if memory_check.status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

        disk_check = check_disk_health()
        checks.append(disk_check)
        if disk_check.status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

        # Calculate uptime
        uptime = (datetime.now(UTC) - startup_time).total_seconds()

        # Calculate response time
        response_time = (time.time() - start_time) * 1000

        # System information
        system_info = {
            "python_version": get_python_version(),
            "platform": get_platform_info(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(shutil.disk_usage("/").total / (1024**3), 2),
            "process_id": get_process_id(),
            "active_connections": get_active_connections(),
        }

        # Log health check
        if not IS_TEST_MODE:
            logfire.info(
                "Health check performed",
                overall_status=overall_status,
                response_time_ms=response_time,
                checks_count=len(checks),
                uptime_seconds=uptime,
            )

        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now(UTC),
            version="1.0.0",
            uptime_seconds=uptime,
            checks=checks,
            system_info=system_info,
        )

        # Return appropriate HTTP status code
        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=503, content=response.model_dump(mode="json")
            )
        elif overall_status == HealthStatus.DEGRADED:
            return JSONResponse(
                status_code=200, content=response.model_dump(mode="json")
            )
        else:
            return response

    except Exception as e:
        if not IS_TEST_MODE:
            logfire.error(
                "Health check failed",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        # Return unhealthy status
        error_response = HealthResponse(
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(UTC),
            version="1.0.0",
            uptime_seconds=(datetime.now(UTC) - startup_time).total_seconds(),
            checks=[
                HealthCheck(
                    service="health_check",
                    status=HealthStatus.UNHEALTHY,
                    details={"error": str(e)},
                )
            ],
            system_info={},
        )

        return JSONResponse(
            status_code=503, content=error_response.model_dump(mode="json")
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.

    Simple endpoint that returns 200 if the service is running.
    Used by Kubernetes to determine if the pod should be restarted.
    """
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/ready")
async def readiness_probe(db: DatabaseManager = Depends(get_database_manager)):
    """
    Kubernetes readiness probe endpoint.

    Returns 200 if the service is ready to accept traffic.
    Checks critical dependencies like database connectivity.
    """
    try:
        # Check database connectivity
        db_healthy = await check_database_connectivity(db)

        if db_healthy:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "database_unavailable",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    except Exception as e:
        if not IS_TEST_MODE:
            logfire.error("Readiness probe failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get("/health/startup")
async def startup_probe():
    """
    Kubernetes startup probe endpoint.

    Returns 200 when the application has finished starting up.
    Used to indicate when the container is ready for liveness/readiness probes.
    """
    # Check if enough time has passed since startup
    startup_duration = (datetime.utcnow() - startup_time).total_seconds()

    if startup_duration > 30:  # Application should be ready after 30 seconds
        return {
            "status": "started",
            "startup_duration_seconds": startup_duration,
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "startup_duration_seconds": startup_duration,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


async def check_database_health(db: DatabaseManager) -> HealthCheck:
    """Check database health with detailed diagnostics."""
    start_time = time.time()

    try:
        # Test basic connectivity
        is_connected = await db.health_check()
        response_time = (time.time() - start_time) * 1000

        if is_connected:
            # Additional database checks
            details = {
                "connection_status": "connected",
                "response_time_ms": response_time,
                "connection_pool": await get_connection_pool_status(db),
            }

            status = HealthStatus.HEALTHY
            if response_time > 1000:  # > 1 second is slow
                status = HealthStatus.DEGRADED
                details["warning"] = "slow_response"
        else:
            details = {
                "connection_status": "disconnected",
                "response_time_ms": response_time,
            }
            status = HealthStatus.UNHEALTHY

        return HealthCheck(
            service="database",
            status=status,
            response_time_ms=response_time,
            details=details,
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return HealthCheck(
            service="database",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": str(e), "connection_status": "error"},
        )


async def check_redis_health() -> HealthCheck:
    """Check Redis health (placeholder for future Redis integration)."""
    start_time = time.time()

    try:
        # TODO: Implement actual Redis health check when Redis is integrated
        response_time = (time.time() - start_time) * 1000

        return HealthCheck(
            service="redis",
            status=HealthStatus.HEALTHY,
            response_time_ms=response_time,
            details={
                "connection_status": "not_configured",
                "note": "Redis integration pending",
            },
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return HealthCheck(
            service="redis",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=response_time,
            details={"error": str(e)},
        )


def check_cpu_health() -> HealthCheck:
    """Check CPU health and usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0]

        # Determine health status based on CPU usage
        if cpu_percent < 70:
            status = HealthStatus.HEALTHY
        elif cpu_percent < 90:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        details = {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "load_average_1m": load_avg[0],
            "load_average_5m": load_avg[1],
            "load_average_15m": load_avg[2],
        }

        return HealthCheck(service="cpu", status=status, details=details)

    except Exception as e:
        return HealthCheck(
            service="cpu", status=HealthStatus.UNHEALTHY, details={"error": str(e)}
        )


def check_memory_health() -> HealthCheck:
    """Check memory health and usage."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Determine health status based on memory usage
        if memory.percent < 80:
            status = HealthStatus.HEALTHY
        elif memory.percent < 95:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        details = {
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "swap_percent": swap.percent,
            "swap_total_gb": round(swap.total / (1024**3), 2),
        }

        return HealthCheck(service="memory", status=status, details=details)

    except Exception as e:
        return HealthCheck(
            service="memory", status=HealthStatus.UNHEALTHY, details={"error": str(e)}
        )


def check_disk_health() -> HealthCheck:
    """Check disk health and usage."""
    try:
        disk_usage = shutil.disk_usage("/")
        total_gb = disk_usage.total / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100

        # Determine health status based on disk usage
        if used_percent < 80:
            status = HealthStatus.HEALTHY
        elif used_percent < 95:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        details = {
            "disk_usage_percent": round(used_percent, 2),
            "disk_free_gb": round(free_gb, 2),
            "disk_total_gb": round(total_gb, 2),
        }

        return HealthCheck(service="disk", status=status, details=details)

    except Exception as e:
        return HealthCheck(
            service="disk", status=HealthStatus.UNHEALTHY, details={"error": str(e)}
        )


async def check_database_connectivity(db: DatabaseManager) -> bool:
    """Simple database connectivity check for readiness probe."""
    try:
        return await db.health_check()
    except:
        return False


async def get_connection_pool_status(db: DatabaseManager) -> Dict[str, Any]:
    """Get database connection pool status."""
    try:
        # TODO: Implement actual connection pool status check
        return {
            "active_connections": 1,
            "idle_connections": 9,
            "max_connections": 10,
            "status": "healthy",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_python_version() -> str:
    """Get Python version information."""
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_platform_info() -> str:
    """Get platform information."""
    import platform

    return f"{platform.system()} {platform.release()}"


def get_process_id() -> int:
    """Get current process ID."""
    import os

    return os.getpid()


def get_active_connections() -> int:
    """Get number of active network connections."""
    try:
        connections = psutil.net_connections(kind="inet")
        return len([conn for conn in connections if conn.status == "ESTABLISHED"])
    except:
        return 0

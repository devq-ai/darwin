"""
Darwin Health Monitoring System

This module provides comprehensive health checking capabilities for the Darwin platform,
including database connectivity, API endpoint health, MCP server status, WebSocket
connections, and system resource monitoring.

Features:
- Database connection health checks
- API endpoint availability monitoring
- MCP server status verification
- WebSocket connection health
- System resource monitoring (CPU, memory, disk)
- Custom health check registration
- Health status aggregation and reporting
- Integration with Logfire for health event logging
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    text = None
    AsyncSession = None

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system resource monitoring disabled")


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    critical: bool = False


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
    environment: str = "development"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "environment": self.environment,
            "checks": {
                check.name: {
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                    "timestamp": check.timestamp.isoformat(),
                    "duration_ms": check.duration_ms,
                    "critical": check.critical,
                }
                for check in self.checks
            },
        }


class HealthChecker:
    """Main health checking system for Darwin platform."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize health checker.

        Args:
            config: Health check configuration
        """
        self.config = config or {}
        self.check_interval = self.config.get("check_interval", 60)
        self.timeout = self.config.get("timeout", 10)
        self.critical_services = self.config.get(
            "critical_services", ["database", "api", "mcp_server", "websocket"]
        )

        self.start_time = time.time()
        self.custom_checks: Dict[str, Callable] = {}
        self.last_health_status: Optional[SystemHealth] = None
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 100

        # Database session factory (to be set externally)
        self.db_session_factory: Optional[Callable] = None

        logger.info("Health checker initialized")

    def register_db_session_factory(self, session_factory: Callable):
        """Register database session factory for health checks."""
        self.db_session_factory = session_factory

    def register_custom_check(self, name: str, check_func: Callable):
        """
        Register a custom health check function.

        Args:
            name: Unique name for the health check
            check_func: Async function that returns HealthCheck
        """
        self.custom_checks[name] = check_func
        logger.info(f"Registered custom health check: {name}")

    async def check_database_health(self) -> HealthCheck:
        """Check database connectivity and health."""
        start_time = time.time()

        try:
            if not SQLALCHEMY_AVAILABLE:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="SQLAlchemy not available - database monitoring disabled",
                    details={"sqlalchemy_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True,
                )

            if not self.db_session_factory:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="Database session factory not configured",
                    critical=True,
                )

            async with self.db_session_factory() as session:
                # Simple connectivity test
                result = await session.execute(text("SELECT 1"))
                result.scalar()

                # Check if we can perform basic operations
                await session.execute(text("SELECT current_timestamp"))

                duration_ms = (time.time() - start_time) * 1000

                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    details={"connection_time_ms": duration_ms, "query_test": "passed"},
                    duration_ms=duration_ms,
                    critical=True,
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=True,
            )

    async def check_api_health(self) -> HealthCheck:
        """Check API endpoint health."""
        start_time = time.time()

        try:
            if not HTTPX_AVAILABLE:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.UNKNOWN,
                    message="httpx not available - API monitoring disabled",
                    details={"httpx_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True,
                )

            # Check if we can reach the API internally
            api_url = self.config.get("api_url", "http://localhost:8000")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{api_url}/health")

                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return HealthCheck(
                        name="api",
                        status=HealthStatus.HEALTHY,
                        message="API endpoint responding",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                        },
                        duration_ms=duration_ms,
                        critical=True,
                    )
                else:
                    return HealthCheck(
                        name="api",
                        status=HealthStatus.DEGRADED,
                        message=f"API returned status {response.status_code}",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                        },
                        duration_ms=duration_ms,
                        critical=True,
                    )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=True,
            )

    async def check_mcp_server_health(self) -> HealthCheck:
        """Check MCP server health."""
        start_time = time.time()

        try:
            if not HTTPX_AVAILABLE:
                return HealthCheck(
                    name="mcp_server",
                    status=HealthStatus.UNKNOWN,
                    message="httpx not available - MCP server monitoring disabled",
                    details={"httpx_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True,
                )

            # Check if MCP server process is running
            mcp_port = self.config.get("mcp_port", 3000)
            mcp_host = self.config.get("mcp_host", "localhost")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try to connect to MCP server health endpoint
                response = await client.get(f"http://{mcp_host}:{mcp_port}/health")

                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return HealthCheck(
                        name="mcp_server",
                        status=HealthStatus.HEALTHY,
                        message="MCP server responding",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                            "host": mcp_host,
                            "port": mcp_port,
                        },
                        duration_ms=duration_ms,
                        critical=True,
                    )
                else:
                    return HealthCheck(
                        name="mcp_server",
                        status=HealthStatus.DEGRADED,
                        message=f"MCP server returned status {response.status_code}",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                        },
                        duration_ms=duration_ms,
                        critical=True,
                    )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="mcp_server",
                status=HealthStatus.UNHEALTHY,
                message=f"MCP server check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=True,
            )

    async def check_websocket_health(self) -> HealthCheck:
        """Check WebSocket server health."""
        start_time = time.time()

        try:
            # For now, we'll do a basic port check
            # In production, this could be enhanced with actual WebSocket connection test
            websocket_port = self.config.get("websocket_port", 8000)

            # Check if the port is accessible (simplified check)
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex(("localhost", websocket_port))
            sock.close()

            duration_ms = (time.time() - start_time) * 1000

            if result == 0:
                return HealthCheck(
                    name="websocket",
                    status=HealthStatus.HEALTHY,
                    message="WebSocket port accessible",
                    details={"port": websocket_port, "connection_time_ms": duration_ms},
                    duration_ms=duration_ms,
                    critical=False,
                )
            else:
                return HealthCheck(
                    name="websocket",
                    status=HealthStatus.UNHEALTHY,
                    message=f"WebSocket port {websocket_port} not accessible",
                    details={"port": websocket_port, "error_code": result},
                    duration_ms=duration_ms,
                    critical=False,
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="websocket",
                status=HealthStatus.UNHEALTHY,
                message=f"WebSocket check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=False,
            )

    async def check_memory_health(self) -> HealthCheck:
        """Check system memory health."""
        start_time = time.time()

        try:
            if not PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available - memory monitoring disabled",
                    details={"psutil_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=False,
                )

            memory = psutil.virtual_memory()
            memory_threshold = self.config.get("memory_threshold", 0.85)  # 85%

            duration_ms = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory.percent}%"

            if memory.percent / 100 > memory_threshold:
                status = (
                    HealthStatus.DEGRADED
                    if memory.percent < 95
                    else HealthStatus.UNHEALTHY
                )
                message = f"High memory usage: {memory.percent}%"

            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "used_percent": memory.percent,
                    "used_bytes": memory.used,
                    "available_bytes": memory.available,
                    "total_bytes": memory.total,
                    "threshold_percent": memory_threshold * 100,
                },
                duration_ms=duration_ms,
                critical=False,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=False,
            )

    async def check_cpu_health(self) -> HealthCheck:
        """Check system CPU health."""
        start_time = time.time()

        try:
            if not PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="cpu",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available - CPU monitoring disabled",
                    details={"psutil_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=False,
                )

            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_threshold = self.config.get("cpu_threshold", 0.90)  # 90%

            duration_ms = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY
            message = f"CPU usage: {cpu_percent}%"

            if cpu_percent / 100 > cpu_threshold:
                status = (
                    HealthStatus.DEGRADED
                    if cpu_percent < 95
                    else HealthStatus.UNHEALTHY
                )
                message = f"High CPU usage: {cpu_percent}%"

            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                details={
                    "usage_percent": cpu_percent,
                    "threshold_percent": cpu_threshold * 100,
                    "cpu_count": psutil.cpu_count(),
                    "load_average": os.getloadavg()
                    if hasattr(os, "getloadavg")
                    else None,
                },
                duration_ms=duration_ms,
                critical=False,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=False,
            )

    async def check_disk_health(self) -> HealthCheck:
        """Check system disk health."""
        start_time = time.time()

        try:
            if not PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available - disk monitoring disabled",
                    details={"psutil_available": False},
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=False,
                )

            # Check root disk usage
            disk = psutil.disk_usage("/")
            disk_threshold = self.config.get("disk_threshold", 0.85)  # 85%

            duration_ms = (time.time() - start_time) * 1000

            used_percent = (disk.used / disk.total) * 100

            status = HealthStatus.HEALTHY
            message = f"Disk usage: {used_percent:.1f}%"

            if used_percent / 100 > disk_threshold:
                status = (
                    HealthStatus.DEGRADED
                    if used_percent < 95
                    else HealthStatus.UNHEALTHY
                )
                message = f"High disk usage: {used_percent:.1f}%"

            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                details={
                    "used_percent": used_percent,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                    "total_bytes": disk.total,
                    "threshold_percent": disk_threshold * 100,
                },
                duration_ms=duration_ms,
                critical=False,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                critical=False,
            )

    async def run_all_checks(self) -> SystemHealth:
        """Run all health checks and return system health status."""
        checks = []

        # Run core health checks
        check_tasks = [
            self.check_database_health(),
            self.check_api_health(),
            self.check_mcp_server_health(),
            self.check_websocket_health(),
            self.check_memory_health(),
            self.check_cpu_health(),
            self.check_disk_health(),
        ]

        # Add custom checks
        for name, check_func in self.custom_checks.items():
            check_tasks.append(check_func())

        # Run all checks concurrently
        try:
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

            for i, result in enumerate(check_results):
                if isinstance(result, Exception):
                    # Handle check that threw an exception
                    check_name = f"check_{i}"
                    checks.append(
                        HealthCheck(
                            name=check_name,
                            status=HealthStatus.UNKNOWN,
                            message=f"Check failed with exception: {str(result)}",
                            details={"error": str(result)},
                            critical=False,
                        )
                    )
                else:
                    checks.append(result)

        except Exception as e:
            logger.error(f"Failed to run health checks: {e}")
            checks.append(
                HealthCheck(
                    name="health_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check system failed: {str(e)}",
                    details={"error": str(e)},
                    critical=True,
                )
            )

        # Determine overall system health
        overall_status = self._determine_overall_status(checks)

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        system_health = SystemHealth(
            status=overall_status,
            checks=checks,
            uptime_seconds=uptime_seconds,
            version=self.config.get("version", "1.0.0"),
            environment=self.config.get("environment", "development"),
        )

        # Store in history
        self.last_health_status = system_health
        self.health_history.append(system_health)

        # Trim history if needed
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size :]

        return system_health

    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN

        # Check for critical failures
        critical_unhealthy = any(
            check.status == HealthStatus.UNHEALTHY and check.critical
            for check in checks
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # Check for any unhealthy services
        any_unhealthy = any(check.status == HealthStatus.UNHEALTHY for check in checks)

        if any_unhealthy:
            return HealthStatus.DEGRADED

        # Check for degraded services
        any_degraded = any(check.status == HealthStatus.DEGRADED for check in checks)

        if any_degraded:
            return HealthStatus.DEGRADED

        # Check for unknown status
        any_unknown = any(check.status == HealthStatus.UNKNOWN for check in checks)

        if any_unknown:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        return [health.to_dict() for health in self.health_history[-limit:]]

    def add_health_endpoints(self, app: FastAPI):
        """Add health check endpoints to FastAPI application."""

        @app.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            try:
                health_status = await self.run_all_checks()

                if health_status.status == HealthStatus.HEALTHY:
                    return JSONResponse(
                        status_code=200, content=health_status.to_dict()
                    )
                elif health_status.status == HealthStatus.DEGRADED:
                    return JSONResponse(
                        status_code=200,  # Still operational but degraded
                        content=health_status.to_dict(),
                    )
                else:
                    return JSONResponse(
                        status_code=503,  # Service unavailable
                        content=health_status.to_dict(),
                    )

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "unhealthy",
                        "message": f"Health check system error: {str(e)}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        @app.get("/health/detailed")
        async def detailed_health_check():
            """Detailed health check with all component status."""
            try:
                health_status = await self.run_all_checks()
                return JSONResponse(status_code=200, content=health_status.to_dict())
            except Exception as e:
                logger.error(f"Detailed health check failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Health check system error: {str(e)}"
                )

        @app.get("/health/history")
        async def health_history(limit: int = 10):
            """Get health check history."""
            try:
                history = self.get_health_history(limit)
                return JSONResponse(
                    status_code=200, content={"history": history, "count": len(history)}
                )
            except Exception as e:
                logger.error(f"Health history check failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Health history error: {str(e)}"
                )

        logger.info("Health check endpoints added to FastAPI application")

    async def start_periodic_checks(self):
        """Start periodic health checks in background."""
        logger.info(
            f"Starting periodic health checks every {self.check_interval} seconds"
        )

        while True:
            try:
                health_status = await self.run_all_checks()
                logger.info(f"Health check completed: {health_status.status.value}")

                # Log critical issues
                critical_issues = [
                    check
                    for check in health_status.checks
                    if check.status == HealthStatus.UNHEALTHY and check.critical
                ]

                if critical_issues:
                    logger.error(
                        f"Critical health issues detected: {[c.name for c in critical_issues]}"
                    )

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Periodic health check failed: {e}")
                await asyncio.sleep(self.check_interval)

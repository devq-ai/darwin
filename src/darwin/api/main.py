"""
Darwin Genetic Algorithm Solver - FastAPI Main Application

This module implements the core FastAPI application for the Darwin genetic algorithm
optimization platform. It provides REST API endpoints for creating, managing, and
monitoring genetic algorithm optimization runs.

Features:
- RESTful API endpoints for optimizer management
- Real-time progress monitoring
- LogFire integration for comprehensive logging
- OpenAPI documentation generation
- CORS support for frontend integration
- Input validation with Pydantic models
- Error handling and health checks
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from darwin.api.routes import algorithms, health, metrics, optimizers, templates
from darwin.core.optimizer import GeneticOptimizer as GeneticAlgorithm
from darwin.core.problem import OptimizationProblem
from darwin.db.manager import DatabaseManager

# Check if we're in test mode
IS_TEST_MODE = os.getenv("TESTING", "false").lower() == "true"

# Only import and configure LogFire if not in test mode
if not IS_TEST_MODE:
    import logfire

    from darwin.api.middleware.logging import setup_logfire_middleware

    # Configure LogFire
    logfire.configure(
        service_name="darwin-api", service_version="1.0.0", environment="development"
    )

logger = logging.getLogger(__name__)

# Global state for optimization runs
optimization_runs: Dict[str, Dict] = {}
database_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting Darwin API application")

    global database_manager
    try:
        database_manager = DatabaseManager()
        await database_manager.connect()
        logger.info("Database connection established")
    except Exception as e:
        if IS_TEST_MODE:
            logger.warning(f"Test mode: Ignoring database connection error: {e}")
            database_manager = DatabaseManager()
            database_manager.connected = True
        else:
            logger.error(f"Failed to connect to database: {e}")
            raise

    # Configure LogFire for the application only if not in test mode
    if not IS_TEST_MODE:
        logfire.instrument_fastapi(app)

    yield

    # Shutdown
    logger.info("Shutting down Darwin API application")
    if database_manager:
        await database_manager.disconnect()
        logger.info("Database connection closed")


# Create FastAPI application
app = FastAPI(
    title="Darwin Genetic Algorithm Solver API",
    description="REST API for the Darwin genetic algorithm optimization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Setup LogFire middleware only if not in test mode
if not IS_TEST_MODE:
    setup_logfire_middleware(app)

# Include routers
app.include_router(optimizers.router, prefix="/api/v1", tags=["optimizers"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(templates.router, prefix="/api/v1", tags=["templates"])
app.include_router(algorithms.router, prefix="/api/v1", tags=["algorithms"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with LogFire integration."""
    if not IS_TEST_MODE:
        logfire.error(
            "Unhandled exception in API request",
            request_url=str(request.url),
            request_method=request.method,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
    else:
        logger.error(
            f"Unhandled exception in API request: {exc}",
            extra={
                "request_url": str(request.url),
                "request_method": request.method,
                "exception_type": type(exc).__name__,
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with LogFire integration."""
    if not IS_TEST_MODE:
        logfire.warning(
            "HTTP exception in API request",
            request_url=str(request.url),
            request_method=request.method,
            status_code=exc.status_code,
            detail=exc.detail,
        )
    else:
        logger.warning(
            f"HTTP exception in API request: {exc.detail}",
            extra={
                "request_url": str(request.url),
                "request_method": request.method,
                "status_code": exc.status_code,
            },
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Darwin Genetic Algorithm Solver API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/api/v1/health",
        "timestamp": datetime.now(UTC).isoformat(),
    }


# Dependency to get database manager
async def get_database_manager() -> DatabaseManager:
    """Dependency to provide database manager instance."""
    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    return database_manager


# Utility functions for optimization management
async def create_optimizer_instance(
    problem: OptimizationProblem, config: Dict
) -> GeneticAlgorithm:
    """Create a new genetic algorithm optimizer instance."""
    try:
        optimizer = GeneticAlgorithm(
            problem=problem,
            population_size=config.get("population_size", 50),
            max_generations=config.get("max_generations", 100),
            selection_type=config.get("selection_type", "tournament"),
            crossover_type=config.get("crossover_type", "single_point"),
            mutation_type=config.get("mutation_type", "uniform"),
            crossover_probability=config.get("crossover_probability", 0.8),
            mutation_probability=config.get("mutation_probability", 0.1),
            elitism=config.get("elitism", True),
            adaptive_params=config.get("adaptive_params", False),
        )

        return optimizer

    except Exception as e:
        if not IS_TEST_MODE:
            logfire.error(
                "Failed to create optimizer instance",
                problem_name=problem.name,
                config=config,
                error=str(e),
            )
        else:
            logger.error(f"Failed to create optimizer instance: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to create optimizer: {str(e)}"
        )


async def run_optimization_background(
    optimizer_id: str, optimizer: GeneticAlgorithm, db_manager: DatabaseManager
):
    """Run optimization in background task."""
    try:
        if not IS_TEST_MODE:
            logfire.info(
                "Starting optimization run",
                optimizer_id=optimizer_id,
                problem_name=optimizer.problem.name,
            )
        else:
            logger.info(f"Starting optimization run: {optimizer_id}")

        # Update status to running
        optimization_runs[optimizer_id]["status"] = "running"
        optimization_runs[optimizer_id]["start_time"] = datetime.now(UTC)

        # Store initial state in database
        await db_manager.store_optimizer_run(
            optimizer_id, optimization_runs[optimizer_id]
        )

        # Run optimization
        results = await optimizer.optimize()

        # Update with results
        optimization_runs[optimizer_id]["status"] = "completed"
        optimization_runs[optimizer_id]["end_time"] = datetime.now(UTC)
        optimization_runs[optimizer_id]["results"] = results
        optimization_runs[optimizer_id]["best_fitness"] = float(results.best_fitness)
        optimization_runs[optimizer_id]["generations_completed"] = results.generations

        # Store final results in database
        await db_manager.store_optimizer_results(optimizer_id, results)

        if not IS_TEST_MODE:
            logfire.info(
                "Optimization completed successfully",
                optimizer_id=optimizer_id,
                best_fitness=float(results.best_fitness),
                generations=results.generations,
            )
        else:
            logger.info(f"Optimization completed successfully: {optimizer_id}")

    except Exception as e:
        if not IS_TEST_MODE:
            logfire.error(
                "Optimization run failed", optimizer_id=optimizer_id, error=str(e)
            )
        else:
            logger.error(f"Optimization run failed: {optimizer_id} - {e}")

        optimization_runs[optimizer_id]["status"] = "failed"
        optimization_runs[optimizer_id]["error"] = str(e)
        optimization_runs[optimizer_id]["end_time"] = datetime.now(UTC)


# Health check utilities
async def check_database_connection() -> bool:
    """Check database connection health."""
    try:
        if database_manager:
            return await database_manager.health_check()
        return False
    except Exception:
        return False


async def check_redis_connection() -> bool:
    """Check Redis connection health."""
    try:
        # TODO: Implement Redis health check when Redis is integrated
        return True
    except Exception:
        return False


def check_disk_space() -> bool:
    """Check available disk space."""
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100
        return free_percent > 10  # At least 10% free space
    except Exception:
        return False


def check_memory_usage() -> bool:
    """Check memory usage."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% memory usage
    except ImportError:
        # psutil not available, assume healthy
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "darwin.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

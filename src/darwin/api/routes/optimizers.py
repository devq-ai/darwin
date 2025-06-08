"""
Darwin API Optimizer Routes

This module implements the core API endpoints for managing genetic algorithm optimizers.
Provides full CRUD operations and optimization run management.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from darwin.api.models.requests import OptimizationRunRequest, OptimizerCreateRequest
from darwin.api.models.responses import (
    HistoryResponse,
    OptimizerListResponse,
    OptimizerResponse,
    OptimizerStatus,
    ProgressResponse,
    ResultsResponse,
)
from darwin.core.optimizer import GeneticOptimizer as GeneticAlgorithm
from darwin.db.manager import DatabaseManager

# Check if we're in test mode
IS_TEST_MODE = os.getenv("TESTING", "false").lower() == "true"

# Only import logfire and middleware if not in test mode
if not IS_TEST_MODE:
    import logfire

    from darwin.api.middleware.logging import (
        log_optimization_event,
        performance_tracker,
    )

router = APIRouter()

# Global state for active optimizers
active_optimizers: Dict[str, Dict] = {}


async def get_database_manager() -> DatabaseManager:
    """Dependency to get database manager."""
    from darwin.api.main import database_manager

    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    return database_manager


@router.post("/optimizers", response_model=OptimizerResponse, status_code=201)
async def create_optimizer(
    request: OptimizerCreateRequest, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Create a new genetic algorithm optimizer.

    Creates an optimizer instance with the specified problem and configuration.
    The optimizer is ready to run but not automatically started.
    """
    start_time = datetime.utcnow()
    optimizer_id = str(uuid.uuid4())

    try:
        logfire.info(
            "Creating new optimizer",
            optimizer_id=optimizer_id,
            problem_name=request.problem.name,
            population_size=request.config.population_size,
            max_generations=request.config.max_generations,
        )

        # Create genetic algorithm instance
        optimizer = GeneticAlgorithm(
            problem=request.problem,
            population_size=request.config.population_size,
            max_generations=request.config.max_generations,
            selection_type=request.config.selection_type,
            crossover_type=request.config.crossover_type,
            mutation_type=request.config.mutation_type,
            crossover_probability=request.config.crossover_probability,
            mutation_probability=request.config.mutation_probability,
            elitism=request.config.elitism,
            adaptive_params=request.config.adaptive_params,
        )

        # Store optimizer data
        optimizer_data = {
            "optimizer_id": optimizer_id,
            "name": request.name or f"Optimizer_{optimizer_id[:8]}",
            "status": OptimizerStatus.CREATED,
            "created_at": start_time,
            "updated_at": start_time,
            "problem": request.problem.dict(),
            "config": request.config.dict(),
            "tags": request.tags,
            "optimizer_instance": optimizer,
        }

        # Store in memory and database
        active_optimizers[optimizer_id] = optimizer_data
        await db.store_optimizer(optimizer_id, optimizer_data)

        # Log creation event
        log_optimization_event(
            "optimizer_created",
            optimizer_id,
            {
                "problem_name": request.problem.name,
                "variables_count": len(request.problem.variables),
                "constraints_count": len(request.problem.constraints),
            },
        )

        # Track performance
        creation_time = (datetime.utcnow() - start_time).total_seconds()
        performance_tracker.track_operation("optimizer_creation", creation_time)

        return OptimizerResponse(
            optimizer_id=optimizer_id,
            name=optimizer_data["name"],
            status=optimizer_data["status"],
            created_at=optimizer_data["created_at"],
            updated_at=optimizer_data["updated_at"],
            problem_name=request.problem.name,
            algorithm_config=request.config.dict(),
            tags=request.tags,
        )

    except Exception as e:
        logfire.error(
            "Failed to create optimizer",
            optimizer_id=optimizer_id,
            error=str(e),
            problem_name=request.problem.name,
        )
        raise HTTPException(
            status_code=400, detail=f"Failed to create optimizer: {str(e)}"
        )


@router.get("/optimizers/{optimizer_id}", response_model=OptimizerResponse)
async def get_optimizer(
    optimizer_id: str, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Get optimizer information by ID.

    Returns current status and configuration of the specified optimizer.
    """
    try:
        # Check in-memory first
        if optimizer_id in active_optimizers:
            data = active_optimizers[optimizer_id]
            return OptimizerResponse(
                optimizer_id=optimizer_id,
                name=data["name"],
                status=data["status"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                problem_name=data["problem"]["name"],
                algorithm_config=data["config"],
                tags=data["tags"],
            )

        # Check database
        data = await db.get_optimizer(optimizer_id)
        if not data:
            raise HTTPException(
                status_code=404, detail=f"Optimizer {optimizer_id} not found"
            )

        return OptimizerResponse(**data)

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to retrieve optimizer", optimizer_id=optimizer_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve optimizer information"
        )


@router.post("/optimizers/{optimizer_id}/run", response_model=ResultsResponse)
async def start_optimization(
    optimizer_id: str,
    request: OptimizationRunRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_database_manager),
):
    """
    Start optimization run for the specified optimizer.

    Begins the genetic algorithm optimization process in the background.
    Returns immediately with run information.
    """
    try:
        if optimizer_id not in active_optimizers:
            raise HTTPException(
                status_code=404, detail=f"Optimizer {optimizer_id} not found"
            )

        optimizer_data = active_optimizers[optimizer_id]

        # Check if already running
        if optimizer_data["status"] in [
            OptimizerStatus.RUNNING,
            OptimizerStatus.COMPLETED,
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Optimizer is already {optimizer_data['status']}",
            )

        # Update status
        optimizer_data["status"] = OptimizerStatus.RUNNING
        optimizer_data["updated_at"] = datetime.utcnow()
        optimizer_data["run_config"] = request.dict()

        # Start optimization in background
        background_tasks.add_task(
            run_optimization_background,
            optimizer_id,
            optimizer_data["optimizer_instance"],
            db,
            request,
        )

        log_optimization_event(
            "optimization_started",
            optimizer_id,
            {
                "run_name": request.run_name,
                "save_history": request.save_history,
                "timeout_seconds": request.timeout_seconds,
            },
        )

        return ResultsResponse(
            optimizer_id=optimizer_id,
            status=OptimizerStatus.RUNNING,
            results=None,
            start_time=datetime.utcnow(),
            progress_percentage=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to start optimization", optimizer_id=optimizer_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to start optimization")


@router.post("/optimizers/{optimizer_id}/stop")
async def stop_optimization(
    optimizer_id: str, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Stop running optimization.

    Gracefully stops the optimization process and saves current progress.
    """
    try:
        if optimizer_id not in active_optimizers:
            raise HTTPException(
                status_code=404, detail=f"Optimizer {optimizer_id} not found"
            )

        optimizer_data = active_optimizers[optimizer_id]

        if optimizer_data["status"] != OptimizerStatus.RUNNING:
            raise HTTPException(
                status_code=400, detail="Optimizer is not currently running"
            )

        # Set stop flag
        optimizer_data["status"] = OptimizerStatus.CANCELLED
        optimizer_data["updated_at"] = datetime.utcnow()
        optimizer_data["stop_requested"] = True

        # Update database
        await db.update_optimizer_status(optimizer_id, OptimizerStatus.CANCELLED)

        log_optimization_event(
            "optimization_stopped",
            optimizer_id,
            {"stopped_at": datetime.utcnow().isoformat()},
        )

        return {"message": "Optimization stop requested", "status": "stopping"}

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to stop optimization", optimizer_id=optimizer_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to stop optimization")


@router.get("/optimizers/{optimizer_id}/results", response_model=ResultsResponse)
async def get_optimization_results(
    optimizer_id: str, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Get optimization results.

    Returns the current or final results of the optimization run.
    """
    try:
        if optimizer_id not in active_optimizers:
            # Check database for completed optimizations
            results = await db.get_optimization_results(optimizer_id)
            if not results:
                raise HTTPException(
                    status_code=404, detail=f"Optimizer {optimizer_id} not found"
                )
            return ResultsResponse(**results)

        optimizer_data = active_optimizers[optimizer_id]

        # Calculate progress
        progress = 0.0
        if "current_generation" in optimizer_data:
            max_gen = optimizer_data["config"]["max_generations"]
            current_gen = optimizer_data.get("current_generation", 0)
            progress = (current_gen / max_gen) * 100 if max_gen > 0 else 0.0

        return ResultsResponse(
            optimizer_id=optimizer_id,
            status=optimizer_data["status"],
            results=optimizer_data.get("results"),
            start_time=optimizer_data.get("start_time"),
            end_time=optimizer_data.get("end_time"),
            progress_percentage=progress,
        )

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to get optimization results",
            optimizer_id=optimizer_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve optimization results"
        )


@router.delete("/optimizers/{optimizer_id}", status_code=204)
async def delete_optimizer(
    optimizer_id: str, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Delete optimizer and all associated data.

    Permanently removes the optimizer and its optimization history.
    """
    try:
        # Check if optimizer exists
        if optimizer_id not in active_optimizers:
            stored_data = await db.get_optimizer(optimizer_id)
            if not stored_data:
                raise HTTPException(
                    status_code=404, detail=f"Optimizer {optimizer_id} not found"
                )
        else:
            # Stop if running
            optimizer_data = active_optimizers[optimizer_id]
            if optimizer_data["status"] == OptimizerStatus.RUNNING:
                optimizer_data["stop_requested"] = True

            # Remove from memory
            del active_optimizers[optimizer_id]

        # Delete from database
        await db.delete_optimizer(optimizer_id)

        log_optimization_event(
            "optimizer_deleted",
            optimizer_id,
            {"deleted_at": datetime.utcnow().isoformat()},
        )

        return

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to delete optimizer", optimizer_id=optimizer_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to delete optimizer")


@router.get("/optimizers/{optimizer_id}/progress", response_model=ProgressResponse)
async def get_optimization_progress(
    optimizer_id: str, db: DatabaseManager = Depends(get_database_manager)
):
    """
    Get real-time optimization progress.

    Returns current generation progress and performance metrics.
    """
    try:
        if optimizer_id not in active_optimizers:
            raise HTTPException(
                status_code=404, detail=f"Optimizer {optimizer_id} not found"
            )

        optimizer_data = active_optimizers[optimizer_id]

        # Get current progress from optimizer instance
        current_progress = optimizer_data.get(
            "current_progress",
            {
                "generation": 0,
                "best_fitness": 0.0,
                "average_fitness": 0.0,
                "worst_fitness": 0.0,
                "diversity": 0.0,
                "convergence_rate": 0.0,
            },
        )

        # Get progress history
        history = optimizer_data.get("progress_history", [])

        return ProgressResponse(
            optimizer_id=optimizer_id,
            status=optimizer_data["status"],
            current_progress=current_progress,
            history=history[-100:],  # Last 100 entries
            last_updated=optimizer_data["updated_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to get optimization progress",
            optimizer_id=optimizer_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve optimization progress"
        )


@router.get("/optimizers/{optimizer_id}/history", response_model=HistoryResponse)
async def get_evolution_history(
    optimizer_id: str,
    generations: Optional[int] = Query(
        None, description="Number of recent generations"
    ),
    db: DatabaseManager = Depends(get_database_manager),
):
    """
    Get evolution history data.

    Returns detailed history of the evolutionary process.
    """
    try:
        if optimizer_id not in active_optimizers:
            # Get from database
            history_data = await db.get_evolution_history(optimizer_id, generations)
            if not history_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No history found for optimizer {optimizer_id}",
                )
            return HistoryResponse(**history_data)

        optimizer_data = active_optimizers[optimizer_id]
        full_history = optimizer_data.get("evolution_history", [])

        # Apply generation limit if specified
        if generations:
            history = full_history[-generations:]
        else:
            history = full_history

        return HistoryResponse(
            optimizer_id=optimizer_id,
            total_generations=len(full_history),
            history=history,
            sampling_rate=1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to get evolution history", optimizer_id=optimizer_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve evolution history"
        )


@router.get("/optimizers", response_model=OptimizerListResponse)
async def list_optimizers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[OptimizerStatus] = Query(None, description="Filter by status"),
    db: DatabaseManager = Depends(get_database_manager),
):
    """
    List all optimizers with pagination and filtering.

    Returns a paginated list of optimizers with optional status filtering.
    """
    try:
        # Get from database with pagination
        offset = (page - 1) * page_size
        optimizers_data = await db.list_optimizers(
            offset=offset, limit=page_size, status_filter=status
        )

        total_count = await db.count_optimizers(status_filter=status)
        total_pages = (total_count + page_size - 1) // page_size

        optimizers = [OptimizerResponse(**opt) for opt in optimizers_data]

        return OptimizerListResponse(
            optimizers=optimizers,
            page=page,
            page_size=page_size,
            total_items=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    except Exception as e:
        logfire.error(
            "Failed to list optimizers", error=str(e), page=page, page_size=page_size
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve optimizers list"
        )


async def run_optimization_background(
    optimizer_id: str,
    optimizer: GeneticAlgorithm,
    db: DatabaseManager,
    run_config: OptimizationRunRequest,
):
    """
    Run optimization in background task.

    Executes the genetic algorithm and updates progress in real-time.
    """
    try:
        optimizer_data = active_optimizers[optimizer_id]

        logfire.info(
            "Starting background optimization",
            optimizer_id=optimizer_id,
            max_generations=optimizer.max_generations,
        )

        # Initialize progress tracking
        optimizer_data["start_time"] = datetime.utcnow()
        optimizer_data["progress_history"] = []
        optimizer_data["evolution_history"] = []

        # Run optimization with progress callbacks
        def progress_callback(generation, population, fitness_values):
            """Callback for tracking optimization progress."""
            try:
                # Update current progress
                progress = {
                    "generation": generation,
                    "best_fitness": float(max(fitness_values)),
                    "average_fitness": float(sum(fitness_values) / len(fitness_values)),
                    "worst_fitness": float(min(fitness_values)),
                    "diversity": calculate_diversity(population),
                    "convergence_rate": calculate_convergence_rate(
                        optimizer_data["progress_history"]
                    ),
                }

                optimizer_data["current_progress"] = progress
                optimizer_data["current_generation"] = generation
                optimizer_data["progress_history"].append(progress)
                optimizer_data["updated_at"] = datetime.utcnow()

                # Store evolution history
                evolution_entry = {
                    "generation": generation,
                    "timestamp": datetime.utcnow(),
                    "best_fitness": progress["best_fitness"],
                    "average_fitness": progress["average_fitness"],
                    "diversity": progress["diversity"],
                }
                optimizer_data["evolution_history"].append(evolution_entry)

                # Check for stop request
                if optimizer_data.get("stop_requested"):
                    return False  # Signal to stop optimization

                return True  # Continue optimization

            except Exception as e:
                logfire.error(
                    "Error in progress callback",
                    optimizer_id=optimizer_id,
                    generation=generation,
                    error=str(e),
                )
                return True  # Continue despite callback error

        # Run the optimization
        results = await optimizer.optimize(progress_callback=progress_callback)

        # Update final results
        optimizer_data["status"] = OptimizerStatus.COMPLETED
        optimizer_data["end_time"] = datetime.utcnow()
        optimizer_data["results"] = results
        optimizer_data["updated_at"] = datetime.utcnow()

        # Store final results in database
        await db.store_optimization_results(optimizer_id, results)

        log_optimization_event(
            "optimization_completed",
            optimizer_id,
            {
                "best_fitness": float(results.best_fitness),
                "generations": results.generations,
                "total_evaluations": results.total_evaluations,
            },
        )

    except Exception as e:
        # Handle optimization failure
        optimizer_data = active_optimizers.get(optimizer_id, {})
        optimizer_data["status"] = OptimizerStatus.FAILED
        optimizer_data["error"] = str(e)
        optimizer_data["end_time"] = datetime.utcnow()
        optimizer_data["updated_at"] = datetime.utcnow()

        logfire.error("Optimization failed", optimizer_id=optimizer_id, error=str(e))

        log_optimization_event("optimization_failed", optimizer_id, {"error": str(e)})


def calculate_diversity(population) -> float:
    """Calculate population diversity measure."""
    try:
        # Simple diversity calculation based on variance
        import numpy as np

        if len(population) == 0:
            return 0.0

        population_array = np.array(population)
        return float(np.std(population_array))
    except:
        return 0.0


def calculate_convergence_rate(history) -> float:
    """Calculate convergence rate from progress history."""
    try:
        if len(history) < 2:
            return 0.0

        recent_fitness = [h["best_fitness"] for h in history[-10:]]
        if len(recent_fitness) < 2:
            return 0.0

        # Simple convergence rate calculation
        improvement = recent_fitness[-1] - recent_fitness[0]
        return float(abs(improvement) / len(recent_fitness))
    except:
        return 0.0

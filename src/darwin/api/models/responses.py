"""
Darwin API Response Models

This module defines Pydantic models for API response data.
All response models include comprehensive typing and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OptimizerStatus(str, Enum):
    """Optimizer status enumeration."""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class OptimizerResponse(BaseModel):
    """Response model for optimizer operations."""

    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    name: Optional[str] = Field(None, description="Optimizer name")
    status: OptimizerStatus = Field(..., description="Current optimizer status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    problem_name: str = Field(..., description="Optimization problem name")
    algorithm_config: Dict[str, Any] = Field(..., description="Algorithm configuration")
    tags: List[str] = Field(default=[], description="Optimizer tags")


class OptimizationResults(BaseModel):
    """Optimization results data."""

    best_fitness: float = Field(..., description="Best fitness value achieved")
    best_solution: List[float] = Field(..., description="Best solution variables")
    generations_completed: int = Field(
        ..., description="Number of generations completed"
    )
    total_evaluations: int = Field(
        ..., description="Total fitness evaluations performed"
    )
    convergence_achieved: bool = Field(
        ..., description="Whether convergence was achieved"
    )
    final_diversity: float = Field(..., description="Final population diversity")
    execution_time_seconds: float = Field(
        ..., description="Total execution time in seconds"
    )
    pareto_front: Optional[List[Dict[str, float]]] = Field(
        None, description="Pareto front for multi-objective optimization"
    )


class ResultsResponse(BaseModel):
    """Response model for optimization results."""

    optimizer_id: str = Field(..., description="Optimizer identifier")
    status: OptimizerStatus = Field(..., description="Optimization status")
    results: Optional[OptimizationResults] = Field(
        None, description="Optimization results"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    start_time: Optional[datetime] = Field(None, description="Optimization start time")
    end_time: Optional[datetime] = Field(None, description="Optimization end time")
    progress_percentage: float = Field(..., description="Completion percentage")


class ProgressUpdate(BaseModel):
    """Real-time progress update data."""

    generation: int = Field(..., description="Current generation number")
    best_fitness: float = Field(..., description="Best fitness in current generation")
    average_fitness: float = Field(
        ..., description="Average fitness in current generation"
    )
    worst_fitness: float = Field(..., description="Worst fitness in current generation")
    diversity: float = Field(..., description="Population diversity measure")
    convergence_rate: float = Field(..., description="Rate of convergence")
    estimated_time_remaining: Optional[float] = Field(
        None, description="Estimated time remaining in seconds"
    )


class ProgressResponse(BaseModel):
    """Response model for progress monitoring."""

    optimizer_id: str = Field(..., description="Optimizer identifier")
    status: OptimizerStatus = Field(..., description="Current status")
    current_progress: ProgressUpdate = Field(..., description="Current progress data")
    history: List[ProgressUpdate] = Field(default=[], description="Progress history")
    last_updated: datetime = Field(..., description="Last update timestamp")


class EvolutionHistory(BaseModel):
    """Evolution history data."""

    generation: int = Field(..., description="Generation number")
    timestamp: datetime = Field(..., description="Generation timestamp")
    best_fitness: float = Field(..., description="Best fitness value")
    average_fitness: float = Field(..., description="Average fitness value")
    diversity: float = Field(..., description="Population diversity")
    selected_parents: int = Field(..., description="Number of selected parents")
    offspring_created: int = Field(..., description="Number of offspring created")
    mutations_applied: int = Field(..., description="Number of mutations applied")


class HistoryResponse(BaseModel):
    """Response model for evolution history."""

    optimizer_id: str = Field(..., description="Optimizer identifier")
    total_generations: int = Field(..., description="Total generations completed")
    history: List[EvolutionHistory] = Field(..., description="Evolution history data")
    sampling_rate: int = Field(..., description="History sampling rate")


class VisualizationData(BaseModel):
    """Visualization data container."""

    plot_type: str = Field(..., description="Type of visualization")
    data: Dict[str, Any] = Field(..., description="Plot data")
    metadata: Dict[str, Any] = Field(default={}, description="Plot metadata")
    generated_at: datetime = Field(..., description="Generation timestamp")


class VisualizationResponse(BaseModel):
    """Response model for visualizations."""

    optimizer_id: str = Field(..., description="Optimizer identifier")
    available_plots: List[str] = Field(..., description="Available plot types")
    generated_plot: Optional[VisualizationData] = Field(
        None, description="Generated plot data"
    )


class HealthCheck(BaseModel):
    """Individual health check result."""

    service: str = Field(..., description="Service name")
    status: HealthStatus = Field(..., description="Service health status")
    response_time_ms: Optional[float] = Field(
        None, description="Response time in milliseconds"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional health details"
    )


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: HealthStatus = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    checks: List[HealthCheck] = Field(
        ..., description="Individual service health checks"
    )
    system_info: Dict[str, Any] = Field(default={}, description="System information")


class ProblemTemplate(BaseModel):
    """Problem template definition."""

    id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Problem category")
    difficulty: str = Field(..., description="Problem difficulty level")
    variables_count: int = Field(..., description="Number of variables")
    constraints_count: int = Field(..., description="Number of constraints")
    default_config: Dict[str, Any] = Field(
        ..., description="Default algorithm configuration"
    )
    example_usage: str = Field(..., description="Example usage instructions")


class TemplatesResponse(BaseModel):
    """Response model for problem templates."""

    templates: List[ProblemTemplate] = Field(
        ..., description="Available problem templates"
    )
    categories: List[str] = Field(..., description="Available categories")
    total_count: int = Field(..., description="Total number of templates")


class AlgorithmInfo(BaseModel):
    """Algorithm information."""

    name: str = Field(..., description="Algorithm name")
    description: str = Field(..., description="Algorithm description")
    parameters: Dict[str, Any] = Field(..., description="Algorithm parameters")
    suitable_for: List[str] = Field(
        ..., description="Problem types suitable for this algorithm"
    )
    complexity: str = Field(..., description="Algorithm complexity")
    references: List[str] = Field(default=[], description="Academic references")


class AlgorithmsResponse(BaseModel):
    """Response model for available algorithms."""

    algorithms: List[AlgorithmInfo] = Field(..., description="Available algorithms")
    total_count: int = Field(..., description="Total number of algorithms")


class SystemMetrics(BaseModel):
    """System performance metrics."""

    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    active_optimizers: int = Field(..., description="Number of active optimizers")
    total_requests: int = Field(..., description="Total API requests handled")
    average_response_time_ms: float = Field(..., description="Average response time")
    uptime_seconds: float = Field(..., description="System uptime")


class APIMetrics(BaseModel):
    """API-specific metrics."""

    requests_per_minute: float = Field(..., description="Requests per minute")
    error_rate_percent: float = Field(..., description="Error rate percentage")
    p95_response_time_ms: float = Field(
        ..., description="95th percentile response time"
    )
    p99_response_time_ms: float = Field(
        ..., description="99th percentile response time"
    )
    active_connections: int = Field(..., description="Active WebSocket connections")
    cache_hit_rate_percent: float = Field(..., description="Cache hit rate percentage")


class MetricsResponse(BaseModel):
    """Response model for system metrics."""

    timestamp: datetime = Field(..., description="Metrics timestamp")
    system: SystemMetrics = Field(..., description="System metrics")
    api: APIMetrics = Field(..., description="API metrics")
    optimization_stats: Dict[str, Any] = Field(
        default={}, description="Optimization-specific statistics"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class OptimizerListResponse(PaginatedResponse):
    """Response model for listing optimizers."""

    optimizers: List[OptimizerResponse] = Field(..., description="List of optimizers")


class ExportResponse(BaseModel):
    """Response model for data export."""

    export_id: str = Field(..., description="Export identifier")
    format: str = Field(..., description="Export format")
    file_size_bytes: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="Download URL")
    expires_at: datetime = Field(..., description="Download URL expiration")
    checksum: str = Field(..., description="File checksum for verification")

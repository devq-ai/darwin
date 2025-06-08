"""
Darwin API Models Package

This package contains Pydantic models for API request and response validation.
Provides comprehensive data models for the Darwin genetic algorithm optimization API.
"""

from .requests import (
    Constraint,
    CrossoverType,
    ExportRequest,
    GeneticAlgorithmConfig,
    MutationType,
    ObjectiveType,
    OptimizationProblem,
    OptimizationRunRequest,
    OptimizerCreateRequest,
    SelectionType,
    Variable,
    VariableEncoding,
    VariableType,
    VisualizationRequest,
)
from .responses import (
    AlgorithmInfo,
    AlgorithmsResponse,
    APIMetrics,
    ErrorResponse,
    EvolutionHistory,
    ExportResponse,
    HealthCheck,
    HealthResponse,
    HealthStatus,
    HistoryResponse,
    MetricsResponse,
    OptimizationResults,
    OptimizerListResponse,
    OptimizerResponse,
    OptimizerStatus,
    PaginatedResponse,
    ProblemTemplate,
    ProgressResponse,
    ProgressUpdate,
    ResultsResponse,
    SystemMetrics,
    TemplatesResponse,
    VisualizationData,
    VisualizationResponse,
)

__all__ = [
    # Request models
    "OptimizerCreateRequest",
    "OptimizationRunRequest",
    "VisualizationRequest",
    "ExportRequest",
    "OptimizationProblem",
    "GeneticAlgorithmConfig",
    "Variable",
    "Constraint",
    # Request enums
    "ObjectiveType",
    "VariableType",
    "VariableEncoding",
    "SelectionType",
    "CrossoverType",
    "MutationType",
    # Response models
    "OptimizerResponse",
    "ResultsResponse",
    "ProgressResponse",
    "HistoryResponse",
    "VisualizationResponse",
    "HealthResponse",
    "TemplatesResponse",
    "AlgorithmsResponse",
    "MetricsResponse",
    "ExportResponse",
    "OptimizerListResponse",
    "ErrorResponse",
    "PaginatedResponse",
    # Response enums
    "OptimizerStatus",
    "HealthStatus",
    # Response components
    "HealthCheck",
    "SystemMetrics",
    "APIMetrics",
    "OptimizationResults",
    "ProgressUpdate",
    "EvolutionHistory",
    "VisualizationData",
    "ProblemTemplate",
    "AlgorithmInfo",
]

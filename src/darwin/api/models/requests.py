"""
Darwin API Request Models

This module defines Pydantic models for validating API request data.
All request models include comprehensive validation and documentation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ObjectiveType(str, Enum):
    """Optimization objective types."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MULTI_OBJECTIVE = "multi_objective"


class VariableType(str, Enum):
    """Variable types for optimization problems."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class VariableEncoding(str, Enum):
    """Variable encoding types."""

    REAL = "real"
    BINARY = "binary"
    PERMUTATION = "permutation"


class SelectionType(str, Enum):
    """Selection operator types."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    STEADY_STATE = "steady_state"


class CrossoverType(str, Enum):
    """Crossover operator types."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"


class MutationType(str, Enum):
    """Mutation operator types."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    RANDOM_RESETTING = "random_resetting"


class Variable(BaseModel):
    """Optimization variable definition."""

    name: str = Field(..., description="Variable name", min_length=1, max_length=100)
    type: VariableType = Field(..., description="Variable type")
    bounds: tuple[float, float] = Field(..., description="Variable bounds (min, max)")
    gene_space: Optional[List[Union[float, int, str]]] = Field(
        None, description="Discrete gene space for categorical/discrete variables"
    )
    encoding: VariableEncoding = Field(
        default=VariableEncoding.REAL, description="Variable encoding"
    )

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v):
        """Validate variable bounds."""
        if len(v) != 2:
            raise ValueError("Bounds must be a tuple of exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v

    @field_validator("gene_space")
    @classmethod
    def validate_gene_space(cls, v, info):
        """Validate gene space for discrete/categorical variables."""
        if info.data.get("type") in [VariableType.DISCRETE, VariableType.CATEGORICAL]:
            if v is None or len(v) == 0:
                raise ValueError(
                    "Gene space is required for discrete/categorical variables"
                )
        return v


class Constraint(BaseModel):
    """Optimization constraint definition."""

    name: str = Field(..., description="Constraint name", min_length=1, max_length=100)
    type: str = Field(..., description="Constraint type (equality, inequality)")
    expression: str = Field(..., description="Constraint expression")
    tolerance: float = Field(default=1e-6, description="Constraint tolerance")

    @field_validator("type")
    @classmethod
    def validate_constraint_type(cls, v):
        """Validate constraint type."""
        valid_types = ["equality", "inequality", "<=", ">=", "=="]
        if v not in valid_types:
            raise ValueError(f"Invalid constraint type: {v}")
        return v


class OptimizationProblem(BaseModel):
    """Optimization problem definition."""

    name: str = Field(..., description="Problem name", min_length=1, max_length=200)
    description: str = Field(..., description="Problem description", max_length=1000)
    objective_type: ObjectiveType = Field(
        ..., description="Optimization objective type"
    )
    variables: List[Variable] = Field(
        ..., description="Problem variables", min_items=1, max_items=100
    )
    constraints: List[Constraint] = Field(
        default=[], description="Problem constraints", max_items=50
    )
    fitness_function: Optional[str] = Field(
        None, description="Custom fitness function code (Python)", max_length=10000
    )
    metadata: Dict[str, Any] = Field(
        default={}, description="Additional problem metadata"
    )

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, v):
        """Validate variables list."""
        if not v:
            raise ValueError("At least one variable must be defined")

        # Check for duplicate variable names
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return v


class GeneticAlgorithmConfig(BaseModel):
    """Genetic algorithm configuration."""

    population_size: int = Field(
        default=50, ge=10, le=1000, description="Population size"
    )
    max_generations: int = Field(
        default=100, ge=1, le=10000, description="Maximum number of generations"
    )
    selection_type: SelectionType = Field(
        default=SelectionType.TOURNAMENT, description="Selection operator"
    )
    crossover_type: CrossoverType = Field(
        default=CrossoverType.SINGLE_POINT, description="Crossover operator"
    )
    mutation_type: MutationType = Field(
        default=MutationType.UNIFORM, description="Mutation operator"
    )
    crossover_probability: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Crossover probability"
    )
    mutation_probability: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Mutation probability"
    )
    elitism: bool = Field(default=True, description="Enable elitism")
    elitism_size: int = Field(
        default=1, ge=0, le=50, description="Number of elite individuals to preserve"
    )
    tournament_size: int = Field(
        default=3, ge=2, le=20, description="Tournament size for tournament selection"
    )
    adaptive_params: bool = Field(
        default=False, description="Enable adaptive parameters"
    )
    convergence_threshold: float = Field(
        default=1e-6, gt=0, description="Convergence threshold"
    )
    diversity_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Diversity threshold"
    )
    parallel_evaluation: bool = Field(
        default=True, description="Enable parallel fitness evaluation"
    )
    random_seed: Optional[int] = Field(
        None, ge=0, description="Random seed for reproducibility"
    )


class OptimizerCreateRequest(BaseModel):
    """Request model for creating a new optimizer."""

    problem: OptimizationProblem = Field(
        ..., description="Optimization problem definition"
    )
    config: GeneticAlgorithmConfig = Field(
        default_factory=GeneticAlgorithmConfig,
        description="Genetic algorithm configuration",
    )
    name: Optional[str] = Field(
        None, max_length=200, description="Optional optimizer name"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Optional optimizer description"
    )
    tags: List[str] = Field(
        default=[], max_items=10, description="Optional tags for categorization"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags."""
        for tag in v:
            if not isinstance(tag, str) or len(tag.strip()) == 0:
                raise ValueError("Tags must be non-empty strings")
            if len(tag) > 50:
                raise ValueError("Tags must be 50 characters or less")
        return [tag.strip() for tag in v]


class OptimizationRunRequest(BaseModel):
    """Request model for starting an optimization run."""

    run_name: Optional[str] = Field(
        None, max_length=200, description="Optional name for this run"
    )
    config_overrides: Optional[GeneticAlgorithmConfig] = Field(
        None, description="Override default algorithm configuration"
    )
    save_history: bool = Field(default=True, description="Save evolution history")
    save_population: bool = Field(default=False, description="Save population data")
    callback_url: Optional[str] = Field(
        None, description="Webhook URL for completion notification"
    )
    timeout_seconds: Optional[int] = Field(
        None, ge=1, le=86400, description="Maximum run time in seconds"
    )


class VisualizationRequest(BaseModel):
    """Request model for generating visualizations."""

    plot_type: str = Field(..., description="Type of plot to generate")
    parameters: Dict[str, Any] = Field(
        default={}, description="Plot-specific parameters"
    )
    format: str = Field(default="png", description="Output format (png, svg, pdf)")
    width: int = Field(default=800, ge=100, le=2000, description="Plot width in pixels")
    height: int = Field(
        default=600, ge=100, le=2000, description="Plot height in pixels"
    )

    @field_validator("plot_type")
    @classmethod
    def validate_plot_type(cls, v):
        """Validate plot type."""
        valid_types = [
            "fitness_evolution",
            "population_diversity",
            "pareto_frontier",
            "convergence_analysis",
            "parameter_correlation",
            "solution_distribution",
        ]
        if v not in valid_types:
            raise ValueError(f"plot_type must be one of: {valid_types}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate output format."""
        valid_formats = ["png", "svg", "pdf", "json"]
        if v not in valid_formats:
            raise ValueError(f"format must be one of: {valid_formats}")
        return v


class ExportRequest(BaseModel):
    """Request model for exporting results."""

    format: str = Field(..., description="Export format")
    include_population: bool = Field(
        default=False, description="Include population data"
    )
    include_history: bool = Field(default=True, description="Include evolution history")
    include_metadata: bool = Field(
        default=True, description="Include optimizer metadata"
    )
    compression: bool = Field(default=True, description="Compress exported data")

    @field_validator("format")
    @classmethod
    def validate_export_format(cls, v):
        """Validate export format."""
        valid_formats = ["csv", "json", "xlsx", "parquet"]
        if v not in valid_formats:
            raise ValueError(f"export format must be one of: {valid_formats}")
        return v

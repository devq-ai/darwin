"""
Darwin Core Configuration Module

This module provides configuration classes for genetic algorithm optimization
and general Darwin platform settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm optimization."""

    # Population parameters
    population_size: int = 50
    max_generations: int = 100

    # Genetic operators
    selection_type: SelectionType = SelectionType.TOURNAMENT
    crossover_type: CrossoverType = CrossoverType.SINGLE_POINT
    mutation_type: MutationType = MutationType.UNIFORM

    # Operator probabilities
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1

    # Elitism
    elitism: bool = True
    elitism_size: int = 1

    # Selection parameters
    tournament_size: int = 3

    # Adaptive parameters
    adaptive_params: bool = False

    # Convergence criteria
    convergence_threshold: float = 1e-6
    diversity_threshold: float = 0.1

    # Performance settings
    parallel_evaluation: bool = True
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")

        if self.max_generations < 1:
            raise ValueError("Max generations must be at least 1")

        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("Crossover probability must be between 0 and 1")

        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("Mutation probability must be between 0 and 1")

        if self.elitism_size >= self.population_size:
            raise ValueError("Elitism size must be less than population size")

        if self.tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "selection_type": self.selection_type.value,
            "crossover_type": self.crossover_type.value,
            "mutation_type": self.mutation_type.value,
            "crossover_probability": self.crossover_probability,
            "mutation_probability": self.mutation_probability,
            "elitism": self.elitism,
            "elitism_size": self.elitism_size,
            "tournament_size": self.tournament_size,
            "adaptive_params": self.adaptive_params,
            "convergence_threshold": self.convergence_threshold,
            "diversity_threshold": self.diversity_threshold,
            "parallel_evaluation": self.parallel_evaluation,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneticAlgorithmConfig":
        """Create configuration from dictionary."""
        return cls(
            population_size=data.get("population_size", 50),
            max_generations=data.get("max_generations", 100),
            selection_type=SelectionType(data.get("selection_type", "tournament")),
            crossover_type=CrossoverType(data.get("crossover_type", "single_point")),
            mutation_type=MutationType(data.get("mutation_type", "uniform")),
            crossover_probability=data.get("crossover_probability", 0.8),
            mutation_probability=data.get("mutation_probability", 0.1),
            elitism=data.get("elitism", True),
            elitism_size=data.get("elitism_size", 1),
            tournament_size=data.get("tournament_size", 3),
            adaptive_params=data.get("adaptive_params", False),
            convergence_threshold=data.get("convergence_threshold", 1e-6),
            diversity_threshold=data.get("diversity_threshold", 0.1),
            parallel_evaluation=data.get("parallel_evaluation", True),
            random_seed=data.get("random_seed"),
        )


@dataclass
class RunConfig:
    """Configuration for optimization runs."""

    run_name: Optional[str] = None
    save_history: bool = True
    save_population: bool = False
    callback_url: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_memory_mb: int = 1024
    checkpoint_interval: int = 10

    def __post_init__(self):
        """Validate run configuration."""
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        if self.max_memory_mb <= 0:
            raise ValueError("Max memory must be positive")

        if self.checkpoint_interval <= 0:
            raise ValueError("Checkpoint interval must be positive")


@dataclass
class OptimizerSettings:
    """General optimizer settings."""

    log_level: str = "INFO"
    enable_monitoring: bool = True
    enable_profiling: bool = False
    cache_results: bool = True
    max_cache_size: int = 1000

    def __post_init__(self):
        """Validate optimizer settings."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")

        if self.max_cache_size <= 0:
            raise ValueError("Max cache size must be positive")


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = "ws://localhost:8000/rpc"
    username: str = "root"
    password: str = "root"
    namespace: str = "darwin"
    database: str = "optimization"
    connection_timeout: int = 30
    max_connections: int = 10

    def __post_init__(self):
        """Validate database configuration."""
        if not self.url:
            raise ValueError("Database URL is required")

        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")

        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://localhost:5173",
        ]
    )
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    def __post_init__(self):
        """Validate API configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")

        if self.rate_limit_requests <= 0:
            raise ValueError("Rate limit requests must be positive")

        if self.rate_limit_window <= 0:
            raise ValueError("Rate limit window must be positive")


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    logfire_token: Optional[str] = None
    service_name: str = "darwin-api"
    service_version: str = "1.0.0"
    environment: str = "development"
    enable_tracing: bool = True
    enable_metrics: bool = True
    metrics_port: int = 9090

    def __post_init__(self):
        """Validate monitoring configuration."""
        if not 1 <= self.metrics_port <= 65535:
            raise ValueError("Metrics port must be between 1 and 65535")


@dataclass
class DarwinConfig:
    """Main Darwin platform configuration."""

    genetic_algorithm: GeneticAlgorithmConfig = field(
        default_factory=GeneticAlgorithmConfig
    )
    run: RunConfig = field(default_factory=RunConfig)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DarwinConfig":
        """Create configuration from dictionary."""
        return cls(
            genetic_algorithm=GeneticAlgorithmConfig.from_dict(
                data.get("genetic_algorithm", {})
            ),
            run=RunConfig(**data.get("run", {})),
            optimizer=OptimizerSettings(**data.get("optimizer", {})),
            database=DatabaseConfig(**data.get("database", {})),
            api=APIConfig(**data.get("api", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "genetic_algorithm": self.genetic_algorithm.to_dict(),
            "run": {
                "run_name": self.run.run_name,
                "save_history": self.run.save_history,
                "save_population": self.run.save_population,
                "callback_url": self.run.callback_url,
                "timeout_seconds": self.run.timeout_seconds,
                "max_memory_mb": self.run.max_memory_mb,
                "checkpoint_interval": self.run.checkpoint_interval,
            },
            "optimizer": {
                "log_level": self.optimizer.log_level,
                "enable_monitoring": self.optimizer.enable_monitoring,
                "enable_profiling": self.optimizer.enable_profiling,
                "cache_results": self.optimizer.cache_results,
                "max_cache_size": self.optimizer.max_cache_size,
            },
            "database": {
                "url": self.database.url,
                "username": self.database.username,
                "password": self.database.password,
                "namespace": self.database.namespace,
                "database": self.database.database,
                "connection_timeout": self.database.connection_timeout,
                "max_connections": self.database.max_connections,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "reload": self.api.reload,
                "log_level": self.api.log_level,
                "cors_origins": self.api.cors_origins,
                "rate_limit_requests": self.api.rate_limit_requests,
                "rate_limit_window": self.api.rate_limit_window,
            },
            "monitoring": {
                "logfire_token": self.monitoring.logfire_token,
                "service_name": self.monitoring.service_name,
                "service_version": self.monitoring.service_version,
                "environment": self.monitoring.environment,
                "enable_tracing": self.monitoring.enable_tracing,
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
            },
        }

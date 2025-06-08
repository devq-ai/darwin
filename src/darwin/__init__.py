"""
Darwin: Genetic Algorithm Solver
===============================

A comprehensive genetic algorithm optimization platform providing both standalone
application capabilities and Model Context Protocol (MCP) server integration.

Features:
- Advanced genetic algorithms with PyGAD integration
- Multi-objective optimization (NSGA-II, NSGA-III)
- Interactive Panel dashboard with real-time visualization
- MCP server for AI agent integration
- Enterprise-grade monitoring with Logfire
- Constraint handling and adaptive operators
- Production-ready deployment with Docker/Kubernetes

Usage:
    # Import core components
    from darwin import GeneticOptimizer, OptimizationProblem
    from darwin.algorithms import NSGAII, AdaptiveGA
    from darwin.dashboard import DarwinDashboard
    from darwin.mcp import DarwinMCPServer

    # Create optimization problem
    problem = OptimizationProblem(
        name="Function Optimization",
        variables=[
            {"name": "x", "type": "continuous", "bounds": [-5, 5]},
            {"name": "y", "type": "continuous", "bounds": [-5, 5]}
        ],
        fitness_function="rastrigin_2d",
        objective_type="minimize"
    )

    # Initialize optimizer
    optimizer = GeneticOptimizer(problem)

    # Run optimization
    result = optimizer.run(max_generations=100)

    # Start MCP server
    server = DarwinMCPServer()
    server.run()

    # Launch dashboard
    dashboard = DarwinDashboard()
    dashboard.serve()

For more information, visit: https://darwin.devq.ai
"""

__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "team@devq.ai"
__license__ = "BSD-3-Clause"
__copyright__ = "Copyright (c) 2025 DevQ.ai"

# Initialize logging early
import logging
import sys

# Package-level logger (initialized early)
logger = logging.getLogger(__name__)

# Core exports (required)
try:
    from darwin.core.config import (
        DarwinConfig,
        GeneticAlgorithmConfig,
        OptimizerSettings,
        RunConfig,
    )
    from darwin.core.optimizer import GeneticOptimizer
    from darwin.core.problem import Constraint, OptimizationProblem, Variable
except ImportError as e:
    logger.warning(f"Core modules not available: {e}")

    # Create placeholder classes if core modules are missing
    class GeneticOptimizer:
        pass

    class OptimizationProblem:
        pass

    class Variable:
        pass

    class Constraint:
        pass

    class GeneticAlgorithmConfig:
        pass

    class RunConfig:
        pass

    class OptimizerSettings:
        pass

    class DarwinConfig:
        pass


# Algorithm implementations (optional)
try:
    from darwin.algorithms.adaptive import AdaptiveGA
    from darwin.algorithms.constraints import ConstraintGA
    from darwin.algorithms.genetic import EnhancedGA
    from darwin.algorithms.multi_objective import NSGAII, NSGAIII
except ImportError as e:
    logger.warning(f"Algorithm modules not available: {e}")

    # Create placeholder classes
    class EnhancedGA:
        pass

    class NSGAII:
        pass

    class NSGAIII:
        pass

    class AdaptiveGA:
        pass

    class ConstraintGA:
        pass


# Visualization and dashboard (optional)
try:
    from darwin.dashboard.app import DarwinDashboard
    from darwin.dashboard.components import (
        OptimizationMonitor,
        ProblemEditor,
        ResultsViewer,
    )
except ImportError as e:
    logger.warning(f"Dashboard modules not available: {e}")

    # Create placeholder classes
    class DarwinDashboard:
        pass

    class OptimizationMonitor:
        pass

    class ResultsViewer:
        pass

    class ProblemEditor:
        pass


# MCP server (optional)
try:
    from darwin.mcp.server import DarwinMCPServer
    from darwin.mcp.tools import (
        CreateOptimizerTool,
        GetResultsTool,
        RunOptimizationTool,
        VisualizationTool,
    )
except ImportError as e:
    logger.warning(f"MCP modules not available: {e}")

    # Create placeholder classes
    class DarwinMCPServer:
        pass

    class CreateOptimizerTool:
        pass

    class RunOptimizationTool:
        pass

    class GetResultsTool:
        pass

    class VisualizationTool:
        pass


# Utilities (optional)
try:
    from darwin.utils.benchmarks import BenchmarkSuite
    from darwin.utils.logging import setup_logging
    from darwin.utils.templates import ProblemTemplates
    from darwin.utils.validators import validate_config, validate_problem
except ImportError as e:
    logger.warning(f"Utility modules not available: {e}")

    # Create placeholder classes and functions
    class BenchmarkSuite:
        pass

    class ProblemTemplates:
        pass

    def validate_problem(*args, **kwargs):
        return True

    def validate_config(*args, **kwargs):
        return True

    def setup_logging(*args, **kwargs):
        pass


# Exception classes (optional)
try:
    from darwin.exceptions import (
        ConfigurationError,
        DarwinError,
        MCPError,
        OptimizationError,
        ValidationError,
    )
except ImportError as e:
    logger.warning(f"Exception modules not available: {e}")

    # Create placeholder exception classes
    class DarwinError(Exception):
        pass

    class OptimizationError(DarwinError):
        pass

    class ValidationError(DarwinError):
        pass

    class ConfigurationError(DarwinError):
        pass

    class MCPError(DarwinError):
        pass


# Version information
__all__ = [
    # Core classes
    "GeneticOptimizer",
    "OptimizationProblem",
    "Variable",
    "Constraint",
    # Configuration
    "GeneticAlgorithmConfig",
    "RunConfig",
    "OptimizerSettings",
    "DarwinConfig",
    # Algorithms
    "EnhancedGA",
    "NSGAII",
    "NSGAIII",
    "AdaptiveGA",
    "ConstraintGA",
    # Dashboard
    "DarwinDashboard",
    "OptimizationMonitor",
    "ResultsViewer",
    "ProblemEditor",
    # MCP Server
    "DarwinMCPServer",
    "CreateOptimizerTool",
    "RunOptimizationTool",
    "GetResultsTool",
    "VisualizationTool",
    # Utilities
    "BenchmarkSuite",
    "ProblemTemplates",
    "validate_problem",
    "validate_config",
    "setup_logging",
    # Exceptions
    "DarwinError",
    "OptimizationError",
    "ValidationError",
    "ConfigurationError",
    "MCPError",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

# Configure default logging
from typing import Optional


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    enable_structlog: bool = True,
) -> None:
    """
    Configure Darwin logging system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        enable_structlog: Whether to use structured logging
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )

    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, stream=sys.stdout
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    if enable_structlog:
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        except ImportError:
            # Fallback to standard logging if structlog not available
            pass


# Initialize default logging
configure_logging()

logger.info(f"Darwin v{__version__} initialized")

# Feature flags for conditional imports
FEATURES = {
    "mcp_server": True,
    "dashboard": True,
    "monitoring": True,
    "distributed": False,  # Future feature
    "gpu_acceleration": False,  # Future feature
}


def get_feature_status() -> dict:
    """Get current feature availability status."""
    status = FEATURES.copy()

    # Check for optional dependencies
    try:
        import panel

        status["dashboard"] = True
    except ImportError:
        status["dashboard"] = False
        logger.warning("Panel not available - dashboard features disabled")

    try:
        import logfire

        status["monitoring"] = True
    except ImportError:
        status["monitoring"] = False
        logger.warning("Logfire not available - advanced monitoring disabled")

    try:
        import redis

        status["distributed"] = True
    except ImportError:
        status["distributed"] = False

    return status


# Lazy loading for heavy dependencies
def __getattr__(name: str):
    """Lazy loading of heavy modules."""
    if name == "benchmarks":
        from darwin.utils import benchmarks

        return benchmarks

    if name == "visualization":
        from darwin.dashboard import visualization

        return visualization

    if name == "templates":
        from darwin.utils import templates

        return templates

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Development mode warnings
import os

if os.getenv("DARWIN_ENV") == "development":
    import warnings

    warnings.filterwarnings("default", category=DeprecationWarning, module="darwin")
    logger.info("Darwin running in development mode - showing all warnings")

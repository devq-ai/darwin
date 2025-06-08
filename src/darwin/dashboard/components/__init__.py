"""
Darwin Dashboard Components Package

This package contains all the individual UI components that make up the Darwin
genetic algorithm optimization dashboard. Each component is designed to be
modular, reusable, and integrated with the Darwin API backend.

Components:
- ProblemEditor: Interactive interface for defining optimization problems
- MonitoringDashboard: Real-time monitoring of genetic algorithm runs
- VisualizationEngine: Advanced analytics and interactive visualizations
- TemplateManager: Management of problem templates and configurations
- ExperimentManager: Tracking and analysis of optimization experiments

Each component follows the Panel framework conventions and integrates with:
- Darwin API client for backend communication
- WebSocket manager for real-time updates
- Shared configuration and state management
- Consistent UI/UX patterns and styling

Usage:
    from darwin.dashboard.components import ProblemEditor, MonitoringDashboard

    # Create API client
    api_client = DarwinAPIClient("http://localhost:8000")

    # Initialize components
    problem_editor = ProblemEditor(api_client=api_client)
    monitoring = MonitoringDashboard(api_client=api_client)

    # Create interfaces
    problem_interface = problem_editor.create_interface()
    monitoring_interface = monitoring.create_interface()
"""

from .experiments import ExperimentFilter, ExperimentManager
from .monitoring import MonitoringDashboard, OptimizationMonitor
from .problem_editor import ProblemConfig, ProblemEditor, Variable
from .templates import TemplateManager, TemplateMetadata
from .visualizations import VisualizationConfig, VisualizationEngine

__version__ = "1.0.0"

__all__ = [
    # Main component classes
    "ProblemEditor",
    "MonitoringDashboard",
    "VisualizationEngine",
    "TemplateManager",
    "ExperimentManager",
    # Supporting parameter classes
    "ProblemConfig",
    "Variable",
    "OptimizationMonitor",
    "VisualizationConfig",
    "TemplateMetadata",
    "ExperimentFilter",
]

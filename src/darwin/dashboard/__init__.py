"""
Darwin Dashboard Package

This package provides the Panel-based web dashboard for the Darwin genetic algorithm
optimization platform. It includes interactive components for problem definition,
real-time monitoring, visualization, template management, and experiment tracking.

Main Components:
- DarwinDashboard: Main dashboard application
- ProblemEditor: Interactive problem definition interface
- MonitoringDashboard: Real-time optimization monitoring
- VisualizationEngine: Advanced analytics and plotting
- TemplateManager: Problem template management
- ExperimentManager: Experiment tracking and analysis

Usage:
    from darwin.dashboard import DarwinDashboard

    # Create dashboard application
    dashboard = DarwinDashboard(api_base_url="http://localhost:8000")

    # Serve the dashboard
    dashboard.serve(port=5007, show=True)
"""

from darwin.dashboard.app import DarwinDashboard, create_app
from darwin.dashboard.components.experiments import ExperimentManager
from darwin.dashboard.components.monitoring import MonitoringDashboard
from darwin.dashboard.components.problem_editor import ProblemEditor
from darwin.dashboard.components.templates import TemplateManager
from darwin.dashboard.components.visualizations import VisualizationEngine
from darwin.dashboard.utils.api_client import DarwinAPIClient, get_api_client
from darwin.dashboard.utils.websocket_manager import WebSocketManager

__version__ = "1.0.0"

__all__ = [
    "DarwinDashboard",
    "create_app",
    "ProblemEditor",
    "MonitoringDashboard",
    "VisualizationEngine",
    "TemplateManager",
    "ExperimentManager",
    "DarwinAPIClient",
    "get_api_client",
    "WebSocketManager",
]

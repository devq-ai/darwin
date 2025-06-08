"""
Darwin Genetic Algorithm Optimizer - Dashboard Package

This package provides the Panel-based web dashboard for the Darwin optimization platform.
It includes interactive components for problem configuration, optimization monitoring,
and result visualization.

Components:
- main: Core dashboard application and UI components
- templates: Problem templates interface and management
"""

from typing import Optional

try:
    from .main import (
        APIClient,
        DarwinDashboard,
        ProblemConfig,
        create_dashboard,
        serve_dashboard,
    )
    from .templates import ProblemTemplate, TemplateManager, TemplatesInterface

    __all__ = [
        "DarwinDashboard",
        "APIClient",
        "ProblemConfig",
        "create_dashboard",
        "serve_dashboard",
        "TemplatesInterface",
        "ProblemTemplate",
        "TemplateManager",
    ]

    DASHBOARD_AVAILABLE = True

except ImportError as e:
    # Panel or other dashboard dependencies not available
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Dashboard components not available: {e}")

    __all__ = []
    DASHBOARD_AVAILABLE = False


def is_available() -> bool:
    """Check if dashboard components are available."""
    return DASHBOARD_AVAILABLE


def get_version() -> str:
    """Get dashboard version."""
    return "1.0.0"


def get_info() -> dict:
    """Get dashboard package information."""
    return {
        "name": "darwin-dashboard",
        "version": get_version(),
        "available": DASHBOARD_AVAILABLE,
        "components": __all__,
        "description": "Interactive Panel-based dashboard for genetic algorithm optimization",
        "dependencies": ["panel", "bokeh", "param", "requests"],
    }

"""
Darwin Genetic Algorithm Optimizer - UI Module

This module provides user interface components for the Darwin platform,
including the Panel-based web dashboard for interactive optimization.

Features:
- Panel dashboard for problem configuration and monitoring
- Problem templates interface
- Real-time visualization components
- FastAPI backend integration
"""

from typing import Optional

# Dashboard components (optional imports to avoid hard dependencies)
try:
    from .dashboard.main import DarwinDashboard, create_dashboard, serve_dashboard
    from .dashboard.templates import ProblemTemplate, TemplatesInterface

    __all__ = [
        "DarwinDashboard",
        "create_dashboard",
        "serve_dashboard",
        "TemplatesInterface",
        "ProblemTemplate",
    ]

    DASHBOARD_AVAILABLE = True

except ImportError as e:
    # Panel/dashboard dependencies not available
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Dashboard components not available: {e}")

    __all__ = []
    DASHBOARD_AVAILABLE = False


def check_dashboard_dependencies() -> bool:
    """Check if dashboard dependencies are available."""
    return DASHBOARD_AVAILABLE


def get_dashboard_info() -> dict:
    """Get information about dashboard availability and components."""
    return {
        "available": DASHBOARD_AVAILABLE,
        "components": __all__ if DASHBOARD_AVAILABLE else [],
        "description": "Panel-based web dashboard for genetic algorithm optimization",
    }

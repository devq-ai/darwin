"""
Darwin Visualization Engine

This package provides advanced visualization capabilities for genetic algorithm
optimization results, including interactive plots, statistical analysis,
performance comparisons, and solution space exploration.

Features:
- Interactive Bokeh-based visualizations
- Real-time optimization monitoring
- Multi-objective optimization analysis
- Solution space exploration with dimensionality reduction
- Performance benchmarking and comparison
- Statistical analysis and correlation plots
- Export capabilities for plots and data
- Responsive design for different screen sizes
"""

from .analytics import (
    AnalyticsEngine,
    ConvergenceAnalyzer,
    DiversityAnalyzer,
    PerformanceAnalyzer,
    StatisticalAnalyzer,
)
from .engine import VisualizationEngine
from .export import ExportManager
from .interactive import InteractiveExplorer
from .plots import (
    ConvergencePlot,
    DiversityPlot,
    FitnessLandscapePlot,
    ParetoFrontierPlot,
    PerformancePlot,
    SolutionSpacePlot,
)
from .themes import ThemeManager

__version__ = "1.0.0"
__author__ = "DevQ.ai"

__all__ = [
    # Core engine
    "VisualizationEngine",
    # Plot types
    "ConvergencePlot",
    "DiversityPlot",
    "FitnessLandscapePlot",
    "ParetoFrontierPlot",
    "PerformancePlot",
    "SolutionSpacePlot",
    # Analytics
    "AnalyticsEngine",
    "ConvergenceAnalyzer",
    "DiversityAnalyzer",
    "PerformanceAnalyzer",
    "StatisticalAnalyzer",
    # Interactive features
    "InteractiveExplorer",
    # Utilities
    "ExportManager",
    "ThemeManager",
]

# Default configuration
DEFAULT_CONFIG = {
    "theme": "light",
    "color_palette": "Category10",
    "plot_width": 700,
    "plot_height": 500,
    "animation_enabled": True,
    "export_format": "png",
    "responsive": True,
    "accessibility": True,
}


def get_version():
    """Get the current version of the visualization engine."""
    return __version__


def create_visualization_engine(**kwargs):
    """Factory function to create a configured visualization engine."""
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return VisualizationEngine(config=config)

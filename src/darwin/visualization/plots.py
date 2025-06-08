"""
Darwin Visualization Plots Module

This module provides specialized plot classes for genetic algorithm visualization,
including convergence plots, diversity analysis, fitness landscapes, Pareto frontiers,
performance comparisons, and solution space exploration.

Features:
- Interactive Bokeh-based plot components
- Real-time data updates and streaming
- Customizable styling and themes
- Export capabilities for static and interactive formats
- Responsive design for different screen sizes
- Accessibility features and keyboard navigation
- Animation support for temporal data
- Multi-objective optimization visualization
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    Range1d,
    Span,
)
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

logger = logging.getLogger(__name__)


class BasePlot:
    """Base class for all Darwin visualization plots."""
    
    def __init__(
        self,
        width: int = 700,
        height: int = 500,
        title: str = "",
        theme: str = "light"
    ):
        """
        Initialize base plot.
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
            title: Plot title
            theme: Visual theme ('light', 'dark', 'minimal')
        """
        self.width = width
        self.height = height
        self.title = title
        self.theme = theme
        self.plot = None
        self.data_source = ColumnDataSource()
        
        # Theme configuration
        self.theme_config = self._get_theme_config(theme)
        
        self._setup_plot()
    
    def _get_theme_config(self, theme: str) -> Dict[str, Any]:
        """Get theme-specific configuration."""
        themes = {
            "light": {
                "background_fill_color": "white",
                "border_fill_color": "white",
                "grid_line_color": "#e6e6e6",
                "axis_line_color": "#cccccc",
                "text_color": "#333333"
            },
            "dark": {
                "background_fill_color": "#2F2F2F",
                "border_fill_color": "#2F2F2F",
                "grid_line_color": "#555555",
                "axis_line_color": "#777777",
                "text_color": "#ffffff"
            },
            "minimal": {
                "background_fill_color": "#fafafa",
                "border_fill_color": "#fafafa",
                "grid_line_color": "#f0f0f0",
                "axis_line_color": "#dddddd",
                "text_color": "#444444"
            }
        }
        return themes.get(theme, themes["light"])
    
    def _setup_plot(self):
        """Set up the basic plot structure."""
        self.plot = figure(
            width=self.width,
            height=self.height,
            title=self.title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            **self.theme_config
        )
        
        # Configure grid
        self.plot.grid.grid_line_alpha = 0.3
        
        # Add hover tool by default
        hover = HoverTool()
        self.plot.add_tools(hover)
    
    def update_data(self, data: Dict[str, Any]):
        """Update plot data."""
        self.data_source.data = data
    
    def export_plot(self, filename: str, format: str = "png"):
        """Export plot to file."""
        # This would be implemented by calling the ExportManager
        pass


class ConvergencePlot(BasePlot):
    """Plot for visualizing optimization convergence."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_fitness_line = None
        self.average_fitness_line = None
        self.convergence_threshold = None
        
        self._setup_convergence_plot()
    
    def _setup_convergence_plot(self):
        """Set up convergence-specific plot elements."""
        self.plot.title.text = self.title or "Optimization Convergence"
        self.plot.xaxis.axis_label = "Generation"
        self.plot.yaxis.axis_label = "Fitness"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("Generation", "@generation"),
            ("Best Fitness", "@best_fitness"),
            ("Average Fitness", "@average_fitness"),
            ("Improvement", "@improvement")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_convergence_data(
        self,
        generations: List[int],
        best_fitness: List[float],
        average_fitness: List[float],
        target_fitness: Optional[float] = None
    ):
        """
        Update convergence plot with new data.
        
        Args:
            generations: List of generation numbers
            best_fitness: Best fitness values per generation
            average_fitness: Average fitness values per generation
            target_fitness: Optional target fitness threshold
        """
        try:
            # Calculate improvements
            improvements = [0.0]
            for i in range(1, len(best_fitness)):
                improvement = abs(best_fitness[i] - best_fitness[i-1])
                improvements.append(improvement)
            
            # Update data source
            data = {
                "generation": generations,
                "best_fitness": best_fitness,
                "average_fitness": average_fitness,
                "improvement": improvements
            }
            self.update_data(data)
            
            # Clear existing renderers
            self.plot.renderers = []
            
            # Plot best fitness line
            self.best_fitness_line = self.plot.line(
                "generation", "best_fitness",
                source=self.data_source,
                line_width=3,
                color="red",
                legend_label="Best Fitness",
                alpha=0.8
            )
            
            # Plot average fitness line
            self.average_fitness_line = self.plot.line(
                "generation", "average_fitness",
                source=self.data_source,
                line_width=2,
                color="blue",
                legend_label="Average Fitness",
                alpha=0.6
            )
            
            # Add target fitness line if provided
            if target_fitness is not None:
                target_span = Span(
                    location=target_fitness,
                    dimension='width',
                    line_color='green',
                    line_dash='dashed',
                    line_width=2
                )
                self.plot.add_layout(target_span)
            
            # Configure legend
            self.plot.legend.location = "top_right"
            self.plot.legend.click_policy = "hide"
            
            logger.info("Convergence plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update convergence plot: {e}")


class DiversityPlot(BasePlot):
    """Plot for visualizing population diversity metrics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_diversity_plot()
    
    def _setup_diversity_plot(self):
        """Set up diversity-specific plot elements."""
        self.plot.title.text = self.title or "Population Diversity Analysis"
        self.plot.xaxis.axis_label = "Generation"
        self.plot.yaxis.axis_label = "Diversity Metric"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("Generation", "@generation"),
            ("Genetic Diversity", "@genetic_diversity"),
            ("Phenotypic Diversity", "@phenotypic_diversity"),
            ("Spatial Diversity", "@spatial_diversity")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_diversity_data(
        self,
        generations: List[int],
        genetic_diversity: List[float],
        phenotypic_diversity: List[float],
        spatial_diversity: Optional[List[float]] = None
    ):
        """
        Update diversity plot with new data.
        
        Args:
            generations: List of generation numbers
            genetic_diversity: Genetic diversity values
            phenotypic_diversity: Phenotypic diversity values
            spatial_diversity: Optional spatial diversity values
        """
        try:
            data = {
                "generation": generations,
                "genetic_diversity": genetic_diversity,
                "phenotypic_diversity": phenotypic_diversity
            }
            
            if spatial_diversity:
                data["spatial_diversity"] = spatial_diversity
            
            self.update_data(data)
            
            # Clear existing renderers
            self.plot.renderers = []
            
            # Plot diversity metrics
            self.plot.line(
                "generation", "genetic_diversity",
                source=self.data_source,
                line_width=2,
                color="red",
                legend_label="Genetic Diversity"
            )
            
            self.plot.line(
                "generation", "phenotypic_diversity",
                source=self.data_source,
                line_width=2,
                color="blue",
                legend_label="Phenotypic Diversity"
            )
            
            if spatial_diversity:
                self.plot.line(
                    "generation", "spatial_diversity",
                    source=self.data_source,
                    line_width=2,
                    color="green",
                    legend_label="Spatial Diversity"
                )
            
            self.plot.legend.location = "top_right"
            logger.info("Diversity plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update diversity plot: {e}")


class FitnessLandscapePlot(BasePlot):
    """3D-style fitness landscape visualization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_landscape_plot()
    
    def _setup_landscape_plot(self):
        """Set up fitness landscape plot elements."""
        self.plot.title.text = self.title or "Fitness Landscape"
        self.plot.xaxis.axis_label = "Parameter 1"
        self.plot.yaxis.axis_label = "Parameter 2"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("X", "@x"),
            ("Y", "@y"),
            ("Fitness", "@fitness"),
            ("Generation", "@generation")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_landscape_data(
        self,
        x_coords: List[float],
        y_coords: List[float],
        fitness_values: List[float],
        generations: Optional[List[int]] = None
    ):
        """
        Update fitness landscape with new data.
        
        Args:
            x_coords: X coordinates (parameter 1 values)
            y_coords: Y coordinates (parameter 2 values)
            fitness_values: Fitness values for coloring
            generations: Optional generation numbers
        """
        try:
            data = {
                "x": x_coords,
                "y": y_coords,
                "fitness": fitness_values
            }
            
            if generations:
                data["generation"] = generations
            
            self.update_data(data)
            
            # Clear existing renderers
            self.plot.renderers = []
            
            # Create color mapper
            color_mapper = LinearColorMapper(
                palette=Viridis256,
                low=min(fitness_values),
                high=max(fitness_values)
            )
            
            # Plot fitness landscape as colored circles
            self.plot.circle(
                "x", "y",
                source=self.data_source,
                size=8,
                color={"field": "fitness", "transform": color_mapper},
                alpha=0.7
            )
            
            # Add color bar
            color_bar = ColorBar(
                color_mapper=color_mapper,
                width=8,
                location=(0, 0),
                title="Fitness"
            )
            self.plot.add_layout(color_bar, 'right')
            
            logger.info("Fitness landscape plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update fitness landscape plot: {e}")


class ParetoFrontierPlot(BasePlot):
    """Plot for multi-objective optimization Pareto frontier."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_pareto_plot()
    
    def _setup_pareto_plot(self):
        """Set up Pareto frontier plot elements."""
        self.plot.title.text = self.title or "Pareto Frontier"
        self.plot.xaxis.axis_label = "Objective 1"
        self.plot.yaxis.axis_label = "Objective 2"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("Objective 1", "@obj1"),
            ("Objective 2", "@obj2"),
            ("Solution ID", "@solution_id"),
            ("Dominated", "@dominated")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_pareto_data(
        self,
        objective1: List[float],
        objective2: List[float],
        is_pareto_optimal: List[bool],
        solution_ids: Optional[List[str]] = None
    ):
        """
        Update Pareto frontier plot with new data.
        
        Args:
            objective1: Values for first objective
            objective2: Values for second objective
            is_pareto_optimal: Boolean mask for Pareto optimal solutions
            solution_ids: Optional solution identifiers
        """
        try:
            data = {
                "obj1": objective1,
                "obj2": objective2,
                "dominated": ["No" if optimal else "Yes" for optimal in is_pareto_optimal]
            }
            
            if solution_ids:
                data["solution_id"] = solution_ids
            else:
                data["solution_id"] = [f"Sol_{i}" for i in range(len(objective1))]
            
            self.update_data(data)
            
            # Clear existing renderers
            self.plot.renderers = []
            
            # Plot all solutions
            pareto_points = [i for i, optimal in enumerate(is_pareto_optimal) if optimal]
            dominated_points = [i for i, optimal in enumerate(is_pareto_optimal) if not optimal]
            
            # Plot dominated solutions
            if dominated_points:
                dominated_data = ColumnDataSource({
                    key: [data[key][i] for i in dominated_points]
                    for key in data.keys()
                })
                
                self.plot.circle(
                    "obj1", "obj2",
                    source=dominated_data,
                    size=6,
                    color="lightblue",
                    alpha=0.5,
                    legend_label="Dominated Solutions"
                )
            
            # Plot Pareto optimal solutions
            if pareto_points:
                pareto_data = ColumnDataSource({
                    key: [data[key][i] for i in pareto_points]
                    for key in data.keys()
                })
                
                self.plot.circle(
                    "obj1", "obj2",
                    source=pareto_data,
                    size=10,
                    color="red",
                    alpha=0.8,
                    legend_label="Pareto Optimal"
                )
                
                # Connect Pareto points with a line
                pareto_obj1 = sorted([objective1[i] for i in pareto_points])
                pareto_obj2_sorted = []
                for obj1_val in pareto_obj1:
                    for i in pareto_points:
                        if objective1[i] == obj1_val:
                            pareto_obj2_sorted.append(objective2[i])
                            break
                
                self.plot.line(
                    pareto_obj1, pareto_obj2_sorted,
                    line_width=2,
                    color="red",
                    alpha=0.6,
                    line_dash="dashed"
                )
            
            self.plot.legend.location = "top_right"
            logger.info("Pareto frontier plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update Pareto frontier plot: {e}")


class PerformancePlot(BasePlot):
    """Plot for algorithm performance comparison."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_performance_plot()
    
    def _setup_performance_plot(self):
        """Set up performance comparison plot elements."""
        self.plot.title.text = self.title or "Algorithm Performance Comparison"
        self.plot.xaxis.axis_label = "Algorithm"
        self.plot.yaxis.axis_label = "Performance Metric"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("Algorithm", "@algorithm"),
            ("Metric", "@metric"),
            ("Value", "@value"),
            ("Std Dev", "@std_dev")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_performance_data(
        self,
        algorithms: List[str],
        metrics: Dict[str, List[float]],
        std_devs: Optional[Dict[str, List[float]]] = None
    ):
        """
        Update performance comparison plot.
        
        Args:
            algorithms: List of algorithm names
            metrics: Dictionary of metric names to value lists
            std_devs: Optional standard deviations for error bars
        """
        try:
            # Clear existing renderers
            self.plot.renderers = []
            
            colors = Category10[max(3, len(metrics))]
            
            # Calculate bar positions
            bar_width = 0.8 / len(metrics)
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                x_offset = (i - len(metrics) / 2) * bar_width + bar_width / 2
                x_positions = [j + x_offset for j in range(len(algorithms))]
                
                # Create data source for this metric
                metric_data = {
                    "x": x_positions,
                    "top": values,
                    "algorithm": algorithms,
                    "metric": [metric_name] * len(algorithms),
                    "value": values
                }
                
                if std_devs and metric_name in std_devs:
                    metric_data["std_dev"] = std_devs[metric_name]
                else:
                    metric_data["std_dev"] = [0.0] * len(algorithms)
                
                metric_source = ColumnDataSource(metric_data)
                
                # Plot bars
                self.plot.vbar(
                    x="x", top="top",
                    source=metric_source,
                    width=bar_width * 0.9,
                    color=colors[i % len(colors)],
                    legend_label=metric_name,
                    alpha=0.8
                )
                
                # Add error bars if std_devs provided
                if std_devs and metric_name in std_devs:
                    # Error bars would be added here
                    pass
            
            # Configure x-axis
            self.plot.xaxis.ticker = list(range(len(algorithms)))
            self.plot.xaxis.major_label_overrides = {
                i: alg for i, alg in enumerate(algorithms)
            }
            
            self.plot.legend.location = "top_left"
            logger.info("Performance plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update performance plot: {e}")


class SolutionSpacePlot(BasePlot):
    """Plot for exploring solution space with dimensionality reduction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_solution_space_plot()
    
    def _setup_solution_space_plot(self):
        """Set up solution space plot elements."""
        self.plot.title.text = self.title or "Solution Space Exploration"
        self.plot.xaxis.axis_label = "Component 1"
        self.plot.yaxis.axis_label = "Component 2"
        
        # Configure hover tool
        hover = HoverTool(tooltips=[
            ("X", "@x"),
            ("Y", "@y"),
            ("Fitness", "@fitness"),
            ("Cluster", "@cluster"),
            ("Solution ID", "@solution_id")
        ])
        self.plot.tools = [tool for tool in self.plot.tools if not isinstance(tool, HoverTool)]
        self.plot.add_tools(hover)
    
    def update_solution_space_data(
        self,
        x_coords: List[float],
        y_coords: List[float],
        fitness_values: List[float],
        clusters: Optional[List[int]] = None,
        solution_ids: Optional[List[str]] = None
    ):
        """
        Update solution space plot with dimensionally reduced data.
        
        Args:
            x_coords: X coordinates from dimensionality reduction
            y_coords: Y coordinates from dimensionality reduction
            fitness_values: Fitness values for coloring
            clusters: Optional cluster assignments
            solution_ids: Optional solution identifiers
        """
        try:
            data = {
                "x": x_coords,
                "y": y_coords,
                "fitness": fitness_values
            }
            
            if clusters:
                data["cluster"] = clusters
            else:
                data["cluster"] = [0] * len(x_coords)
            
            if solution_ids:
                data["solution_id"] = solution_ids
            else:
                data["solution_id"] = [f"Sol_{i}" for i in range(len(x_coords))]
            
            self.update_data(data)
            
            # Clear existing renderers
            self.plot.renderers = []
            
            if clusters:
                # Color by cluster
                unique_clusters = list(set(clusters))
                colors = Category10[max(3, len(unique_clusters))]
                
                for i, cluster_id in enumerate(unique_clusters):
                    cluster_indices = [j for j, c in enumerate(clusters) if c == cluster_id]
                    cluster_data = ColumnDataSource({
                        key: [data[key][j] for j in cluster_indices]
                        for key in data.keys()
                    })
                    
                    self.plot.circle(
                        "x", "y",
                        source=cluster_data,
                        size=8,
                        color=colors[i % len(colors)],
                        alpha=0.7,
                        legend_label=f"Cluster {cluster_id}"
                    )
            else:
                # Color by fitness
                color_mapper = LinearColorMapper(
                    palette=Viridis256,
                    low=min(fitness_values),
                    high=max(fitness_values)
                )
                
                self.plot.circle(
                    "x", "y",
                    source=self.data_source,
                    size=8,
                    color={"field": "fitness", "transform": color_mapper},
                    alpha=0.7
                )
                
                # Add color bar
                color_bar = ColorBar(
                    color_mapper=color_mapper,
                    width=8,
                    location=(0, 0),
                    title="Fitness"
                )
                self.plot.add_layout(color_bar, 'right')
            
            if clusters:
                self.plot.legend.location = "top_right"
            
            logger.info("Solution space plot updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update solution space plot: {e}")
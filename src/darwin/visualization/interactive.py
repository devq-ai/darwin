"""
Darwin Interactive Visualization Explorer

This module provides interactive exploration capabilities for genetic algorithm
optimization results, including dynamic filtering, real-time updates, parameter
sensitivity analysis, and solution space navigation.

Features:
- Interactive parameter exploration with sliders and controls
- Real-time filtering and data manipulation
- Solution space navigation with zoom and pan
- Parameter sensitivity analysis tools
- Interactive selection and highlighting
- Brushing and linking between multiple views
- Animation controls for temporal data
- Export capabilities for interactive sessions
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.events import ButtonClick, SelectionGeometry, Tap
from bokeh.layouts import column, row
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LassoSelectTool,
    RangeSlider,
    Slider,
    TapTool,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

logger = logging.getLogger(__name__)


class InteractiveFilter(param.Parameterized):
    """Interactive filtering controls for optimization data."""
    
    generation_range = param.Range(
        default=(0, 100),
        bounds=(0, 1000),
        doc="Generation range filter"
    )
    
    fitness_range = param.Range(
        default=(0.0, 1.0),
        bounds=(0.0, 1.0),
        doc="Fitness range filter"
    )
    
    diversity_threshold = param.Number(
        default=0.5,
        bounds=(0.0, 1.0),
        doc="Diversity threshold"
    )
    
    algorithm_type = param.Selector(
        default="all",
        objects=["all", "genetic", "evolutionary", "differential"],
        doc="Algorithm type filter"
    )


class ParameterExplorer(param.Parameterized):
    """Interactive parameter exploration interface."""
    
    mutation_rate = param.Number(
        default=0.1,
        bounds=(0.0, 1.0),
        step=0.01,
        doc="Mutation rate"
    )
    
    crossover_rate = param.Number(
        default=0.8,
        bounds=(0.0, 1.0),
        step=0.01,
        doc="Crossover rate"
    )
    
    population_size = param.Integer(
        default=100,
        bounds=(10, 1000),
        step=10,
        doc="Population size"
    )
    
    selection_pressure = param.Number(
        default=2.0,
        bounds=(1.0, 5.0),
        step=0.1,
        doc="Selection pressure"
    )


class InteractiveExplorer:
    """Main interactive exploration interface for optimization data."""
    
    def __init__(self, data_source: Optional[ColumnDataSource] = None):
        """
        Initialize the interactive explorer.
        
        Args:
            data_source: Bokeh ColumnDataSource with optimization data
        """
        self.data_source = data_source or ColumnDataSource()
        self.original_data = dict(self.data_source.data)
        self.filtered_data = dict(self.data_source.data)
        
        self.filter_controls = InteractiveFilter()
        self.parameter_explorer = ParameterExplorer()
        
        # Interactive plots
        self.main_plot = None
        self.detail_plot = None
        self.parameter_plot = None
        
        # Selection state
        self.selected_indices = []
        self.highlighted_solution = None
        
        # Animation state
        self.animation_playing = False
        self.animation_speed = 1.0
        self.current_generation = 0
        
        self._setup_plots()
        self._setup_callbacks()
    
    def _setup_plots(self):
        """Set up the main interactive plots."""
        # Main scatter plot
        self.main_plot = figure(
            width=600,
            height=400,
            title="Solution Space Explorer",
            tools="pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select,tap"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Generation", "@generation"),
            ("Fitness", "@fitness"),
            ("Solution", "@solution_id"),
            ("Parameters", "@parameters")
        ])
        self.main_plot.add_tools(hover)
        
        # Detail plot for selected solutions
        self.detail_plot = figure(
            width=600,
            height=300,
            title="Selected Solution Details",
            tools="pan,wheel_zoom,reset,save"
        )
        
        # Parameter sensitivity plot
        self.parameter_plot = figure(
            width=600,
            height=300,
            title="Parameter Sensitivity Analysis",
            tools="pan,wheel_zoom,reset,save"
        )
    
    def _setup_callbacks(self):
        """Set up interactive callbacks and event handlers."""
        # Selection callbacks
        self.data_source.selected.on_change('indices', self._on_selection_change)
        
        # Filter parameter callbacks
        self.filter_controls.param.watch(self._on_filter_change, 'generation_range')
        self.filter_controls.param.watch(self._on_filter_change, 'fitness_range')
        self.filter_controls.param.watch(self._on_filter_change, 'diversity_threshold')
        
        # Parameter exploration callbacks
        self.parameter_explorer.param.watch(self._on_parameter_change, [
            'mutation_rate', 'crossover_rate', 'population_size', 'selection_pressure'
        ])
    
    def update_data(self, new_data: Dict[str, Any]):
        """
        Update the data source with new optimization data.
        
        Args:
            new_data: New data dictionary to display
        """
        try:
            self.original_data = new_data.copy()
            self.data_source.data = new_data
            self._apply_filters()
            self._update_plots()
            
            logger.info("Interactive explorer data updated")
            
        except Exception as e:
            logger.error(f"Failed to update data: {e}")
    
    def _apply_filters(self):
        """Apply current filter settings to the data."""
        try:
            filtered_data = {}
            
            # Get filter values
            gen_min, gen_max = self.filter_controls.generation_range
            fit_min, fit_max = self.filter_controls.fitness_range
            div_threshold = self.filter_controls.diversity_threshold
            
            # Apply filters
            for key, values in self.original_data.items():
                filtered_values = []
                
                for i, value in enumerate(values):
                    # Generation filter
                    if 'generation' in self.original_data:
                        gen = self.original_data['generation'][i]
                        if not (gen_min <= gen <= gen_max):
                            continue
                    
                    # Fitness filter
                    if 'fitness' in self.original_data:
                        fit = self.original_data['fitness'][i]
                        if not (fit_min <= fit <= fit_max):
                            continue
                    
                    # Diversity filter
                    if 'diversity' in self.original_data:
                        div = self.original_data['diversity'][i]
                        if div < div_threshold:
                            continue
                    
                    filtered_values.append(value)
                
                filtered_data[key] = filtered_values
            
            self.filtered_data = filtered_data
            self.data_source.data = filtered_data
            
        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
    
    def _update_plots(self):
        """Update all plots with current data."""
        try:
            if not self.filtered_data:
                return
            
            # Update main plot
            if 'x' in self.filtered_data and 'y' in self.filtered_data:
                # Clear existing glyphs
                self.main_plot.renderers = []
                
                # Add scatter plot
                if 'fitness' in self.filtered_data:
                    mapper = linear_cmap(
                        field_name='fitness',
                        palette='Viridis256',
                        low=min(self.filtered_data['fitness']),
                        high=max(self.filtered_data['fitness'])
                    )
                    
                    self.main_plot.circle(
                        'x', 'y',
                        source=self.data_source,
                        size=8,
                        color=mapper,
                        alpha=0.7
                    )
                else:
                    self.main_plot.circle(
                        'x', 'y',
                        source=self.data_source,
                        size=8,
                        color='blue',
                        alpha=0.7
                    )
            
            # Update detail plot for selected solutions
            self._update_detail_plot()
            
            # Update parameter sensitivity plot
            self._update_parameter_plot()
            
        except Exception as e:
            logger.error(f"Failed to update plots: {e}")
    
    def _update_detail_plot(self):
        """Update the detail plot with selected solution information."""
        try:
            if not self.selected_indices:
                return
            
            # Clear existing glyphs
            self.detail_plot.renderers = []
            
            # Get selected solution data
            selected_data = {}
            for key, values in self.filtered_data.items():
                selected_data[key] = [values[i] for i in self.selected_indices if i < len(values)]
            
            if 'generation' in selected_data and 'fitness' in selected_data:
                self.detail_plot.line(
                    selected_data['generation'],
                    selected_data['fitness'],
                    line_width=2,
                    color='red'
                )
                
                self.detail_plot.circle(
                    selected_data['generation'],
                    selected_data['fitness'],
                    size=6,
                    color='red',
                    alpha=0.8
                )
            
        except Exception as e:
            logger.error(f"Failed to update detail plot: {e}")
    
    def _update_parameter_plot(self):
        """Update the parameter sensitivity plot."""
        try:
            # Clear existing glyphs
            self.parameter_plot.renderers = []
            
            # Generate parameter sensitivity data
            params = ['mutation_rate', 'crossover_rate', 'population_size', 'selection_pressure']
            current_values = [
                self.parameter_explorer.mutation_rate,
                self.parameter_explorer.crossover_rate,
                self.parameter_explorer.population_size / 100,  # Normalize
                self.parameter_explorer.selection_pressure / 5   # Normalize
            ]
            
            # Create bar chart
            self.parameter_plot.vbar(
                x=params,
                top=current_values,
                width=0.8,
                color='lightblue',
                alpha=0.7
            )
            
            self.parameter_plot.xaxis.major_label_orientation = 45
            
        except Exception as e:
            logger.error(f"Failed to update parameter plot: {e}")
    
    def _on_selection_change(self, attr, old, new):
        """Handle selection changes in the main plot."""
        try:
            self.selected_indices = new
            self._update_detail_plot()
            
            if self.selected_indices:
                logger.info(f"Selected {len(self.selected_indices)} solutions")
            
        except Exception as e:
            logger.error(f"Error handling selection change: {e}")
    
    def _on_filter_change(self, event):
        """Handle filter parameter changes."""
        try:
            self._apply_filters()
            self._update_plots()
            
        except Exception as e:
            logger.error(f"Error handling filter change: {e}")
    
    def _on_parameter_change(self, event):
        """Handle parameter exploration changes."""
        try:
            self._update_parameter_plot()
            self._simulate_parameter_effect()
            
        except Exception as e:
            logger.error(f"Error handling parameter change: {e}")
    
    def _simulate_parameter_effect(self):
        """Simulate the effect of parameter changes on optimization."""
        try:
            # This would integrate with the optimization engine
            # For now, we'll just log the parameter changes
            params = {
                'mutation_rate': self.parameter_explorer.mutation_rate,
                'crossover_rate': self.parameter_explorer.crossover_rate,
                'population_size': self.parameter_explorer.population_size,
                'selection_pressure': self.parameter_explorer.selection_pressure
            }
            
            logger.info(f"Parameter simulation requested: {params}")
            
        except Exception as e:
            logger.error(f"Failed to simulate parameter effect: {e}")
    
    def create_control_panel(self) -> pn.Column:
        """
        Create the control panel for interactive exploration.
        
        Returns:
            Panel Column with all controls
        """
        try:
            # Filter controls
            filter_panel = pn.Param(
                self.filter_controls,
                parameters=['generation_range', 'fitness_range', 'diversity_threshold', 'algorithm_type'],
                widgets={
                    'generation_range': pn.widgets.RangeSlider,
                    'fitness_range': pn.widgets.RangeSlider,
                    'diversity_threshold': pn.widgets.FloatSlider,
                    'algorithm_type': pn.widgets.Select
                },
                name="Data Filters"
            )
            
            # Parameter exploration controls
            param_panel = pn.Param(
                self.parameter_explorer,
                parameters=['mutation_rate', 'crossover_rate', 'population_size', 'selection_pressure'],
                widgets={
                    'mutation_rate': pn.widgets.FloatSlider,
                    'crossover_rate': pn.widgets.FloatSlider,
                    'population_size': pn.widgets.IntSlider,
                    'selection_pressure': pn.widgets.FloatSlider
                },
                name="Parameter Explorer"
            )
            
            # Animation controls
            animation_controls = pn.Column(
                pn.pane.Markdown("## Animation Controls"),
                pn.widgets.Button(name="Play/Pause", button_type="primary"),
                pn.widgets.FloatSlider(name="Speed", start=0.1, end=5.0, value=1.0, step=0.1),
                pn.widgets.IntSlider(name="Generation", start=0, end=100, value=0)
            )
            
            # Export controls
            export_controls = pn.Column(
                pn.pane.Markdown("## Export Options"),
                pn.widgets.Button(name="Export Selection", button_type="primary"),
                pn.widgets.Button(name="Export Current View", button_type="primary"),
                pn.widgets.Button(name="Export Animation", button_type="primary")
            )
            
            return pn.Column(
                pn.pane.Markdown("# Interactive Explorer Controls"),
                filter_panel,
                param_panel,
                animation_controls,
                export_controls
            )
            
        except Exception as e:
            logger.error(f"Failed to create control panel: {e}")
            return pn.pane.Markdown("Error creating control panel")
    
    def create_dashboard(self) -> pn.Row:
        """
        Create the complete interactive dashboard.
        
        Returns:
            Panel Row with plots and controls
        """
        try:
            # Create plot panels
            main_plot_panel = pn.pane.Bokeh(self.main_plot, sizing_mode="stretch_width")
            detail_plot_panel = pn.pane.Bokeh(self.detail_plot, sizing_mode="stretch_width")
            param_plot_panel = pn.pane.Bokeh(self.parameter_plot, sizing_mode="stretch_width")
            
            plots_column = pn.Column(
                main_plot_panel,
                pn.Row(detail_plot_panel, param_plot_panel)
            )
            
            # Create control panel
            controls = self.create_control_panel()
            
            return pn.Row(
                plots_column,
                controls,
                sizing_mode="stretch_width"
            )
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return pn.pane.Markdown("Error creating dashboard")
    
    def export_selected_data(self) -> Dict[str, Any]:
        """
        Export data for currently selected solutions.
        
        Returns:
            Dictionary with selected solution data
        """
        try:
            if not self.selected_indices:
                return {"error": "No solutions selected"}
            
            selected_data = {}
            for key, values in self.filtered_data.items():
                selected_data[key] = [values[i] for i in self.selected_indices if i < len(values)]
            
            selected_data['export_timestamp'] = datetime.now(timezone.utc).isoformat()
            selected_data['selection_count'] = len(self.selected_indices)
            
            return selected_data
            
        except Exception as e:
            logger.error(f"Failed to export selected data: {e}")
            return {"error": str(e)}
    
    def highlight_solution(self, solution_id: str):
        """
        Highlight a specific solution in the visualization.
        
        Args:
            solution_id: ID of the solution to highlight
        """
        try:
            if 'solution_id' in self.filtered_data:
                indices = [
                    i for i, sid in enumerate(self.filtered_data['solution_id'])
                    if sid == solution_id
                ]
                
                if indices:
                    self.data_source.selected.indices = indices
                    self.highlighted_solution = solution_id
                    logger.info(f"Highlighted solution: {solution_id}")
                else:
                    logger.warning(f"Solution not found: {solution_id}")
            
        except Exception as e:
            logger.error(f"Failed to highlight solution: {e}")
    
    def reset_exploration(self):
        """Reset all exploration state to defaults."""
        try:
            # Reset filters
            self.filter_controls.generation_range = (0, 100)
            self.filter_controls.fitness_range = (0.0, 1.0)
            self.filter_controls.diversity_threshold = 0.5
            self.filter_controls.algorithm_type = "all"
            
            # Reset parameters
            self.parameter_explorer.mutation_rate = 0.1
            self.parameter_explorer.crossover_rate = 0.8
            self.parameter_explorer.population_size = 100
            self.parameter_explorer.selection_pressure = 2.0
            
            # Reset selections
            self.selected_indices = []
            self.highlighted_solution = None
            
            # Reset data
            self.data_source.data = self.original_data
            self.filtered_data = self.original_data.copy()
            
            # Update plots
            self._update_plots()
            
            logger.info("Exploration state reset")
            
        except Exception as e:
            logger.error(f"Failed to reset exploration: {e}")
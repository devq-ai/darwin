"""
Monitoring Dashboard Component for Darwin Dashboard

This module provides real-time monitoring capabilities for genetic algorithm
optimization runs. It displays live progress updates, performance metrics,
population statistics, and convergence analysis.

Features:
- Real-time optimization progress tracking
- Live fitness evolution plots
- Population diversity monitoring
- Performance metrics display
- Generation-by-generation statistics
- Control buttons for optimization management
- WebSocket integration for live updates
"""

import logging
from typing import Any, Dict

import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from darwin.dashboard.utils.api_client import DarwinAPIClient

logger = logging.getLogger(__name__)


class OptimizationMonitor(param.Parameterized):
    """Parameter class for optimization monitoring state."""

    selected_optimizer_id = param.String(
        default="", doc="Currently selected optimizer ID"
    )
    is_running = param.Boolean(default=False, doc="Optimization running status")
    current_generation = param.Integer(default=0, doc="Current generation number")
    best_fitness = param.Number(default=0.0, doc="Best fitness achieved")
    average_fitness = param.Number(default=0.0, doc="Average population fitness")
    elapsed_time = param.String(default="00:00:00", doc="Elapsed optimization time")
    progress_percentage = param.Number(
        default=0.0, bounds=(0.0, 100.0), doc="Progress percentage"
    )


class MonitoringDashboard(param.Parameterized):
    """Real-time monitoring dashboard for genetic algorithm optimization runs."""

    # Dashboard state
    monitor = param.Parameter(default=None, doc="Optimization monitor instance")
    update_interval = param.Integer(default=1000, doc="Update interval in milliseconds")
    auto_refresh = param.Boolean(default=True, doc="Enable automatic refresh")

    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)

        self.api_client = api_client
        self.monitor = OptimizationMonitor()

        # Data storage for real-time updates
        self.fitness_history = []
        self.diversity_history = []
        self.performance_history = []

        # Bokeh data sources for live plots
        self.fitness_source = ColumnDataSource(
            data={
                "generation": [],
                "best_fitness": [],
                "avg_fitness": [],
                "worst_fitness": [],
                "std_fitness": [],
            }
        )

        self.diversity_source = ColumnDataSource(
            data={"generation": [], "diversity": [], "entropy": []}
        )

        self.performance_source = ColumnDataSource(
            data={"generation": [], "evaluation_time": [], "memory_usage": []}
        )

        # Create UI components
        self._create_components()

        # Start periodic updates if auto-refresh is enabled
        if self.auto_refresh:
            self._start_periodic_updates()

    def _create_components(self):
        """Create all monitoring dashboard components."""

        # Optimizer selector and controls
        self.controls_section = self._create_controls_section()

        # Progress and status display
        self.status_section = self._create_status_section()

        # Real-time plots
        self.plots_section = self._create_plots_section()

        # Statistics tables
        self.statistics_section = self._create_statistics_section()

        # Performance metrics
        self.performance_section = self._create_performance_section()

    def _create_controls_section(self):
        """Create optimizer controls and selection."""

        # Optimizer selector
        optimizer_selector = pn.Column(
            pn.pane.Markdown("### 🎯 Optimizer Selection"),
            pn.widgets.Select(
                name="Active Optimizers",
                value="",
                options=[],
                width=300,
                sizing_mode="fixed",
            ),
            pn.widgets.Button(name="🔄 Refresh List", button_type="light", width=150),
        )

        # Control buttons
        control_buttons = pn.Column(
            pn.pane.Markdown("### 🎮 Controls"),
            pn.Row(
                pn.widgets.Button(name="▶️ Start", button_type="success", width=100),
                pn.widgets.Button(name="⏸️ Pause", button_type="light", width=100),
                pn.widgets.Button(name="⏹️ Stop", button_type="light", width=100),
            ),
            pn.Row(
                pn.widgets.Button(name="📊 Results", button_type="primary", width=100),
                pn.widgets.Button(name="💾 Export", button_type="light", width=100),
                pn.widgets.Button(name="🗑️ Delete", button_type="light", width=100),
            ),
        )

        # Update settings
        update_settings = pn.Column(
            pn.pane.Markdown("### ⚙️ Settings"),
            pn.widgets.Checkbox(name="Auto Refresh", value=self.auto_refresh),
            pn.widgets.IntSlider(
                name="Update Interval (ms)",
                start=500,
                end=10000,
                step=500,
                value=self.update_interval,
                width=200,
            ),
        )

        # Setup event handlers
        self._setup_controls_handlers(
            optimizer_selector, control_buttons, update_settings
        )

        return pn.Row(
            optimizer_selector,
            control_buttons,
            update_settings,
            sizing_mode="stretch_width",
        )

    def _create_status_section(self):
        """Create status display section."""

        # Progress bar
        progress_bar = pn.widgets.Progress(
            name="Optimization Progress",
            value=0,
            max=100,
            sizing_mode="stretch_width",
            bar_color="success",
        )

        # Status cards
        status_cards = pn.GridBox(
            # Current Generation Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.2em;'>Generation</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='current-generation'>0</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Current</p>
                </div>
                """,
                width=150,
                height=100,
            ),
            # Best Fitness Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.2em;'>Best Fitness</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='best-fitness'>--</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Value</p>
                </div>
                """,
                width=150,
                height=100,
            ),
            # Average Fitness Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.2em;'>Avg Fitness</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='avg-fitness'>--</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Population</p>
                </div>
                """,
                width=150,
                height=100,
            ),
            # Elapsed Time Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                            color: #333; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.2em;'>Elapsed Time</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='elapsed-time'>00:00:00</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Duration</p>
                </div>
                """,
                width=150,
                height=100,
            ),
            ncols=4,
            sizing_mode="stretch_width",
        )

        # Status indicator
        status_indicator = pn.pane.HTML(
            """
            <div style='padding: 10px; border-radius: 5px; margin: 10px 0;
                        background-color: #f5f5f5; border: 1px solid #ddd; text-align: center;'>
                <strong>Status:</strong> <span id='optimization-status'>No optimization selected</span>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## 📈 Real-time Status"),
            progress_bar,
            status_cards,
            status_indicator,
            sizing_mode="stretch_width",
        )

    def _create_plots_section(self):
        """Create real-time plotting section."""

        # Fitness evolution plot
        fitness_plot = self._create_fitness_plot()

        # Population diversity plot
        diversity_plot = self._create_diversity_plot()

        # Performance metrics plot
        performance_plot = self._create_performance_plot()

        # Combine plots in tabs
        plots_tabs = pn.Tabs(
            ("📈 Fitness Evolution", fitness_plot),
            ("🌈 Population Diversity", diversity_plot),
            ("⚡ Performance Metrics", performance_plot),
            ("🎯 Solution Space", self._create_solution_space_plot()),
            dynamic=True,
        )

        return pn.Column(
            pn.pane.Markdown("## 📊 Live Visualizations"),
            plots_tabs,
            sizing_mode="stretch_width",
        )

    def _create_fitness_plot(self):
        """Create the fitness evolution plot."""

        p = figure(
            title="Fitness Evolution Over Generations",
            x_axis_label="Generation",
            y_axis_label="Fitness Value",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # Best fitness line
        p.line(
            "generation",
            "best_fitness",
            source=self.fitness_source,
            line_width=3,
            color="green",
            legend_label="Best Fitness",
            alpha=0.8,
        )

        # Average fitness line
        p.line(
            "generation",
            "avg_fitness",
            source=self.fitness_source,
            line_width=2,
            color="blue",
            legend_label="Average Fitness",
            alpha=0.7,
        )

        # Worst fitness line
        p.line(
            "generation",
            "worst_fitness",
            source=self.fitness_source,
            line_width=1,
            color="red",
            legend_label="Worst Fitness",
            alpha=0.6,
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Generation", "@generation"),
                ("Best", "@best_fitness{0.000}"),
                ("Average", "@avg_fitness{0.000}"),
                ("Worst", "@worst_fitness{0.000}"),
            ]
        )
        p.add_tools(hover)

        # Configure legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p)

    def _create_diversity_plot(self):
        """Create the population diversity plot."""

        p = figure(
            title="Population Diversity Over Time",
            x_axis_label="Generation",
            y_axis_label="Diversity Measure",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # Diversity line
        p.line(
            "generation",
            "diversity",
            source=self.diversity_source,
            line_width=2,
            color="purple",
            legend_label="Population Diversity",
        )

        # Entropy line
        p.line(
            "generation",
            "entropy",
            source=self.diversity_source,
            line_width=2,
            color="orange",
            legend_label="Genetic Entropy",
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Generation", "@generation"),
                ("Diversity", "@diversity{0.000}"),
                ("Entropy", "@entropy{0.000}"),
            ]
        )
        p.add_tools(hover)

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p)

    def _create_performance_plot(self):
        """Create the performance metrics plot."""

        p = figure(
            title="Performance Metrics",
            x_axis_label="Generation",
            y_axis_label="Time (seconds)",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # Evaluation time line
        p.line(
            "generation",
            "evaluation_time",
            source=self.performance_source,
            line_width=2,
            color="red",
            legend_label="Evaluation Time",
        )

        # Memory usage (scaled)
        p.line(
            "generation",
            "memory_usage",
            source=self.performance_source,
            line_width=2,
            color="blue",
            legend_label="Memory Usage (MB/10)",
            y_range_name="memory",
        )

        # Add second y-axis for memory
        p.extra_y_ranges = {"memory": p.y_range}

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Generation", "@generation"),
                ("Eval Time", "@evaluation_time{0.000}s"),
                ("Memory", "@memory_usage{0.0}MB"),
            ]
        )
        p.add_tools(hover)

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p)

    def _create_solution_space_plot(self):
        """Create solution space visualization."""

        # Placeholder for solution space plot
        placeholder = pn.pane.HTML(
            """
            <div style='text-align: center; padding: 40px; background-color: #f5f5f5;
                        border-radius: 8px; border: 2px dashed #ddd;'>
                <h3 style='color: #666; margin-bottom: 10px;'>🎯 Solution Space Visualization</h3>
                <p style='color: #888; margin: 0;'>
                    This section will show the solution space exploration<br>
                    including Pareto fronts for multi-objective problems
                </p>
            </div>
            """,
            height=400,
            sizing_mode="stretch_width",
        )

        return placeholder

    def _create_statistics_section(self):
        """Create statistics tables section."""

        # Generation statistics
        generation_stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Best Fitness",
                    "Average Fitness",
                    "Worst Fitness",
                    "Standard Deviation",
                    "Diversity Index",
                ],
                "Current": ["--", "--", "--", "--", "--"],
                "Previous": ["--", "--", "--", "--", "--"],
                "Change": ["--", "--", "--", "--", "--"],
            }
        )

        generation_stats_table = pn.widgets.Tabulator(
            generation_stats_df,
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            height=200,
            title="Generation Statistics",
        )

        # Population statistics
        population_stats_df = pd.DataFrame(
            {"Variable": [], "Min": [], "Max": [], "Mean": [], "Std": [], "Range": []}
        )

        population_stats_table = pn.widgets.Tabulator(
            population_stats_df,
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            height=200,
            title="Population Statistics",
        )

        return pn.Column(
            pn.pane.Markdown("## 📋 Statistics"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Generation Metrics"), generation_stats_table
                ),
                pn.Column(
                    pn.pane.Markdown("### Population Variables"), population_stats_table
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _create_performance_section(self):
        """Create performance metrics section."""

        # Performance summary
        performance_summary = pn.pane.HTML(
            """
            <div style='padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;'>
                <h4 style='margin-top: 0; color: #495057;'>⚡ Performance Summary</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;'>
                    <div>
                        <strong>Evaluations/sec:</strong><br>
                        <span id='evals-per-sec' style='font-size: 1.2em; color: #28a745;'>--</span>
                    </div>
                    <div>
                        <strong>Memory Usage:</strong><br>
                        <span id='memory-usage' style='font-size: 1.2em; color: #17a2b8;'>--</span>
                    </div>
                    <div>
                        <strong>CPU Usage:</strong><br>
                        <span id='cpu-usage' style='font-size: 1.2em; color: #fd7e14;'>--</span>
                    </div>
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        # Convergence analysis
        convergence_analysis = pn.pane.HTML(
            """
            <div style='padding: 15px; background-color: #fff3cd; border-radius: 8px; border: 1px solid #ffeaa7;'>
                <h4 style='margin-top: 0; color: #856404;'>🎯 Convergence Analysis</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                    <div>
                        <strong>Convergence Rate:</strong><br>
                        <span id='convergence-rate' style='font-size: 1.2em;'>--</span>
                    </div>
                    <div>
                        <strong>Stagnation Counter:</strong><br>
                        <span id='stagnation-counter' style='font-size: 1.2em;'>--</span>
                    </div>
                </div>
                <div style='margin-top: 10px;'>
                    <strong>Recommendation:</strong><br>
                    <span id='convergence-recommendation' style='color: #856404;'>Monitoring optimization progress...</span>
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## ⚡ Performance & Convergence"),
            performance_summary,
            convergence_analysis,
            sizing_mode="stretch_width",
        )

    def _setup_controls_handlers(
        self, optimizer_selector, control_buttons, update_settings
    ):
        """Setup event handlers for control components."""

        # Get specific buttons
        refresh_btn = optimizer_selector[2]
        start_btn = control_buttons[1][0]
        pause_btn = control_buttons[1][1]
        stop_btn = control_buttons[1][2]
        results_btn = control_buttons[2][0]
        export_btn = control_buttons[2][1]
        delete_btn = control_buttons[2][2]

        # Auto refresh checkbox
        auto_refresh_cb = update_settings[1]

        # Setup handlers
        refresh_btn.on_click(self._refresh_optimizer_list)
        start_btn.on_click(self._start_optimization)
        pause_btn.on_click(self._pause_optimization)
        stop_btn.on_click(self._stop_optimization)
        results_btn.on_click(self._show_results)
        export_btn.on_click(self._export_results)
        delete_btn.on_click(self._delete_optimizer)

        # Auto refresh handler
        auto_refresh_cb.param.watch(self._toggle_auto_refresh, "value")

    def create_interface(self):
        """Create the complete monitoring dashboard interface."""

        return pn.Column(
            pn.pane.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">📊 Optimization Monitor</h2>
                    <p style="margin: 10px 0; opacity: 0.9;">Real-time tracking of genetic algorithm evolution</p>
                </div>
                """
            ),
            self.controls_section,
            self.status_section,
            self.plots_section,
            self.statistics_section,
            self.performance_section,
            sizing_mode="stretch_width",
        )

    # Event handlers
    def _refresh_optimizer_list(self, event):
        """Refresh the list of available optimizers."""
        try:
            # TODO: Implement API call to get optimizer list
            logger.info("Refreshing optimizer list...")
        except Exception as e:
            logger.error(f"Error refreshing optimizer list: {e}")

    def _start_optimization(self, event):
        """Start the selected optimization."""
        try:
            if self.monitor.selected_optimizer_id:
                # TODO: Implement start optimization API call
                logger.info(
                    f"Starting optimization: {self.monitor.selected_optimizer_id}"
                )
        except Exception as e:
            logger.error(f"Error starting optimization: {e}")

    def _pause_optimization(self, event):
        """Pause the running optimization."""
        try:
            # TODO: Implement pause functionality
            logger.info("Pausing optimization...")
        except Exception as e:
            logger.error(f"Error pausing optimization: {e}")

    def _stop_optimization(self, event):
        """Stop the running optimization."""
        try:
            if self.monitor.selected_optimizer_id:
                # TODO: Implement stop optimization API call
                logger.info(
                    f"Stopping optimization: {self.monitor.selected_optimizer_id}"
                )
        except Exception as e:
            logger.error(f"Error stopping optimization: {e}")

    def _show_results(self, event):
        """Show optimization results."""
        try:
            # TODO: Navigate to results view
            logger.info("Showing optimization results...")
        except Exception as e:
            logger.error(f"Error showing results: {e}")

    def _export_results(self, event):
        """Export optimization results."""
        try:
            # TODO: Implement export functionality
            logger.info("Exporting optimization results...")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

    def _delete_optimizer(self, event):
        """Delete the selected optimizer."""
        try:
            # TODO: Implement delete functionality with confirmation
            logger.info("Deleting optimizer...")
        except Exception as e:
            logger.error(f"Error deleting optimizer: {e}")

    def _toggle_auto_refresh(self, event):
        """Toggle automatic refresh."""
        self.auto_refresh = event.new
        if self.auto_refresh:
            self._start_periodic_updates()
        else:
            self._stop_periodic_updates()

    def _start_periodic_updates(self):
        """Start periodic updates for real-time monitoring."""
        try:
            # Add periodic callback for updates
            pn.state.add_periodic_callback(
                self._update_monitoring_data, self.update_interval
            )
            logger.info("Started periodic updates for monitoring")
        except Exception as e:
            logger.error(f"Error starting periodic updates: {e}")

    def _stop_periodic_updates(self):
        """Stop periodic updates."""
        try:
            # TODO: Remove periodic callback
            logger.info("Stopped periodic updates for monitoring")
        except Exception as e:
            logger.error(f"Error stopping periodic updates: {e}")

    async def _update_monitoring_data(self):
        """Update monitoring data from API."""
        try:
            if not self.monitor.selected_optimizer_id:
                return

            # Get optimization progress
            progress = await self.api_client.get_optimization_progress(
                self.monitor.selected_optimizer_id
            )

            if progress:
                # Update monitor state
                self.monitor.current_generation = progress.get("current_generation", 0)
                self.monitor.best_fitness = progress.get("best_fitness", 0.0)
                self.monitor.average_fitness = progress.get("average_fitness", 0.0)
                self.monitor.is_running = progress.get("status") == "running"

                # Update progress percentage
                max_generations = progress.get("max_generations", 100)
                self.monitor.progress_percentage = (
                    self.monitor.current_generation / max_generations * 100
                )

                # Update fitness history
                self._update_fitness_data(progress)

                # Update diversity data
                self._update_diversity_data(progress)

                # Update performance data
                self._update_performance_data(progress)

        except Exception as e:
            logger.error(f"Error updating monitoring data: {e}")

    def _update_fitness_data(self, progress_data: Dict[str, Any]):
        """Update fitness evolution data."""
        try:
            new_data = {
                "generation": [progress_data.get("current_generation", 0)],
                "best_fitness": [progress_data.get("best_fitness", 0.0)],
                "avg_fitness": [progress_data.get("average_fitness", 0.0)],
                "worst_fitness": [progress_data.get("worst_fitness", 0.0)],
                "std_fitness": [progress_data.get("std_fitness", 0.0)],
            }

            # Stream new data to the plot
            self.fitness_source.stream(new_data, rollover=1000)

        except Exception as e:
            logger.error(f"Error updating fitness data: {e}")

    def _update_diversity_data(self, progress_data: Dict[str, Any]):
        """Update population diversity data."""
        try:
            new_data = {
                "generation": [progress_data.get("current_generation", 0)],
                "diversity": [progress_data.get("diversity", 0.0)],
                "entropy": [progress_data.get("entropy", 0.0)],
            }

            # Stream new data to the plot
            self.diversity_source.stream(new_data, rollover=1000)

        except Exception as e:
            logger.error(f"Error updating diversity data: {e}")

    def _update_performance_data(self, progress_data: Dict[str, Any]):
        """Update performance metrics data."""
        try:
            new_data = {
                "generation": [progress_data.get("current_generation", 0)],
                "evaluation_time": [progress_data.get("evaluation_time", 0.0)],
                "memory_usage": [
                    progress_data.get("memory_usage", 0.0) / 10
                ],  # Scale for plotting
            }

            # Stream new data to the plot
            self.performance_source.stream(new_data, rollover=1000)

        except Exception as e:
            logger.error(f"Error updating performance data: {e}")

    async def update_progress(self, progress_data: Dict[str, Any]):
        """Update progress from WebSocket message."""
        try:
            await self._update_monitoring_data()
        except Exception as e:
            logger.error(f"Error updating progress from WebSocket: {e}")

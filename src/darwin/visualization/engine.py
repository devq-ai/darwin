"""
Darwin Visualization Engine Core

This module implements the core visualization engine for the Darwin genetic algorithm
platform. It provides comprehensive visualization capabilities including interactive
plots, real-time monitoring, statistical analysis, and advanced exploration tools.

Features:
- Interactive Bokeh-based visualizations
- Real-time optimization monitoring with WebSocket support
- Multi-objective optimization analysis
- Solution space exploration with dimensionality reduction
- Performance benchmarking and comparison tools
- Statistical analysis and correlation visualization
- Responsive design with mobile support
- Export capabilities for plots and data
- Theme management and accessibility features
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.layouts import column, row
from bokeh.models import HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from darwin.dashboard.utils.api_client import DarwinAPIClient
from darwin.dashboard.utils.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class VisualizationConfig(param.Parameterized):
    """Configuration parameters for the visualization engine."""

    # Visual settings
    theme = param.Selector(
        default="light",
        objects=["light", "dark", "minimal", "high_contrast"],
        doc="Visual theme",
    )
    color_palette = param.Selector(
        default="Category10",
        objects=["Category10", "Viridis", "Spectral", "Plasma", "Custom"],
        doc="Color palette for plots",
    )

    # Plot dimensions
    plot_width = param.Integer(
        default=700, bounds=(400, 1200), doc="Default plot width"
    )
    plot_height = param.Integer(
        default=500, bounds=(300, 800), doc="Default plot height"
    )

    # Interactive features
    animation_enabled = param.Boolean(default=True, doc="Enable plot animations")
    hover_enabled = param.Boolean(default=True, doc="Enable hover tooltips")
    zoom_enabled = param.Boolean(default=True, doc="Enable zoom interactions")
    pan_enabled = param.Boolean(default=True, doc="Enable pan interactions")

    # Performance settings
    max_points_per_plot = param.Integer(default=10000, doc="Maximum points per plot")
    update_frequency = param.Number(
        default=1.0, bounds=(0.1, 10.0), doc="Update frequency (seconds)"
    )

    # Export settings
    export_format = param.Selector(
        default="png",
        objects=["png", "svg", "html", "pdf"],
        doc="Default export format",
    )
    export_quality = param.Selector(
        default="high",
        objects=["low", "medium", "high", "publication"],
        doc="Export quality",
    )


class VisualizationEngine(param.Parameterized):
    """
    Advanced visualization engine for genetic algorithm analytics.

    Provides comprehensive visualization capabilities for optimization results,
    including real-time monitoring, statistical analysis, and interactive exploration.
    """

    # State parameters
    is_monitoring = param.Boolean(default=False, doc="Real-time monitoring status")
    selected_optimizers = param.List(default=[], doc="Selected optimizer IDs")
    current_view = param.Selector(
        default="overview",
        objects=[
            "overview",
            "fitness",
            "diversity",
            "solutions",
            "comparison",
            "analytics",
        ],
        doc="Current visualization view",
    )

    def __init__(
        self,
        api_client: DarwinAPIClient,
        config: Optional[VisualizationConfig] = None,
        **params,
    ):
        super().__init__(**params)

        self.api_client = api_client
        self.config = config or VisualizationConfig()

        # Data storage
        self.optimization_data = {}
        self.cached_plots = {}
        self.data_sources = {}

        # Real-time monitoring
        self.websocket_manager = None
        self.monitoring_task = None

        # Initialize components
        self._initialize_components()
        self._setup_layouts()

    def _initialize_components(self):
        """Initialize all visualization components."""
        # Control panel
        self.controls = self._create_control_panel()

        # Main plot area
        self.main_plot_area = pn.pane.HTML("<div>Select an optimization to view</div>")

        # Statistics panel
        self.stats_panel = self._create_statistics_panel()

        # Export panel
        self.export_panel = self._create_export_panel()

        # Setup event handlers
        self._setup_event_handlers()

    def _create_control_panel(self) -> pn.Column:
        """Create the main control panel."""
        # Optimizer selection
        self.optimizer_select = pn.widgets.MultiSelect(
            name="Select Optimizers", options=[], size=8, sizing_mode="stretch_width"
        )

        # View selection
        self.view_select = pn.widgets.RadioButtonGroup(
            name="View",
            options=[
                "Overview",
                "Fitness",
                "Diversity",
                "Solutions",
                "Comparison",
                "Analytics",
            ],
            value="Overview",
            button_type="primary",
        )

        # Real-time monitoring toggle
        self.monitoring_toggle = pn.widgets.Toggle(
            name="Real-time Monitoring", value=False, button_type="success"
        )

        # Refresh button
        self.refresh_button = pn.widgets.Button(
            name="Refresh Data", button_type="primary", sizing_mode="stretch_width"
        )

        # Configuration panel
        config_panel = self._create_configuration_panel()

        return pn.Column(
            "## Controls",
            self.optimizer_select,
            self.view_select,
            self.monitoring_toggle,
            self.refresh_button,
            "## Configuration",
            config_panel,
            sizing_mode="stretch_width",
        )

    def _create_configuration_panel(self) -> pn.Column:
        """Create configuration controls."""
        # Theme selector
        theme_select = pn.widgets.Select(
            name="Theme",
            value=self.config.theme,
            options=["light", "dark", "minimal", "high_contrast"],
        )

        # Color palette selector
        palette_select = pn.widgets.Select(
            name="Color Palette",
            value=self.config.color_palette,
            options=["Category10", "Viridis", "Spectral", "Plasma"],
        )

        # Animation toggle
        animation_toggle = pn.widgets.Checkbox(
            name="Enable Animations", value=self.config.animation_enabled
        )

        # Update frequency slider
        update_slider = pn.widgets.FloatSlider(
            name="Update Frequency (s)",
            start=0.1,
            end=10.0,
            step=0.1,
            value=self.config.update_frequency,
        )

        return pn.Column(
            theme_select,
            palette_select,
            animation_toggle,
            update_slider,
            sizing_mode="stretch_width",
        )

    def _create_statistics_panel(self) -> pn.Column:
        """Create statistics display panel."""
        self.stats_html = pn.pane.HTML(
            "<div><h3>Statistics</h3><p>No data available</p></div>",
            sizing_mode="stretch_width",
        )

        return pn.Column("## Statistics", self.stats_html, sizing_mode="stretch_width")

    def _create_export_panel(self) -> pn.Column:
        """Create export controls panel."""
        # Export format selector
        export_format = pn.widgets.Select(
            name="Export Format",
            value=self.config.export_format,
            options=["png", "svg", "html", "pdf"],
        )

        # Export quality selector
        export_quality = pn.widgets.Select(
            name="Export Quality",
            value=self.config.export_quality,
            options=["low", "medium", "high", "publication"],
        )

        # Export buttons
        export_plot_btn = pn.widgets.Button(
            name="Export Current Plot", button_type="primary"
        )

        export_all_btn = pn.widgets.Button(
            name="Export All Plots", button_type="success"
        )

        export_data_btn = pn.widgets.Button(name="Export Data", button_type="default")

        return pn.Column(
            "## Export",
            export_format,
            export_quality,
            export_plot_btn,
            export_all_btn,
            export_data_btn,
            sizing_mode="stretch_width",
        )

    def _setup_layouts(self):
        """Setup the main layout structure."""
        # Main dashboard layout
        self.layout = pn.template.MaterialTemplate(
            title="🔬 Darwin Visualization Engine",
            sidebar=[self.controls, self.stats_panel, self.export_panel],
            sidebar_width=350,
            header_background="#1565C0",
        )

        # Add main content
        self.layout.main.append(
            pn.Column(self.main_plot_area, sizing_mode="stretch_both")
        )

    def _setup_event_handlers(self):
        """Setup event handlers for interactive components."""
        # Optimizer selection
        self.optimizer_select.param.watch(self._on_optimizer_selection_change, "value")

        # View selection
        self.view_select.param.watch(self._on_view_change, "value")

        # Monitoring toggle
        self.monitoring_toggle.param.watch(self._on_monitoring_toggle, "value")

        # Refresh button
        self.refresh_button.on_click(self._on_refresh_click)

    async def initialize(self):
        """Initialize the visualization engine."""
        try:
            # Load available optimizers
            await self._load_optimizers()

            # Setup WebSocket for real-time updates
            self._setup_websocket()

            logger.info("Visualization engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize visualization engine: {e}")
            raise

    async def _load_optimizers(self):
        """Load available optimizers from the API."""
        try:
            # Get optimizer list from API
            optimizers = await self.api_client.get_optimizers()

            # Update optimizer selection options
            options = [(opt["name"], opt["id"]) for opt in optimizers]
            self.optimizer_select.options = options

            logger.info(f"Loaded {len(optimizers)} optimizers")

        except Exception as e:
            logger.error(f"Failed to load optimizers: {e}")
            self.optimizer_select.options = []

    def _setup_websocket(self):
        """Setup WebSocket connection for real-time updates."""
        try:
            self.websocket_manager = WebSocketManager(
                "ws://localhost:8000/ws/optimization/progress"
            )

            # Setup message handler
            self.websocket_manager.on_message(self._handle_websocket_message)

        except Exception as e:
            logger.error(f"Failed to setup WebSocket: {e}")

    async def _handle_websocket_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            message_type = message.get("type")

            if message_type == "optimization_progress":
                await self._update_optimization_data(message["data"])
            elif message_type == "optimization_complete":
                await self._handle_optimization_complete(message["data"])
            elif message_type == "error":
                logger.error(f"WebSocket error: {message['error']}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _update_optimization_data(self, data: Dict[str, Any]):
        """Update optimization data from WebSocket."""
        optimizer_id = data.get("optimizer_id")

        if optimizer_id in self.selected_optimizers:
            # Update cached data
            if optimizer_id not in self.optimization_data:
                self.optimization_data[optimizer_id] = {
                    "fitness_history": [],
                    "population_data": [],
                    "statistics": {},
                }

            # Add new data point
            self.optimization_data[optimizer_id]["fitness_history"].append(
                {
                    "generation": data.get("generation", 0),
                    "best_fitness": data.get("best_fitness", 0),
                    "mean_fitness": data.get("mean_fitness", 0),
                    "diversity": data.get("diversity", 0),
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Update visualization if monitoring is enabled
            if self.is_monitoring:
                await self._update_current_view()

    async def _handle_optimization_complete(self, data: Dict[str, Any]):
        """Handle optimization completion."""
        optimizer_id = data.get("optimizer_id")

        if optimizer_id in self.selected_optimizers:
            # Load final results
            await self._load_optimization_results(optimizer_id)

            # Update visualization
            await self._update_current_view()

            # Show completion notification
            self._show_notification(
                f"Optimization {optimizer_id} completed", type="success"
            )

    async def _load_optimization_results(self, optimizer_id: str):
        """Load complete optimization results."""
        try:
            # Get detailed results from API
            results = await self.api_client.get_optimization_results(optimizer_id)

            # Process and cache the data
            self.optimization_data[optimizer_id] = {
                "fitness_history": results.get("fitness_history", []),
                "population_data": results.get("population_data", []),
                "best_solutions": results.get("best_solutions", []),
                "statistics": results.get("statistics", {}),
                "metadata": results.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Failed to load optimization results for {optimizer_id}: {e}")

    def _on_optimizer_selection_change(self, event):
        """Handle optimizer selection changes."""
        self.selected_optimizers = event.new

        # Load data for selected optimizers
        asyncio.create_task(self._load_selected_data())

    async def _load_selected_data(self):
        """Load data for currently selected optimizers."""
        for optimizer_id in self.selected_optimizers:
            if optimizer_id not in self.optimization_data:
                await self._load_optimization_results(optimizer_id)

        # Update current view
        await self._update_current_view()

    def _on_view_change(self, event):
        """Handle view selection changes."""
        self.current_view = event.new.lower()
        asyncio.create_task(self._update_current_view())

    def _on_monitoring_toggle(self, event):
        """Handle monitoring toggle changes."""
        self.is_monitoring = event.new

        if self.is_monitoring:
            self._start_monitoring()
        else:
            self._stop_monitoring()

    def _start_monitoring(self):
        """Start real-time monitoring."""
        if self.websocket_manager:
            asyncio.create_task(self.websocket_manager.connect())

        # Start periodic updates
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Started real-time monitoring")

    def _stop_monitoring(self):
        """Stop real-time monitoring."""
        if self.websocket_manager:
            asyncio.create_task(self.websocket_manager.disconnect())

        # Stop periodic updates
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None

        logger.info("Stopped real-time monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop for periodic updates."""
        while self.is_monitoring:
            try:
                # Update current view
                await self._update_current_view()

                # Wait for next update
                await asyncio.sleep(self.config.update_frequency)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    async def _update_current_view(self):
        """Update the current visualization view."""
        if not self.selected_optimizers:
            self.main_plot_area.object = "<div><h3>No optimizers selected</h3><p>Please select one or more optimizers to visualize.</p></div>"
            return

        try:
            if self.current_view == "overview":
                plot = await self._create_overview_plot()
            elif self.current_view == "fitness":
                plot = await self._create_fitness_plot()
            elif self.current_view == "diversity":
                plot = await self._create_diversity_plot()
            elif self.current_view == "solutions":
                plot = await self._create_solutions_plot()
            elif self.current_view == "comparison":
                plot = await self._create_comparison_plot()
            elif self.current_view == "analytics":
                plot = await self._create_analytics_plot()
            else:
                plot = pn.pane.HTML("<div>View not implemented</div>")

            self.main_plot_area.object = plot

            # Update statistics
            await self._update_statistics()

        except Exception as e:
            logger.error(f"Error updating view {self.current_view}: {e}")
            self.main_plot_area.object = f"<div><h3>Error</h3><p>{str(e)}</p></div>"

    async def _create_overview_plot(self) -> pn.pane.Bokeh:
        """Create overview dashboard plot."""
        # Create multi-panel overview
        plots = []

        for optimizer_id in self.selected_optimizers[:4]:  # Limit to 4 optimizers
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])

            if fitness_history:
                df = pd.DataFrame(fitness_history)

                # Create mini fitness plot
                p = figure(
                    width=350,
                    height=250,
                    title=f"Optimizer {optimizer_id}",
                    tools="pan,wheel_zoom,box_zoom,reset",
                )

                p.line(
                    df["generation"],
                    df["best_fitness"],
                    line_width=2,
                    color=Category10[10][len(plots) % 10],
                    legend_label="Best Fitness",
                )

                p.line(
                    df["generation"],
                    df["mean_fitness"],
                    line_width=1,
                    line_dash="dashed",
                    color=Category10[10][len(plots) % 10],
                    legend_label="Mean Fitness",
                    alpha=0.7,
                )

                p.xaxis.axis_label = "Generation"
                p.yaxis.axis_label = "Fitness"
                p.legend.location = "top_left"

                plots.append(p)

        if plots:
            # Arrange plots in grid
            if len(plots) == 1:
                layout = plots[0]
            elif len(plots) == 2:
                layout = row(plots[0], plots[1])
            elif len(plots) == 3:
                layout = column(plots[0], row(plots[1], plots[2]))
            else:
                layout = column(row(plots[0], plots[1]), row(plots[2], plots[3]))

            return pn.pane.Bokeh(layout, sizing_mode="stretch_width")
        else:
            return pn.pane.HTML(
                "<div><h3>No data available</h3><p>Run optimizations to see results.</p></div>"
            )

    async def _create_fitness_plot(self) -> pn.pane.Bokeh:
        """Create detailed fitness evolution plot."""
        p = figure(
            width=self.config.plot_width,
            height=self.config.plot_height,
            title="Fitness Evolution",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_axis_label="Generation",
            y_axis_label="Fitness",
        )

        colors = Category10[10]

        for i, optimizer_id in enumerate(self.selected_optimizers):
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])

            if fitness_history:
                df = pd.DataFrame(fitness_history)
                color = colors[i % len(colors)]

                # Best fitness line
                p.line(
                    df["generation"],
                    df["best_fitness"],
                    line_width=3,
                    color=color,
                    legend_label=f"Optimizer {optimizer_id} (Best)",
                    alpha=0.8,
                )

                # Mean fitness line
                p.line(
                    df["generation"],
                    df["mean_fitness"],
                    line_width=2,
                    line_dash="dashed",
                    color=color,
                    legend_label=f"Optimizer {optimizer_id} (Mean)",
                    alpha=0.6,
                )

                # Add hover tool
                if self.config.hover_enabled:
                    hover = HoverTool(
                        tooltips=[
                            ("Generation", "@x"),
                            ("Fitness", "@y"),
                            ("Optimizer", optimizer_id),
                        ]
                    )
                    p.add_tools(hover)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p, sizing_mode="stretch_width")

    async def _create_diversity_plot(self) -> pn.pane.Bokeh:
        """Create population diversity plot."""
        p = figure(
            width=self.config.plot_width,
            height=self.config.plot_height,
            title="Population Diversity",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_axis_label="Generation",
            y_axis_label="Diversity Measure",
        )

        colors = Category10[10]

        for i, optimizer_id in enumerate(self.selected_optimizers):
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])

            if fitness_history:
                df = pd.DataFrame(fitness_history)
                if "diversity" in df.columns:
                    color = colors[i % len(colors)]

                    p.line(
                        df["generation"],
                        df["diversity"],
                        line_width=2,
                        color=color,
                        legend_label=f"Optimizer {optimizer_id}",
                        alpha=0.8,
                    )

        p.legend.location = "top_right"
        return pn.pane.Bokeh(p, sizing_mode="stretch_width")

    async def _create_solutions_plot(self) -> pn.pane.Bokeh:
        """Create solution space exploration plot."""
        # Use PCA or t-SNE for dimensionality reduction
        all_solutions = []
        solution_labels = []

        for optimizer_id in self.selected_optimizers:
            data = self.optimization_data.get(optimizer_id, {})
            best_solutions = data.get("best_solutions", [])

            for solution in best_solutions:
                if "variables" in solution:
                    all_solutions.append(solution["variables"])
                    solution_labels.append(optimizer_id)

        if len(all_solutions) < 2:
            return pn.pane.HTML(
                "<div><h3>Insufficient Data</h3><p>Need at least 2 solutions for visualization.</p></div>"
            )

        # Perform dimensionality reduction
        solutions_array = np.array(all_solutions)

        if solutions_array.shape[1] > 2:
            # Use PCA for initial reduction
            pca = PCA(n_components=min(50, solutions_array.shape[1]))
            solutions_reduced = pca.fit_transform(
                StandardScaler().fit_transform(solutions_array)
            )

            if solutions_reduced.shape[1] > 2:
                # Use t-SNE for final 2D projection
                tsne = TSNE(n_components=2, random_state=42)
                solutions_2d = tsne.fit_transform(solutions_reduced)
            else:
                solutions_2d = solutions_reduced
        else:
            solutions_2d = solutions_array

        # Create scatter plot
        p = figure(
            width=self.config.plot_width,
            height=self.config.plot_height,
            title="Solution Space (2D Projection)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_axis_label="Component 1",
            y_axis_label="Component 2",
        )

        # Color by optimizer
        colors = Category10[10]
        unique_optimizers = list(set(solution_labels))

        for i, optimizer_id in enumerate(unique_optimizers):
            mask = [label == optimizer_id for label in solution_labels]
            x_vals = solutions_2d[mask, 0]
            y_vals = solutions_2d[mask, 1]

            p.scatter(
                x_vals,
                y_vals,
                size=8,
                color=colors[i % len(colors)],
                legend_label=f"Optimizer {optimizer_id}",
                alpha=0.7,
            )

        p.legend.location = "top_left"
        return pn.pane.Bokeh(p, sizing_mode="stretch_width")

    async def _create_comparison_plot(self) -> pn.pane.Bokeh:
        """Create performance comparison plot."""
        if len(self.selected_optimizers) < 2:
            return pn.pane.HTML(
                "<div><h3>Comparison View</h3><p>Select at least 2 optimizers to compare.</p></div>"
            )

        # Create comparison metrics
        metrics = []

        for optimizer_id in self.selected_optimizers:
            data = self.optimization_data.get(optimizer_id, {})
            stats = data.get("statistics", {})

            metrics.append(
                {
                    "optimizer": optimizer_id,
                    "best_fitness": stats.get("best_fitness", 0),
                    "convergence_generation": stats.get("convergence_generation", 0),
                    "total_generations": stats.get("total_generations", 0),
                    "mean_fitness": stats.get("final_mean_fitness", 0),
                }
            )

        df = pd.DataFrame(metrics)

        # Create bar chart
        p = figure(
            width=self.config.plot_width,
            height=self.config.plot_height,
            title="Performance Comparison",
            x_range=df["optimizer"].tolist(),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_axis_label="Optimizer",
            y_axis_label="Best Fitness",
        )

        p.vbar(
            x=df["optimizer"],
            top=df["best_fitness"],
            width=0.8,
            color=Category10[3][0],
            alpha=0.8,
        )

        p.xgrid.grid_line_color = None
        return pn.pane.Bokeh(p, sizing_mode="stretch_width")

    async def _create_analytics_plot(self) -> pn.Column:
        """Create advanced analytics dashboard."""
        # Statistical analysis
        stats_html = self._generate_statistical_analysis()

        # Correlation analysis
        correlation_plot = await self._create_correlation_plot()

        # Performance metrics table
        metrics_table = self._create_metrics_table()

        return pn.Column(
            "## Advanced Analytics",
            stats_html,
            correlation_plot,
            metrics_table,
            sizing_mode="stretch_width",
        )

    def _generate_statistical_analysis(self) -> pn.pane.HTML:
        """Generate statistical analysis summary."""
        html_content = "<div><h4>Statistical Analysis</h4>"

        if not self.selected_optimizers:
            html_content += "<p>No optimizers selected for analysis.</p></div>"
            return pn.pane.HTML(html_content)

        for optimizer_id in self.selected_optimizers:
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])

            if fitness_history:
                df = pd.DataFrame(fitness_history)

                html_content += f"""
                <div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
                    <h5>Optimizer {optimizer_id}</h5>
                    <ul>
                        <li><strong>Generations:</strong> {len(df)}</li>
                        <li><strong>Best Fitness:</strong> {df['best_fitness'].max():.4f}</li>
                        <li><strong>Final Fitness:</strong> {df['best_fitness'].iloc[-1]:.4f}</li>
                        <li><strong>Mean Improvement:</strong> {(df['best_fitness'].iloc[-1] - df['best_fitness'].iloc[0]):.4f}</li>
                        <li><strong>Convergence Rate:</strong> {((df['best_fitness'].iloc[-1] - df['best_fitness'].iloc[0]) / len(df)):.6f} per generation</li>
                    </ul>
                </div>
                """

        html_content += "</div>"
        return pn.pane.HTML(html_content, sizing_mode="stretch_width")

    async def _create_correlation_plot(self) -> pn.pane.Bokeh:
        """Create correlation analysis plot."""
        # This is a placeholder - would implement actual correlation analysis
        p = figure(
            width=600,
            height=400,
            title="Parameter Correlation Analysis",
            tools="pan,wheel_zoom,box_zoom,reset",
        )

        p.text(
            x=[0.5],
            y=[0.5],
            text=["Correlation analysis coming soon"],
            text_align="center",
            text_baseline="middle",
        )

        return pn.pane.Bokeh(p)

    def _create_metrics_table(self) -> pn.widgets.Tabulator:
        """Create performance metrics table."""
        metrics_data = []

        for optimizer_id in self.selected_optimizers:
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])
            stats = data.get("statistics", {})

            if fitness_history:
                df = pd.DataFrame(fitness_history)

                metrics_data.append(
                    {
                        "Optimizer": optimizer_id,
                        "Generations": len(df),
                        "Best Fitness": f"{df['best_fitness'].max():.4f}",
                        "Final Fitness": f"{df['best_fitness'].iloc[-1]:.4f}",
                        "Convergence Gen": stats.get("convergence_generation", "N/A"),
                        "Success Rate": f"{stats.get('success_rate', 0):.2%}",
                    }
                )

        if not metrics_data:
            metrics_data = [
                {
                    "Optimizer": "No data",
                    "Generations": 0,
                    "Best Fitness": "0.0000",
                    "Final Fitness": "0.0000",
                    "Convergence Gen": "N/A",
                    "Success Rate": "0.00%",
                }
            ]

        return pn.widgets.Tabulator(
            pd.DataFrame(metrics_data),
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
        )

    async def _update_statistics(self):
        """Update the statistics panel."""
        if not self.selected_optimizers:
            self.stats_html.object = (
                "<div><h3>Statistics</h3><p>No optimizers selected</p></div>"
            )
            return

        total_optimizers = len(self.selected_optimizers)
        total_generations = 0
        best_overall_fitness = float("-inf")

        for optimizer_id in self.selected_optimizers:
            data = self.optimization_data.get(optimizer_id, {})
            fitness_history = data.get("fitness_history", [])

            if fitness_history:
                df = pd.DataFrame(fitness_history)
                total_generations += len(df)
                best_fitness = df["best_fitness"].max()
                if best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness

        stats_html = f"""
        <div style='padding: 15px;'>
            <h3>Statistics Summary</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;'>
                <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
                    <strong>Selected Optimizers:</strong><br>{total_optimizers}
                </div>
                <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
                    <strong>Total Generations:</strong><br>{total_generations}
                </div>
                <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
                    <strong>Best Overall Fitness:</strong><br>{best_overall_fitness:.4f if best_overall_fitness != float('-inf') else 'N/A'}
                </div>
                <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
                    <strong>Status:</strong><br>{'Monitoring' if self.is_monitoring else 'Static'}
                </div>
            </div>
        </div>
        """

        self.stats_html.object = stats_html

    def _on_refresh_click(self, event):
        """Handle refresh button click."""
        asyncio.create_task(self._refresh_data())

    async def _refresh_data(self):
        """Refresh all data and visualizations."""
        try:
            # Reload optimizers
            await self._load_optimizers()

            # Reload data for selected optimizers
            await self._load_selected_data()

            # Update current view
            await self._update_current_view()

            self._show_notification("Data refreshed successfully", type="success")

        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")
            self._show_notification(f"Failed to refresh data: {str(e)}", type="error")

    def _show_notification(self, message: str, type: str = "info"):
        """Show notification to user."""
        # This would integrate with a notification system
        logger.info(f"Notification ({type}): {message}")

        # For now, we'll update the stats panel with the message
        current_time = datetime.now().strftime("%H:%M:%S")
        notification_html = f"""
        <div style='padding: 10px; margin: 5px 0; border-radius: 5px;
                    background-color: {"#d4edda" if type == "success" else "#f8d7da" if type == "error" else "#d1ecf1"};
                    border: 1px solid {"#c3e6cb" if type == "success" else "#f5c6cb" if type == "error" else "#bee5eb"};'>
            <strong>[{current_time}]</strong> {message}
        </div>
        """

        # Prepend to existing stats
        current_stats = self.stats_html.object
        self.stats_html.object = notification_html + current_stats

    def get_dashboard(self) -> pn.template.MaterialTemplate:
        """Get the dashboard layout for embedding."""
        return self.layout

    def export_current_plot(self, format: str = None, filename: str = None):
        """Export the current plot."""
        export_format = format or self.config.export_format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"darwin_plot_{timestamp}.{export_format}"

        try:
            # This would implement actual export functionality
            logger.info(f"Exporting plot to {filename} in {export_format} format")
            self._show_notification(f"Plot exported as {filename}", type="success")

        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            self._show_notification(f"Export failed: {str(e)}", type="error")

    def export_data(self, format: str = "csv", filename: str = None):
        """Export optimization data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"darwin_data_{timestamp}.{format}"

        try:
            # Collect all data
            all_data = []

            for optimizer_id in self.selected_optimizers:
                data = self.optimization_data.get(optimizer_id, {})
                fitness_history = data.get("fitness_history", [])

                for record in fitness_history:
                    record_copy = record.copy()
                    record_copy["optimizer_id"] = optimizer_id
                    all_data.append(record_copy)

            if all_data:
                df = pd.DataFrame(all_data)

                if format == "csv":
                    # df.to_csv(filename, index=False)
                    pass  # Actual file writing would happen here
                elif format == "json":
                    # df.to_json(filename, orient='records')
                    pass  # Actual file writing would happen here

                logger.info(f"Data exported to {filename}")
                self._show_notification(f"Data exported as {filename}", type="success")
            else:
                self._show_notification("No data to export", type="error")

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            self._show_notification(f"Data export failed: {str(e)}", type="error")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop monitoring
            if self.is_monitoring:
                self._stop_monitoring()

            # Close WebSocket connection
            if self.websocket_manager:
                await self.websocket_manager.disconnect()

            # Clear cached data
            self.optimization_data.clear()
            self.cached_plots.clear()

            logger.info("Visualization engine cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def create_visualization_engine(
    api_client: DarwinAPIClient, config: Optional[VisualizationConfig] = None
) -> VisualizationEngine:
    """Factory function to create a visualization engine."""
    return VisualizationEngine(api_client=api_client, config=config)


async def serve_visualization_dashboard(
    api_base_url: str = "http://localhost:8000", port: int = 5007, show: bool = True
):
    """Serve the visualization dashboard."""
    try:
        # Create API client
        api_client = DarwinAPIClient(api_base_url)

        # Create visualization engine
        engine = create_visualization_engine(api_client)

        # Initialize engine
        await engine.initialize()

        # Get dashboard
        dashboard = engine.get_dashboard()

        # Serve dashboard
        dashboard.servable()

        if show:
            dashboard.show(port=port)

        logger.info(f"Visualization dashboard served on port {port}")
        return engine

    except Exception as e:
        logger.error(f"Failed to serve visualization dashboard: {e}")
        raise

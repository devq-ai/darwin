"""
Visualization Engine Component for Darwin Dashboard

This module provides comprehensive visualization and analytics capabilities for
genetic algorithm optimization results. It includes interactive plots, statistical
analysis tools, solution space exploration, and export functionality.

Features:
- Interactive Bokeh visualizations with custom tools
- Statistical analysis and correlation plots
- Solution space exploration and clustering
- Pareto frontier visualization for multi-objective problems
- Performance comparison and benchmarking
- Export capabilities for plots and data
- Real-time visualization updates
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import figure
from sklearn.decomposition import PCA

from darwin.dashboard.utils.api_client import DarwinAPIClient

logger = logging.getLogger(__name__)


class VisualizationConfig(param.Parameterized):
    """Configuration parameters for visualizations."""

    plot_theme = param.Selector(
        default="light", objects=["light", "dark", "minimal"], doc="Plot theme"
    )
    color_palette = param.Selector(
        default="Category10",
        objects=["Category10", "Viridis", "Spectral", "Plasma"],
        doc="Color palette",
    )
    plot_width = param.Integer(default=700, bounds=(400, 1200), doc="Plot width")
    plot_height = param.Integer(default=500, bounds=(300, 800), doc="Plot height")
    animation_enabled = param.Boolean(default=True, doc="Enable plot animations")
    export_format = param.Selector(
        default="png", objects=["png", "svg", "html", "pdf"], doc="Export format"
    )


class VisualizationEngine(param.Parameterized):
    """
    Advanced visualization engine for genetic algorithm analytics.

    Provides interactive visualizations for optimization results analysis,
    solution space exploration, and performance comparison.
    """

    # State parameters
    selected_optimizer_ids = param.List(
        default=[], doc="Selected optimizer IDs for comparison"
    )
    current_visualization = param.String(
        default="fitness_evolution", doc="Current visualization type"
    )
    config = param.Parameter(default=None, doc="Visualization configuration")

    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)

        self.api_client = api_client
        self.config = VisualizationConfig()

        # Data storage
        self.optimization_data = {}
        self.analysis_results = {}
        self.cached_plots = {}

        # Create UI components
        self._create_components()

    def _create_components(self):
        """Create all visualization components."""

        # Visualization selector and controls
        self.controls_section = self._create_controls_section()

        # Main visualization area
        self.visualization_area = self._create_visualization_area()

        # Analysis tools
        self.analysis_section = self._create_analysis_section()

        # Export and sharing
        self.export_section = self._create_export_section()

    def _create_controls_section(self):
        """Create visualization controls and configuration."""

        # Optimizer selection
        optimizer_selector = pn.Column(
            pn.pane.Markdown("### 🎯 Data Selection"),
            pn.widgets.MultiSelect(
                name="Optimizers", value=[], options=[], size=6, width=300
            ),
            pn.widgets.Button(name="🔄 Refresh Data", button_type="primary", width=150),
            pn.widgets.Button(name="📊 Load Selected", button_type="success", width=150),
        )

        # Visualization type selector
        viz_type_selector = pn.Column(
            pn.pane.Markdown("### 📈 Visualization Type"),
            pn.widgets.Select(
                name="Chart Type",
                value="fitness_evolution",
                options={
                    "Fitness Evolution": "fitness_evolution",
                    "Population Diversity": "population_diversity",
                    "Solution Space": "solution_space",
                    "Pareto Frontier": "pareto_frontier",
                    "Performance Comparison": "performance_comparison",
                    "Correlation Analysis": "correlation_analysis",
                    "Convergence Analysis": "convergence_analysis",
                    "Parameter Sensitivity": "parameter_sensitivity",
                },
                width=200,
            ),
            pn.widgets.Button(name="🎨 Generate Plot", button_type="primary", width=150),
        )

        # Configuration panel
        config_panel = pn.Column(
            pn.pane.Markdown("### ⚙️ Plot Configuration"),
            pn.Param(
                self.config,
                parameters=["plot_theme", "color_palette", "plot_width", "plot_height"],
                widgets={
                    "plot_theme": pn.widgets.Select,
                    "color_palette": pn.widgets.Select,
                    "plot_width": pn.widgets.IntSlider,
                    "plot_height": pn.widgets.IntSlider,
                },
            ),
        )

        # Setup event handlers
        self._setup_controls_handlers(
            optimizer_selector, viz_type_selector, config_panel
        )

        return pn.Row(
            optimizer_selector,
            viz_type_selector,
            config_panel,
            sizing_mode="stretch_width",
        )

    def _create_visualization_area(self):
        """Create the main visualization display area."""

        # Main plot container
        main_plot = pn.pane.HTML(
            """
            <div style='text-align: center; padding: 50px; background-color: #f8f9fa;
                        border-radius: 8px; border: 2px dashed #dee2e6; min-height: 400px;
                        display: flex; align-items: center; justify-content: center;'>
                <div>
                    <h3 style='color: #6c757d; margin-bottom: 15px;'>📊 Visualization Area</h3>
                    <p style='color: #868e96; margin: 0; font-size: 16px;'>
                        Select optimizers and choose a visualization type to begin analysis
                    </p>
                </div>
            </div>
            """,
            name="main_plot",
            sizing_mode="stretch_width",
            height=500,
        )

        # Plot tabs for multiple views
        plot_tabs = pn.Tabs(
            ("📈 Main View", main_plot),
            ("🔍 Detail View", self._create_detail_view()),
            ("📊 Statistics", self._create_statistics_view()),
            ("🎯 Interactive", self._create_interactive_view()),
            dynamic=True,
        )

        return pn.Column(
            pn.pane.Markdown("## 📊 Visualization Dashboard"),
            plot_tabs,
            sizing_mode="stretch_width",
        )

    def _create_detail_view(self):
        """Create detailed view for in-depth analysis."""

        detail_content = pn.Column(
            pn.pane.HTML(
                """
                <div style='padding: 20px; background-color: #e9ecef; border-radius: 8px;'>
                    <h4 style='margin-top: 0; color: #495057;'>🔍 Detailed Analysis</h4>
                    <p style='color: #6c757d; margin-bottom: 0;'>
                        This view will show detailed breakdowns of selected optimizations
                        including generation-by-generation analysis and parameter evolution.
                    </p>
                </div>
                """,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            height=400,
        )

        return detail_content

    def _create_statistics_view(self):
        """Create statistics summary view."""

        # Statistics tables
        summary_stats = pd.DataFrame(
            {
                "Metric": [
                    "Best Fitness",
                    "Final Generation",
                    "Convergence Rate",
                    "Success Rate",
                ],
                "Value": ["--", "--", "--", "--"],
                "Rank": ["--", "--", "--", "--"],
            }
        )

        stats_table = pn.widgets.Tabulator(
            summary_stats,
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            height=200,
            title="Summary Statistics",
        )

        # Statistical plots
        stats_plots = pn.pane.HTML(
            """
            <div style='padding: 20px; background-color: #fff3cd; border-radius: 8px; border: 1px solid #ffeaa7;'>
                <h4 style='margin-top: 0; color: #856404;'>📊 Statistical Analysis</h4>
                <p style='color: #856404; margin-bottom: 0;'>
                    Distribution plots, box plots, and statistical tests will be displayed here
                    to compare optimization performance across different runs.
                </p>
            </div>
            """,
            sizing_mode="stretch_width",
            height=200,
        )

        return pn.Column(stats_table, stats_plots, sizing_mode="stretch_width")

    def _create_interactive_view(self):
        """Create interactive exploration view."""

        interactive_content = pn.Column(
            pn.pane.HTML(
                """
                <div style='padding: 20px; background-color: #d1ecf1; border-radius: 8px; border: 1px solid #b8daff;'>
                    <h4 style='margin-top: 0; color: #0c5460;'>🎯 Interactive Exploration</h4>
                    <p style='color: #0c5460; margin-bottom: 10px;'>
                        Interactive tools for exploring solution spaces and parameter relationships:
                    </p>
                    <ul style='color: #0c5460; margin-bottom: 0;'>
                        <li>3D solution space visualization</li>
                        <li>Parameter correlation heatmaps</li>
                        <li>Interactive Pareto frontier exploration</li>
                        <li>Solution clustering and analysis</li>
                    </ul>
                </div>
                """,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            height=400,
        )

        return interactive_content

    def _create_analysis_section(self):
        """Create analysis tools section."""

        # Analysis type selector
        analysis_selector = pn.Column(
            pn.pane.Markdown("### 🔬 Analysis Tools"),
            pn.widgets.Select(
                name="Analysis Type",
                value="basic_stats",
                options={
                    "Basic Statistics": "basic_stats",
                    "Convergence Analysis": "convergence_analysis",
                    "Parameter Sensitivity": "parameter_sensitivity",
                    "Solution Clustering": "solution_clustering",
                    "Performance Comparison": "performance_comparison",
                    "Pareto Analysis": "pareto_analysis",
                },
                width=200,
            ),
            pn.widgets.Button(name="🔍 Run Analysis", button_type="primary", width=150),
        )

        # Analysis parameters
        analysis_params = pn.Column(
            pn.pane.Markdown("### ⚙️ Analysis Parameters"),
            pn.widgets.IntSlider(
                name="Cluster Count", start=2, end=10, step=1, value=5, width=200
            ),
            pn.widgets.FloatSlider(
                name="Confidence Level",
                start=0.90,
                end=0.99,
                step=0.01,
                value=0.95,
                width=200,
            ),
            pn.widgets.Checkbox(name="Include Outliers", value=True),
        )

        # Analysis results
        analysis_results = pn.pane.HTML(
            """
            <div style='padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;'>
                <h4 style='margin-top: 0; color: #495057;'>📋 Analysis Results</h4>
                <p style='color: #6c757d; margin-bottom: 0;'>
                    Analysis results will appear here after running the selected analysis.
                </p>
            </div>
            """,
            name="analysis_results",
            sizing_mode="stretch_width",
            height=200,
        )

        return pn.Row(
            analysis_selector,
            analysis_params,
            analysis_results,
            sizing_mode="stretch_width",
        )

    def _create_export_section(self):
        """Create export and sharing section."""

        # Export options
        export_options = pn.Column(
            pn.pane.Markdown("### 💾 Export Options"),
            pn.widgets.Select(
                name="Format",
                value="png",
                options=["png", "svg", "html", "pdf", "json"],
                width=150,
            ),
            pn.widgets.Select(
                name="Resolution",
                value="high",
                options=["low", "medium", "high", "print"],
                width=150,
            ),
            pn.widgets.Button(name="📤 Export Plot", button_type="success", width=150),
        )

        # Sharing options
        sharing_options = pn.Column(
            pn.pane.Markdown("### 🔗 Sharing"),
            pn.widgets.Button(name="📋 Copy Link", button_type="light", width=150),
            pn.widgets.Button(name="📧 Email Report", button_type="light", width=150),
            pn.widgets.Button(name="📱 Generate QR", button_type="light", width=150),
        )

        # Report generation
        report_options = pn.Column(
            pn.pane.Markdown("### 📊 Report Generation"),
            pn.widgets.Checkbox(name="Include Statistics", value=True),
            pn.widgets.Checkbox(name="Include Plots", value=True),
            pn.widgets.Button(
                name="📄 Generate Report", button_type="primary", width=150
            ),
        )

        return pn.Row(
            export_options, sharing_options, report_options, sizing_mode="stretch_width"
        )

    def _setup_controls_handlers(
        self, optimizer_selector, viz_type_selector, config_panel
    ):
        """Setup event handlers for control components."""

        # Get specific buttons and widgets
        refresh_btn = optimizer_selector[2]
        load_btn = optimizer_selector[3]
        generate_btn = viz_type_selector[2]
        viz_select = viz_type_selector[1]

        # Setup handlers
        refresh_btn.on_click(self._refresh_optimizers)
        load_btn.on_click(self._load_selected_data)
        generate_btn.on_click(self._generate_visualization)
        viz_select.param.watch(self._on_visualization_type_change, "value")

    def create_interface(self):
        """Create the complete visualization engine interface."""

        return pn.Column(
            pn.pane.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">📈 Visualization & Analytics Engine</h2>
                    <p style="margin: 10px 0; opacity: 0.9;">Interactive exploration and analysis of optimization results</p>
                </div>
                """
            ),
            self.controls_section,
            self.visualization_area,
            pn.Tabs(
                ("🔬 Analysis", self.analysis_section),
                ("💾 Export", self.export_section),
                dynamic=True,
            ),
            sizing_mode="stretch_width",
        )

    # Visualization generation methods
    def _generate_fitness_evolution_plot(self, data: Dict[str, Any]) -> pn.pane.Bokeh:
        """Generate fitness evolution plot."""

        p = figure(
            title="Fitness Evolution Comparison",
            x_axis_label="Generation",
            y_axis_label="Fitness Value",
            width=self.config.plot_width,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        colors = Category10[10]

        for i, (optimizer_id, opt_data) in enumerate(data.items()):
            if "fitness_history" in opt_data:
                history = opt_data["fitness_history"]
                generations = list(range(len(history["best"])))

                # Best fitness line
                p.line(
                    generations,
                    history["best"],
                    line_width=2,
                    color=colors[i % len(colors)],
                    legend_label=f"{optimizer_id} (Best)",
                    alpha=0.8,
                )

                # Average fitness line
                p.line(
                    generations,
                    history["average"],
                    line_width=1,
                    color=colors[i % len(colors)],
                    legend_label=f"{optimizer_id} (Avg)",
                    alpha=0.6,
                    line_dash="dashed",
                )

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Generation", "$x"),
                ("Fitness", "$y{0.000}"),
                ("Optimizer", "@optimizer_id"),
            ]
        )
        p.add_tools(hover)

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p)

    def _generate_solution_space_plot(self, data: Dict[str, Any]) -> pn.pane.Bokeh:
        """Generate solution space visualization."""

        # Combine all solutions
        all_solutions = []
        optimizer_labels = []

        for optimizer_id, opt_data in data.items():
            if "final_population" in opt_data:
                solutions = opt_data["final_population"]
                all_solutions.extend(solutions)
                optimizer_labels.extend([optimizer_id] * len(solutions))

        if not all_solutions:
            return pn.pane.HTML("<p>No solution data available</p>")

        # Convert to numpy array
        solutions_array = np.array(all_solutions)

        # Reduce dimensionality if needed
        if solutions_array.shape[1] > 2:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            solutions_2d = pca.fit_transform(solutions_array)
        else:
            solutions_2d = solutions_array

        # Create scatter plot
        p = figure(
            title="Solution Space Exploration",
            x_axis_label="Component 1",
            y_axis_label="Component 2",
            width=self.config.plot_width,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # Color by optimizer
        unique_optimizers = list(set(optimizer_labels))
        colors = Category10[max(3, len(unique_optimizers))]

        for i, optimizer_id in enumerate(unique_optimizers):
            mask = np.array(optimizer_labels) == optimizer_id
            opt_solutions = solutions_2d[mask]

            p.scatter(
                opt_solutions[:, 0],
                opt_solutions[:, 1],
                size=8,
                color=colors[i % len(colors)],
                alpha=0.6,
                legend_label=optimizer_id,
            )

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return pn.pane.Bokeh(p)

    def _generate_pareto_frontier_plot(self, data: Dict[str, Any]) -> pn.pane.Bokeh:
        """Generate Pareto frontier visualization for multi-objective problems."""

        p = figure(
            title="Pareto Frontier Analysis",
            x_axis_label="Objective 1",
            y_axis_label="Objective 2",
            width=self.config.plot_width,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        colors = Category10[10]

        for i, (optimizer_id, opt_data) in enumerate(data.items()):
            if "pareto_front" in opt_data:
                front = opt_data["pareto_front"]

                # Plot Pareto front
                p.line(
                    front[:, 0],
                    front[:, 1],
                    line_width=2,
                    color=colors[i % len(colors)],
                    legend_label=f"{optimizer_id} Front",
                )

                # Plot points
                p.scatter(
                    front[:, 0],
                    front[:, 1],
                    size=8,
                    color=colors[i % len(colors)],
                    alpha=0.7,
                )

        p.legend.location = "top_right"

        return pn.pane.Bokeh(p)

    def _generate_performance_comparison_plot(
        self, data: Dict[str, Any]
    ) -> pn.pane.Bokeh:
        """Generate performance comparison plot."""

        # Extract performance metrics
        optimizer_ids = list(data.keys())
        metrics = ["best_fitness", "convergence_rate", "runtime", "evaluations"]

        # Create DataFrame for easier plotting
        performance_data = []
        for optimizer_id in optimizer_ids:
            opt_data = data[optimizer_id]
            for metric in metrics:
                if metric in opt_data:
                    performance_data.append(
                        {
                            "optimizer": optimizer_id,
                            "metric": metric,
                            "value": opt_data[metric],
                        }
                    )

        if not performance_data:
            return pn.pane.HTML("<p>No performance data available</p>")

        df = pd.DataFrame(performance_data)

        # Create grouped bar chart
        p = figure(
            title="Performance Metrics Comparison",
            x_range=optimizer_ids,
            width=self.config.plot_width,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # For simplicity, show only best fitness comparison
        best_fitness_data = df[df["metric"] == "best_fitness"]

        if not best_fitness_data.empty:
            p.vbar(
                x=best_fitness_data["optimizer"],
                top=best_fitness_data["value"],
                width=0.8,
                color="steelblue",
                alpha=0.7,
            )

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        return pn.pane.Bokeh(p)

    def _generate_correlation_analysis_plot(
        self, data: Dict[str, Any]
    ) -> pn.pane.Bokeh:
        """Generate correlation analysis heatmap."""

        # Placeholder for correlation analysis
        p = figure(
            title="Parameter Correlation Analysis",
            width=self.config.plot_width,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        # Add placeholder text
        p.text(
            [0.5],
            [0.5],
            text=["Correlation analysis will be implemented here"],
            text_align="center",
            text_baseline="middle",
        )

        return pn.pane.Bokeh(p)

    # Event handlers
    def _refresh_optimizers(self, event):
        """Refresh the list of available optimizers."""
        try:
            # TODO: Implement API call to get optimizer list
            logger.info("Refreshing optimizer list...")
        except Exception as e:
            logger.error(f"Error refreshing optimizers: {e}")

    def _load_selected_data(self, event):
        """Load data for selected optimizers."""
        try:
            # TODO: Implement data loading from API
            logger.info("Loading selected optimizer data...")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _generate_visualization(self, event):
        """Generate the selected visualization."""
        try:
            viz_type = self.current_visualization

            # Mock data for demonstration
            mock_data = self._generate_mock_data()

            # Generate appropriate plot
            if viz_type == "fitness_evolution":
                plot = self._generate_fitness_evolution_plot(mock_data)
            elif viz_type == "solution_space":
                plot = self._generate_solution_space_plot(mock_data)
            elif viz_type == "pareto_frontier":
                plot = self._generate_pareto_frontier_plot(mock_data)
            elif viz_type == "performance_comparison":
                plot = self._generate_performance_comparison_plot(mock_data)
            elif viz_type == "correlation_analysis":
                plot = self._generate_correlation_analysis_plot(mock_data)
            else:
                plot = pn.pane.HTML(
                    f"<p>Visualization type '{viz_type}' not implemented yet</p>"
                )

            # Update the main plot area
            self._update_main_plot(plot)

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")

    def _on_visualization_type_change(self, event):
        """Handle visualization type change."""
        self.current_visualization = event.new
        logger.info(f"Visualization type changed to: {event.new}")

    def _update_main_plot(self, plot):
        """Update the main plot area with a new visualization."""
        try:
            # TODO: Properly update the plot in the interface
            logger.info("Updated main plot area")
        except Exception as e:
            logger.error(f"Error updating main plot: {e}")

    def _generate_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for demonstration purposes."""

        mock_data = {}

        for i in range(3):
            optimizer_id = f"optimizer_{i+1}"

            # Generate mock fitness history
            generations = 100
            best_fitness = np.cummin(np.random.exponential(1, generations))
            avg_fitness = best_fitness + np.random.exponential(0.5, generations)

            # Generate mock population
            population_size = 50
            num_variables = 5
            final_population = np.random.random((population_size, num_variables))

            # Generate mock Pareto front (for multi-objective)
            pareto_size = 20
            pareto_front = np.random.random((pareto_size, 2))

            mock_data[optimizer_id] = {
                "fitness_history": {
                    "best": best_fitness.tolist(),
                    "average": avg_fitness.tolist(),
                },
                "final_population": final_population.tolist(),
                "pareto_front": pareto_front,
                "best_fitness": float(best_fitness[-1]),
                "convergence_rate": np.random.uniform(0.1, 0.9),
                "runtime": np.random.uniform(10, 300),
                "evaluations": generations * population_size,
            }

        return mock_data

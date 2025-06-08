"""
Experiment Manager Component for Darwin Dashboard

This module provides experiment management functionality for genetic algorithm
optimization runs. It allows users to track, compare, analyze, and manage
multiple optimization experiments with comprehensive history and analytics.

Features:
- Experiment tracking and history
- Run comparison and analysis
- Experiment grouping and tagging
- Performance benchmarking
- Export and reporting capabilities
- Experiment notes and documentation
- Batch experiment management
"""

import asyncio
import logging

import pandas as pd
import panel as pn
import param

from darwin.dashboard.utils.api_client import DarwinAPIClient

logger = logging.getLogger(__name__)


class ExperimentFilter(param.Parameterized):
    """Experiment filtering and search parameters."""

    search_query = param.String(default="", doc="Search query")
    status_filter = param.Selector(
        default="all",
        objects=["all", "running", "completed", "failed", "paused"],
        doc="Status filter",
    )
    date_range = param.Selector(
        default="all_time",
        objects=["today", "week", "month", "year", "all_time"],
        doc="Date range filter",
    )
    problem_type = param.Selector(
        default="all",
        objects=["all", "single_objective", "multi_objective", "constrained"],
        doc="Problem type filter",
    )


class ExperimentManager(param.Parameterized):
    """
    Experiment management component for optimization experiments.

    Provides comprehensive experiment tracking, comparison, and analysis
    capabilities for genetic algorithm optimization runs.
    """

    # State parameters
    selected_experiments = param.List(default=[], doc="Selected experiment IDs")
    current_experiment_id = param.String(default="", doc="Current experiment ID")
    filter_config = param.Parameter(default=None, doc="Filter configuration")
    view_mode = param.Selector(
        default="table", objects=["table", "cards", "timeline"], doc="View mode"
    )

    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)

        self.api_client = api_client
        self.filter_config = ExperimentFilter()

        # Data storage
        self.experiments_data = []
        self.experiment_details = {}

        # UI state
        self.experiments_df = pd.DataFrame()

        # Create UI components
        self._create_components()

        # Load initial experiments
        self._load_experiments()

    def _create_components(self):
        """Create all experiment manager components."""

        # Header and controls
        self.header_section = self._create_header_section()

        # Filter and search section
        self.filter_section = self._create_filter_section()

        # Experiments view section
        self.experiments_section = self._create_experiments_section()

        # Experiment details section
        self.details_section = self._create_details_section()

        # Comparison and analysis section
        self.analysis_section = self._create_analysis_section()

    def _create_header_section(self):
        """Create header with statistics and quick actions."""

        # Statistics cards
        stats_cards = pn.GridBox(
            # Total Experiments
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.1em;'>Total Experiments</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='total-experiments'>0</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>All time</p>
                </div>
                """,
                width=180,
                height=100,
            ),
            # Running Experiments
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.1em;'>Running</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='running-experiments'>0</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Active now</p>
                </div>
                """,
                width=180,
                height=100,
            ),
            # Completed Today
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.1em;'>Completed Today</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='completed-today'>0</h2>
                    <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Last 24h</p>
                </div>
                """,
                width=180,
                height=100,
            ),
            # Success Rate
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                            color: #333; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; font-size: 1.1em;'>Success Rate</h4>
                    <h2 style='margin: 5px 0; font-size: 2em;' id='success-rate'>--</h2>
                    <p style='margin: 0; opacity: 0.7; font-size: 0.9em;'>This month</p>
                </div>
                """,
                width=180,
                height=100,
            ),
            ncols=4,
            sizing_mode="stretch_width",
        )

        # Quick actions
        quick_actions = pn.Row(
            pn.widgets.Button(
                name="🚀 New Experiment", button_type="success", width=150
            ),
            pn.widgets.Button(
                name="📊 Compare Selected", button_type="primary", width=150
            ),
            pn.widgets.Button(name="📈 Analytics", button_type="light", width=150),
            pn.widgets.Button(name="📤 Export", button_type="light", width=150),
            pn.widgets.Button(name="🗑️ Batch Delete", button_type="light", width=150),
        )

        # Setup event handlers
        self._setup_header_handlers(quick_actions)

        return pn.Column(
            stats_cards,
            pn.layout.Spacer(height=15),
            quick_actions,
            sizing_mode="stretch_width",
        )

    def _create_filter_section(self):
        """Create filtering and search interface."""

        # Search bar
        search_bar = pn.widgets.TextInput(
            name="🔍 Search Experiments",
            placeholder="Search by name, problem type, or tags...",
            width=300,
            value=self.filter_config.search_query,
        )

        # Filter controls
        filter_controls = pn.Row(
            pn.widgets.Select(
                name="Status",
                value=self.filter_config.status_filter,
                options=["all", "running", "completed", "failed", "paused"],
                width=120,
            ),
            pn.widgets.Select(
                name="Date Range",
                value=self.filter_config.date_range,
                options=["today", "week", "month", "year", "all_time"],
                width=120,
            ),
            pn.widgets.Select(
                name="Problem Type",
                value=self.filter_config.problem_type,
                options=["all", "single_objective", "multi_objective", "constrained"],
                width=150,
            ),
            pn.widgets.Select(
                name="View Mode",
                value=self.view_mode,
                options=["table", "cards", "timeline"],
                width=120,
            ),
        )

        # Filter actions
        filter_actions = pn.Row(
            pn.widgets.Button(name="🔍 Apply Filters", button_type="primary", width=120),
            pn.widgets.Button(name="🔄 Reset", button_type="light", width=100),
            pn.widgets.Button(name="💾 Save Filter", button_type="light", width=120),
        )

        # Setup event handlers
        self._setup_filter_handlers(search_bar, filter_controls, filter_actions)

        return pn.Column(
            pn.Row(search_bar, filter_actions),
            filter_controls,
            sizing_mode="stretch_width",
        )

    def _create_experiments_section(self):
        """Create experiments display section."""

        # Experiments table
        self.experiments_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=[
                    "Name",
                    "Status",
                    "Problem Type",
                    "Started",
                    "Duration",
                    "Best Fitness",
                    "Generation",
                    "Actions",
                ]
            ),
            pagination="remote",
            page_size=15,
            sizing_mode="stretch_width",
            height=400,
            selectable="checkbox",
            name="experiments_table",
        )

        # Experiments cards view (alternative to table)
        self.experiments_cards = pn.Column(
            pn.pane.HTML(
                """
                <div style='text-align: center; padding: 40px; background-color: #f8f9fa;
                            border-radius: 8px; border: 2px dashed #dee2e6;'>
                    <h4 style='color: #6c757d; margin-bottom: 10px;'>📊 Experiments Cards View</h4>
                    <p style='color: #868e96; margin: 0;'>
                        Card-based experiment display will be implemented here
                    </p>
                </div>
                """,
                height=400,
                sizing_mode="stretch_width",
            ),
            visible=False,
            name="experiments_cards",
        )

        # Timeline view (alternative to table)
        self.experiments_timeline = pn.Column(
            pn.pane.HTML(
                """
                <div style='text-align: center; padding: 40px; background-color: #f8f9fa;
                            border-radius: 8px; border: 2px dashed #dee2e6;'>
                    <h4 style='color: #6c757d; margin-bottom: 10px;'>📅 Experiments Timeline</h4>
                    <p style='color: #868e96; margin: 0;'>
                        Timeline view of experiments will be implemented here
                    </p>
                </div>
                """,
                height=400,
                sizing_mode="stretch_width",
            ),
            visible=False,
            name="experiments_timeline",
        )

        # View container
        experiments_view = pn.Column(
            self.experiments_table,
            self.experiments_cards,
            self.experiments_timeline,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## 🧪 Experiments"),
            experiments_view,
            sizing_mode="stretch_width",
        )

    def _create_details_section(self):
        """Create experiment details section."""

        # Experiment information
        experiment_info = pn.Column(
            pn.pane.Markdown("### 📋 Experiment Details"),
            pn.pane.HTML(
                """
                <div style='padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;'>
                    <div id='experiment-details'>
                        <p style='color: #6c757d; text-align: center; margin: 0;'>
                            Select an experiment to view details
                        </p>
                    </div>
                </div>
                """,
                name="experiment_info",
                sizing_mode="stretch_width",
                height=200,
            ),
        )

        # Experiment parameters
        experiment_params = pn.Column(
            pn.pane.Markdown("### ⚙️ Parameters"),
            pn.widgets.Tabulator(
                pd.DataFrame(columns=["Parameter", "Value", "Description"]),
                pagination="remote",
                page_size=10,
                sizing_mode="stretch_width",
                height=200,
                name="params_table",
            ),
        )

        # Experiment notes
        experiment_notes = pn.Column(
            pn.pane.Markdown("### 📝 Notes"),
            pn.widgets.TextAreaInput(
                name="Experiment Notes",
                placeholder="Add notes about this experiment...",
                height=150,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.widgets.Button(
                    name="💾 Save Notes", button_type="primary", width=120
                ),
                pn.widgets.Button(name="📋 Copy Info", button_type="light", width=120),
            ),
        )

        return pn.Row(
            experiment_info,
            pn.Column(experiment_params, experiment_notes),
            sizing_mode="stretch_width",
        )

    def _create_analysis_section(self):
        """Create comparison and analysis section."""

        # Comparison controls
        comparison_controls = pn.Column(
            pn.pane.Markdown("### 📊 Comparison Analysis"),
            pn.widgets.MultiSelect(
                name="Select Experiments to Compare",
                value=[],
                options=[],
                size=6,
                width=300,
            ),
            pn.Row(
                pn.widgets.Button(name="📈 Compare", button_type="primary", width=120),
                pn.widgets.Button(name="📊 Benchmark", button_type="light", width=120),
            ),
        )

        # Analysis results
        analysis_results = pn.Column(
            pn.pane.Markdown("### 📋 Analysis Results"),
            pn.pane.HTML(
                """
                <div style='padding: 20px; background-color: #e9ecef; border-radius: 8px; min-height: 300px;'>
                    <div style='text-align: center; color: #6c757d; padding-top: 50px;'>
                        <h4>Comparative Analysis</h4>
                        <p>Select experiments and click 'Compare' to see detailed analysis</p>
                    </div>
                </div>
                """,
                name="analysis_results",
                sizing_mode="stretch_width",
            ),
        )

        # Performance metrics
        performance_metrics = pn.Column(
            pn.pane.Markdown("### ⚡ Performance Metrics"),
            pn.widgets.Tabulator(
                pd.DataFrame(columns=["Metric", "Best", "Average", "Worst", "Std Dev"]),
                pagination="remote",
                page_size=8,
                sizing_mode="stretch_width",
                height=250,
                name="metrics_table",
            ),
        )

        return pn.Row(
            comparison_controls,
            pn.Column(analysis_results, performance_metrics),
            sizing_mode="stretch_width",
        )

    def _setup_header_handlers(self, quick_actions):
        """Setup event handlers for header actions."""

        new_exp_btn = quick_actions[0]
        compare_btn = quick_actions[1]
        analytics_btn = quick_actions[2]
        export_btn = quick_actions[3]
        delete_btn = quick_actions[4]

        new_exp_btn.on_click(self._create_new_experiment)
        compare_btn.on_click(self._compare_selected_experiments)
        analytics_btn.on_click(self._show_analytics)
        export_btn.on_click(self._export_experiments)
        delete_btn.on_click(self._batch_delete_experiments)

    def _setup_filter_handlers(self, search_bar, filter_controls, filter_actions):
        """Setup event handlers for filter components."""

        # Search and filter change handlers
        search_bar.param.watch(self._on_search_change, "value")

        for control in filter_controls:
            if hasattr(control, "param"):
                control.param.watch(self._on_filter_change, "value")

        # Filter action handlers
        apply_btn = filter_actions[0]
        reset_btn = filter_actions[1]
        save_btn = filter_actions[2]

        apply_btn.on_click(self._apply_filters)
        reset_btn.on_click(self._reset_filters)
        save_btn.on_click(self._save_filter_preset)

    def create_interface(self):
        """Create the complete experiment manager interface."""

        return pn.Column(
            pn.pane.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">🧪 Experiment Manager</h2>
                    <p style="margin: 10px 0; opacity: 0.9;">Track, compare, and analyze optimization experiments</p>
                </div>
                """
            ),
            self.header_section,
            self.filter_section,
            self.experiments_section,
            pn.Tabs(
                ("📋 Details", self.details_section),
                ("📊 Analysis", self.analysis_section),
                dynamic=True,
            ),
            sizing_mode="stretch_width",
        )

    def _load_experiments(self):
        """Load experiments from API."""
        try:
            # TODO: Load experiments from API
            # For now, use mock data
            self._load_mock_experiments()

        except Exception as e:
            logger.error(f"Error loading experiments: {e}")

    def _load_mock_experiments(self):
        """Load mock experiments for demonstration."""

        mock_experiments = [
            {
                "id": "exp_001",
                "name": "Sphere Function Test",
                "status": "completed",
                "problem_type": "single_objective",
                "started": "2024-01-15 10:30:00",
                "duration": "00:05:23",
                "best_fitness": 0.0001234,
                "generation": 87,
                "population_size": 50,
                "max_generations": 100,
                "success": True,
            },
            {
                "id": "exp_002",
                "name": "Portfolio Optimization Run",
                "status": "running",
                "problem_type": "multi_objective",
                "started": "2024-01-15 14:15:00",
                "duration": "01:23:45",
                "best_fitness": 0.234567,
                "generation": 156,
                "population_size": 100,
                "max_generations": 200,
                "success": None,
            },
            {
                "id": "exp_003",
                "name": "Neural Architecture Search",
                "status": "failed",
                "problem_type": "constrained",
                "started": "2024-01-14 09:00:00",
                "duration": "00:15:30",
                "best_fitness": None,
                "generation": 25,
                "population_size": 30,
                "max_generations": 500,
                "success": False,
            },
            {
                "id": "exp_004",
                "name": "Engineering Design Opt",
                "status": "paused",
                "problem_type": "constrained",
                "started": "2024-01-14 16:45:00",
                "duration": "02:30:12",
                "best_fitness": 1.456789,
                "generation": 234,
                "population_size": 75,
                "max_generations": 300,
                "success": None,
            },
        ]

        self.experiments_data = mock_experiments
        self._update_experiments_table()
        self._update_statistics()

    def _update_experiments_table(self):
        """Update the experiments table with current data."""
        try:
            # Convert to DataFrame
            df_data = []
            for exp in self.experiments_data:
                status_icon = {
                    "running": "🟢",
                    "completed": "✅",
                    "failed": "❌",
                    "paused": "⏸️",
                }.get(exp["status"], "❓")

                df_data.append(
                    {
                        "Name": exp["name"],
                        "Status": f"{status_icon} {exp['status'].title()}",
                        "Problem Type": exp["problem_type"].replace("_", " ").title(),
                        "Started": exp["started"],
                        "Duration": exp["duration"],
                        "Best Fitness": f"{exp['best_fitness']:.6f}"
                        if exp["best_fitness"] is not None
                        else "--",
                        "Generation": f"{exp['generation']}/{exp['max_generations']}",
                        "Actions": "👁️ 📊 🗑️",
                    }
                )

            self.experiments_df = pd.DataFrame(df_data)
            self.experiments_table.value = self.experiments_df

        except Exception as e:
            logger.error(f"Error updating experiments table: {e}")

    def _update_statistics(self):
        """Update header statistics."""
        try:
            total = len(self.experiments_data)
            running = len(
                [e for e in self.experiments_data if e["status"] == "running"]
            )
            completed_today = len(
                [e for e in self.experiments_data if e["status"] == "completed"]
            )
            success_rate = 85.7  # Mock value

            # TODO: Update actual HTML elements with these values
            logger.info(
                f"Stats - Total: {total}, Running: {running}, Completed: {completed_today}, Success: {success_rate}%"
            )

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    # Event handlers
    def _on_search_change(self, event):
        """Handle search query change."""
        self.filter_config.search_query = event.new
        self._apply_filters_auto()

    def _on_filter_change(self, event):
        """Handle filter change."""
        self._apply_filters_auto()

    def _apply_filters_auto(self):
        """Apply filters automatically (with debouncing)."""
        try:
            # TODO: Implement actual filtering logic
            logger.info("Auto-applying filters...")

        except Exception as e:
            logger.error(f"Error auto-applying filters: {e}")

    def _create_new_experiment(self, event):
        """Create a new experiment."""
        try:
            # TODO: Navigate to problem editor or show creation dialog
            self._show_notification(
                "Navigate to Problem Editor to create new experiment", "info"
            )

        except Exception as e:
            logger.error(f"Error creating new experiment: {e}")

    def _compare_selected_experiments(self, event):
        """Compare selected experiments."""
        try:
            if len(self.selected_experiments) < 2:
                self._show_notification(
                    "Select at least 2 experiments to compare", "warning"
                )
                return

            # TODO: Implement experiment comparison
            self._show_notification("Experiment comparison not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error comparing experiments: {e}")

    def _show_analytics(self, event):
        """Show analytics dashboard."""
        try:
            # TODO: Navigate to analytics view
            self._show_notification("Analytics dashboard not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error showing analytics: {e}")

    def _export_experiments(self, event):
        """Export experiment data."""
        try:
            # TODO: Implement export functionality
            self._show_notification("Export functionality not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error exporting experiments: {e}")

    def _batch_delete_experiments(self, event):
        """Delete selected experiments."""
        try:
            if not self.selected_experiments:
                self._show_notification(
                    "No experiments selected for deletion", "warning"
                )
                return

            # TODO: Implement batch deletion with confirmation
            self._show_notification("Batch deletion not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error deleting experiments: {e}")

    def _apply_filters(self, event):
        """Apply current filter settings."""
        try:
            # TODO: Apply filters to experiment list
            self._show_notification("Filters applied successfully", "success")

        except Exception as e:
            logger.error(f"Error applying filters: {e}")

    def _reset_filters(self, event):
        """Reset all filters to default values."""
        try:
            self.filter_config.search_query = ""
            self.filter_config.status_filter = "all"
            self.filter_config.date_range = "all_time"
            self.filter_config.problem_type = "all"

            self._load_experiments()
            self._show_notification("Filters reset successfully", "success")

        except Exception as e:
            logger.error(f"Error resetting filters: {e}")

    def _save_filter_preset(self, event):
        """Save current filter settings as a preset."""
        try:
            # TODO: Implement filter preset saving
            self._show_notification("Filter preset saving not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error saving filter preset: {e}")

    async def refresh_experiments(self):
        """Refresh experiments from API (called by WebSocket updates)."""
        try:
            await asyncio.sleep(0)  # Yield control
            self._load_experiments()

        except Exception as e:
            logger.error(f"Error refreshing experiments: {e}")

    def _show_notification(self, message: str, type: str = "info"):
        """Show a notification message."""
        colors = {
            "success": "#4CAF50",
            "error": "#f44336",
            "warning": "#ff9800",
            "info": "#2196F3",
        }

        color = colors.get(type, colors["info"])

        # TODO: Implement proper notification system
        logger.info(f"Notification ({type}): {message}")

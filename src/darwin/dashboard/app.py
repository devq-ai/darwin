"""
Darwin Panel Dashboard Application

This module implements the main Panel dashboard application for the Darwin genetic
algorithm optimization platform. It provides an interactive web interface for
creating, managing, and monitoring genetic algorithm optimization runs.

Features:
- Interactive problem definition interface
- Real-time optimization monitoring
- Visualization of evolution progress
- Template management system
- Experiment history and analysis
- FastAPI backend integration
- WebSocket support for real-time updates
"""

import asyncio
import logging

import panel as pn
import param

from darwin.dashboard.components.experiments import ExperimentManager
from darwin.dashboard.components.monitoring import MonitoringDashboard
from darwin.dashboard.components.problem_editor import ProblemEditor
from darwin.dashboard.components.templates import TemplateManager
from darwin.dashboard.components.visualizations import VisualizationEngine
from darwin.dashboard.utils.api_client import DarwinAPIClient
from darwin.dashboard.utils.websocket_manager import WebSocketManager

# Configure Panel
pn.extension(
    "bokeh", "tabulator", "ace", template="material", sizing_mode="stretch_width"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DarwinDashboard(param.Parameterized):
    """Main Darwin Dashboard application class."""

    # Dashboard state parameters
    current_optimizer_id = param.String(
        default="", doc="Currently selected optimizer ID"
    )
    is_connected = param.Boolean(default=False, doc="API connection status")

    def __init__(self, api_base_url: str = "http://localhost:8000", **params):
        super().__init__(**params)

        # Initialize API client
        self.api_client = DarwinAPIClient(api_base_url)

        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager(
            "ws://localhost:8000/ws/optimization/progress"
        )

        # Initialize dashboard components
        self.problem_editor = ProblemEditor(api_client=self.api_client)
        self.monitoring_dashboard = MonitoringDashboard(api_client=self.api_client)
        self.visualization_engine = VisualizationEngine(api_client=self.api_client)
        self.template_manager = TemplateManager(api_client=self.api_client)
        self.experiment_manager = ExperimentManager(api_client=self.api_client)

        # Create the main template
        self.template = pn.template.MaterialTemplate(
            title="🧬 Darwin Genetic Algorithm Optimizer",
            sidebar_width=300,
            header_background="#1f77b4",
            sidebar_footer="Darwin v1.0.0 | DevQ.ai",
        )

        # Setup the dashboard
        self._setup_dashboard()

        # Start background tasks
        self._start_background_tasks()

    def _setup_dashboard(self):
        """Setup the main dashboard layout and components."""
        # Create sidebar navigation
        self._create_sidebar()

        # Create main content area
        self._create_main_content()

        # Setup event handlers
        self._setup_event_handlers()

    def _create_sidebar(self):
        """Create the sidebar with navigation and controls."""
        # Connection status indicator
        connection_status = pn.pane.HTML(
            """
            <div style='padding: 10px; border-radius: 5px; margin-bottom: 15px;
                        background-color: #f44336; color: white; text-align: center;'>
                <strong>🔴 Disconnected</strong><br>
                <small>Unable to connect to API</small>
            </div>
            """,
            name="connection_status",
        )

        # Navigation menu
        nav_buttons = pn.Column(
            pn.pane.Markdown("## 📋 Navigation"),
            pn.widgets.Button(
                name="🏠 Dashboard", button_type="primary", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="📝 Problem Editor", button_type="default", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="📊 Monitoring", button_type="default", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="📈 Visualizations", button_type="default", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="📚 Templates", button_type="default", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="🧪 Experiments", button_type="default", width=250, margin=(5, 5)
            ),
            sizing_mode="stretch_width",
        )

        # Quick actions
        quick_actions = pn.Column(
            pn.pane.Markdown("## ⚡ Quick Actions"),
            pn.widgets.Button(
                name="➕ New Optimization",
                button_type="success",
                width=250,
                margin=(5, 5),
            ),
            pn.widgets.Button(
                name="⏹️ Stop All", button_type="light", width=250, margin=(5, 5)
            ),
            pn.widgets.Button(
                name="🔄 Refresh", button_type="light", width=250, margin=(5, 5)
            ),
            sizing_mode="stretch_width",
        )

        # System status
        system_status = pn.Column(
            pn.pane.Markdown("## 🖥️ System Status"),
            pn.pane.HTML(
                """
                <div style='font-size: 12px; color: #666;'>
                    <div>CPU Usage: <span style='color: #2196F3;'>--</span></div>
                    <div>Memory: <span style='color: #4CAF50;'>--</span></div>
                    <div>Active Runs: <span style='color: #FF9800;'>0</span></div>
                    <div>Queue: <span style='color: #9C27B0;'>0</span></div>
                </div>
                """,
                name="system_metrics",
            ),
            sizing_mode="stretch_width",
        )

        # Add to sidebar
        self.template.sidebar.extend(
            [connection_status, nav_buttons, quick_actions, system_status]
        )

    def _create_main_content(self):
        """Create the main content area with tabs."""
        # Create main tabs
        self.main_tabs = pn.Tabs(
            ("🏠 Dashboard", self._create_dashboard_tab()),
            ("📝 Problem Editor", self.problem_editor.create_interface()),
            ("📊 Monitoring", self.monitoring_dashboard.create_interface()),
            ("📈 Visualizations", self.visualization_engine.create_interface()),
            ("📚 Templates", self.template_manager.create_interface()),
            ("🧪 Experiments", self.experiment_manager.create_interface()),
            dynamic=True,
            tabs_location="above",
            sizing_mode="stretch_width",
        )

        # Add to main template
        self.template.main.append(self.main_tabs)

    def _create_dashboard_tab(self):
        """Create the main dashboard overview tab."""
        # Welcome header
        welcome_header = pn.pane.HTML(
            """
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; color: white; margin-bottom: 20px;'>
                <h1 style='margin: 0; font-size: 2.5em;'>🧬 Darwin</h1>
                <h2 style='margin: 5px 0; font-weight: 300;'>Genetic Algorithm Optimization Platform</h2>
                <p style='margin: 10px 0; opacity: 0.9;'>Evolve solutions to complex optimization problems</p>
            </div>
            """
        )

        # Quick stats cards
        stats_cards = pn.GridBox(
            # Total Optimizations Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='margin: 0; font-size: 2em;'>0</h3>
                    <p style='margin: 5px 0; opacity: 0.9;'>Total Optimizations</p>
                </div>
                """,
                width=200,
                height=120,
            ),
            # Active Runs Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            color: white; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='margin: 0; font-size: 2em;'>0</h3>
                    <p style='margin: 5px 0; opacity: 0.9;'>Active Runs</p>
                </div>
                """,
                width=200,
                height=120,
            ),
            # Best Fitness Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                            color: #333; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='margin: 0; font-size: 2em;'>--</h3>
                    <p style='margin: 5px 0; opacity: 0.8;'>Best Fitness</p>
                </div>
                """,
                width=200,
                height=120,
            ),
            # Success Rate Card
            pn.pane.HTML(
                """
                <div style='background: linear-gradient(135deg, #96fbc4 0%, #f9f586 100%);
                            color: #333; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='margin: 0; font-size: 2em;'>--</h3>
                    <p style='margin: 5px 0; opacity: 0.8;'>Success Rate</p>
                </div>
                """,
                width=200,
                height=120,
            ),
            ncols=4,
            sizing_mode="stretch_width",
        )

        # Recent activity section
        recent_activity = pn.Column(
            pn.pane.Markdown("## 📋 Recent Activity"),
            pn.pane.HTML(
                """
                <div style='padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;'>
                    <p style='text-align: center; color: #666; margin: 0;'>
                        No recent activity. Start your first optimization to see results here.
                    </p>
                </div>
                """,
                name="recent_activity_content",
            ),
            sizing_mode="stretch_width",
        )

        # Getting started section
        getting_started = pn.Column(
            pn.pane.Markdown("## 🚀 Getting Started"),
            pn.pane.HTML(
                """
                <div style='padding: 20px; border: 1px solid #2196F3; border-radius: 5px; background: #e3f2fd;'>
                    <h4 style='margin-top: 0; color: #1976D2;'>Quick Start Guide</h4>
                    <ol style='color: #333; line-height: 1.6;'>
                        <li><strong>Define Problem:</strong> Go to "Problem Editor" to create your optimization problem</li>
                        <li><strong>Configure Algorithm:</strong> Set population size, generations, and other parameters</li>
                        <li><strong>Run Optimization:</strong> Start the genetic algorithm and monitor progress</li>
                        <li><strong>Analyze Results:</strong> Use visualizations to understand the evolution</li>
                        <li><strong>Save Templates:</strong> Store successful configurations for future use</li>
                    </ol>
                    <p style='margin-bottom: 0; font-style: italic; color: #666;'>
                        💡 Tip: Start with one of the pre-built templates to get familiar with the system.
                    </p>
                </div>
                """
            ),
            sizing_mode="stretch_width",
        )

        # Combine all dashboard elements
        dashboard_content = pn.Column(
            welcome_header,
            stats_cards,
            pn.layout.Spacer(height=20),
            pn.Row(recent_activity, getting_started, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

        return dashboard_content

    def _setup_event_handlers(self):
        """Setup event handlers for dashboard interactions."""
        # Navigation button handlers
        for widget in self.template.sidebar:
            if hasattr(widget, "objects"):
                for obj in widget.objects:
                    if isinstance(obj, pn.widgets.Button):
                        if "Dashboard" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(0))
                        elif "Problem Editor" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(1))
                        elif "Monitoring" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(2))
                        elif "Visualizations" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(3))
                        elif "Templates" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(4))
                        elif "Experiments" in obj.name:
                            obj.on_click(lambda event: self._switch_tab(5))
                        elif "New Optimization" in obj.name:
                            obj.on_click(self._create_new_optimization)
                        elif "Stop All" in obj.name:
                            obj.on_click(self._stop_all_optimizations)
                        elif "Refresh" in obj.name:
                            obj.on_click(self._refresh_dashboard)

    def _switch_tab(self, tab_index: int):
        """Switch to a specific tab."""
        try:
            self.main_tabs.active = tab_index
        except Exception as e:
            logger.error(f"Error switching tab: {e}")

    def _create_new_optimization(self, event):
        """Create a new optimization run."""
        try:
            # Switch to problem editor tab
            self.main_tabs.active = 1
            # Reset problem editor to create new problem
            self.problem_editor.reset_form()
        except Exception as e:
            logger.error(f"Error creating new optimization: {e}")

    def _stop_all_optimizations(self, event):
        """Stop all running optimizations."""
        try:
            # TODO: Implement stop all functionality
            logger.info("Stopping all optimizations...")
        except Exception as e:
            logger.error(f"Error stopping optimizations: {e}")

    def _refresh_dashboard(self, event):
        """Refresh dashboard data."""
        try:
            logger.info("Refreshing dashboard...")
            # Update all components
            self._update_dashboard_stats()
            self._update_system_status()
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")

    def _start_background_tasks(self):
        """Start background tasks for real-time updates."""
        try:
            # Start periodic updates
            pn.state.add_periodic_callback(
                self._update_dashboard_stats, 5000
            )  # Every 5 seconds
            pn.state.add_periodic_callback(
                self._update_system_status, 10000
            )  # Every 10 seconds

            # Start WebSocket connection for real-time updates
            asyncio.create_task(self._websocket_listener())

        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")

    async def _update_dashboard_stats(self):
        """Update dashboard statistics."""
        try:
            # Get stats from API
            stats = await self.api_client.get_optimization_stats()

            if stats:
                # Update stats cards (this would need to be implemented properly)
                logger.info(f"Updated dashboard stats: {stats}")

        except Exception as e:
            logger.error(f"Error updating dashboard stats: {e}")

    async def _update_system_status(self):
        """Update system status information."""
        try:
            # Check API connection
            health = await self.api_client.check_health()

            if health and health.get("status") == "healthy":
                self.is_connected = True
                # Update connection status indicator
                # This would need proper widget reference management
            else:
                self.is_connected = False

        except Exception as e:
            logger.error(f"Error updating system status: {e}")
            self.is_connected = False

    async def _websocket_listener(self):
        """Listen for WebSocket updates."""
        try:
            await self.websocket_manager.connect()

            async for message in self.websocket_manager.listen():
                # Handle real-time updates
                await self._handle_websocket_message(message)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_websocket_message(self, message: dict):
        """Handle incoming WebSocket messages."""
        try:
            msg_type = message.get("type")

            if msg_type == "optimization_progress":
                # Update monitoring dashboard
                await self.monitoring_dashboard.update_progress(message["data"])

            elif msg_type == "optimization_complete":
                # Update experiment manager
                await self.experiment_manager.refresh_experiments()

            elif msg_type == "system_status":
                # Update system status
                await self._update_system_status()

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def serve(self, port: int = 5007, show: bool = True, autoreload: bool = False):
        """Serve the dashboard application."""
        logger.info(f"Starting Darwin Dashboard on port {port}")

        # Configure Panel server
        pn.serve(
            self.template,
            port=port,
            show=show,
            autoreload=autoreload,
            title="Darwin Genetic Algorithm Optimizer",
            favicon="https://darwin.devq.ai/favicon.ico",
            allow_websocket_origin=["localhost:5007"],
            oauth_provider=None,  # TODO: Add authentication if needed
        )


def create_app(api_base_url: str = "http://localhost:8000") -> DarwinDashboard:
    """Create and configure the Darwin dashboard application."""
    return DarwinDashboard(api_base_url=api_base_url)


def main():
    """Main entry point for the dashboard application."""
    import argparse

    parser = argparse.ArgumentParser(description="Darwin Dashboard Application")
    parser.add_argument("--port", type=int, default=5007, help="Port to serve on")
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument("--no-show", action="store_true", help="Don't open browser")
    parser.add_argument("--autoreload", action="store_true", help="Enable autoreload")

    args = parser.parse_args()

    # Create and serve the dashboard
    app = create_app(api_base_url=args.api_url)
    app.serve(port=args.port, show=not args.no_show, autoreload=args.autoreload)


if __name__ == "__main__":
    main()

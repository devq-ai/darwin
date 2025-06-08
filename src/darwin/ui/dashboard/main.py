"""
Darwin Genetic Algorithm Optimizer - Panel Dashboard

This module implements the main Panel dashboard application for the Darwin
genetic algorithm optimization platform. It provides an interactive web interface
for creating, configuring, and monitoring genetic algorithm optimization runs.

Features:
- Problem definition interface
- Real-time optimization monitoring
- Visualization dashboard
- Template management
- Experiment tracking
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import panel as pn
import param
import requests
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Configure Panel
pn.extension("bokeh", "tabulator", template="material")

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with Darwin FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available problem templates."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/templates")
            response.raise_for_status()
            return response.json().get("templates", [])
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return []

    def create_optimizer(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new optimizer."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/optimizers", json=problem_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            return {"error": str(e)}

    def get_optimizer_status(self, optimizer_id: str) -> Dict[str, Any]:
        """Get optimizer status."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/optimizers/{optimizer_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get optimizer status: {e}")
            return {"error": str(e)}

    def run_optimizer(self, optimizer_id: str) -> Dict[str, Any]:
        """Start optimizer run."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/optimizers/{optimizer_id}/run"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to run optimizer: {e}")
            return {"error": str(e)}

    def get_optimizer_history(self, optimizer_id: str) -> List[Dict[str, Any]]:
        """Get optimization history."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/optimizers/{optimizer_id}/history"
            )
            response.raise_for_status()
            return response.json().get("history", [])
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


class ProblemConfig(param.Parameterized):
    """Configuration for optimization problem."""

    name = param.String(default="My Problem", doc="Problem name")
    description = param.String(default="", doc="Problem description")
    objective_type = param.Selector(
        default="minimize",
        objects=["minimize", "maximize", "multi_objective"],
        doc="Optimization objective",
    )
    population_size = param.Integer(
        default=50, bounds=(10, 1000), doc="Population size"
    )
    max_generations = param.Integer(
        default=100, bounds=(1, 10000), doc="Maximum generations"
    )
    crossover_probability = param.Number(
        default=0.8, bounds=(0.0, 1.0), doc="Crossover probability"
    )
    mutation_probability = param.Number(
        default=0.1, bounds=(0.0, 1.0), doc="Mutation probability"
    )


class VariableConfig(param.Parameterized):
    """Configuration for optimization variables."""

    name = param.String(default="x", doc="Variable name")
    type = param.Selector(
        default="continuous",
        objects=["continuous", "discrete", "categorical"],
        doc="Variable type",
    )
    lower_bound = param.Number(default=-10.0, doc="Lower bound")
    upper_bound = param.Number(default=10.0, doc="Upper bound")


class DarwinDashboard:
    """Main Darwin optimization dashboard."""

    def __init__(
        self, api_base_url: str = "http://localhost:8000", enable_callbacks: bool = True
    ):
        self.api_client = APIClient(api_base_url)
        self.problem_config = ProblemConfig()
        self.current_optimizer_id = None
        self.optimization_history = []

        # Initialize UI components
        self._create_components()
        self._setup_layout()

        # Set up periodic updates (only if callbacks enabled)
        if enable_callbacks:
            try:
                pn.state.add_periodic_callback(self._update_dashboard, 2000, start=True)
            except RuntimeError:
                # No event loop available (e.g., in tests)
                pass

    def _create_components(self):
        """Create dashboard UI components."""

        # Header
        self.header = pn.pane.Markdown(
            "# 🧬 Darwin Genetic Algorithm Optimizer",
            styles={"color": "#2E8B57", "text-align": "center"},
        )

        # Status indicator
        self.status_indicator = pn.indicators.LoadingSpinner(
            value=False, size=20, color="primary"
        )

        # Problem configuration panel
        self.problem_panel = pn.Param(
            self.problem_config,
            parameters=[
                "name",
                "description",
                "objective_type",
                "population_size",
                "max_generations",
                "crossover_probability",
                "mutation_probability",
            ],
            widgets={
                "description": pn.widgets.TextAreaInput,
                "objective_type": pn.widgets.Select,
            },
            name="Problem Configuration",
        )

        # Variables configuration
        variables_data = pd.DataFrame(
            [
                {
                    "name": "x",
                    "type": "continuous",
                    "lower_bound": -10,
                    "upper_bound": 10,
                },
                {
                    "name": "y",
                    "type": "continuous",
                    "lower_bound": -10,
                    "upper_bound": 10,
                },
            ]
        )
        self.variables_table = pn.widgets.Tabulator(
            value=variables_data,
            configuration={
                "columns": [
                    {"title": "Name", "field": "name", "editor": "input"},
                    {
                        "title": "Type",
                        "field": "type",
                        "editor": "select",
                        "editorParams": {
                            "values": ["continuous", "discrete", "categorical"]
                        },
                    },
                    {
                        "title": "Lower Bound",
                        "field": "lower_bound",
                        "editor": "number",
                    },
                    {
                        "title": "Upper Bound",
                        "field": "upper_bound",
                        "editor": "number",
                    },
                ]
            },
            name="Variables",
            height=200,
        )

        # Fitness function editor
        self.fitness_function = pn.widgets.CodeEditor(
            value="""def fitness(solution):
    # Example: Sphere function
    x, y = solution[0], solution[1]
    return x**2 + y**2""",
            language="python",
            theme="github",
            height=150,
            name="Fitness Function",
        )

        # Control buttons
        self.create_button = pn.widgets.Button(
            name="Create Optimizer", button_type="primary", width=150
        )
        self.run_button = pn.widgets.Button(
            name="Run Optimization", button_type="success", disabled=True, width=150
        )
        self.stop_button = pn.widgets.Button(
            name="Stop", button_type="light", disabled=True, width=150
        )

        # Results display
        self.results_text = pn.pane.Markdown("## Results\nNo optimization run yet.")

        # Fitness plot
        self.fitness_plot = self._create_fitness_plot()

        # Progress bar
        self.progress_bar = pn.indicators.Progress(
            name="Optimization Progress", value=0, max=100, bar_color="success"
        )

        # Set up callbacks
        self.create_button.on_click(self._create_optimizer)
        self.run_button.on_click(self._run_optimization)
        self.stop_button.on_click(self._stop_optimization)

    def _create_fitness_plot(self):
        """Create fitness evolution plot."""
        p = figure(
            title="Fitness Evolution",
            x_axis_label="Generation",
            y_axis_label="Fitness",
            width=600,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        # Empty plot initially
        source = ColumnDataSource(data=dict(x=[], y=[]))
        p.line(
            "x",
            "y",
            source=source,
            line_width=2,
            color="#2E8B57",
            legend_label="Best Fitness",
        )
        p.legend.location = "top_right"

        return pn.pane.Bokeh(p, name="Fitness Evolution")

    def _setup_layout(self):
        """Setup dashboard layout."""

        # Sidebar with configuration
        sidebar = pn.Column(
            self.header,
            pn.Spacer(height=20),
            self.problem_panel,
            pn.Spacer(height=10),
            self.variables_table,
            pn.Spacer(height=10),
            self.fitness_function,
            pn.Spacer(height=20),
            pn.Row(self.create_button, self.run_button, self.stop_button),
            width=400,
            margin=(10, 10),
        )

        # Main content area
        main_content = pn.Column(
            pn.Row(
                pn.pane.Markdown("### Optimization Status"),
                pn.Spacer(),
                self.status_indicator,
            ),
            self.progress_bar,
            pn.Spacer(height=20),
            self.fitness_plot,
            pn.Spacer(height=20),
            self.results_text,
            margin=(10, 10),
        )

        # Complete layout
        self.layout = pn.Row(
            sidebar, pn.VSpacer(), main_content, sizing_mode="stretch_width"
        )

    def _create_optimizer(self, event):
        """Create a new optimizer."""
        try:
            self.create_button.loading = True

            # Prepare problem data
            variables = []
            var_df = self.variables_table.value
            if isinstance(var_df, pd.DataFrame):
                for _, var_data in var_df.iterrows():
                    variables.append(
                        {
                            "name": var_data["name"],
                            "type": var_data["type"],
                            "bounds": [
                                var_data["lower_bound"],
                                var_data["upper_bound"],
                            ],
                        }
                    )
            else:
                # Fallback for list format
                for var_data in var_df:
                    variables.append(
                        {
                            "name": var_data["name"],
                            "type": var_data["type"],
                            "bounds": [
                                var_data["lower_bound"],
                                var_data["upper_bound"],
                            ],
                        }
                    )

            problem_data = {
                "problem": {
                    "name": self.problem_config.name,
                    "description": self.problem_config.description,
                    "objective_type": self.problem_config.objective_type,
                    "variables": variables,
                    "constraints": [],
                    "fitness_function": self.fitness_function.value,
                },
                "config": {
                    "population_size": self.problem_config.population_size,
                    "max_generations": self.problem_config.max_generations,
                    "crossover_probability": self.problem_config.crossover_probability,
                    "mutation_probability": self.problem_config.mutation_probability,
                },
            }

            # Create optimizer via API
            result = self.api_client.create_optimizer(problem_data)

            if "error" in result:
                self.results_text.object = f"## Error\n{result['error']}"
            else:
                self.current_optimizer_id = result.get("optimizer_id")
                self.results_text.object = (
                    f"## Optimizer Created\nID: {self.current_optimizer_id}"
                )
                self.run_button.disabled = False

        except Exception as e:
            logger.error(f"Error creating optimizer: {e}")
            self.results_text.object = f"## Error\nFailed to create optimizer: {str(e)}"
        finally:
            self.create_button.loading = False

    def _run_optimization(self, event):
        """Start optimization run."""
        if not self.current_optimizer_id:
            return

        try:
            self.run_button.loading = True
            self.run_button.disabled = True
            self.stop_button.disabled = False

            result = self.api_client.run_optimizer(self.current_optimizer_id)

            if "error" in result:
                self.results_text.object = f"## Error\n{result['error']}"
                self.run_button.disabled = False
                self.stop_button.disabled = True
            else:
                self.results_text.object = (
                    "## Optimization Running\nMonitoring progress..."
                )
                self.status_indicator.value = True

        except Exception as e:
            logger.error(f"Error starting optimization: {e}")
            self.results_text.object = (
                f"## Error\nFailed to start optimization: {str(e)}"
            )
        finally:
            self.run_button.loading = False

    def _stop_optimization(self, event):
        """Stop optimization run."""
        # Implementation for stopping optimization
        self.status_indicator.value = False
        self.run_button.disabled = False
        self.stop_button.disabled = True
        self.results_text.object = "## Optimization Stopped"

    def _update_dashboard(self):
        """Periodic update of dashboard data."""
        if not self.current_optimizer_id:
            return

        try:
            # Get optimizer status
            status = self.api_client.get_optimizer_status(self.current_optimizer_id)

            if "error" not in status:
                # Update progress
                if "progress" in status:
                    self.progress_bar.value = int(status["progress"])

                # Update results if completed
                if status.get("status") == "completed":
                    self.status_indicator.value = False
                    self.run_button.disabled = False
                    self.stop_button.disabled = True

                    if "best_fitness" in status:
                        self.results_text.object = f"""## Optimization Complete
**Best Fitness:** {status['best_fitness']:.6f}
**Generations:** {status.get('generations_completed', 'N/A')}
**Status:** {status['status']}"""

                # Update fitness plot
                history = self.api_client.get_optimizer_history(
                    self.current_optimizer_id
                )
                if history and len(history) > len(self.optimization_history):
                    self.optimization_history = history
                    self._update_fitness_plot()

        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")

    def _update_fitness_plot(self):
        """Update fitness evolution plot."""
        if not self.optimization_history:
            return

        try:
            generations = [h["generation"] for h in self.optimization_history]
            fitness_values = [h["best_fitness"] for h in self.optimization_history]

            # Update plot data
            p = figure(
                title="Fitness Evolution",
                x_axis_label="Generation",
                y_axis_label="Fitness",
                width=600,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )

            p.line(
                generations,
                fitness_values,
                line_width=2,
                color="#2E8B57",
                legend_label="Best Fitness",
            )
            p.circle(generations, fitness_values, size=6, color="#2E8B57", alpha=0.7)
            p.legend.location = "top_right"

            self.fitness_plot.object = p

        except Exception as e:
            logger.error(f"Error updating fitness plot: {e}")

    def servable(self):
        """Return the servable layout."""
        return self.layout


def create_dashboard(
    api_base_url: str = "http://localhost:8000", enable_callbacks: bool = True
) -> DarwinDashboard:
    """Create and configure the Darwin dashboard."""
    return DarwinDashboard(api_base_url, enable_callbacks)


def serve_dashboard(port: int = 5006, api_base_url: str = "http://localhost:8000"):
    """Serve the Darwin dashboard."""
    dashboard = create_dashboard(api_base_url)

    # Configure Panel server
    pn.serve(
        dashboard.servable(),
        port=port,
        allow_websocket_origin=["localhost:5006"],
        show=True,
        title="Darwin Genetic Algorithm Optimizer",
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Serve dashboard
    serve_dashboard()

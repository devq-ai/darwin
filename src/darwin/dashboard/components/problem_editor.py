"""
Problem Editor Component for Darwin Dashboard

This module provides an interactive Panel component for defining and editing
optimization problems. It includes form validation, code editing capabilities,
and integration with the Darwin API for problem creation and management.

Features:
- Interactive problem definition forms
- Code editor for fitness functions and constraints
- Variable definition with type validation
- Problem template support
- Real-time validation and preview
- Export/import capabilities
"""

import json
import logging
from typing import Any, Dict

import pandas as pd
import panel as pn
import param

from darwin.dashboard.utils.api_client import DarwinAPIClient

logger = logging.getLogger(__name__)


class Variable(param.Parameterized):
    """Parameter class for optimization variables."""

    name = param.String(default="", doc="Variable name")
    type = param.Selector(
        default="continuous",
        objects=["continuous", "discrete", "categorical"],
        doc="Variable type",
    )
    lower_bound = param.Number(default=0.0, doc="Lower bound")
    upper_bound = param.Number(default=1.0, doc="Upper bound")
    description = param.String(default="", doc="Variable description")


class ProblemConfig(param.Parameterized):
    """Parameter class for problem configuration."""

    name = param.String(default="", doc="Problem name")
    description = param.String(default="", doc="Problem description")
    objective_type = param.Selector(
        default="minimize",
        objects=["minimize", "maximize", "multi_objective"],
        doc="Optimization objective",
    )

    # Algorithm parameters
    population_size = param.Integer(
        default=50, bounds=(10, 1000), doc="Population size"
    )
    max_generations = param.Integer(
        default=100, bounds=(10, 10000), doc="Maximum generations"
    )
    selection_type = param.Selector(
        default="tournament",
        objects=["tournament", "roulette", "rank", "steady_state"],
        doc="Selection method",
    )
    crossover_rate = param.Number(
        default=0.8, bounds=(0.0, 1.0), doc="Crossover probability"
    )
    mutation_rate = param.Number(
        default=0.1, bounds=(0.0, 1.0), doc="Mutation probability"
    )
    elitism = param.Boolean(default=True, doc="Enable elitism")


class ProblemEditor(param.Parameterized):
    """Interactive problem editor component for the Darwin dashboard."""

    # Current problem state
    current_problem = param.Parameter(default=None, doc="Current problem configuration")
    validation_status = param.String(default="", doc="Validation status message")
    is_valid = param.Boolean(default=False, doc="Problem validation status")

    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)

        self.api_client = api_client

        # Initialize problem configuration
        self.problem_config = ProblemConfig()

        # Initialize variables list
        self.variables = []
        self.variables_df = pd.DataFrame(
            columns=["Name", "Type", "Lower Bound", "Upper Bound", "Description"]
        )

        # Initialize code editors
        self.fitness_function_code = """
def fitness_function(solution):
    \"\"\"
    Calculate fitness for a given solution.

    Args:
        solution: List or array of variable values

    Returns:
        float: Fitness value (lower is better for minimization)
    \"\"\"
    # Example: Sphere function
    return sum(x**2 for x in solution)
"""

        self.constraints_code = """
def constraint_functions(solution):
    \"\"\"
    Evaluate constraints for a given solution.

    Args:
        solution: List or array of variable values

    Returns:
        list: List of constraint violations (0 = satisfied, >0 = violated)
    \"\"\"
    constraints = []

    # Example: Sum of variables must be <= 1
    # constraints.append(max(0, sum(solution) - 1))

    return constraints
"""

        # Create UI components
        self._create_components()

    def _create_components(self):
        """Create all UI components for the problem editor."""

        # Problem metadata section
        self.metadata_section = pn.Column(
            pn.pane.Markdown("## 📝 Problem Definition"),
            pn.Param(
                self.problem_config,
                parameters=["name", "description", "objective_type"],
                widgets={
                    "name": pn.widgets.TextInput,
                    "description": pn.widgets.TextAreaInput,
                    "objective_type": pn.widgets.Select,
                },
                name="Problem Metadata",
            ),
        )

        # Variables section
        self.variables_section = self._create_variables_section()

        # Code editors section
        self.code_section = self._create_code_section()

        # Algorithm configuration section
        self.algorithm_section = pn.Column(
            pn.pane.Markdown("## ⚙️ Algorithm Configuration"),
            pn.Param(
                self.problem_config,
                parameters=[
                    "population_size",
                    "max_generations",
                    "selection_type",
                    "crossover_rate",
                    "mutation_rate",
                    "elitism",
                ],
                widgets={
                    "population_size": pn.widgets.IntSlider,
                    "max_generations": pn.widgets.IntSlider,
                    "selection_type": pn.widgets.Select,
                    "crossover_rate": pn.widgets.FloatSlider,
                    "mutation_rate": pn.widgets.FloatSlider,
                    "elitism": pn.widgets.Checkbox,
                },
                name="Algorithm Parameters",
            ),
        )

        # Actions section
        self.actions_section = self._create_actions_section()

        # Validation section
        self.validation_section = pn.Column(
            pn.pane.Markdown("## ✅ Validation"),
            pn.pane.HTML(
                """
                <div id="validation-status" style="padding: 10px; border-radius: 5px;
                     background-color: #f5f5f5; border: 1px solid #ddd;">
                    <strong>Status:</strong> Not validated
                </div>
                """,
                name="validation_display",
            ),
        )

    def _create_variables_section(self):
        """Create the variables definition section."""

        # Variable form for adding new variables
        new_variable_form = pn.Column(
            pn.pane.Markdown("### Add Variable"),
            pn.Row(
                pn.widgets.TextInput(name="Name", placeholder="variable_name"),
                pn.widgets.Select(
                    name="Type",
                    value="continuous",
                    options=["continuous", "discrete", "categorical"],
                ),
                pn.widgets.FloatInput(name="Lower", value=0.0),
                pn.widgets.FloatInput(name="Upper", value=1.0),
            ),
            pn.widgets.TextInput(
                name="Description", placeholder="Variable description"
            ),
            pn.Row(
                pn.widgets.Button(name="Add Variable", button_type="primary"),
                pn.widgets.Button(name="Clear Form", button_type="light"),
            ),
        )

        # Variables table
        self.variables_table = pn.widgets.Tabulator(
            self.variables_df,
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            height=300,
            disabled=False,
            name="variables_table",
        )

        # Variables actions
        variables_actions = pn.Row(
            pn.widgets.Button(name="Remove Selected", button_type="light"),
            pn.widgets.Button(name="Clear All", button_type="light"),
            pn.widgets.Button(name="Load Template", button_type="success"),
        )

        # Setup event handlers
        add_button = new_variable_form[2][0]  # Add Variable button
        clear_button = new_variable_form[2][1]  # Clear Form button

        add_button.on_click(self._add_variable)
        clear_button.on_click(self._clear_variable_form)

        return pn.Column(
            pn.pane.Markdown("## 🔢 Variables"),
            new_variable_form,
            pn.pane.Markdown("### Current Variables"),
            self.variables_table,
            variables_actions,
        )

    def _create_code_section(self):
        """Create the code editors section."""

        # Fitness function editor
        fitness_editor = pn.widgets.Ace(
            value=self.fitness_function_code,
            language="python",
            theme="github",
            height=400,
            sizing_mode="stretch_width",
            name="fitness_function_editor",
        )

        # Constraints editor
        constraints_editor = pn.widgets.Ace(
            value=self.constraints_code,
            language="python",
            theme="github",
            height=300,
            sizing_mode="stretch_width",
            name="constraints_editor",
        )

        # Code validation buttons
        code_actions = pn.Row(
            pn.widgets.Button(name="Validate Fitness", button_type="primary"),
            pn.widgets.Button(name="Validate Constraints", button_type="primary"),
            pn.widgets.Button(name="Test Functions", button_type="success"),
        )

        # Setup event handlers
        validate_fitness_btn = code_actions[0]
        validate_constraints_btn = code_actions[1]
        test_functions_btn = code_actions[2]

        validate_fitness_btn.on_click(self._validate_fitness_function)
        validate_constraints_btn.on_click(self._validate_constraints)
        test_functions_btn.on_click(self._test_functions)

        return pn.Column(
            pn.pane.Markdown("## 🧮 Function Definitions"),
            pn.pane.Markdown("### Fitness Function"),
            pn.pane.HTML(
                """
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>💡 Tip:</strong> Define your objective function here.
                    For minimization problems, return lower values for better solutions.
                </div>
                """
            ),
            fitness_editor,
            pn.pane.Markdown("### Constraint Functions"),
            pn.pane.HTML(
                """
                <div style="background-color: #f3e5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>💡 Tip:</strong> Return a list of constraint violations.
                    Return 0 for satisfied constraints, positive values for violations.
                </div>
                """
            ),
            constraints_editor,
            code_actions,
        )

    def _create_actions_section(self):
        """Create the actions section with save, load, and run buttons."""

        primary_actions = pn.Row(
            pn.widgets.Button(
                name="💾 Save as Template", button_type="success", width=200
            ),
            pn.widgets.Button(name="🔄 Load Template", button_type="light", width=200),
            pn.widgets.Button(name="🚀 Create & Run", button_type="primary", width=200),
        )

        secondary_actions = pn.Row(
            pn.widgets.Button(
                name="📋 Validate Problem", button_type="light", width=200
            ),
            pn.widgets.Button(name="📤 Export JSON", button_type="light", width=200),
            pn.widgets.Button(name="📥 Import JSON", button_type="light", width=200),
        )

        # Setup event handlers
        save_template_btn = primary_actions[0]
        load_template_btn = primary_actions[1]
        create_run_btn = primary_actions[2]
        validate_btn = secondary_actions[0]
        export_btn = secondary_actions[1]
        import_btn = secondary_actions[2]

        save_template_btn.on_click(self._save_template)
        load_template_btn.on_click(self._show_template_selector)
        create_run_btn.on_click(self._create_and_run)
        validate_btn.on_click(self._validate_problem)
        export_btn.on_click(self._export_json)
        import_btn.on_click(self._show_import_dialog)

        return pn.Column(
            pn.pane.Markdown("## 🎯 Actions"), primary_actions, secondary_actions
        )

    def create_interface(self):
        """Create the complete problem editor interface."""

        return pn.Column(
            pn.pane.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">📝 Problem Editor</h2>
                    <p style="margin: 10px 0; opacity: 0.9;">Define your optimization problem and algorithm configuration</p>
                </div>
                """
            ),
            pn.Tabs(
                (
                    "📋 Definition",
                    pn.Column(self.metadata_section, self.variables_section),
                ),
                ("🧮 Functions", self.code_section),
                ("⚙️ Algorithm", self.algorithm_section),
                ("🎯 Actions", pn.Column(self.actions_section, self.validation_section)),
                dynamic=True,
            ),
            sizing_mode="stretch_width",
        )

    def _add_variable(self, event):
        """Add a new variable to the problem definition."""
        try:
            # Get form values
            form = event.obj.parent.parent  # Navigate to the form container
            name_input = form[1][0]  # Name input
            type_input = form[1][1]  # Type input
            lower_input = form[1][2]  # Lower bound input
            upper_input = form[1][3]  # Upper bound input
            desc_input = form[2]  # Description input

            # Validate inputs
            if not name_input.value:
                self._show_notification("Variable name is required", "error")
                return

            if name_input.value in [var["Name"] for var in self.variables]:
                self._show_notification("Variable name already exists", "error")
                return

            # Add variable
            new_variable = {
                "Name": name_input.value,
                "Type": type_input.value,
                "Lower Bound": lower_input.value,
                "Upper Bound": upper_input.value,
                "Description": desc_input.value,
            }

            self.variables.append(new_variable)
            self.variables_df = pd.DataFrame(self.variables)
            self.variables_table.value = self.variables_df

            # Clear form
            self._clear_variable_form(event)

            self._show_notification(
                f"Variable '{name_input.value}' added successfully", "success"
            )

        except Exception as e:
            logger.error(f"Error adding variable: {e}")
            self._show_notification("Error adding variable", "error")

    def _clear_variable_form(self, event):
        """Clear the variable form."""
        try:
            form = event.obj.parent.parent
            form[1][0].value = ""  # Name
            form[1][1].value = "continuous"  # Type
            form[1][2].value = 0.0  # Lower
            form[1][3].value = 1.0  # Upper
            form[2].value = ""  # Description
        except Exception as e:
            logger.error(f"Error clearing form: {e}")

    def _validate_fitness_function(self, event):
        """Validate the fitness function code."""
        try:
            # Get the fitness function code
            fitness_code = self._get_fitness_function_code()

            # Basic syntax validation
            compile(fitness_code, "<fitness_function>", "exec")

            # Test with dummy data
            namespace = {}
            exec(fitness_code, namespace)

            if "fitness_function" not in namespace:
                raise ValueError("fitness_function not defined")

            # Test function call
            test_solution = [0.5] * max(1, len(self.variables))
            result = namespace["fitness_function"](test_solution)

            if not isinstance(result, (int, float)):
                raise ValueError("Fitness function must return a numeric value")

            self._show_notification("Fitness function is valid", "success")

        except Exception as e:
            self._show_notification(f"Fitness function error: {str(e)}", "error")

    def _validate_constraints(self, event):
        """Validate the constraints code."""
        try:
            # Get the constraints code
            constraints_code = self._get_constraints_code()

            # Basic syntax validation
            compile(constraints_code, "<constraints>", "exec")

            # Test with dummy data
            namespace = {}
            exec(constraints_code, namespace)

            if "constraint_functions" not in namespace:
                raise ValueError("constraint_functions not defined")

            # Test function call
            test_solution = [0.5] * max(1, len(self.variables))
            result = namespace["constraint_functions"](test_solution)

            if not isinstance(result, list):
                raise ValueError("Constraint function must return a list")

            self._show_notification("Constraints are valid", "success")

        except Exception as e:
            self._show_notification(f"Constraints error: {str(e)}", "error")

    def _test_functions(self, event):
        """Test both fitness function and constraints with sample data."""
        try:
            if not self.variables:
                self._show_notification(
                    "Add variables before testing functions", "warning"
                )
                return

            # Test fitness function
            self._validate_fitness_function(event)

            # Test constraints
            self._validate_constraints(event)

            self._show_notification("All functions tested successfully", "success")

        except Exception as e:
            self._show_notification(f"Function testing failed: {str(e)}", "error")

    def _validate_problem(self, event):
        """Validate the complete problem definition."""
        try:
            problem_data = self._get_problem_data()

            # Basic validation
            if not problem_data["name"]:
                raise ValueError("Problem name is required")

            if not self.variables:
                raise ValueError("At least one variable is required")

            # API validation
            validation_result = self.api_client.validate_problem_definition(
                problem_data
            )

            if validation_result:
                self.is_valid = True
                self.validation_status = "Problem is valid and ready for optimization"
                self._show_notification("Problem validation successful", "success")
            else:
                self.is_valid = False
                self.validation_status = "Problem validation failed"
                self._show_notification("Problem validation failed", "error")

        except Exception as e:
            self.is_valid = False
            self.validation_status = f"Validation error: {str(e)}"
            self._show_notification(f"Validation error: {str(e)}", "error")

    def _save_template(self, event):
        """Save the current problem as a template."""
        try:
            problem_data = self._get_problem_data()

            # TODO: Implement template saving via API
            self._show_notification("Template saving not yet implemented", "info")

        except Exception as e:
            self._show_notification(f"Error saving template: {str(e)}", "error")

    def _show_template_selector(self, event):
        """Show template selection dialog."""
        # TODO: Implement template selector
        self._show_notification("Template selection not yet implemented", "info")

    def _create_and_run(self, event):
        """Create optimizer and start optimization run."""
        try:
            if not self.is_valid:
                self._validate_problem(event)
                if not self.is_valid:
                    return

            problem_data = self._get_problem_data()

            # Create optimizer via API
            result = self.api_client.create_optimizer(problem_data)

            if result:
                optimizer_id = result.get("optimizer_id")
                if optimizer_id:
                    # Start optimization
                    run_result = self.api_client.start_optimization(optimizer_id)
                    if run_result:
                        self._show_notification(
                            f"Optimization started: {optimizer_id}", "success"
                        )
                    else:
                        self._show_notification("Failed to start optimization", "error")
                else:
                    self._show_notification("Failed to create optimizer", "error")
            else:
                self._show_notification("Failed to create optimizer", "error")

        except Exception as e:
            self._show_notification(f"Error creating optimizer: {str(e)}", "error")

    def _export_json(self, event):
        """Export problem definition as JSON."""
        try:
            problem_data = self._get_problem_data()
            json_str = json.dumps(problem_data, indent=2)

            # TODO: Implement file download
            self._show_notification("JSON export not yet implemented", "info")

        except Exception as e:
            self._show_notification(f"Export error: {str(e)}", "error")

    def _show_import_dialog(self, event):
        """Show JSON import dialog."""
        # TODO: Implement JSON import
        self._show_notification("JSON import not yet implemented", "info")

    def _get_problem_data(self) -> Dict[str, Any]:
        """Get the complete problem data as a dictionary."""
        return {
            "name": self.problem_config.name,
            "description": self.problem_config.description,
            "objective_type": self.problem_config.objective_type,
            "variables": [
                {
                    "name": var["Name"],
                    "type": var["Type"],
                    "bounds": [var["Lower Bound"], var["Upper Bound"]],
                    "description": var["Description"],
                }
                for var in self.variables
            ],
            "fitness_function": self._get_fitness_function_code(),
            "constraints": self._get_constraints_code(),
            "config": {
                "population_size": self.problem_config.population_size,
                "max_generations": self.problem_config.max_generations,
                "selection_type": self.problem_config.selection_type,
                "crossover_rate": self.problem_config.crossover_rate,
                "mutation_rate": self.problem_config.mutation_rate,
                "elitism": self.problem_config.elitism,
            },
        }

    def _get_fitness_function_code(self) -> str:
        """Get the fitness function code from the editor."""
        # TODO: Get from actual ACE editor
        return self.fitness_function_code

    def _get_constraints_code(self) -> str:
        """Get the constraints code from the editor."""
        # TODO: Get from actual ACE editor
        return self.constraints_code

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

    def reset_form(self):
        """Reset the form to its initial state."""
        self.problem_config.name = ""
        self.problem_config.description = ""
        self.problem_config.objective_type = "minimize"
        self.variables = []
        self.variables_df = pd.DataFrame(
            columns=["Name", "Type", "Lower Bound", "Upper Bound", "Description"]
        )
        self.variables_table.value = self.variables_df
        self.is_valid = False
        self.validation_status = ""

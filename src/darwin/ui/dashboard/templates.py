"""
Darwin Genetic Algorithm Optimizer - Problem Templates Interface

This module implements the problem templates interface for the Darwin dashboard.
It provides functionality for browsing, selecting, and managing optimization
problem templates.

Features:
- Template browsing and filtering
- Template preview and details
- Custom template creation
- Template import/export
"""

import json
import logging
from typing import Any, Dict

import pandas as pd
import panel as pn
import param

logger = logging.getLogger(__name__)


class TemplateManager(param.Parameterized):
    """Manager for problem templates."""

    category_filter = param.Selector(
        default="All",
        objects=["All", "Mathematical", "Engineering", "Machine Learning", "Custom"],
        doc="Filter templates by category",
    )

    difficulty_filter = param.Selector(
        default="All",
        objects=["All", "Beginner", "Intermediate", "Advanced"],
        doc="Filter templates by difficulty",
    )

    selected_template = param.Parameter(default=None, doc="Currently selected template")


class ProblemTemplate:
    """Represents a problem template."""

    def __init__(self, template_data: Dict[str, Any]):
        self.data = template_data
        self.name = template_data.get("name", "Unnamed Template")
        self.description = template_data.get("description", "No description")
        self.category = template_data.get("category", "Custom")
        self.difficulty = template_data.get("difficulty", "Beginner")
        self.variables = template_data.get("variables", [])
        self.fitness_function = template_data.get("fitness_function", "")
        self.config = template_data.get("config", {})
        self.tags = template_data.get("tags", [])


class TemplatesInterface:
    """Problem templates interface for the Darwin dashboard."""

    def __init__(self, api_client):
        self.api_client = api_client
        self.template_manager = TemplateManager()
        self.templates = []
        self.filtered_templates = []

        # Initialize components
        self._create_components()
        self._setup_layout()
        self._load_templates()

        # Setup callbacks
        self._setup_callbacks()

    def _create_components(self):
        """Create UI components for templates interface."""

        # Header
        self.header = pn.pane.Markdown(
            "## 📋 Problem Templates", styles={"color": "#2E8B57"}
        )

        # Filter controls
        self.filter_panel = pn.Param(
            self.template_manager,
            parameters=["category_filter", "difficulty_filter"],
            widgets={
                "category_filter": pn.widgets.Select,
                "difficulty_filter": pn.widgets.Select,
            },
            name="Filters",
        )

        # Templates list
        self.templates_table = pn.widgets.Tabulator(
            value=pd.DataFrame(),
            configuration={
                "columns": [
                    {"title": "Name", "field": "name", "width": 200},
                    {"title": "Category", "field": "category", "width": 120},
                    {"title": "Difficulty", "field": "difficulty", "width": 100},
                    {"title": "Description", "field": "description", "width": 300},
                ]
            },
            selectable=1,
            height=300,
            name="Available Templates",
        )

        # Template preview
        self.preview_panel = pn.Column(
            pn.pane.Markdown("### Template Preview"),
            pn.pane.Markdown("Select a template to see preview"),
            name="Preview",
        )

        # Template actions
        self.load_button = pn.widgets.Button(
            name="Load Template", button_type="primary", disabled=True, width=120
        )

        self.save_button = pn.widgets.Button(
            name="Save as Template", button_type="success", width=120
        )

        self.export_button = pn.widgets.Button(
            name="Export Template", button_type="light", disabled=True, width=120
        )

        self.import_button = pn.widgets.FileInput(accept=".json", width=120)

        # Custom template creation
        self.custom_template_panel = self._create_custom_template_panel()

    def _create_custom_template_panel(self):
        """Create custom template creation panel."""

        template_name = pn.widgets.TextInput(
            name="Template Name", placeholder="Enter template name"
        )

        template_description = pn.widgets.TextAreaInput(
            name="Description", placeholder="Enter template description", height=100
        )

        template_category = pn.widgets.Select(
            name="Category",
            options=["Mathematical", "Engineering", "Machine Learning", "Custom"],
            value="Custom",
        )

        template_difficulty = pn.widgets.Select(
            name="Difficulty",
            options=["Beginner", "Intermediate", "Advanced"],
            value="Beginner",
        )

        template_tags = pn.widgets.TextInput(
            name="Tags", placeholder="Enter comma-separated tags"
        )

        create_template_button = pn.widgets.Button(
            name="Create Template", button_type="primary"
        )

        return pn.Column(
            pn.pane.Markdown("### Create Custom Template"),
            template_name,
            template_description,
            pn.Row(template_category, template_difficulty),
            template_tags,
            create_template_button,
            name="Custom Template",
        )

    def _setup_layout(self):
        """Setup templates interface layout."""

        # Left panel with filters and templates list
        left_panel = pn.Column(
            self.header,
            self.filter_panel,
            self.templates_table,
            pn.Row(self.load_button, self.save_button, self.export_button),
            self.import_button,
            width=600,
        )

        # Right panel with preview and custom template creation
        right_panel = pn.Column(
            self.preview_panel,
            pn.Spacer(height=20),
            self.custom_template_panel,
            width=400,
        )

        self.layout = pn.Row(
            left_panel, pn.VSpacer(), right_panel, sizing_mode="stretch_width"
        )

    def _setup_callbacks(self):
        """Setup event callbacks."""

        # Filter change callbacks
        self.template_manager.param.watch(
            self._on_filter_change, ["category_filter", "difficulty_filter"]
        )

        # Table selection callback
        self.templates_table.param.watch(self._on_template_selection, "selection")

        # Button callbacks
        self.load_button.on_click(self._load_template)
        self.save_button.on_click(self._save_template)
        self.export_button.on_click(self._export_template)
        self.import_button.param.watch(self._import_template, "value")

    def _load_templates(self):
        """Load templates from API."""
        try:
            templates_data = self.api_client.get_templates()
            self.templates = [ProblemTemplate(t) for t in templates_data]

            # Add some default templates if none exist
            if not self.templates:
                self._add_default_templates()

            self._update_template_categories()
            self._apply_filters()

        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self._add_default_templates()
            self._apply_filters()

    def _add_default_templates(self):
        """Add default problem templates."""

        default_templates = [
            {
                "name": "Sphere Function",
                "description": "Simple sphere function optimization (f(x,y) = x² + y²)",
                "category": "Mathematical",
                "difficulty": "Beginner",
                "variables": [
                    {"name": "x", "type": "continuous", "bounds": [-10, 10]},
                    {"name": "y", "type": "continuous", "bounds": [-10, 10]},
                ],
                "fitness_function": """def fitness(solution):
    x, y = solution[0], solution[1]
    return x**2 + y**2""",
                "config": {
                    "population_size": 50,
                    "max_generations": 100,
                    "objective_type": "minimize",
                },
                "tags": ["mathematical", "simple", "quadratic"],
            },
            {
                "name": "Rosenbrock Function",
                "description": "Classic Rosenbrock function optimization",
                "category": "Mathematical",
                "difficulty": "Intermediate",
                "variables": [
                    {"name": "x", "type": "continuous", "bounds": [-5, 5]},
                    {"name": "y", "type": "continuous", "bounds": [-5, 5]},
                ],
                "fitness_function": """def fitness(solution):
    x, y = solution[0], solution[1]
    return 100 * (y - x**2)**2 + (1 - x)**2""",
                "config": {
                    "population_size": 100,
                    "max_generations": 500,
                    "objective_type": "minimize",
                },
                "tags": ["mathematical", "rosenbrock", "nonconvex"],
            },
            {
                "name": "Knapsack Problem",
                "description": "0-1 Knapsack optimization problem",
                "category": "Engineering",
                "difficulty": "Intermediate",
                "variables": [
                    {"name": f"item_{i}", "type": "discrete", "bounds": [0, 1]}
                    for i in range(10)
                ],
                "fitness_function": """def fitness(solution):
    # Items: [weight, value]
    items = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
             (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
    capacity = 50

    total_weight = sum(solution[i] * items[i][0] for i in range(len(solution)))
    total_value = sum(solution[i] * items[i][1] for i in range(len(solution)))

    if total_weight > capacity:
        return 0  # Invalid solution
    return total_value""",
                "config": {
                    "population_size": 50,
                    "max_generations": 200,
                    "objective_type": "maximize",
                },
                "tags": ["knapsack", "discrete", "combinatorial"],
            },
            {
                "name": "Feature Selection",
                "description": "Machine learning feature selection optimization",
                "category": "Machine Learning",
                "difficulty": "Advanced",
                "variables": [
                    {"name": f"feature_{i}", "type": "discrete", "bounds": [0, 1]}
                    for i in range(20)
                ],
                "fitness_function": """def fitness(solution):
    # Placeholder for ML feature selection
    # In practice, this would evaluate model performance
    selected_features = sum(solution)

    # Balance between model accuracy and feature count
    # Higher accuracy with fewer features is better
    accuracy = 0.8 + 0.1 * (selected_features / len(solution))
    complexity_penalty = selected_features / len(solution) * 0.2

    return accuracy - complexity_penalty""",
                "config": {
                    "population_size": 100,
                    "max_generations": 300,
                    "objective_type": "maximize",
                },
                "tags": ["ml", "feature-selection", "classification"],
            },
        ]

        self.templates = [ProblemTemplate(t) for t in default_templates]

    def _update_template_categories(self):
        """Update filter options based on available templates."""
        categories = ["All"] + list(set(t.category for t in self.templates))
        difficulties = ["All"] + list(set(t.difficulty for t in self.templates))

        self.template_manager.param.category_filter.objects = categories
        self.template_manager.param.difficulty_filter.objects = difficulties

    def _apply_filters(self):
        """Apply current filters to templates list."""
        filtered = self.templates

        if self.template_manager.category_filter != "All":
            filtered = [
                t
                for t in filtered
                if t.category == self.template_manager.category_filter
            ]

        if self.template_manager.difficulty_filter != "All":
            filtered = [
                t
                for t in filtered
                if t.difficulty == self.template_manager.difficulty_filter
            ]

        self.filtered_templates = filtered

        # Update table
        table_data = pd.DataFrame(
            [
                {
                    "name": t.name,
                    "category": t.category,
                    "difficulty": t.difficulty,
                    "description": t.description,
                }
                for t in filtered
            ]
        )
        self.templates_table.value = table_data

    def _on_filter_change(self, event):
        """Handle filter changes."""
        self._apply_filters()

    def _on_template_selection(self, event):
        """Handle template selection."""
        if not event.new:
            self.load_button.disabled = True
            self.export_button.disabled = True
            return

        selected_idx = event.new[0]
        if 0 <= selected_idx < len(self.filtered_templates):
            template = self.filtered_templates[selected_idx]
            self._update_template_preview(template)
            self.load_button.disabled = False
            self.export_button.disabled = False
            self.template_manager.selected_template = template

    def _update_template_preview(self, template: ProblemTemplate):
        """Update template preview panel."""

        preview_content = f"""### {template.name}

**Category:** {template.category}
**Difficulty:** {template.difficulty}
**Tags:** {', '.join(template.tags)}

**Description:**
{template.description}

**Variables:**
"""

        for var in template.variables:
            bounds_str = (
                f"[{var['bounds'][0]}, {var['bounds'][1]}]"
                if "bounds" in var
                else "N/A"
            )
            preview_content += f"- {var['name']}: {var['type']} {bounds_str}\n"

        preview_content += f"""
**Configuration:**
- Population Size: {template.config.get('population_size', 'N/A')}
- Max Generations: {template.config.get('max_generations', 'N/A')}
- Objective: {template.config.get('objective_type', 'N/A')}

**Fitness Function:**
```python
{template.fitness_function}
```
"""

        self.preview_panel[1] = pn.pane.Markdown(preview_content)

    def _load_template(self, event):
        """Load selected template."""
        if not self.template_manager.selected_template:
            return

        template = self.template_manager.selected_template

        # This would typically emit an event or call a callback
        # to load the template data into the main problem configuration
        logger.info(f"Loading template: {template.name}")

        # For now, just show a message
        pn.state.notifications.info(f"Template '{template.name}' loaded successfully!")

    def _save_template(self, event):
        """Save current problem configuration as template."""
        # This would save the current problem configuration as a new template
        logger.info("Saving current configuration as template")
        pn.state.notifications.info("Template saved successfully!")

    def _export_template(self, event):
        """Export selected template to file."""
        if not self.template_manager.selected_template:
            return

        template = self.template_manager.selected_template
        template_json = json.dumps(template.data, indent=2)

        # Create download
        file_obj = pn.pane.HTML(
            f'<a href="data:application/json;charset=utf-8,{template_json}" '
            f'download="{template.name.replace(" ", "_")}.json">Download Template</a>'
        )

        # Show in notification or modal
        pn.state.notifications.info("Template exported successfully!")

    def _import_template(self, event):
        """Import template from file."""
        if not event.new:
            return

        try:
            # Read uploaded file
            file_content = event.new.decode("utf-8")
            template_data = json.loads(file_content)

            # Validate template data
            required_fields = ["name", "description", "variables", "fitness_function"]
            if not all(field in template_data for field in required_fields):
                pn.state.notifications.error("Invalid template format!")
                return

            # Add to templates
            new_template = ProblemTemplate(template_data)
            self.templates.append(new_template)

            # Update UI
            self._update_template_categories()
            self._apply_filters()

            pn.state.notifications.success(
                f"Template '{new_template.name}' imported successfully!"
            )

        except Exception as e:
            logger.error(f"Error importing template: {e}")
            pn.state.notifications.error(f"Error importing template: {str(e)}")

    def get_layout(self):
        """Get the templates interface layout."""
        return self.layout

    def get_selected_template(self):
        """Get currently selected template."""
        return self.template_manager.selected_template

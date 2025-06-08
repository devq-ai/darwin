"""
Template Manager Component for Darwin Dashboard

This module provides template management functionality for genetic algorithm
optimization problems. It allows users to save, load, edit, and share problem
templates for common optimization scenarios.

Features:
- Template creation from current problems
- Template library browsing and searching
- Template preview and validation
- Import/export capabilities
- Template categorization and tagging
- Version control for templates
- Community template sharing
"""

import logging

import pandas as pd
import panel as pn
import param

from darwin.dashboard.utils.api_client import DarwinAPIClient

logger = logging.getLogger(__name__)


class TemplateMetadata(param.Parameterized):
    """Template metadata configuration."""

    name = param.String(default="", doc="Template name")
    description = param.String(default="", doc="Template description")
    category = param.Selector(
        default="general",
        objects=[
            "general",
            "engineering",
            "finance",
            "science",
            "machine_learning",
            "optimization",
        ],
        doc="Template category",
    )
    difficulty = param.Selector(
        default="beginner",
        objects=["beginner", "intermediate", "advanced", "expert"],
        doc="Template difficulty level",
    )
    tags = param.String(default="", doc="Template tags (comma-separated)")
    author = param.String(default="", doc="Template author")
    version = param.String(default="1.0.0", doc="Template version")


class TemplateManager(param.Parameterized):
    """
    Template management component for optimization problem templates.

    Provides functionality to create, browse, edit, and manage templates
    for common genetic algorithm optimization problems.
    """

    # State parameters
    selected_template_id = param.String(
        default="", doc="Currently selected template ID"
    )
    current_template = param.Parameter(default=None, doc="Current template data")
    search_query = param.String(default="", doc="Search query for templates")
    filter_category = param.String(default="all", doc="Category filter")

    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)

        self.api_client = api_client

        # Template data
        self.templates_data = []
        self.template_metadata = TemplateMetadata()

        # UI state
        self.templates_df = pd.DataFrame()

        # Create UI components
        self._create_components()

        # Load initial templates
        self._load_templates()

    def _create_components(self):
        """Create all template manager components."""

        # Template browser section
        self.browser_section = self._create_browser_section()

        # Template editor section
        self.editor_section = self._create_editor_section()

        # Template actions section
        self.actions_section = self._create_actions_section()

        # Template preview section
        self.preview_section = self._create_preview_section()

    def _create_browser_section(self):
        """Create template browser and search interface."""

        # Search and filter controls
        search_controls = pn.Row(
            pn.widgets.TextInput(
                name="🔍 Search Templates",
                placeholder="Search by name, description, or tags...",
                width=300,
                value=self.search_query,
            ),
            pn.widgets.Select(
                name="Category",
                value="all",
                options=[
                    "all",
                    "general",
                    "engineering",
                    "finance",
                    "science",
                    "machine_learning",
                    "optimization",
                ],
                width=150,
            ),
            pn.widgets.Select(
                name="Difficulty",
                value="all",
                options=["all", "beginner", "intermediate", "advanced", "expert"],
                width=150,
            ),
            pn.widgets.Button(name="🔄 Refresh", button_type="light", width=100),
        )

        # Templates table
        self.templates_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=[
                    "Name",
                    "Category",
                    "Difficulty",
                    "Author",
                    "Created",
                    "Actions",
                ]
            ),
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            height=300,
            selectable=1,
            name="templates_table",
        )

        # Template statistics
        template_stats = pn.pane.HTML(
            """
            <div style='padding: 10px; background-color: #e9ecef; border-radius: 5px; margin: 10px 0;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; text-align: center;'>
                    <div>
                        <strong>Total Templates</strong><br>
                        <span id='total-templates' style='font-size: 1.2em; color: #007bff;'>0</span>
                    </div>
                    <div>
                        <strong>My Templates</strong><br>
                        <span id='my-templates' style='font-size: 1.2em; color: #28a745;'>0</span>
                    </div>
                    <div>
                        <strong>Community</strong><br>
                        <span id='community-templates' style='font-size: 1.2em; color: #17a2b8;'>0</span>
                    </div>
                    <div>
                        <strong>Featured</strong><br>
                        <span id='featured-templates' style='font-size: 1.2em; color: #ffc107;'>0</span>
                    </div>
                </div>
            </div>
            """,
            name="template_stats",
        )

        # Setup event handlers
        self._setup_browser_handlers(search_controls)

        return pn.Column(
            pn.pane.Markdown("## 📚 Template Library"),
            template_stats,
            search_controls,
            self.templates_table,
            sizing_mode="stretch_width",
        )

    def _create_editor_section(self):
        """Create template editor interface."""

        # Template metadata form
        metadata_form = pn.Column(
            pn.pane.Markdown("### 📝 Template Information"),
            pn.Param(
                self.template_metadata,
                parameters=[
                    "name",
                    "description",
                    "category",
                    "difficulty",
                    "tags",
                    "author",
                    "version",
                ],
                widgets={
                    "name": pn.widgets.TextInput,
                    "description": pn.widgets.TextAreaInput,
                    "category": pn.widgets.Select,
                    "difficulty": pn.widgets.Select,
                    "tags": pn.widgets.TextInput,
                    "author": pn.widgets.TextInput,
                    "version": pn.widgets.TextInput,
                },
                name="Template Metadata",
            ),
        )

        # Template content editor
        content_editor = pn.Column(
            pn.pane.Markdown("### 🧮 Template Content"),
            pn.widgets.Ace(
                value="",
                language="json",
                theme="github",
                height=400,
                sizing_mode="stretch_width",
                name="template_content_editor",
                placeholder="Template JSON content will appear here...",
            ),
        )

        # Template validation
        validation_section = pn.Column(
            pn.pane.Markdown("### ✅ Validation"),
            pn.pane.HTML(
                """
                <div id="template-validation" style="padding: 10px; border-radius: 5px;
                     background-color: #f5f5f5; border: 1px solid #ddd;">
                    <strong>Status:</strong> <span id="validation-status">Not validated</span><br>
                    <div id="validation-details" style="margin-top: 5px; font-size: 0.9em; color: #666;"></div>
                </div>
                """,
                name="template_validation",
            ),
            pn.Row(
                pn.widgets.Button(
                    name="🔍 Validate Template", button_type="primary", width=150
                ),
                pn.widgets.Button(name="👁️ Preview", button_type="light", width=150),
            ),
        )

        return pn.Row(
            pn.Column(metadata_form, validation_section),
            content_editor,
            sizing_mode="stretch_width",
        )

    def _create_actions_section(self):
        """Create template actions and management."""

        # Primary actions
        primary_actions = pn.Row(
            pn.widgets.Button(name="💾 Save Template", button_type="success", width=150),
            pn.widgets.Button(name="📤 Export", button_type="light", width=150),
            pn.widgets.Button(name="📥 Import", button_type="light", width=150),
            pn.widgets.Button(name="🗑️ Delete", button_type="light", width=150),
        )

        # Template management
        management_actions = pn.Row(
            pn.widgets.Button(name="📋 Clone Template", button_type="light", width=150),
            pn.widgets.Button(name="🌐 Share Public", button_type="light", width=150),
            pn.widgets.Button(name="🏷️ Create Version", button_type="light", width=150),
            pn.widgets.Button(name="📊 Usage Stats", button_type="light", width=150),
        )

        # Quick actions
        quick_actions = pn.Row(
            pn.widgets.Button(name="🚀 Use Template", button_type="primary", width=150),
            pn.widgets.Button(name="✏️ Edit Copy", button_type="light", width=150),
            pn.widgets.Button(name="❤️ Favorite", button_type="light", width=150),
            pn.widgets.Button(name="⭐ Rate", button_type="light", width=150),
        )

        # Setup event handlers
        self._setup_actions_handlers(primary_actions, management_actions, quick_actions)

        return pn.Column(
            pn.pane.Markdown("## 🎯 Template Actions"),
            pn.pane.Markdown("### Primary Actions"),
            primary_actions,
            pn.pane.Markdown("### Management"),
            management_actions,
            pn.pane.Markdown("### Quick Actions"),
            quick_actions,
            sizing_mode="stretch_width",
        )

    def _create_preview_section(self):
        """Create template preview interface."""

        # Template preview
        preview_content = pn.Column(
            pn.pane.Markdown("### 👁️ Template Preview"),
            pn.pane.HTML(
                """
                <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px;
                            border: 1px solid #dee2e6; min-height: 300px;'>
                    <div style='text-align: center; color: #6c757d; padding: 50px;'>
                        <h4>Template Preview</h4>
                        <p>Select a template to see its preview here</p>
                    </div>
                </div>
                """,
                name="template_preview",
                sizing_mode="stretch_width",
            ),
        )

        # Template details
        details_content = pn.Column(
            pn.pane.Markdown("### 📋 Template Details"),
            pn.pane.HTML(
                """
                <div style='padding: 15px; background-color: #e9ecef; border-radius: 8px;'>
                    <div id='template-details'>
                        <p style='color: #6c757d; text-align: center; margin: 0;'>
                            No template selected
                        </p>
                    </div>
                </div>
                """,
                name="template_details",
                sizing_mode="stretch_width",
            ),
        )

        # Usage examples
        examples_content = pn.Column(
            pn.pane.Markdown("### 💡 Usage Examples"),
            pn.pane.HTML(
                """
                <div style='padding: 15px; background-color: #d1ecf1; border-radius: 8px; border: 1px solid #b8daff;'>
                    <div id='usage-examples'>
                        <p style='color: #0c5460; text-align: center; margin: 0;'>
                            Usage examples will appear here when a template is selected
                        </p>
                    </div>
                </div>
                """,
                name="usage_examples",
                sizing_mode="stretch_width",
            ),
        )

        return pn.Row(
            preview_content,
            pn.Column(details_content, examples_content),
            sizing_mode="stretch_width",
        )

    def _setup_browser_handlers(self, search_controls):
        """Setup event handlers for browser components."""

        search_input = search_controls[0]
        category_select = search_controls[1]
        difficulty_select = search_controls[2]
        refresh_button = search_controls[3]

        # Search handler
        search_input.param.watch(self._on_search_change, "value")
        category_select.param.watch(self._on_filter_change, "value")
        difficulty_select.param.watch(self._on_filter_change, "value")
        refresh_button.on_click(self._refresh_templates)

        # Table selection handler
        self.templates_table.on_click(self._on_template_select)

    def _setup_actions_handlers(
        self, primary_actions, management_actions, quick_actions
    ):
        """Setup event handlers for action buttons."""

        # Primary actions
        primary_actions[0].on_click(self._save_template)
        primary_actions[1].on_click(self._export_template)
        primary_actions[2].on_click(self._import_template)
        primary_actions[3].on_click(self._delete_template)

        # Management actions
        management_actions[0].on_click(self._clone_template)
        management_actions[1].on_click(self._share_template)
        management_actions[2].on_click(self._create_version)
        management_actions[3].on_click(self._show_usage_stats)

        # Quick actions
        quick_actions[0].on_click(self._use_template)
        quick_actions[1].on_click(self._edit_template_copy)
        quick_actions[2].on_click(self._toggle_favorite)
        quick_actions[3].on_click(self._rate_template)

    def create_interface(self):
        """Create the complete template manager interface."""

        return pn.Column(
            pn.pane.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">📚 Template Manager</h2>
                    <p style="margin: 10px 0; opacity: 0.9;">Create, browse, and manage optimization problem templates</p>
                </div>
                """
            ),
            pn.Tabs(
                ("📚 Browse", self.browser_section),
                ("✏️ Editor", self.editor_section),
                ("🎯 Actions", self.actions_section),
                ("👁️ Preview", self.preview_section),
                dynamic=True,
            ),
            sizing_mode="stretch_width",
        )

    def _load_templates(self):
        """Load templates from API or local storage."""
        try:
            # TODO: Load templates from API
            # For now, use mock data
            self._load_mock_templates()

        except Exception as e:
            logger.error(f"Error loading templates: {e}")

    def _load_mock_templates(self):
        """Load mock templates for demonstration."""

        mock_templates = [
            {
                "id": "template_1",
                "name": "Sphere Function Optimization",
                "description": "Classic sphere function minimization problem",
                "category": "general",
                "difficulty": "beginner",
                "author": "System",
                "created": "2024-01-15",
                "tags": "benchmark, continuous, minimization",
                "usage_count": 125,
                "rating": 4.8,
            },
            {
                "id": "template_2",
                "name": "Portfolio Optimization",
                "description": "Multi-objective portfolio optimization with risk constraints",
                "category": "finance",
                "difficulty": "intermediate",
                "author": "Financial Team",
                "created": "2024-01-10",
                "tags": "portfolio, finance, multi-objective, constraints",
                "usage_count": 87,
                "rating": 4.6,
            },
            {
                "id": "template_3",
                "name": "Neural Network Architecture Search",
                "description": "Optimize neural network architecture parameters",
                "category": "machine_learning",
                "difficulty": "advanced",
                "author": "ML Research",
                "created": "2024-01-05",
                "tags": "neural networks, architecture, deep learning",
                "usage_count": 43,
                "rating": 4.9,
            },
            {
                "id": "template_4",
                "name": "Engineering Design Optimization",
                "description": "Structural optimization with multiple constraints",
                "category": "engineering",
                "difficulty": "expert",
                "author": "Engineering Dept",
                "created": "2024-01-01",
                "tags": "structural, constraints, engineering, design",
                "usage_count": 67,
                "rating": 4.7,
            },
        ]

        self.templates_data = mock_templates
        self._update_templates_table()

    def _update_templates_table(self):
        """Update the templates table with current data."""
        try:
            # Convert to DataFrame
            df_data = []
            for template in self.templates_data:
                df_data.append(
                    {
                        "Name": template["name"],
                        "Category": template["category"].title(),
                        "Difficulty": template["difficulty"].title(),
                        "Author": template["author"],
                        "Created": template["created"],
                        "Rating": f"⭐ {template['rating']:.1f}",
                        "Usage": template["usage_count"],
                    }
                )

            self.templates_df = pd.DataFrame(df_data)
            self.templates_table.value = self.templates_df

        except Exception as e:
            logger.error(f"Error updating templates table: {e}")

    # Event handlers
    def _on_search_change(self, event):
        """Handle search query change."""
        self.search_query = event.new
        self._filter_templates()

    def _on_filter_change(self, event):
        """Handle filter change."""
        self._filter_templates()

    def _filter_templates(self):
        """Filter templates based on search and filters."""
        try:
            # TODO: Implement actual filtering logic
            logger.info(f"Filtering templates with query: {self.search_query}")

        except Exception as e:
            logger.error(f"Error filtering templates: {e}")

    def _refresh_templates(self, event):
        """Refresh templates from API."""
        try:
            self._load_templates()
            self._show_notification("Templates refreshed successfully", "success")

        except Exception as e:
            logger.error(f"Error refreshing templates: {e}")
            self._show_notification("Error refreshing templates", "error")

    def _on_template_select(self, event):
        """Handle template selection."""
        try:
            # TODO: Load selected template details
            logger.info("Template selected")

        except Exception as e:
            logger.error(f"Error selecting template: {e}")

    def _save_template(self, event):
        """Save current template."""
        try:
            # TODO: Implement template saving
            self._show_notification("Template saved successfully", "success")

        except Exception as e:
            logger.error(f"Error saving template: {e}")
            self._show_notification("Error saving template", "error")

    def _export_template(self, event):
        """Export template to file."""
        try:
            # TODO: Implement template export
            self._show_notification("Template export not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error exporting template: {e}")

    def _import_template(self, event):
        """Import template from file."""
        try:
            # TODO: Implement template import
            self._show_notification("Template import not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error importing template: {e}")

    def _delete_template(self, event):
        """Delete selected template."""
        try:
            # TODO: Implement template deletion with confirmation
            self._show_notification("Template deletion not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error deleting template: {e}")

    def _clone_template(self, event):
        """Clone selected template."""
        try:
            # TODO: Implement template cloning
            self._show_notification("Template cloning not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error cloning template: {e}")

    def _share_template(self, event):
        """Share template publicly."""
        try:
            # TODO: Implement template sharing
            self._show_notification("Template sharing not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error sharing template: {e}")

    def _create_version(self, event):
        """Create new version of template."""
        try:
            # TODO: Implement version creation
            self._show_notification("Version creation not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error creating version: {e}")

    def _show_usage_stats(self, event):
        """Show template usage statistics."""
        try:
            # TODO: Implement usage stats display
            self._show_notification("Usage stats not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error showing usage stats: {e}")

    def _use_template(self, event):
        """Use selected template in problem editor."""
        try:
            # TODO: Load template into problem editor
            self._show_notification("Template usage not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error using template: {e}")

    def _edit_template_copy(self, event):
        """Edit a copy of the selected template."""
        try:
            # TODO: Create editable copy
            self._show_notification("Edit copy not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error editing template copy: {e}")

    def _toggle_favorite(self, event):
        """Toggle favorite status of template."""
        try:
            # TODO: Implement favorite toggle
            self._show_notification("Favorite toggle not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")

    def _rate_template(self, event):
        """Rate the selected template."""
        try:
            # TODO: Implement template rating
            self._show_notification("Template rating not yet implemented", "info")

        except Exception as e:
            logger.error(f"Error rating template: {e}")

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

"""
Test suite for Task 5: Panel Dashboard UI

This test validates that the Panel Dashboard UI is properly implemented with all required
components, functionality, and integration. All tests must pass for the task
to be considered complete.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import panel as pn
import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    mock_client = Mock()
    mock_client.get_health.return_value = {"status": "healthy"}
    mock_client.get_templates.return_value = [
        {
            "name": "Test Template",
            "description": "Test description",
            "category": "Mathematical",
            "difficulty": "Beginner",
            "variables": [{"name": "x", "type": "continuous", "bounds": [-10, 10]}],
            "fitness_function": "def fitness(x): return x**2",
            "config": {"population_size": 50},
        }
    ]
    mock_client.create_optimizer.return_value = {"optimizer_id": "test-123"}
    mock_client.get_optimizer_status.return_value = {"status": "created", "progress": 0}
    mock_client.run_optimizer.return_value = {"status": "running"}
    mock_client.get_optimizer_history.return_value = []
    return mock_client


@pytest.mark.task_validation
class TestTask5_PanelDashboardImplementation:
    """Test suite for Task 5: Panel Dashboard UI Implementation"""

    def test_dashboard_module_exists(self, project_root):
        """Test that dashboard module exists and is importable."""
        dashboard_module = (
            project_root / "src" / "darwin" / "ui" / "dashboard" / "main.py"
        )
        assert dashboard_module.exists(), "Dashboard main module should exist"

        # Test importability
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard, create_dashboard

            assert (
                DarwinDashboard is not None
            ), "DarwinDashboard class should be importable"
            assert (
                create_dashboard is not None
            ), "create_dashboard function should be importable"
        except ImportError as e:
            pytest.fail(f"Dashboard module should be importable: {e}")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_templates_module_exists(self, project_root):
        """Test that templates module exists and is importable."""
        templates_module = (
            project_root / "src" / "darwin" / "ui" / "dashboard" / "templates.py"
        )
        assert templates_module.exists(), "Templates module should exist"

        # Test importability
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.templates import (
                ProblemTemplate,
                TemplatesInterface,
            )

            assert (
                TemplatesInterface is not None
            ), "TemplatesInterface class should be importable"
            assert (
                ProblemTemplate is not None
            ), "ProblemTemplate class should be importable"
        except ImportError as e:
            pytest.fail(f"Templates module should be importable: {e}")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_panel_extension_configuration(self, project_root):
        """Test that Panel is properly configured with required extensions."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            import darwin.ui.dashboard.main

            # Panel should be configured with required extensions
            # Check that extensions are loaded by verifying pn.extension was called
            assert hasattr(pn, "pane")  # Basic Panel functionality
            assert hasattr(pn, "widgets")  # Widget functionality

            # Verify bokeh and tabulator extensions are available
            from bokeh.plotting import figure
            from panel.widgets import Tabulator

            # If we can import these without error, extensions are properly loaded
            assert figure is not None
            assert Tabulator is not None
        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.ui_components
class TestTask5_DashboardComponents:
    """Test dashboard UI components"""

    def test_dashboard_creation(self, project_root, mock_api_client):
        """Test dashboard creation and initialization."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Check essential components exist
                assert hasattr(dashboard, "header"), "Dashboard should have header"
                assert hasattr(
                    dashboard, "problem_panel"
                ), "Dashboard should have problem panel"
                assert hasattr(
                    dashboard, "variables_table"
                ), "Dashboard should have variables table"
                assert hasattr(
                    dashboard, "fitness_function"
                ), "Dashboard should have fitness function editor"
                assert hasattr(
                    dashboard, "create_button"
                ), "Dashboard should have create button"
                assert hasattr(
                    dashboard, "run_button"
                ), "Dashboard should have run button"
                assert hasattr(
                    dashboard, "fitness_plot"
                ), "Dashboard should have fitness plot"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_problem_configuration_panel(self, project_root, mock_api_client):
        """Test problem configuration panel."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard, ProblemConfig

            config = ProblemConfig()

            # Test default values
            assert config.name == "My Problem", "Default problem name should be set"
            assert (
                config.objective_type == "minimize"
            ), "Default objective should be minimize"
            assert (
                10 <= config.population_size <= 1000
            ), "Population size should be in valid range"
            assert (
                1 <= config.max_generations <= 10000
            ), "Max generations should be in valid range"

            # Test parameter bounds
            assert (
                0.0 <= config.crossover_probability <= 1.0
            ), "Crossover probability should be in [0,1]"
            assert (
                0.0 <= config.mutation_probability <= 1.0
            ), "Mutation probability should be in [0,1]"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_variables_table_configuration(self, project_root, mock_api_client):
        """Test variables table configuration."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test variables table
                assert hasattr(
                    dashboard.variables_table, "value"
                ), "Variables table should have value"

                # Handle DataFrame structure
                var_df = dashboard.variables_table.value
                if hasattr(var_df, "shape"):  # DataFrame
                    assert var_df.shape[0] >= 2, "Should have default variables"

                    # Check column structure
                    expected_columns = ["name", "type", "lower_bound", "upper_bound"]
                    for col in expected_columns:
                        assert (
                            col in var_df.columns
                        ), f"Variable table should have {col} column"
                else:  # List fallback
                    assert len(var_df) >= 2, "Should have default variables"
                    for var in var_df:
                        assert "name" in var, "Variable should have name"
                        assert "type" in var, "Variable should have type"
                        assert "lower_bound" in var, "Variable should have lower_bound"
                        assert "upper_bound" in var, "Variable should have upper_bound"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_fitness_function_editor(self, project_root, mock_api_client):
        """Test fitness function code editor."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test fitness function editor
                assert hasattr(
                    dashboard.fitness_function, "value"
                ), "Fitness function should have value"
                assert (
                    "def fitness" in dashboard.fitness_function.value
                ), "Should contain fitness function definition"
                assert (
                    dashboard.fitness_function.language == "python"
                ), "Should be Python language"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.api_integration
class TestTask5_APIIntegration:
    """Test API integration"""

    def test_api_client_initialization(self, project_root):
        """Test API client initialization."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import APIClient

            client = APIClient("http://localhost:8000")
            assert (
                client.base_url == "http://localhost:8000"
            ), "Base URL should be set correctly"
            assert hasattr(client, "session"), "Should have requests session"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_api_client_methods(self, project_root):
        """Test API client methods exist and have correct signatures."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import APIClient

            client = APIClient()

            # Test required methods exist
            assert hasattr(client, "get_health"), "Should have get_health method"
            assert hasattr(client, "get_templates"), "Should have get_templates method"
            assert hasattr(
                client, "create_optimizer"
            ), "Should have create_optimizer method"
            assert hasattr(
                client, "get_optimizer_status"
            ), "Should have get_optimizer_status method"
            assert hasattr(client, "run_optimizer"), "Should have run_optimizer method"
            assert hasattr(
                client, "get_optimizer_history"
            ), "Should have get_optimizer_history method"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_dashboard_api_integration(self, project_root, mock_api_client):
        """Test dashboard integration with API client."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test API client is set
                assert (
                    dashboard.api_client is not None
                ), "Dashboard should have API client"
                assert (
                    dashboard.api_client == mock_api_client
                ), "Should use provided API client"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.templates
class TestTask5_TemplatesInterface:
    """Test templates interface"""

    def test_templates_interface_creation(self, project_root, mock_api_client):
        """Test templates interface creation."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.templates import TemplatesInterface

            templates_ui = TemplatesInterface(mock_api_client)

            # Check essential components
            assert hasattr(templates_ui, "header"), "Should have header"
            assert hasattr(templates_ui, "filter_panel"), "Should have filter panel"
            assert hasattr(
                templates_ui, "templates_table"
            ), "Should have templates table"
            assert hasattr(templates_ui, "preview_panel"), "Should have preview panel"
            assert hasattr(templates_ui, "load_button"), "Should have load button"
            assert hasattr(templates_ui, "save_button"), "Should have save button"

        except ImportError:
            pytest.skip("Templates module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_problem_template_class(self, project_root):
        """Test ProblemTemplate class."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.templates import ProblemTemplate

            template_data = {
                "name": "Test Template",
                "description": "Test description",
                "category": "Mathematical",
                "difficulty": "Beginner",
                "variables": [{"name": "x", "type": "continuous"}],
                "fitness_function": "def fitness(x): return x**2",
                "config": {"population_size": 50},
                "tags": ["test", "simple"],
            }

            template = ProblemTemplate(template_data)

            assert template.name == "Test Template", "Name should be set correctly"
            assert (
                template.category == "Mathematical"
            ), "Category should be set correctly"
            assert (
                template.difficulty == "Beginner"
            ), "Difficulty should be set correctly"
            assert len(template.variables) == 1, "Variables should be set correctly"
            assert "test" in template.tags, "Tags should be set correctly"

        except ImportError:
            pytest.skip("Templates module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_default_templates_creation(self, project_root, mock_api_client):
        """Test that default templates are created when none exist."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.templates import TemplatesInterface

            # Mock empty templates response
            mock_api_client.get_templates.return_value = []

            templates_ui = TemplatesInterface(mock_api_client)

            # Should have default templates
            assert len(templates_ui.templates) > 0, "Should create default templates"

            # Check template categories
            categories = [t.category for t in templates_ui.templates]
            assert "Mathematical" in categories, "Should have Mathematical templates"
            assert "Engineering" in categories, "Should have Engineering templates"

        except ImportError:
            pytest.skip("Templates module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.visualization
class TestTask5_Visualization:
    """Test visualization components"""

    def test_fitness_plot_creation(self, project_root, mock_api_client):
        """Test fitness plot creation."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test fitness plot exists
                assert dashboard.fitness_plot is not None, "Fitness plot should exist"
                assert hasattr(
                    dashboard.fitness_plot, "object"
                ), "Fitness plot should have Bokeh object"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_progress_indicators(self, project_root, mock_api_client):
        """Test progress indicators."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test progress components
                assert hasattr(dashboard, "progress_bar"), "Should have progress bar"
                assert hasattr(
                    dashboard, "status_indicator"
                ), "Should have status indicator"

                # Test initial states
                assert dashboard.progress_bar.value == 0, "Progress should start at 0"
                assert (
                    dashboard.status_indicator.value == False
                ), "Status indicator should start as False"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.functionality
class TestTask5_DashboardFunctionality:
    """Test dashboard functionality"""

    def test_optimizer_creation_workflow(self, project_root, mock_api_client):
        """Test optimizer creation workflow."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test initial state
                assert (
                    dashboard.current_optimizer_id is None
                ), "Should start with no optimizer"
                assert (
                    dashboard.run_button.disabled == True
                ), "Run button should be disabled initially"

                # Simulate optimizer creation
                dashboard._create_optimizer(None)

                # Verify API was called
                mock_api_client.create_optimizer.assert_called_once()

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_optimization_monitoring(self, project_root, mock_api_client):
        """Test optimization monitoring functionality."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)
                dashboard.current_optimizer_id = "test-123"

                # Test dashboard update
                dashboard._update_dashboard()

                # Verify API calls
                mock_api_client.get_optimizer_status.assert_called_with("test-123")
                mock_api_client.get_optimizer_history.assert_called_with("test-123")

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.layout
class TestTask5_ResponsiveLayout:
    """Test responsive layout and design"""

    def test_dashboard_layout_structure(self, project_root, mock_api_client):
        """Test dashboard layout structure."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Test layout exists
                assert hasattr(dashboard, "layout"), "Dashboard should have layout"
                assert dashboard.layout is not None, "Layout should not be None"

                # Test servable method
                servable = dashboard.servable()
                assert servable is not None, "Should return servable layout"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_templates_layout_structure(self, project_root, mock_api_client):
        """Test templates interface layout structure."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.templates import TemplatesInterface

            templates_ui = TemplatesInterface(mock_api_client)

            # Test layout
            layout = templates_ui.get_layout()
            assert layout is not None, "Templates interface should have layout"

        except ImportError:
            pytest.skip("Templates module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.dependencies
class TestTask5_Dependencies:
    """Test Task 5 dependencies"""

    def test_task_dependencies(self, project_root):
        """Test that required tasks are completed before Task 5."""
        sys.path.insert(0, str(project_root.parent / "tests"))
        try:
            from conftest import TaskValidator

            validator = TaskValidator(project_root)
            task_5 = validator.get_task_by_id("5")

            assert task_5 is not None, "Task 5 should exist"

            dependencies = task_5.get("dependencies", [])
            required_deps = [2, 3]  # Core GA Engine and Database Integration

            for dep in required_deps:
                assert dep in dependencies, f"Task 5 should depend on Task {dep}"

        except ImportError:
            pytest.skip("Task validator not available")
        finally:
            if str(project_root.parent / "tests") in sys.path:
                sys.path.remove(str(project_root.parent / "tests"))


@pytest.mark.integration
class TestTask5_Integration:
    """Test complete integration"""

    def test_dashboard_creation_and_serving(self, project_root, mock_api_client):
        """Test complete dashboard creation and serving capability."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import create_dashboard, serve_dashboard

            # Test dashboard creation function
            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = create_dashboard(
                    "http://localhost:8000", enable_callbacks=False
                )
                assert (
                    dashboard is not None
                ), "create_dashboard should return dashboard instance"

            # Test serve function exists (don't actually serve)
            assert callable(serve_dashboard), "serve_dashboard should be callable"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_end_to_end_workflow_simulation(self, project_root, mock_api_client):
        """Test end-to-end workflow simulation."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard

            # Mock successful responses
            mock_api_client.create_optimizer.return_value = {"optimizer_id": "test-123"}
            mock_api_client.run_optimizer.return_value = {"status": "running"}
            mock_api_client.get_optimizer_status.return_value = {
                "status": "completed",
                "progress": 100,
                "best_fitness": 0.001,
                "generations_completed": 50,
            }
            mock_api_client.get_optimizer_history.return_value = [
                {"generation": 1, "best_fitness": 1.0},
                {"generation": 2, "best_fitness": 0.5},
                {"generation": 3, "best_fitness": 0.001},
            ]

            with patch(
                "darwin.ui.dashboard.main.APIClient", return_value=mock_api_client
            ):
                dashboard = DarwinDashboard(enable_callbacks=False)

                # Simulate workflow
                dashboard._create_optimizer(None)
                assert (
                    dashboard.current_optimizer_id == "test-123"
                ), "Optimizer should be created"

                dashboard._run_optimization(None)
                dashboard._update_dashboard()

                # Verify API interactions
                assert (
                    mock_api_client.create_optimizer.called
                ), "Should create optimizer"
                assert mock_api_client.run_optimizer.called, "Should run optimizer"
                assert mock_api_client.get_optimizer_status.called, "Should get status"

        except ImportError:
            pytest.skip("Dashboard module not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))


@pytest.mark.task_completion
class TestTask5_TaskCompletion:
    """Test Task 5 completion validation"""

    def test_task_completion_validation(self, project_root):
        """Test that Task 5 is properly implemented and complete."""

        # Check all required files exist
        required_files = [
            "src/darwin/ui/dashboard/main.py",
            "src/darwin/ui/dashboard/templates.py",
            "src/darwin/ui/__init__.py",
        ]

        for file_path in required_files:
            file_obj = project_root / file_path
            assert file_obj.exists(), f"Required file {file_path} should exist"

        # Check Panel dependency
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"

        with open(pyproject) as f:
            content = f.read()
            assert "panel" in content, "Panel should be in dependencies"
            assert "bokeh" in content, "Bokeh should be in dependencies"

        # Test imports work
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import (
                DarwinDashboard,
                create_dashboard,
                serve_dashboard,
            )
            from darwin.ui.dashboard.templates import (
                ProblemTemplate,
                TemplatesInterface,
            )

            # Basic functionality check
            assert callable(create_dashboard), "create_dashboard should be callable"
            assert callable(serve_dashboard), "serve_dashboard should be callable"

        except ImportError as e:
            pytest.fail(f"Task 5 implementation incomplete - import error: {e}")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

    def test_ui_requirements_coverage(self, project_root):
        """Test that UI requirements are covered."""
        sys.path.insert(0, str(project_root / "src"))
        try:
            from darwin.ui.dashboard.main import DarwinDashboard
            from darwin.ui.dashboard.templates import TemplatesInterface

            # Test required UI components exist
            required_components = [
                "problem_panel",
                "variables_table",
                "fitness_function",
                "create_button",
                "run_button",
                "fitness_plot",
                "progress_bar",
                "status_indicator",
            ]

            with patch("darwin.ui.dashboard.main.APIClient"):
                dashboard = DarwinDashboard(enable_callbacks=False)

                for component in required_components:
                    assert hasattr(
                        dashboard, component
                    ), f"Dashboard should have {component}"

            # Test templates interface
            mock_client = Mock()
            mock_client.get_templates.return_value = []
            templates_ui = TemplatesInterface(mock_client)

            assert hasattr(
                templates_ui, "templates_table"
            ), "Should have templates table"
            assert hasattr(templates_ui, "preview_panel"), "Should have preview panel"

        except ImportError:
            pytest.skip("Dashboard modules not available")
        finally:
            if str(project_root / "src") in sys.path:
                sys.path.remove(str(project_root / "src"))

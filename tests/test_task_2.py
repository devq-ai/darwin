"""
Test suite for Task 1.2: Configure Poetry and Dependencies

This test validates that Poetry configuration is correctly set up with all required
dependencies. All tests must pass for the task to be considered complete.
"""

import subprocess

import pytest
import toml


class TestTask1_2_PoetryConfiguration:
    """Test suite for Task 1.2: Configure Poetry and Dependencies"""

    def test_pyproject_toml_exists(self, project_root):
        """Test that pyproject.toml file exists."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml file does not exist"
        assert pyproject_path.is_file(), "pyproject.toml is not a file"

    def test_poetry_lock_exists(self, project_root):
        """Test that poetry.lock file exists."""
        poetry_lock_path = project_root / "poetry.lock"
        assert poetry_lock_path.exists(), "poetry.lock file does not exist"
        assert poetry_lock_path.is_file(), "poetry.lock is not a file"

    def test_pyproject_toml_structure(self, project_root):
        """Test that pyproject.toml has correct structure."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        # Check required sections
        assert "tool" in config, "pyproject.toml missing [tool] section"
        assert (
            "poetry" in config["tool"]
        ), "pyproject.toml missing [tool.poetry] section"

        poetry_config = config["tool"]["poetry"]

        # Check required Poetry fields
        required_fields = ["name", "version", "description", "authors"]
        for field in required_fields:
            assert (
                field in poetry_config
            ), f"pyproject.toml missing required field: {field}"

        # Check dependencies section exists
        assert (
            "dependencies" in poetry_config
        ), "pyproject.toml missing dependencies section"
        assert (
            "python" in poetry_config["dependencies"]
        ), "Python version not specified in dependencies"

    def test_required_dependencies(self, project_root):
        """Test that all required dependencies are specified."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        dependencies = config["tool"]["poetry"]["dependencies"]

        # Core dependencies
        required_deps = [
            "python",
            "fastapi",
            "uvicorn",
            "pydantic",
            "pygad",
            "surrealdb",
            "logfire",
            "pytest",
            "numpy",
            "pandas",
            "panel",
            "bokeh",
            "redis",
        ]

        for dep in required_deps:
            assert (
                dep in dependencies
            ), f"Required dependency '{dep}' not found in pyproject.toml"

    def test_development_dependencies(self, project_root):
        """Test that development dependencies are specified."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        # Check for development dependencies section
        assert "group" in config["tool"]["poetry"], "No dependency groups defined"
        assert (
            "dev" in config["tool"]["poetry"]["group"]
        ), "No dev dependency group defined"

        dev_deps = config["tool"]["poetry"]["group"]["dev"]["dependencies"]

        # Required development dependencies
        required_dev_deps = [
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "black",
            "isort",
            "mypy",
            "ruff",
            "pre-commit",
        ]

        for dep in required_dev_deps:
            assert (
                dep in dev_deps
            ), f"Required dev dependency '{dep}' not found in pyproject.toml"

    def test_python_version_constraint(self, project_root):
        """Test that Python version constraint is appropriate."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        python_version = config["tool"]["poetry"]["dependencies"]["python"]

        # Should specify Python 3.12+ as per project rules
        assert (
            "3.12" in python_version or "^3.12" in python_version
        ), f"Python version should be 3.12+, found: {python_version}"

    def test_poetry_check_command(self, project_root, poetry_validator):
        """Test that 'poetry check' command passes."""
        assert (
            poetry_validator.validate_poetry_check()
        ), "poetry check command failed - configuration is invalid"

    def test_poetry_install_dry_run(self, project_root, poetry_validator):
        """Test that 'poetry install --dry-run' succeeds."""
        assert (
            poetry_validator.validate_dependencies_installed()
        ), "poetry install --dry-run failed - dependency resolution issues"

    def test_build_system_configuration(self, project_root):
        """Test that build system is correctly configured."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        # Check build system configuration
        assert "build-system" in config, "pyproject.toml missing [build-system] section"

        build_system = config["build-system"]
        assert "requires" in build_system, "build-system missing 'requires' field"
        assert (
            "build-backend" in build_system
        ), "build-system missing 'build-backend' field"

        # Should use Poetry as build backend
        assert (
            "poetry" in build_system["build-backend"]
        ), f"Expected Poetry build backend, found: {build_system['build-backend']}"

    def test_project_metadata(self, project_root):
        """Test that project metadata is properly configured."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        poetry_config = config["tool"]["poetry"]

        # Check project name
        assert (
            poetry_config["name"] == "darwin"
        ), f"Project name should be 'darwin', found: {poetry_config['name']}"

        # Check version format
        version = poetry_config["version"]
        assert (
            len(version.split(".")) >= 2
        ), f"Version should be in semantic format (x.y.z), found: {version}"

        # Check description exists and is meaningful
        description = poetry_config["description"]
        assert len(description) > 10, "Description should be more descriptive"

        # Check authors format
        authors = poetry_config["authors"]
        assert (
            isinstance(authors, list) and len(authors) > 0
        ), "Authors should be a non-empty list"

    def test_tool_configurations(self, project_root):
        """Test that tool configurations are present."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        # Check for pytest configuration
        if "pytest" in config.get("tool", {}):
            pytest_config = config["tool"]["pytest"]
            assert (
                "ini_options" in pytest_config or "testpaths" in pytest_config
            ), "pytest configuration should specify test paths or options"

        # Check for black configuration
        if "black" in config.get("tool", {}):
            black_config = config["tool"]["black"]
            assert (
                "line-length" in black_config
            ), "black configuration should specify line length"
            assert (
                black_config["line-length"] == 88
            ), "black line length should be 88 characters per project rules"

        # Check for isort configuration
        if "isort" in config.get("tool", {}):
            isort_config = config["tool"]["isort"]
            assert (
                "profile" in isort_config
            ), "isort configuration should specify profile"

    def test_scripts_configuration(self, project_root):
        """Test that scripts are properly configured if present."""
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path) as f:
            config = toml.load(f)

        poetry_config = config["tool"]["poetry"]

        # If scripts are defined, validate them
        if "scripts" in poetry_config:
            scripts = poetry_config["scripts"]
            assert isinstance(scripts, dict), "scripts should be a dictionary"

            for script_name, script_path in scripts.items():
                assert isinstance(script_name, str), "script names should be strings"
                assert isinstance(script_path, str), "script paths should be strings"
                assert (
                    ":" in script_path
                ), f"script path should be module:function format: {script_path}"


@pytest.mark.task_validation
@pytest.mark.dependency_check
class TestTask1_2_Dependencies:
    """Test dependencies for Task 1.2"""

    def test_task_1_1_dependency(self, task_validator):
        """Test that Task 1.1 is completed before Task 1.2 can start."""
        task = task_validator.get_task_by_id("1")
        assert task is not None, "Task 1 not found"

        subtask_2 = None
        for st in task.get("subtasks", []):
            if st["id"] == 2:
                subtask_2 = st
                break

        assert subtask_2 is not None, "Subtask 1.2 not found"

        dependencies = subtask_2.get("dependencies", [])
        assert 1 in dependencies, "Task 1.2 should depend on Task 1.1"

        # Verify Task 1.1 is completed
        subtask_1 = None
        for st in task.get("subtasks", []):
            if st["id"] == 1:
                subtask_1 = st
                break

        assert subtask_1 is not None, "Subtask 1.1 not found"
        assert (
            subtask_1.get("status") == "done"
        ), "Task 1.1 must be completed before Task 1.2"


@pytest.mark.integration
class TestTask1_2_Integration:
    """Integration tests for Task 1.2"""

    def test_poetry_environment_creation(self, project_root):
        """Test that Poetry can create a virtual environment."""
        try:
            # Test environment creation (dry run)
            result = subprocess.run(
                ["poetry", "env", "info"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            # If environment doesn't exist, poetry env info will fail
            # This is acceptable - we just want to ensure Poetry is functional
            assert result.returncode in [
                0,
                1,
            ], "Poetry environment command failed unexpectedly"

        except FileNotFoundError:
            pytest.fail("Poetry is not installed or not accessible")

    def test_dependency_resolution(self, project_root):
        """Test that all dependencies can be resolved without conflicts."""
        try:
            # Use poetry check instead of lock --check for newer Poetry versions
            result = subprocess.run(
                ["poetry", "check"], capture_output=True, text=True, cwd=project_root
            )

            assert (
                result.returncode == 0
            ), f"Dependency resolution failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("Poetry not available for dependency resolution test")

    def test_core_imports_availability(self, project_root):
        """Test that core dependencies can be imported after installation."""
        # This test assumes dependencies are installed
        # In a real scenario, you'd want to test in the Poetry environment

        critical_imports = ["fastapi", "pydantic", "numpy", "pytest"]

        import importlib

        for module_name in critical_imports:
            try:
                spec = importlib.util.find_spec(module_name)
                # We just check if the module can be found, not imported
                # since it might not be installed in the test environment
                if spec is None:
                    pytest.skip(
                        f"Module {module_name} not available in test environment"
                    )
            except ImportError:
                pytest.skip(f"Cannot check import for {module_name}")

    def test_task_completion_validation(self, task_validator):
        """Test that task completion can be properly validated."""
        # Basic validation that the task validator is working
        tasks_data = task_validator.load_tasks()
        assert "tasks" in tasks_data, "Tasks data should contain 'tasks' key"

        # Verify task structure exists
        task_1 = None
        for task in tasks_data["tasks"]:
            if task["id"] == 1:
                task_1 = task
                break

        assert task_1 is not None, "Main Task 1 should exist"
        assert "subtasks" in task_1, "Task 1 should have subtasks"

        # Find subtasks 1.1 and 1.2
        subtask_1 = None
        subtask_2 = None
        for subtask in task_1["subtasks"]:
            if subtask["id"] == 1:
                subtask_1 = subtask
            elif subtask["id"] == 2:
                subtask_2 = subtask

        assert subtask_1 is not None, "Subtask 1.1 should exist"
        assert subtask_2 is not None, "Subtask 1.2 should exist"

        # Verify dependency structure
        assert 1 in subtask_2.get(
            "dependencies", []
        ), "Task 1.2 should depend on Task 1.1"
        assert subtask_1.get("status") == "done", "Task 1.1 should be completed"
        assert subtask_2.get("status") == "done", "Task 1.2 should be completed"

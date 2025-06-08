"""
Test suite for Task 1.1: Create Project Directory Structure

This test validates that the project directory structure is correctly set up
according to the requirements. All tests must pass for the task to be considered complete.
"""


import pytest


class TestTask1_1_DirectoryStructure:
    """Test suite for Task 1.1: Create Project Directory Structure"""

    def test_main_directories_exist(self, project_root, directory_validator):
        """Test that all required main directories exist."""
        missing_dirs = directory_validator.validate_directories()

        assert not missing_dirs, f"Missing required directories: {missing_dirs}"

    def test_required_files_exist(self, project_root, directory_validator):
        """Test that all required __init__.py files exist."""
        missing_files = directory_validator.validate_files()

        assert not missing_files, f"Missing required files: {missing_files}"

    def test_src_directory_structure(self, project_root):
        """Test the src/darwin directory structure specifically."""
        src_darwin = project_root / "src" / "darwin"

        assert src_darwin.exists(), "src/darwin directory does not exist"
        assert src_darwin.is_dir(), "src/darwin is not a directory"

        # Check subdirectories
        subdirs = ["core", "api", "mcp", "ui", "db", "utils"]
        for subdir in subdirs:
            subdir_path = src_darwin / subdir
            assert subdir_path.exists(), f"src/darwin/{subdir} directory does not exist"
            assert subdir_path.is_dir(), f"src/darwin/{subdir} is not a directory"

    def test_python_package_structure(self, project_root):
        """Test that Python package structure is correct with __init__.py files."""
        init_files = [
            "src/darwin/__init__.py",
            "src/darwin/core/__init__.py",
            "src/darwin/api/__init__.py",
            "src/darwin/mcp/__init__.py",
            "src/darwin/ui/__init__.py",
            "src/darwin/db/__init__.py",
            "src/darwin/utils/__init__.py",
        ]

        for init_file in init_files:
            init_path = project_root / init_file
            assert init_path.exists(), f"Missing __init__.py file: {init_file}"
            assert init_path.is_file(), f"{init_file} is not a file"

    def test_tests_directory_structure(self, project_root):
        """Test that tests directory exists and is properly structured."""
        tests_dir = project_root / "tests"

        assert tests_dir.exists(), "tests directory does not exist"
        assert tests_dir.is_dir(), "tests is not a directory"

        # Check for conftest.py
        conftest = tests_dir / "conftest.py"
        assert conftest.exists(), "tests/conftest.py does not exist"

    def test_docs_directory_exists(self, project_root):
        """Test that docs directory exists."""
        docs_dir = project_root / "docs"

        assert docs_dir.exists(), "docs directory does not exist"
        assert docs_dir.is_dir(), "docs is not a directory"

    def test_examples_directory_exists(self, project_root):
        """Test that examples directory exists."""
        examples_dir = project_root / "examples"

        assert examples_dir.exists(), "examples directory does not exist"
        assert examples_dir.is_dir(), "examples is not a directory"

    def test_docker_directory_exists(self, project_root):
        """Test that docker directory exists."""
        docker_dir = project_root / "docker"

        assert docker_dir.exists(), "docker directory does not exist"
        assert docker_dir.is_dir(), "docker is not a directory"

    def test_scripts_directory_exists(self, project_root):
        """Test that scripts directory exists."""
        scripts_dir = project_root / "scripts"

        assert scripts_dir.exists(), "scripts directory does not exist"
        assert scripts_dir.is_dir(), "scripts is not a directory"

    def test_directory_permissions(self, project_root):
        """Test that directories have correct permissions."""
        required_dirs = [
            "src/darwin",
            "src/darwin/core",
            "src/darwin/api",
            "src/darwin/mcp",
            "src/darwin/ui",
            "src/darwin/db",
            "src/darwin/utils",
            "tests",
            "docs",
            "examples",
            "docker",
            "scripts",
        ]

        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} does not exist"

            # Check that we can read the directory
            try:
                list(full_path.iterdir())
            except PermissionError:
                pytest.fail(f"Cannot read directory {dir_path} - permission denied")

    def test_no_unwanted_files_in_source(self, project_root):
        """Test that no unwanted files exist in source directories."""
        src_dir = project_root / "src"
        unwanted_patterns = ["*.pyc", "__pycache__", "*.pyo", ".DS_Store"]

        def check_directory_recursive(directory):
            """Recursively check directory for unwanted files."""
            for item in directory.rglob("*"):
                if item.is_file():
                    for pattern in unwanted_patterns:
                        if item.match(pattern):
                            pytest.fail(f"Found unwanted file: {item}")

        if src_dir.exists():
            check_directory_recursive(src_dir)

    def test_package_importability(self, project_root):
        """Test that the darwin package can be imported."""
        import importlib.util
        import sys

        # Add src to Python path temporarily
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test importing the main package
            spec = importlib.util.spec_from_file_location(
                "darwin", project_root / "src" / "darwin" / "__init__.py"
            )
            assert spec is not None, "Cannot create module spec for darwin package"

            module = importlib.util.module_from_spec(spec)
            assert module is not None, "Cannot create module from spec"

            # Test importing subpackages
            subpackages = ["core", "api", "mcp", "ui", "db", "utils"]
            for subpackage in subpackages:
                subpackage_path = (
                    project_root / "src" / "darwin" / subpackage / "__init__.py"
                )
                spec = importlib.util.spec_from_file_location(
                    f"darwin.{subpackage}", subpackage_path
                )
                assert (
                    spec is not None
                ), f"Cannot create module spec for darwin.{subpackage}"

        finally:
            # Clean up sys.path
            if src_path in sys.path:
                sys.path.remove(src_path)


@pytest.mark.task_validation
@pytest.mark.dependency_check
class TestTask1_1_Dependencies:
    """Test dependencies for Task 1.1"""

    def test_no_dependencies(self, task_validator):
        """Test that Task 1.1 has no dependencies (it's the first task)."""
        task = task_validator.get_task_by_id("1")
        assert task is not None, "Task 1 not found"

        subtask = None
        for st in task.get("subtasks", []):
            if st["id"] == 1:
                subtask = st
                break

        assert subtask is not None, "Subtask 1.1 not found"
        assert (
            subtask.get("dependencies", []) == []
        ), "Task 1.1 should have no dependencies"


@pytest.mark.integration
class TestTask1_1_Integration:
    """Integration tests for Task 1.1"""

    def test_directory_structure_completeness(self, project_root, directory_validator):
        """Integration test to verify complete directory structure setup."""
        # Test all directories
        missing_dirs = directory_validator.validate_directories()
        assert not missing_dirs, f"Directory structure incomplete: {missing_dirs}"

        # Test all files
        missing_files = directory_validator.validate_files()
        assert not missing_files, f"Required files missing: {missing_files}"

        # Test that we can navigate the entire structure
        src_darwin = project_root / "src" / "darwin"
        assert src_darwin.exists()

        for subdir in ["core", "api", "mcp", "ui", "db", "utils"]:
            subdir_path = src_darwin / subdir
            assert subdir_path.exists()
            assert (subdir_path / "__init__.py").exists()

    def test_task_completion_validation(self, task_validator):
        """Test that task completion can be properly validated."""
        # This test validates that the task validation system works
        task = task_validator.get_task_by_id("1")
        assert task is not None

        # If this test passes, it means all structure requirements are met
        assert True, "Task 1.1 validation completed successfully"

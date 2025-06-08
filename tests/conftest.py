"""
Test configuration and fixtures for Darwin project task validation.

This module provides comprehensive test infrastructure to validate task completion
before allowing dependent tasks to proceed. All tests must pass for a task to be
considered complete.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import docker
from docker.errors import DockerException

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root():
    """Fixture providing the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def docker_client():
    """Fixture providing Docker client for container testing."""
    try:
        client = docker.from_env()
        # Test Docker connection
        client.ping()
        return client
    except DockerException as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="session")
def poetry_env():
    """Fixture ensuring Poetry environment is available."""
    try:
        result = subprocess.run(
            ["poetry", "--version"], capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode != 0:
            pytest.skip("Poetry not available")
        return True
    except FileNotFoundError:
        pytest.skip("Poetry not installed")


class TaskValidator:
    """Validator class for task completion verification."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks_file = project_root / ".taskmaster" / "tasks" / "tasks.json"

    def load_tasks(self) -> Dict:
        """Load tasks from tasks.json file."""
        if not self.tasks_file.exists():
            raise FileNotFoundError(f"Tasks file not found: {self.tasks_file}")

        with open(self.tasks_file) as f:
            return json.load(f)

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Get task by ID from tasks.json."""
        tasks = self.load_tasks()

        # First pass: Check main tasks only (prioritize main tasks over subtasks)
        for task in tasks.get("tasks", []):
            if str(task["id"]) == str(task_id):
                return task

        # Second pass: Check subtasks only if no main task found
        for task in tasks.get("tasks", []):
            for subtask in task.get("subtasks", []):
                if str(subtask["id"]) == str(task_id):
                    return subtask

        return None

    def validate_dependencies(self, task_id: str) -> bool:
        """Validate that all dependencies are completed with passing tests."""
        task = self.get_task_by_id(task_id)
        if not task:
            return False

        dependencies = task.get("dependencies", [])
        for dep_id in dependencies:
            dep_task = self.get_task_by_id(dep_id)
            if not dep_task:
                return False

            if dep_task.get("status") != "done":
                return False

            # Verify tests passed for dependency
            if not self.run_task_tests(dep_id):
                return False

        return True

    def run_task_tests(self, task_id: str) -> bool:
        """Run tests for a specific task."""
        test_file = self.project_root / "tests" / f"test_task_{task_id}.py"
        if not test_file.exists():
            # No tests defined means task cannot be validated
            return False

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False

    def can_start_task(self, task_id: str) -> bool:
        """Check if a task can be started based on dependencies."""
        return self.validate_dependencies(task_id)


@pytest.fixture(scope="session")
def task_validator(project_root):
    """Fixture providing task validator instance."""
    return TaskValidator(project_root)


@pytest.fixture(scope="session")
def client(project_root):
    """Create test client for FastAPI app."""
    import sys

    from fastapi.testclient import TestClient

    # Set testing environment variable
    os.environ["TESTING"] = "true"

    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from darwin.api.main import app

        return TestClient(app)
    except ImportError:
        # Return None if FastAPI app not yet implemented
        # Tests should check if client is None before proceeding
        return None
    except Exception as e:
        # Log the error for debugging but return None
        print(f"Warning: Failed to create test client: {e}")
        return None
    finally:
        if src_path in sys.path:
            sys.path.remove(src_path)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    # Set testing environment variable
    os.environ["TESTING"] = "true"

    config.addinivalue_line(
        "markers", "task_validation: mark test as task validation test"
    )
    config.addinivalue_line(
        "markers", "dependency_check: mark test as dependency validation"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "test_task_" in item.nodeid:
            item.add_marker(pytest.mark.task_validation)

        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)


class DirectoryStructureValidator:
    """Validator for project directory structure."""

    REQUIRED_DIRECTORIES = [
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

    REQUIRED_FILES = [
        "src/darwin/__init__.py",
        "src/darwin/core/__init__.py",
        "src/darwin/api/__init__.py",
        "src/darwin/mcp/__init__.py",
        "src/darwin/ui/__init__.py",
        "src/darwin/db/__init__.py",
        "src/darwin/utils/__init__.py",
        "pyproject.toml",
        "README.md",
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def validate_directories(self) -> List[str]:
        """Validate required directories exist."""
        missing = []
        for directory in self.REQUIRED_DIRECTORIES:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                missing.append(directory)
        return missing

    def validate_files(self) -> List[str]:
        """Validate required files exist."""
        missing = []
        for file_path in self.REQUIRED_FILES:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing.append(file_path)
        return missing


@pytest.fixture(scope="session")
def directory_validator(project_root):
    """Fixture providing directory structure validator."""
    return DirectoryStructureValidator(project_root)


class PoetryValidator:
    """Validator for Poetry configuration and dependencies."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.poetry_lock_path = project_root / "poetry.lock"

    def validate_pyproject_exists(self) -> bool:
        """Check if pyproject.toml exists."""
        return self.pyproject_path.exists()

    def validate_poetry_lock_exists(self) -> bool:
        """Check if poetry.lock exists."""
        return self.poetry_lock_path.exists()

    def validate_poetry_check(self) -> bool:
        """Run poetry check to validate configuration."""
        try:
            result = subprocess.run(
                ["poetry", "check"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False

    def validate_dependencies_installed(self) -> bool:
        """Validate that dependencies are properly installed."""
        try:
            result = subprocess.run(
                ["poetry", "install", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False


@pytest.fixture(scope="session")
def poetry_validator(project_root):
    """Fixture providing Poetry validator."""
    return PoetryValidator(project_root)


class DockerValidator:
    """Validator for Docker configuration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dockerfile_dev = project_root / "docker" / "Dockerfile.dev"
        self.dockerfile_prod = project_root / "docker" / "Dockerfile.prod"
        self.docker_compose = project_root / "docker-compose.yml"

    def validate_dockerfiles_exist(self) -> List[str]:
        """Check if required Dockerfiles exist."""
        missing = []
        if not self.dockerfile_dev.exists():
            missing.append("docker/Dockerfile.dev")
        if not self.dockerfile_prod.exists():
            missing.append("docker/Dockerfile.prod")
        if not self.docker_compose.exists():
            missing.append("docker-compose.yml")
        return missing

    def validate_docker_build(self, dockerfile: str) -> bool:
        """Validate Docker build succeeds."""
        try:
            client = docker.from_env()
            dockerfile_path = self.project_root / dockerfile

            # Build image without using cache
            image, logs = client.images.build(
                path=str(self.project_root),
                dockerfile=str(dockerfile_path),
                tag=f"darwin-test-{dockerfile.replace('/', '-')}",
                rm=True,
                forcerm=True,
            )

            # Clean up test image
            client.images.remove(image.id, force=True)
            return True

        except Exception as e:
            print(f"Docker build failed: {e}")
            return False

    def validate_docker_compose(self) -> bool:
        """Validate docker-compose configuration."""
        try:
            # Try modern docker compose command first
            result = subprocess.run(
                ["docker", "compose", "config", "--quiet"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return True

            # Fall back to legacy docker-compose command
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False


@pytest.fixture(scope="session")
def docker_validator(project_root):
    """Fixture providing Docker validator."""
    return DockerValidator(project_root)


class GitValidator:
    """Validator for Git repository setup."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.git_dir = project_root / ".git"
        self.gitignore = project_root / ".gitignore"
        self.precommit_config = project_root / ".pre-commit-config.yaml"

    def validate_git_repo(self) -> bool:
        """Check if Git repository is initialized."""
        return self.git_dir.exists() and self.git_dir.is_dir()

    def validate_gitignore(self) -> bool:
        """Check if .gitignore file exists and has required patterns."""
        if not self.gitignore.exists():
            return False

        content = self.gitignore.read_text()
        required_patterns = [
            "__pycache__",
            "*.pyc",
            ".env",
            ".venv",
            "node_modules",
            ".DS_Store",
            "*.log",
        ]

        return all(pattern in content for pattern in required_patterns)

    def validate_precommit_config(self) -> bool:
        """Check if pre-commit configuration exists."""
        return self.precommit_config.exists()

    def validate_precommit_installed(self) -> bool:
        """Check if pre-commit hooks are installed."""
        try:
            result = subprocess.run(
                ["pre-commit", "--version"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False


@pytest.fixture(scope="session")
def git_validator(project_root):
    """Fixture providing Git validator."""
    return GitValidator(project_root)


# Utility functions for test helpers
def run_command(
    command: List[str], cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        command, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT
    )


def file_contains(file_path: Path, content: str) -> bool:
    """Check if a file contains specific content."""
    if not file_path.exists():
        return False

    try:
        return content in file_path.read_text()
    except Exception:
        return False


def create_test_report(task_id: str, results: Dict) -> None:
    """Create a test report for a task."""
    reports_dir = PROJECT_ROOT / ".taskmaster" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"task_{task_id}_test_report.json"

    with open(report_file, "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "timestamp": pytest.current_timestamp
                if hasattr(pytest, "current_timestamp")
                else None,
                "results": results,
                "passed": all(results.values()),
            },
            f,
            indent=2,
        )

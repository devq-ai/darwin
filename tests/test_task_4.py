"""
Test suite for Task 1.4: Initialize Git Repository and Code Quality Tools

This test validates that Git repository is properly initialized with code quality
tools and pre-commit hooks. All tests must pass for the task to be considered complete.
"""

import subprocess

import pytest
import yaml


class TestTask1_4_GitRepository:
    """Test suite for Task 1.4: Initialize Git Repository and Code Quality Tools"""

    def test_git_repository_initialized(self, project_root):
        """Test that Git repository is properly initialized."""
        git_dir = project_root / ".git"
        assert git_dir.exists(), ".git directory does not exist"
        assert git_dir.is_dir(), ".git is not a directory"

    def test_gitignore_exists(self, project_root):
        """Test that .gitignore file exists."""
        gitignore = project_root / ".gitignore"
        assert gitignore.exists(), ".gitignore file does not exist"
        assert gitignore.is_file(), ".gitignore is not a file"

    def test_gitignore_content(self, project_root):
        """Test that .gitignore has appropriate content for Python projects."""
        gitignore = project_root / ".gitignore"
        content = gitignore.read_text()

        # Required patterns for Python projects
        required_patterns = [
            "__pycache__",
            "*.pyc",
            ".env",
            ".venv",
            "*.log",
            ".DS_Store",
            "node_modules",
            "*.egg-info",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
        ]

        for pattern in required_patterns:
            assert pattern in content, f".gitignore should include {pattern}"

    def test_precommit_config_exists(self, project_root):
        """Test that pre-commit configuration file exists."""
        precommit_config = project_root / ".pre-commit-config.yaml"
        assert precommit_config.exists(), ".pre-commit-config.yaml does not exist"
        assert precommit_config.is_file(), ".pre-commit-config.yaml is not a file"

    def test_precommit_config_structure(self, project_root):
        """Test that pre-commit configuration has proper structure."""
        precommit_config = project_root / ".pre-commit-config.yaml"

        with open(precommit_config) as f:
            config = yaml.safe_load(f)

        # Check required structure
        assert "repos" in config, "pre-commit config missing 'repos' section"
        assert isinstance(config["repos"], list), "repos should be a list"
        assert len(config["repos"]) > 0, "repos should not be empty"

    def test_precommit_hooks_configuration(self, project_root):
        """Test that required pre-commit hooks are configured."""
        precommit_config = project_root / ".pre-commit-config.yaml"

        with open(precommit_config) as f:
            config = yaml.safe_load(f)

        # Extract all hook IDs from all repos
        hook_ids = []
        for repo in config["repos"]:
            if "hooks" in repo:
                for hook in repo["hooks"]:
                    hook_ids.append(hook["id"])

        # Required hooks
        required_hooks = ["black", "isort", "ruff", "mypy"]

        for hook in required_hooks:
            assert (
                hook in hook_ids
            ), f"Required hook '{hook}' not found in pre-commit config"

    def test_git_config_basic(self, project_root):
        """Test basic Git configuration."""
        try:
            result = subprocess.run(
                ["git", "config", "--list"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert result.returncode == 0, "Git config command failed"

            config_lines = result.stdout.split("\n")
            config_dict = {}
            for line in config_lines:
                if "=" in line:
                    key, value = line.split("=", 1)
                    config_dict[key] = value

            # Check for basic configuration
            assert (
                "user.name" in config_dict or "user.email" in config_dict
            ), "Git should have user configuration"

        except FileNotFoundError:
            pytest.skip("Git not available")

    def test_git_status_clean(self, project_root):
        """Test that git status works properly."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert result.returncode == 0, "Git status command failed"

        except FileNotFoundError:
            pytest.skip("Git not available")

    def test_precommit_installation(self, project_root):
        """Test that pre-commit can be installed."""
        try:
            # Check if pre-commit is available
            result = subprocess.run(
                ["pre-commit", "--version"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            if result.returncode != 0:
                pytest.skip("pre-commit not available")

            # Try to validate the configuration
            result = subprocess.run(
                ["pre-commit", "validate-config"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert (
                result.returncode == 0
            ), f"pre-commit config validation failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("pre-commit not available")


@pytest.mark.task_validation
@pytest.mark.dependency_check
class TestTask1_4_Dependencies:
    """Test dependencies for Task 1.4"""

    def test_task_1_1_dependency(self, task_validator):
        """Test that Task 1.1 is completed before Task 1.4 can start."""
        # Get Task 1 and its subtasks
        tasks_data = task_validator.load_tasks()
        task_1 = None
        for task in tasks_data["tasks"]:
            if task["id"] == 1:
                task_1 = task
                break

        assert task_1 is not None, "Main Task 1 should exist"

        # Find subtask 1.4
        subtask_4 = None
        for subtask in task_1.get("subtasks", []):
            if subtask["id"] == 4:
                subtask_4 = subtask
                break

        assert subtask_4 is not None, "Subtask 1.4 should exist"

        # Check dependencies
        dependencies = subtask_4.get("dependencies", [])
        assert 1 in dependencies, "Task 1.4 should depend on Task 1.1"

        # Verify Task 1.1 is completed
        subtask_1 = None
        for subtask in task_1.get("subtasks", []):
            if subtask["id"] == 1:
                subtask_1 = subtask
                break

        assert subtask_1 is not None, "Subtask 1.1 should exist"
        assert (
            subtask_1.get("status") == "done"
        ), "Task 1.1 must be completed before Task 1.4"


@pytest.mark.integration
class TestTask1_4_Integration:
    """Integration tests for Task 1.4"""

    def test_git_and_precommit_integration(self, project_root, git_validator):
        """Test complete Git and pre-commit integration."""
        # Validate Git repository
        assert git_validator.validate_git_repo(), "Git repository should be initialized"
        assert (
            git_validator.validate_gitignore()
        ), ".gitignore should be properly configured"
        assert (
            git_validator.validate_precommit_config()
        ), "Pre-commit config should exist"

    def test_code_quality_tools_available(self, project_root):
        """Test that code quality tools are available."""
        tools = ["black", "isort", "ruff", "mypy"]

        for tool in tools:
            try:
                result = subprocess.run(
                    ["poetry", "run", tool, "--version"],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                )

                # Tool should either work or be available for installation
                if result.returncode != 0:
                    # Check if tool is in dependencies
                    pyproject = project_root / "pyproject.toml"
                    content = pyproject.read_text()
                    assert (
                        tool in content
                    ), f"Tool {tool} should be available or in dependencies"

            except FileNotFoundError:
                pytest.skip(f"{tool} not available in environment")

    def test_git_repository_functionality(self, project_root):
        """Test basic Git repository functionality."""
        try:
            # Test git add functionality
            result = subprocess.run(
                ["git", "add", "--dry-run", "."],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert result.returncode == 0, "Git add should work"

            # Test git status
            result = subprocess.run(
                ["git", "status"], capture_output=True, text=True, cwd=project_root
            )

            assert result.returncode == 0, "Git status should work"

        except FileNotFoundError:
            pytest.skip("Git not available")

    def test_github_remote_configuration(self, project_root):
        """Test GitHub remote configuration if available."""
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            if result.returncode == 0 and result.stdout:
                # If remotes exist, check for GitHub
                output = result.stdout.lower()
                if "github.com" in output:
                    assert (
                        "devq-ai/darwin" in output
                    ), "GitHub remote should point to devq-ai/darwin"

        except FileNotFoundError:
            pytest.skip("Git not available")

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

        # Find subtask 1.4
        subtask_4 = None
        for subtask in task_1["subtasks"]:
            if subtask["id"] == 4:
                subtask_4 = subtask
                break

        assert subtask_4 is not None, "Subtask 1.4 should exist"

        # Verify that the validation system recognizes the task structure
        assert True, "Task 1.4 validation structure is correct"


@pytest.mark.unit
class TestTask1_4_FileContent:
    """Unit tests for specific file content validation"""

    def test_gitignore_python_patterns(self, project_root):
        """Test that .gitignore includes Python-specific patterns."""
        gitignore = project_root / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore file not created yet")

        content = gitignore.read_text()

        python_patterns = [
            "# Byte-compiled",
            "__pycache__/",
            "*.py[cod]",
            "*.so",
            "# Virtual environments",
            ".env",
            ".venv/",
            "# Testing",
            ".pytest_cache/",
            ".coverage",
            "# IDE",
            ".vscode/",
            ".idea/",
        ]

        # At least some Python patterns should be present
        found_patterns = sum(1 for pattern in python_patterns if pattern in content)
        assert (
            found_patterns >= 5
        ), "Should have comprehensive Python .gitignore patterns"

    def test_precommit_hooks_yaml_syntax(self, project_root):
        """Test that pre-commit config has valid YAML syntax."""
        precommit_config = project_root / ".pre-commit-config.yaml"
        if not precommit_config.exists():
            pytest.skip(".pre-commit-config.yaml not created yet")

        try:
            with open(precommit_config) as f:
                config = yaml.safe_load(f)
            assert config is not None, "YAML should be valid"
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax: {e}")

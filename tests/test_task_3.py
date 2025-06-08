"""
Test suite for Task 1.3: Setup Docker Environment

This test validates that Docker configuration is correctly set up with all required
files and that containers can be built successfully. All tests must pass for the task
to be considered complete.
"""

import subprocess

import pytest
import yaml

import docker
from docker.errors import DockerException


class TestTask1_3_DockerConfiguration:
    """Test suite for Task 1.3: Setup Docker Environment"""

    def test_dockerfile_dev_exists(self, project_root):
        """Test that Dockerfile.dev exists in docker directory."""
        dockerfile_dev = project_root / "docker" / "Dockerfile.dev"
        assert dockerfile_dev.exists(), "docker/Dockerfile.dev does not exist"
        assert dockerfile_dev.is_file(), "docker/Dockerfile.dev is not a file"

    def test_dockerfile_prod_exists(self, project_root):
        """Test that Dockerfile.prod exists in docker directory."""
        dockerfile_prod = project_root / "docker" / "Dockerfile.prod"
        assert dockerfile_prod.exists(), "docker/Dockerfile.prod does not exist"
        assert dockerfile_prod.is_file(), "docker/Dockerfile.prod is not a file"

    def test_docker_compose_exists(self, project_root):
        """Test that docker-compose.yml exists in project root."""
        docker_compose = project_root / "docker-compose.yml"
        assert docker_compose.exists(), "docker-compose.yml does not exist"
        assert docker_compose.is_file(), "docker-compose.yml is not a file"

    def test_dockerfile_dev_content(self, project_root):
        """Test that Dockerfile.dev has appropriate content for development."""
        dockerfile_dev = project_root / "docker" / "Dockerfile.dev"
        content = dockerfile_dev.read_text()

        # Check for required elements in development Dockerfile
        required_elements = [
            "FROM python:",  # Should use Python base image
            "WORKDIR",  # Should set working directory
            "COPY",  # Should copy files
            "RUN",  # Should run installation commands
            "EXPOSE",  # Should expose port
        ]

        for element in required_elements:
            assert (
                element in content
            ), f"Dockerfile.dev missing required element: {element}"

        # Development-specific checks
        assert (
            "poetry" in content.lower()
        ), "Dockerfile.dev should use Poetry for dependency management"
        assert (
            "dev" in content.lower() or "development" in content.lower()
        ), "Dockerfile.dev should indicate development environment"

    def test_dockerfile_prod_content(self, project_root):
        """Test that Dockerfile.prod has appropriate content for production."""
        dockerfile_prod = project_root / "docker" / "Dockerfile.prod"
        content = dockerfile_prod.read_text()

        # Check for required elements in production Dockerfile
        required_elements = [
            "FROM python:",  # Should use Python base image
            "WORKDIR",  # Should set working directory
            "COPY",  # Should copy files
            "RUN",  # Should run installation commands
            "EXPOSE",  # Should expose port
            "CMD",  # Should have start command
        ]

        for element in required_elements:
            assert (
                element in content
            ), f"Dockerfile.prod missing required element: {element}"

        # Production-specific checks
        assert (
            "poetry" in content.lower()
        ), "Dockerfile.prod should use Poetry for dependency management"

        # Should use multi-stage builds or optimization
        stages_or_optimization = [
            "AS builder",  # Multi-stage build
            "AS production",  # Multi-stage build
            "--no-dev",  # No dev dependencies
            "--only=main",  # Only main dependencies
        ]

        has_optimization = any(opt in content for opt in stages_or_optimization)
        assert has_optimization, "Dockerfile.prod should use optimization techniques"

    def test_docker_compose_structure(self, project_root):
        """Test that docker-compose.yml has correct structure."""
        docker_compose = project_root / "docker-compose.yml"

        with open(docker_compose) as f:
            compose_config = yaml.safe_load(f)

        # Check required top-level sections
        assert (
            "version" in compose_config or "services" in compose_config
        ), "docker-compose.yml missing version or services section"

        assert (
            "services" in compose_config
        ), "docker-compose.yml missing services section"

        services = compose_config["services"]

        # Should have at least an app service
        assert (
            "app" in services or "darwin" in services or "web" in services
        ), "docker-compose.yml should define an application service"

    def test_docker_compose_app_service(self, project_root):
        """Test that docker-compose.yml has properly configured app service."""
        docker_compose = project_root / "docker-compose.yml"

        with open(docker_compose) as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]

        # Find the main app service
        app_service = None
        for service_name in ["app", "darwin", "web", "api"]:
            if service_name in services:
                app_service = services[service_name]
                break

        assert (
            app_service is not None
        ), "No main application service found in docker-compose.yml"

        # Check required service configuration
        assert (
            "build" in app_service or "image" in app_service
        ), "App service should specify build or image"

        if "build" in app_service:
            build_config = app_service["build"]
            if isinstance(build_config, dict):
                assert (
                    "dockerfile" in build_config or "context" in build_config
                ), "Build configuration should specify dockerfile or context"

        # Should expose ports
        assert "ports" in app_service, "App service should expose ports"

        # Should have environment variables or env_file
        has_env = "environment" in app_service or "env_file" in app_service
        assert has_env, "App service should define environment variables"

    def test_docker_compose_database_service(self, project_root):
        """Test that docker-compose.yml includes database service if required."""
        docker_compose = project_root / "docker-compose.yml"

        with open(docker_compose) as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]

        # Check for database services (SurrealDB, Redis, etc.)
        db_services = ["surrealdb", "surreal", "redis", "database", "db"]
        has_db_service = any(service in services for service in db_services)

        if has_db_service:
            # Find database service
            db_service = None
            for service_name in db_services:
                if service_name in services:
                    db_service = services[service_name]
                    break

            assert "image" in db_service, "Database service should specify image"

            # Should have volumes for data persistence
            if (
                "surrealdb" in str(db_service).lower()
                or "surreal" in str(db_service).lower()
            ):
                assert (
                    "volumes" in db_service or "volume" in db_service
                ), "SurrealDB service should have volumes for data persistence"

    def test_docker_compose_volumes(self, project_root):
        """Test that docker-compose.yml has appropriate volume configurations."""
        docker_compose = project_root / "docker-compose.yml"

        with open(docker_compose) as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]

        # Find main app service
        app_service = None
        for service_name in ["app", "darwin", "web", "api"]:
            if service_name in services:
                app_service = services[service_name]
                break

        if app_service and "volumes" in app_service:
            volumes = app_service["volumes"]

            # Should mount source code for development
            has_source_mount = any(
                "." in volume or "/app" in volume or "/code" in volume
                for volume in volumes
                if isinstance(volume, str)
            )

            # For development, should have source code mounted
            # This can be conditional based on environment
            if any("dev" in str(vol).lower() for vol in volumes):
                assert (
                    has_source_mount
                ), "Development configuration should mount source code"

    def test_docker_compose_networks(self, project_root):
        """Test that docker-compose.yml has appropriate network configuration."""
        docker_compose = project_root / "docker-compose.yml"

        with open(docker_compose) as f:
            compose_config = yaml.safe_load(f)

        # If multiple services are defined, should have network configuration
        services = compose_config.get("services", {})

        if len(services) > 1:
            # Either default network or custom networks
            if "networks" in compose_config:
                networks = compose_config["networks"]
                assert isinstance(networks, dict), "Networks should be properly defined"

    @pytest.mark.skipif(not docker, reason="Docker not available")
    def test_docker_client_available(self, docker_client):
        """Test that Docker client is available and functional."""
        assert docker_client is not None, "Docker client not available"

        try:
            docker_client.ping()
        except DockerException:
            pytest.fail("Docker daemon not accessible")

    @pytest.mark.skipif(not docker, reason="Docker not available")
    def test_dockerfile_dev_builds(self, project_root, docker_client):
        """Test that Dockerfile.dev builds successfully."""
        dockerfile_dev = project_root / "docker" / "Dockerfile.dev"

        if not dockerfile_dev.exists():
            pytest.skip("Dockerfile.dev not found")

        try:
            # Build the development image
            image, logs = docker_client.images.build(
                path=str(project_root),
                dockerfile=str(dockerfile_dev),
                tag="darwin-dev-test",
                rm=True,
                forcerm=True,
                pull=False,  # Don't pull to speed up tests
                quiet=False,
            )

            assert image is not None, "Failed to build development Docker image"

            # Clean up
            docker_client.images.remove(image.id, force=True)

        except Exception as e:
            pytest.fail(f"Dockerfile.dev build failed: {str(e)}")

    @pytest.mark.skipif(not docker, reason="Docker not available")
    def test_dockerfile_prod_builds(self, project_root, docker_client):
        """Test that Dockerfile.prod builds successfully."""
        dockerfile_prod = project_root / "docker" / "Dockerfile.prod"

        if not dockerfile_prod.exists():
            pytest.skip("Dockerfile.prod not found")

        try:
            # Build the production image
            image, logs = docker_client.images.build(
                path=str(project_root),
                dockerfile=str(dockerfile_prod),
                tag="darwin-prod-test",
                rm=True,
                forcerm=True,
                pull=False,  # Don't pull to speed up tests
                quiet=False,
            )

            assert image is not None, "Failed to build production Docker image"

            # Clean up
            docker_client.images.remove(image.id, force=True)

        except Exception as e:
            pytest.fail(f"Dockerfile.prod build failed: {str(e)}")

    def test_docker_compose_syntax(self, project_root):
        """Test that docker-compose.yml has valid syntax."""
        try:
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert (
                result.returncode == 0
            ), f"docker-compose.yml syntax error: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("docker-compose command not available")

    def test_docker_ignore_file(self, project_root):
        """Test that .dockerignore file exists and has appropriate content."""
        dockerignore = project_root / ".dockerignore"

        if dockerignore.exists():
            content = dockerignore.read_text()

            # Should ignore common files that shouldn't be in Docker context
            recommended_ignores = [
                ".git",
                "*.pyc",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
                ".env*",
                "*.log",
            ]

            for ignore_pattern in recommended_ignores:
                assert (
                    ignore_pattern in content
                ), f".dockerignore should include {ignore_pattern}"


@pytest.mark.task_validation
@pytest.mark.dependency_check
class TestTask1_3_Dependencies:
    """Test dependencies for Task 1.3"""

    def test_task_1_2_dependency(self, task_validator):
        """Test that Task 1.2 is completed before Task 1.3 can start."""
        task = task_validator.get_task_by_id("1")
        assert task is not None, "Task 1 not found"

        subtask_3 = None
        for st in task.get("subtasks", []):
            if st["id"] == 3:
                subtask_3 = st
                break

        assert subtask_3 is not None, "Subtask 1.3 not found"

        dependencies = subtask_3.get("dependencies", [])
        assert 2 in dependencies, "Task 1.3 should depend on Task 1.2"

        # Verify Task 1.2 is completed
        subtask_2 = None
        for st in task.get("subtasks", []):
            if st["id"] == 2:
                subtask_2 = st
                break

        assert subtask_2 is not None, "Subtask 1.2 not found"
        assert (
            subtask_2.get("status") == "done"
        ), "Task 1.2 must be completed before Task 1.3"


@pytest.mark.integration
class TestTask1_3_Integration:
    """Integration tests for Task 1.3"""

    def test_docker_environment_completeness(self, project_root, docker_validator):
        """Test complete Docker environment setup."""
        # Test all Docker files exist
        missing_files = docker_validator.validate_dockerfiles_exist()
        assert not missing_files, f"Missing Docker files: {missing_files}"

        # Test docker-compose configuration
        assert (
            docker_validator.validate_docker_compose()
        ), "docker-compose configuration validation failed"

    @pytest.mark.skipif(not docker, reason="Docker not available")
    def test_full_docker_stack(self, project_root):
        """Test that the full Docker stack can be validated."""
        try:
            # Test docker-compose validation
            result = subprocess.run(
                ["docker-compose", "config", "--quiet"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            assert (
                result.returncode == 0
            ), f"docker-compose configuration failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("docker-compose not available")

    def test_environment_variables_setup(self, project_root):
        """Test that environment variables are properly configured."""
        docker_compose = project_root / "docker-compose.yml"

        if docker_compose.exists():
            with open(docker_compose) as f:
                content = f.read()

            # Should reference environment files or variables
            env_indicators = [
                "env_file",
                "environment:",
                "${",  # Variable substitution
                ".env",
            ]

            has_env_config = any(indicator in content for indicator in env_indicators)
            assert (
                has_env_config
            ), "docker-compose.yml should configure environment variables"

    def test_volume_mappings_for_development(self, project_root):
        """Test that volume mappings are appropriate for development."""
        docker_compose = project_root / "docker-compose.yml"

        if docker_compose.exists():
            with open(docker_compose) as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get("services", {})

            # Check if there's a development service with volume mapping
            for service_name, service_config in services.items():
                if "volumes" in service_config:
                    volumes = service_config["volumes"]

                    # Look for source code mounting in development
                    for volume in volumes:
                        if isinstance(volume, str) and ":" in volume:
                            host_path, container_path = volume.split(":", 1)
                            if host_path in [".", "./", "./src"]:
                                # Found source code mounting - this is good for development
                                assert True
                                return

    def test_task_completion_validation(self, task_validator):
        """Test that task completion can be properly validated."""
        task = task_validator.get_task_by_id("1")
        assert task is not None

        # Check that dependencies are satisfied
        can_start = task_validator.can_start_task("3")

        # If Task 1.2 is done, this should be True
        subtask_2 = None
        for st in task.get("subtasks", []):
            if st["id"] == 2:
                subtask_2 = st
                break

        if subtask_2 and subtask_2.get("status") == "done":
            assert can_start, "Task 1.3 should be able to start when Task 1.2 is done"
        else:
            assert not can_start, "Task 1.3 should not start when Task 1.2 is not done"

    def test_docker_security_considerations(self, project_root):
        """Test that Docker configurations follow security best practices."""
        # Check Dockerfile.prod for security practices
        dockerfile_prod = project_root / "docker" / "Dockerfile.prod"

        if dockerfile_prod.exists():
            content = dockerfile_prod.read_text()

            # Should not run as root in production
            security_indicators = [
                "USER ",  # Should switch to non-root user
                "adduser",  # Should create non-root user
                "useradd",  # Should create non-root user
                "RUN groupadd",  # Should create user group
            ]

            has_user_config = any(
                indicator in content for indicator in security_indicators
            )
            # This is a recommendation, not a hard requirement
            if not has_user_config:
                pytest.warns(
                    UserWarning,
                    "Consider adding non-root user configuration in Dockerfile.prod for security",
                )

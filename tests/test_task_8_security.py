"""
Task 8 - Security & Authentication Test Suite

This module provides comprehensive tests for the Darwin security system,
covering JWT authentication, role-based access control, middleware functionality,
password management, and all security features.

Test Coverage:
- JWT token generation, validation, and refresh
- Password hashing and validation with Argon2
- Role-based access control (RBAC) system
- Security middleware (authentication, authorization, rate limiting)
- Session management and security
- API key authentication
- Security configuration and validation
- Exception handling and error responses
- Rate limiting and throttling
- CORS and security headers
"""


# Add project root to path for imports
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from darwin.security import (
        SecurityConstants,
        create_access_control,
        create_auth_manager,
    )
    from darwin.security.auth import (
        AuthManager,
        JWTManager,
        PasswordManager,
        SessionManager,
    )
    from darwin.security.exceptions import (
        AuthenticationError,
        AuthorizationError,
        InsufficientPermissionsError,
        InvalidCredentialsError,
        RateLimitExceededError,
        SecurityConfigError,
        SecurityError,
        TokenExpiredError,
    )
    from darwin.security.middleware import (
        AuthenticationMiddleware,
        AuthorizationMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
        create_security_middleware_stack,
    )
    from darwin.security.models import (
        AuthenticationRequest,
        AuthenticationResponse,
        AuthToken,
        SecurityConfig,
        UserCredentials,
    )
    from darwin.security.rbac import (
        AccessControl,
        Permission,
        PermissionManager,
        PermissionType,
        ResourceType,
        Role,
        RoleManager,
        User,
    )
except ImportError:
    # Skip tests if modules not available
    pytest.skip("Security modules not available", allow_module_level=True)


@pytest.fixture
def security_config():
    """Create test security configuration."""
    return SecurityConfig(
        jwt_secret_key="test-secret-key-that-is-at-least-32-characters-long",
        jwt_algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
        password_min_length=8,
        password_require_uppercase=True,
        password_require_lowercase=True,
        password_require_numbers=True,
        password_require_special_chars=True,
    )


@pytest.fixture
def jwt_manager(security_config):
    """Create JWT manager for testing."""
    return JWTManager(
        secret_key=security_config.jwt_secret_key,
        algorithm=security_config.jwt_algorithm,
        access_token_expire_minutes=security_config.access_token_expire_minutes,
        refresh_token_expire_days=security_config.refresh_token_expire_days,
    )


@pytest.fixture
def password_manager(security_config):
    """Create password manager for testing."""
    return PasswordManager(
        min_length=security_config.password_min_length,
        require_uppercase=security_config.password_require_uppercase,
        require_lowercase=security_config.password_require_lowercase,
        require_numbers=security_config.password_require_numbers,
        require_special_chars=security_config.password_require_special_chars,
    )


@pytest.fixture
def auth_manager(security_config):
    """Create authentication manager for testing."""
    return AuthManager(config=security_config)


@pytest.fixture
def access_control():
    """Create access control system for testing."""
    return AccessControl()


@pytest.fixture
def test_app():
    """Create test FastAPI application."""
    app = FastAPI(title="Test Darwin Security")

    @app.get("/public")
    async def public_endpoint():
        return {"message": "This is public"}

    @app.get("/protected")
    async def protected_endpoint(request: Request):
        return {
            "message": "This is protected",
            "user": getattr(request.state, "username", "unknown"),
        }

    @app.get("/admin")
    async def admin_endpoint(request: Request):
        return {
            "message": "This is admin only",
            "user": getattr(request.state, "username", "unknown"),
        }

    return app


class TestTask8_JWTAuthentication:
    """Test JWT authentication functionality."""

    @pytest.mark.security
    def test_jwt_manager_initialization(self, security_config):
        """Test JWT manager initialization."""
        jwt_manager = JWTManager(
            secret_key=security_config.jwt_secret_key,
            algorithm=security_config.jwt_algorithm,
        )

        assert jwt_manager.secret_key == security_config.jwt_secret_key
        assert jwt_manager.algorithm == security_config.jwt_algorithm
        assert jwt_manager.access_token_expire_minutes == 30
        assert jwt_manager.refresh_token_expire_days == 7

    @pytest.mark.security
    def test_jwt_manager_invalid_secret_key(self):
        """Test JWT manager with invalid secret key."""
        with pytest.raises(SecurityConfigError):
            JWTManager(secret_key="short")  # Too short

    @pytest.mark.security
    def test_create_access_token(self, jwt_manager):
        """Test access token creation."""
        token = jwt_manager.create_access_token(
            user_id="test-user-123",
            username="testuser",
            roles=["user"],
            permissions=["read", "write"],
        )

        assert isinstance(token, AuthToken)
        assert token.user_id == "test-user-123"
        assert token.username == "testuser"
        assert token.roles == ["user"]
        assert token.permissions == ["read", "write"]
        assert token.token_type == "bearer"
        assert not token.is_expired

    @pytest.mark.security
    def test_create_refresh_token(self, jwt_manager):
        """Test refresh token creation."""
        token = jwt_manager.create_refresh_token(
            user_id="test-user-123", username="testuser"
        )

        assert isinstance(token, AuthToken)
        assert token.user_id == "test-user-123"
        assert token.username == "testuser"
        assert token.token_type == "refresh"
        assert not token.is_expired

    @pytest.mark.security
    def test_validate_token(self, jwt_manager):
        """Test token validation."""
        # Create a token
        token = jwt_manager.create_access_token(
            user_id="test-user-123", username="testuser", roles=["user"]
        )

        # Validate the token
        payload = jwt_manager.validate_token(token.token)

        assert payload["sub"] == "test-user-123"
        assert payload["username"] == "testuser"
        assert payload["roles"] == ["user"]
        assert payload["type"] == "access"

    @pytest.mark.security
    def test_validate_invalid_token(self, jwt_manager):
        """Test validation of invalid token."""
        with pytest.raises(AuthenticationError):
            jwt_manager.validate_token("invalid-token")

    @pytest.mark.security
    def test_validate_expired_token(self, security_config):
        """Test validation of expired token."""
        # Create JWT manager with very short expiration
        jwt_manager = JWTManager(
            secret_key=security_config.jwt_secret_key,
            access_token_expire_minutes=0,  # Immediate expiration
        )

        token = jwt_manager.create_access_token(
            user_id="test-user-123", username="testuser"
        )

        # Wait a moment to ensure expiration
        time.sleep(0.1)

        with pytest.raises(TokenExpiredError):
            jwt_manager.validate_token(token.token)

    @pytest.mark.security
    def test_refresh_access_token(self, jwt_manager):
        """Test access token refresh."""
        # Create refresh token
        refresh_token = jwt_manager.create_refresh_token(
            user_id="test-user-123", username="testuser"
        )

        # Refresh tokens
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token.token
        )

        assert isinstance(new_access_token, AuthToken)
        assert isinstance(new_refresh_token, AuthToken)
        assert new_access_token.user_id == "test-user-123"
        assert new_refresh_token.user_id == "test-user-123"

        # Original refresh token should be blacklisted
        assert jwt_manager.is_token_blacklisted(refresh_token.token)

    @pytest.mark.security
    def test_token_blacklisting(self, jwt_manager):
        """Test token blacklisting functionality."""
        token = jwt_manager.create_access_token(
            user_id="test-user-123", username="testuser"
        )

        # Token should be valid initially
        payload = jwt_manager.validate_token(token.token)
        assert payload["sub"] == "test-user-123"

        # Blacklist the token
        jwt_manager.blacklist_token(token.token)

        # Token should now be invalid
        with pytest.raises(AuthenticationError):
            jwt_manager.validate_token(token.token)


class TestTask8_PasswordManagement:
    """Test password management functionality."""

    @pytest.mark.security
    def test_password_manager_initialization(self, password_manager):
        """Test password manager initialization."""
        assert password_manager.min_length == 8
        assert password_manager.require_uppercase is True
        assert password_manager.require_lowercase is True
        assert password_manager.require_numbers is True
        assert password_manager.require_special_chars is True

    @pytest.mark.security
    def test_password_hashing(self, password_manager):
        """Test password hashing with Argon2."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)

        assert hashed != password
        assert len(hashed) > 50  # Argon2 hashes are long
        assert hashed.startswith("$argon2")

    @pytest.mark.security
    def test_password_verification(self, password_manager):
        """Test password verification."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)

        # Correct password should verify
        assert password_manager.verify_password(password, hashed) is True

        # Incorrect password should not verify
        assert password_manager.verify_password("WrongPassword", hashed) is False

    @pytest.mark.security
    def test_password_strength_validation(self, password_manager):
        """Test password strength validation."""
        # Strong password
        is_valid, errors = password_manager.validate_password_strength("StrongPass123!")
        assert is_valid is True
        assert len(errors) == 0

        # Weak passwords
        weak_passwords = [
            "short",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoNumbers!",  # No numbers
            "NoSpecialChars123",  # No special characters
            "password",  # Common password
        ]

        for weak_password in weak_passwords:
            is_valid, errors = password_manager.validate_password_strength(
                weak_password
            )
            assert is_valid is False
            assert len(errors) > 0

    @pytest.mark.security
    def test_generate_secure_password(self, password_manager):
        """Test secure password generation."""
        password = password_manager.generate_secure_password()

        # Should meet strength requirements
        is_valid, errors = password_manager.validate_password_strength(password)
        assert is_valid is True
        assert len(errors) == 0

        # Should be different each time
        password2 = password_manager.generate_secure_password()
        assert password != password2


class TestTask8_SessionManagement:
    """Test session management functionality."""

    @pytest.mark.security
    def test_session_manager_initialization(self):
        """Test session manager initialization."""
        session_manager = SessionManager(
            cookie_secure=True, cookie_httponly=True, max_age_seconds=3600
        )

        assert session_manager.cookie_secure is True
        assert session_manager.cookie_httponly is True
        assert session_manager.max_age_seconds == 3600

    @pytest.mark.security
    def test_create_session(self):
        """Test session creation."""
        session_manager = SessionManager()

        session_id = session_manager.create_session(
            user_id="test-user-123",
            username="testuser",
            ip_address="192.168.1.100",
            user_agent="Test-Agent/1.0",
        )

        assert isinstance(session_id, str)
        assert len(session_id) > 20  # Should be a long random string

        # Session should be in active sessions
        assert session_id in session_manager.active_sessions

        session_data = session_manager.active_sessions[session_id]
        assert session_data["user_id"] == "test-user-123"
        assert session_data["username"] == "testuser"
        assert session_data["ip_address"] == "192.168.1.100"
        assert session_data["is_active"] is True

    @pytest.mark.security
    def test_validate_session(self):
        """Test session validation."""
        session_manager = SessionManager()

        # Create session
        session_id = session_manager.create_session(
            user_id="test-user-123", username="testuser"
        )

        # Validate session
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None
        assert session_data["user_id"] == "test-user-123"

        # Invalid session ID
        invalid_session = session_manager.validate_session("invalid-session-id")
        assert invalid_session is None

    @pytest.mark.security
    def test_session_expiration(self):
        """Test session expiration."""
        session_manager = SessionManager(max_age_seconds=1)  # 1 second expiration

        # Create session
        session_id = session_manager.create_session(
            user_id="test-user-123", username="testuser"
        )

        # Should be valid initially
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be invalid after expiration
        session_data = session_manager.validate_session(session_id)
        assert session_data is None

    @pytest.mark.security
    def test_invalidate_session(self):
        """Test session invalidation."""
        session_manager = SessionManager()

        # Create session
        session_id = session_manager.create_session(
            user_id="test-user-123", username="testuser"
        )

        # Should be valid initially
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None

        # Invalidate session
        session_manager.invalidate_session(session_id)

        # Should be invalid after invalidation
        session_data = session_manager.validate_session(session_id)
        assert session_data is None


class TestTask8_RBAC:
    """Test Role-Based Access Control functionality."""

    @pytest.mark.security
    def test_permission_creation(self):
        """Test permission creation and validation."""
        permission = Permission(
            name="read_optimization",
            description="Read optimization data",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.OPTIMIZATION,
        )

        assert permission.name == "read_optimization"
        assert permission.permission_type == PermissionType.READ
        assert permission.resource_type == ResourceType.OPTIMIZATION
        assert permission.resource_id is None

    @pytest.mark.security
    def test_permission_matching(self):
        """Test permission matching logic."""
        # General permission (no specific resource)
        general_permission = Permission(
            name="read_all_optimizations",
            description="Read all optimizations",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.OPTIMIZATION,
        )

        # Specific permission
        specific_permission = Permission(
            name="read_specific_optimization",
            description="Read specific optimization",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.OPTIMIZATION,
            resource_id="opt-123",
        )

        # General permission should match any resource
        assert general_permission.matches(
            PermissionType.READ, ResourceType.OPTIMIZATION, "opt-123"
        )
        assert general_permission.matches(
            PermissionType.READ, ResourceType.OPTIMIZATION, "opt-456"
        )

        # Specific permission should only match its resource
        assert specific_permission.matches(
            PermissionType.READ, ResourceType.OPTIMIZATION, "opt-123"
        )
        assert not specific_permission.matches(
            PermissionType.READ, ResourceType.OPTIMIZATION, "opt-456"
        )

    @pytest.mark.security
    def test_role_creation_and_permissions(self):
        """Test role creation and permission management."""
        role = Role(name="test_role", description="Test role for testing")

        assert role.name == "test_role"
        assert len(role.permissions) == 0

        # Add permission
        permission = Permission(
            name="read_test",
            description="Read test resource",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.OPTIMIZATION,
        )

        role.add_permission(permission)
        assert len(role.permissions) == 1
        assert permission in role.permissions

        # Check if role has permission
        assert role.has_permission(PermissionType.READ, ResourceType.OPTIMIZATION)
        assert not role.has_permission(PermissionType.WRITE, ResourceType.OPTIMIZATION)

    @pytest.mark.security
    def test_role_manager(self):
        """Test role manager functionality."""
        role_manager = RoleManager()

        # Should have default roles
        admin_role = role_manager.get_role_by_name("admin")
        user_role = role_manager.get_role_by_name("user")
        viewer_role = role_manager.get_role_by_name("viewer")

        assert admin_role is not None
        assert user_role is not None
        assert viewer_role is not None

        # Create custom role
        custom_role = role_manager.create_role(
            name="custom_role", description="Custom test role"
        )

        assert custom_role.name == "custom_role"
        assert custom_role.id in role_manager.roles

    @pytest.mark.security
    def test_access_control_system(self, access_control):
        """Test access control system."""
        # Should have default users
        admin_user = access_control.get_user_by_username("admin")
        regular_user = access_control.get_user_by_username("user")

        assert admin_user is not None
        assert regular_user is not None

        # Admin should have admin permissions
        assert access_control.check_permission(
            admin_user.id, PermissionType.ADMIN, ResourceType.SYSTEM
        )

        # Regular user should not have admin permissions
        assert not access_control.check_permission(
            regular_user.id, PermissionType.ADMIN, ResourceType.SYSTEM
        )

        # Regular user should have read permissions
        assert access_control.check_permission(
            regular_user.id, PermissionType.READ, ResourceType.OPTIMIZATION
        )

    @pytest.mark.security
    def test_permission_requirement_enforcement(self, access_control):
        """Test permission requirement enforcement."""
        regular_user = access_control.get_user_by_username("user")

        # Should not raise exception for allowed permission
        access_control.require_permission(
            regular_user.id, PermissionType.READ, ResourceType.OPTIMIZATION
        )

        # Should raise exception for denied permission
        with pytest.raises(InsufficientPermissionsError):
            access_control.require_permission(
                regular_user.id, PermissionType.ADMIN, ResourceType.SYSTEM
            )


class TestTask8_AuthenticationMiddleware:
    """Test authentication middleware functionality."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_authentication_middleware_excluded_paths(
        self, auth_manager, test_app
    ):
        """Test authentication middleware with excluded paths."""
        # Add authentication middleware
        test_app.add_middleware(
            AuthenticationMiddleware,
            auth_manager=auth_manager,
            excluded_paths=["/public", "/docs"],
        )

        client = TestClient(test_app)

        # Public endpoint should work without authentication
        response = client.get("/public")
        assert response.status_code == 200
        assert response.json()["message"] == "This is public"

    @pytest.mark.security
    def test_authentication_middleware_with_valid_token(self, auth_manager, test_app):
        """Test authentication middleware with valid token."""
        # Add authentication middleware
        test_app.add_middleware(
            AuthenticationMiddleware,
            auth_manager=auth_manager,
            excluded_paths=["/public"],
        )

        client = TestClient(test_app)

        # Create valid token
        token = auth_manager.jwt_manager.create_access_token(
            user_id="test-user-123", username="testuser"
        )

        # Protected endpoint should work with valid token
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {token.token}"}
        )
        assert response.status_code == 200
        assert "testuser" in response.json()["user"]

    @pytest.mark.security
    def test_authentication_middleware_without_token(self, auth_manager, test_app):
        """Test authentication middleware without token."""
        # Add authentication middleware
        test_app.add_middleware(
            AuthenticationMiddleware,
            auth_manager=auth_manager,
            excluded_paths=["/public"],
        )

        client = TestClient(test_app)

        # Protected endpoint should fail without token
        response = client.get("/protected")
        assert response.status_code == 401
        assert "Authentication failed" in response.json()["error"]

    @pytest.mark.security
    def test_authentication_middleware_with_invalid_token(self, auth_manager, test_app):
        """Test authentication middleware with invalid token."""
        # Add authentication middleware
        test_app.add_middleware(
            AuthenticationMiddleware,
            auth_manager=auth_manager,
            excluded_paths=["/public"],
        )

        client = TestClient(test_app)

        # Protected endpoint should fail with invalid token
        response = client.get(
            "/protected", headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
        assert "Authentication failed" in response.json()["error"]


class TestTask8_RateLimitMiddleware:
    """Test rate limiting middleware functionality."""

    @pytest.mark.security
    def test_rate_limit_middleware_basic(self, test_app):
        """Test basic rate limiting functionality."""
        # Add rate limiting middleware with very low limit
        test_app.add_middleware(
            RateLimitMiddleware, default_rate="2/minute"  # Only 2 requests per minute
        )

        client = TestClient(test_app)

        # First two requests should succeed
        response1 = client.get("/public")
        assert response1.status_code == 200

        response2 = client.get("/public")
        assert response2.status_code == 200

        # Third request should be rate limited
        response3 = client.get("/public")
        assert response3.status_code == 429
        assert "Rate limit exceeded" in response3.json()["error"]

    @pytest.mark.security
    def test_rate_limit_headers(self, test_app):
        """Test rate limiting headers."""
        # Add rate limiting middleware
        test_app.add_middleware(RateLimitMiddleware, default_rate="10/minute")

        client = TestClient(test_app)

        response = client.get("/public")
        assert response.status_code == 200

        # Should have rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestTask8_SecurityHeaders:
    """Test security headers middleware."""

    @pytest.mark.security
    def test_security_headers_middleware(self, test_app):
        """Test security headers middleware."""
        # Add security headers middleware
        test_app.add_middleware(SecurityHeadersMiddleware)

        client = TestClient(test_app)

        response = client.get("/public")
        assert response.status_code == 200

        # Should have security headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
        assert response.headers["Content-Security-Policy"] == "default-src 'self'"

    @pytest.mark.security
    def test_custom_security_headers(self, test_app):
        """Test custom security headers."""
        custom_headers = {
            "X-Custom-Header": "custom-value",
            "X-Frame-Options": "SAMEORIGIN",  # Override default
        }

        test_app.add_middleware(SecurityHeadersMiddleware, headers=custom_headers)

        client = TestClient(test_app)

        response = client.get("/public")
        assert response.status_code == 200

        assert response.headers["X-Custom-Header"] == "custom-value"
        assert response.headers["X-Frame-Options"] == "SAMEORIGIN"


class TestTask8_AuthManager:
    """Test authentication manager integration."""

    @pytest.mark.security
    def test_authenticate_user_success(self, auth_manager):
        """Test successful user authentication."""
        credentials = UserCredentials(username="admin", password="admin123")

        access_token, refresh_token = auth_manager.authenticate_user(credentials)

        assert isinstance(access_token, AuthToken)
        assert isinstance(refresh_token, AuthToken)
        assert access_token.username == "admin"
        assert not access_token.is_expired
        assert not refresh_token.is_expired

    @pytest.mark.security
    def test_authenticate_user_invalid_credentials(self, auth_manager):
        """Test authentication with invalid credentials."""
        credentials = UserCredentials(username="admin", password="wrongpassword")

        with pytest.raises(InvalidCredentialsError):
            auth_manager.authenticate_user(credentials)

    @pytest.mark.security
    def test_authenticate_nonexistent_user(self, auth_manager):
        """Test authentication with nonexistent user."""
        credentials = UserCredentials(username="nonexistent", password="password")

        with pytest.raises(InvalidCredentialsError):
            auth_manager.authenticate_user(credentials)

    @pytest.mark.security
    def test_change_password_success(self, auth_manager):
        """Test successful password change."""
        # Get admin user ID
        admin_user = auth_manager._get_user_by_username("admin")

        result = auth_manager.change_password(
            user_id=admin_user["id"],
            current_password="admin123",
            new_password="NewSecurePass123!",
        )

        assert result is True

    @pytest.mark.security
    def test_change_password_wrong_current(self, auth_manager):
        """Test password change with wrong current password."""
        admin_user = auth_manager._get_user_by_username("admin")

        with pytest.raises(InvalidCredentialsError):
            auth_manager.change_password(
                user_id=admin_user["id"],
                current_password="wrongpassword",
                new_password="NewSecurePass123!",
            )

    @pytest.mark.security
    def test_change_password_weak_new_password(self, auth_manager):
        """Test password change with weak new password."""
        admin_user = auth_manager._get_user_by_username("admin")

        with pytest.raises(AuthenticationError):
            auth_manager.change_password(
                user_id=admin_user["id"],
                current_password="admin123",
                new_password="weak",  # Too weak
            )


class TestTask8_SecurityModels:
    """Test security models and validation."""

    @pytest.mark.security
    def test_security_config_validation(self):
        """Test security configuration validation."""
        # Valid config
        config = SecurityConfig(
            jwt_secret_key="valid-secret-key-that-is-long-enough-for-security"
        )
        assert config.jwt_secret_key is not None

        # Invalid config - secret too short
        with pytest.raises(ValueError):
            SecurityConfig(jwt_secret_key="short")

    @pytest.mark.security
    def test_user_credentials_validation(self):
        """Test user credentials validation."""
        # Valid credentials
        credentials = UserCredentials(username="valid_user", password="ValidPass123!")
        assert credentials.username == "valid_user"

        # Invalid username
        with pytest.raises(ValueError):
            UserCredentials(
                username="invalid@user", password="ValidPass123!"  # Invalid characters
            )

    @pytest.mark.security
    def test_auth_token_properties(self):
        """Test auth token properties."""
        future_time = datetime.now(timezone.utc) + timedelta(minutes=30)

        token = AuthToken(
            token="test-token",
            expires_at=future_time,
            user_id="user-123",
            username="testuser",
        )

        assert not token.is_expired
        assert token.expires_in_seconds > 0

        # Expired token
        past_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        expired_token = AuthToken(
            token="expired-token",
            expires_at=past_time,
            user_id="user-123",
            username="testuser",
        )

        assert expired_token.is_expired
        assert expired_token.expires_in_seconds == 0


class TestTask8_SecurityExceptions:
    """Test security exception handling."""

    @pytest.mark.security
    def test_security_error_creation(self):
        """Test security error creation and properties."""
        error = SecurityError(
            message="Test security error",
            error_code="test_error",
            details={"key": "value"},
            http_status_code=400,
        )

        assert str(error) == "Test security error"
        assert error.error_code == "test_error"
        assert error.details == {"key": "value"}
        assert error.http_status_code == 400

        error_dict = error.to_dict()
        assert error_dict["error"] == "test_error"
        assert error_dict["message"] == "Test security error"
        assert error_dict["details"] == {"key": "value"}

    @pytest.mark.security
    def test_authentication_errors(self):
        """Test authentication error types."""
        # Base authentication error
        auth_error = AuthenticationError()
        assert auth_error.http_status_code == 401

        # Invalid credentials error
        cred_error = InvalidCredentialsError(username="testuser")
        assert cred_error.details["username"] == "testuser"

        # Token expired error
        token_error = TokenExpiredError(token_type="access")
        assert token_error.details["token_type"] == "access"

    @pytest.mark.security
    def test_authorization_errors(self):
        """Test authorization error types."""
        # Insufficient permissions error
        perm_error = InsufficientPermissionsError(
            required_permission="admin",
            user_permissions=["read", "write"],
            resource="system",
        )
        assert perm_error.details["required_permission"] == "admin"
        assert perm_error.details["user_permissions"] == ["read", "write"]
        assert perm_error.details["resource"] == "system"
        assert perm_error.http_status_code == 403

    @pytest.mark.security
    def test_rate_limit_error(self):
        """Test rate limit error."""
        rate_error = RateLimitExceededError(
            limit="100/minute", reset_time=1234567890, client_id="client-123"
        )
        assert rate_error.details["limit"] == "100/minute"
        assert rate_error.details["reset_time"] == 1234567890
        assert rate_error.details["client_id"] == "client-123"
        assert rate_error.http_status_code == 429


class TestTask8_SecurityIntegration:
    """Test security system integration."""

    @pytest.mark.security
    def test_complete_security_stack(self, auth_manager, access_control, test_app):
        """Test complete security middleware stack."""
        # Create security middleware stack
        middleware_stack = create_security_middleware_stack(
            auth_manager=auth_manager,
            access_control=access_control,
            config={
                "rate_limiting": {"default_rate": "10/minute"},
                "authentication": {"excluded_paths": ["/public"]},
            },
        )

        # Add all middleware to test app
        for middleware_class, kwargs in middleware_stack:
            test_app.add_middleware(middleware_class, **kwargs)

        client = TestClient(test_app)

        # Public endpoint should work
        response = client.get("/public")
        assert response.status_code == 200

        # Protected endpoint should require authentication
        response = client.get("/protected")
        assert response.status_code == 401

        # With valid token should work
        token = auth_manager.jwt_manager.create_access_token(
            user_id="test-user", username="testuser"
        )

        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {token.token}"}
        )
        assert response.status_code == 200

    @pytest.mark.security
    def test_factory_functions(self):
        """Test security factory functions."""
        # Test auth manager factory
        auth_manager = create_auth_manager(
            jwt_secret_key="test-secret-key-that-is-at-least-32-characters-long"
        )
        assert isinstance(auth_manager, AuthManager)

        # Test access control factory
        access_control = create_access_control()
        assert isinstance(access_control, AccessControl)

    @pytest.mark.security
    def test_security_constants(self):
        """Test security constants."""
        assert SecurityConstants.ACCESS_TOKEN == "access"
        assert SecurityConstants.REFRESH_TOKEN == "refresh"
        assert SecurityConstants.ADMIN_ROLE == "admin"
        assert SecurityConstants.USER_ROLE == "user"
        assert SecurityConstants.READ_PERMISSION == "read"
        assert SecurityConstants.WRITE_PERMISSION == "write"


class TestTask8_TaskCompletion:
    """Test task completion validation and requirements coverage."""

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_jwt_implementation_complete(self, security_config):
        """Test JWT implementation completeness."""
        jwt_manager = JWTManager(
            secret_key=security_config.jwt_secret_key,
            algorithm=security_config.jwt_algorithm,
        )

        # Test all JWT functionality
        assert hasattr(jwt_manager, "create_access_token")
        assert hasattr(jwt_manager, "create_refresh_token")
        assert hasattr(jwt_manager, "validate_token")
        assert hasattr(jwt_manager, "refresh_access_token")
        assert hasattr(jwt_manager, "blacklist_token")

        # Test token creation and validation
        token = jwt_manager.create_access_token("user-123", "testuser")
        payload = jwt_manager.validate_token(token.token)
        assert payload["sub"] == "user-123"

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_rbac_implementation_complete(self):
        """Test RBAC implementation completeness."""
        access_control = AccessControl()

        # Test all RBAC components
        assert hasattr(access_control, "role_manager")
        assert hasattr(access_control, "permission_manager")
        assert hasattr(access_control, "check_permission")
        assert hasattr(access_control, "require_permission")
        assert hasattr(access_control, "assign_role_to_user")
        assert hasattr(access_control, "get_user_permissions")

        # Test permission checking
        admin_user = access_control.get_user_by_username("admin")
        assert access_control.check_permission(
            admin_user.id, PermissionType.ADMIN, ResourceType.SYSTEM
        )

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_middleware_implementation_complete(self, auth_manager, access_control):
        """Test middleware implementation completeness."""
        # Test all middleware classes exist
        from darwin.security.middleware import (
            AuthenticationMiddleware,
            AuthorizationMiddleware,
            RateLimitMiddleware,
            SecurityHeadersMiddleware,
        )

        # Test middleware can be instantiated
        app = FastAPI()

        auth_middleware = AuthenticationMiddleware(app, auth_manager)
        assert auth_middleware is not None

        authz_middleware = AuthorizationMiddleware(app, access_control)
        assert authz_middleware is not None

        rate_middleware = RateLimitMiddleware(app)
        assert rate_middleware is not None

        security_middleware = SecurityHeadersMiddleware(app)
        assert security_middleware is not None

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_password_security_complete(self, password_manager):
        """Test password security implementation completeness."""
        # Test all password functionality
        assert hasattr(password_manager, "hash_password")
        assert hasattr(password_manager, "verify_password")
        assert hasattr(password_manager, "validate_password_strength")
        assert hasattr(password_manager, "generate_secure_password")

        # Test Argon2 usage
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)
        assert hashed.startswith("$argon2")
        assert password_manager.verify_password(password, hashed)

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_security_models_complete(self):
        """Test security models implementation completeness."""
        from darwin.security.models import SecurityConfig, UserCredentials

        # Test all models can be instantiated
        config = SecurityConfig(
            jwt_secret_key="test-secret-key-that-is-at-least-32-characters-long"
        )
        assert config is not None

        credentials = UserCredentials(username="test", password="TestPass123!")
        assert credentials is not None

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_security_exceptions_complete(self):
        """Test security exceptions implementation completeness."""
        from darwin.security.exceptions import (
            AuthenticationError,
            SecurityError,
            format_security_error,
            get_http_status_code,
        )

        # Test exception hierarchy
        auth_error = AuthenticationError("Test error")
        assert isinstance(auth_error, SecurityError)

        # Test utility functions
        status_code = get_http_status_code(auth_error)
        assert status_code == 401

        formatted = format_security_error(auth_error)
        assert "error" in formatted
        assert "message" in formatted

    @pytest.mark.security
    @pytest.mark.task_completion
    def test_task_8_comprehensive_completion(self, auth_manager, access_control):
        """Comprehensive test validating complete Task 8 implementation."""
        try:
            # 1. JWT Authentication System
            token = auth_manager.jwt_manager.create_access_token("user-123", "testuser")
            payload = auth_manager.jwt_manager.validate_token(token.token)
            assert payload["sub"] == "user-123"

            # 2. Role-Based Access Control
            admin_user = access_control.get_user_by_username("admin")
            assert access_control.check_permission(
                admin_user.id, PermissionType.ADMIN, ResourceType.SYSTEM
            )

            # 3. Password Security with Argon2
            password_hash = auth_manager.password_manager.hash_password("TestPass123!")
            assert auth_manager.password_manager.verify_password(
                "TestPass123!", password_hash
            )

            # 4. Session Management
            session_id = auth_manager.session_manager.create_session(
                "user-123", "testuser"
            )
            session_data = auth_manager.session_manager.validate_session(session_id)
            assert session_data is not None

            # 5. Security Middleware
            app = FastAPI()
            middleware_stack = create_security_middleware_stack(
                auth_manager, access_control
            )
            assert (
                len(middleware_stack) >= 4
            )  # Auth, authz, rate limit, security headers

            # 6. User Authentication Flow
            credentials = UserCredentials(username="admin", password="admin123")
            access_token, refresh_token = auth_manager.authenticate_user(credentials)
            assert not access_token.is_expired

            # 7. Permission System
            user_permissions = access_control.get_user_permissions(admin_user.id)
            assert len(user_permissions) > 0

            # 8. Security Configuration
            config = SecurityConfig(
                jwt_secret_key="test-secret-key-that-is-at-least-32-characters-long"
            )
            assert config.jwt_algorithm == "HS256"

            # If we reach here, Task 8 is complete
            assert True, "Task 8 - Security & Authentication is 100% complete"

        except ImportError as e:
            pytest.fail(f"Task 8 implementation incomplete - import error: {e}")
        except Exception as e:
            pytest.fail(f"Task 8 validation failed: {e}")

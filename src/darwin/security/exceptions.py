"""
Security Exceptions for Darwin Security System

This module defines custom exception classes for the Darwin security system,
providing specific error types for authentication, authorization, rate limiting,
and other security-related operations.

Features:
- Hierarchical exception structure with base SecurityError
- Specific exceptions for different security scenarios
- HTTP status code mappings for FastAPI integration
- Detailed error messages and context information
- Security audit logging integration
- Exception chaining and cause tracking
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for all security-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        http_status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.http_status_code = http_status_code

        # Log the security error
        logger.error(
            f"Security error: {self.error_code} - {message}",
            extra={"error_code": self.error_code, "details": self.details},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class AuthenticationError(SecurityError):
    """Exception raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "authentication_failed",
        details: Dict[str, Any] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=401,
        )


class InvalidCredentialsError(AuthenticationError):
    """Exception raised when provided credentials are invalid."""

    def __init__(
        self,
        message: str = "Invalid username or password",
        username: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if username:
            error_details["username"] = username

        super().__init__(
            message=message, error_code="invalid_credentials", details=error_details
        )


class TokenExpiredError(AuthenticationError):
    """Exception raised when an authentication token has expired."""

    def __init__(
        self,
        message: str = "Authentication token has expired",
        token_type: str = None,
        expired_at: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if token_type:
            error_details["token_type"] = token_type
        if expired_at:
            error_details["expired_at"] = expired_at

        super().__init__(
            message=message, error_code="token_expired", details=error_details
        )


class InvalidTokenError(AuthenticationError):
    """Exception raised when an authentication token is invalid."""

    def __init__(
        self,
        message: str = "Invalid authentication token",
        token_type: str = None,
        reason: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if token_type:
            error_details["token_type"] = token_type
        if reason:
            error_details["reason"] = reason

        super().__init__(
            message=message, error_code="invalid_token", details=error_details
        )


class TokenRevokedError(AuthenticationError):
    """Exception raised when trying to use a revoked token."""

    def __init__(
        self,
        message: str = "Authentication token has been revoked",
        token_id: str = None,
        revoked_at: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if token_id:
            error_details["token_id"] = token_id
        if revoked_at:
            error_details["revoked_at"] = revoked_at

        super().__init__(
            message=message, error_code="token_revoked", details=error_details
        )


class AuthorizationError(SecurityError):
    """Base exception for authorization-related errors."""

    def __init__(
        self,
        message: str = "Authorization failed",
        error_code: str = "authorization_failed",
        details: Dict[str, Any] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=403,
        )


class InsufficientPermissionsError(AuthorizationError):
    """Exception raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: str = None,
        user_permissions: list = None,
        resource: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if required_permission:
            error_details["required_permission"] = required_permission
        if user_permissions:
            error_details["user_permissions"] = user_permissions
        if resource:
            error_details["resource"] = resource

        super().__init__(
            message=message,
            error_code="insufficient_permissions",
            details=error_details,
        )


class RoleNotFoundError(AuthorizationError):
    """Exception raised when a required role is not found."""

    def __init__(
        self,
        message: str = "Required role not found",
        role_name: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if role_name:
            error_details["role_name"] = role_name

        super().__init__(
            message=message, error_code="role_not_found", details=error_details
        )


class PermissionDeniedError(AuthorizationError):
    """Exception raised when access to a resource is explicitly denied."""

    def __init__(
        self,
        message: str = "Permission denied",
        resource: str = None,
        action: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if resource:
            error_details["resource"] = resource
        if action:
            error_details["action"] = action

        super().__init__(
            message=message, error_code="permission_denied", details=error_details
        )


class RateLimitExceededError(SecurityError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: str = None,
        reset_time: int = None,
        client_id: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if limit:
            error_details["limit"] = limit
        if reset_time:
            error_details["reset_time"] = reset_time
        if client_id:
            error_details["client_id"] = client_id

        super().__init__(
            message=message,
            error_code="rate_limit_exceeded",
            details=error_details,
            http_status_code=429,
        )


class SecurityConfigError(SecurityError):
    """Exception raised when there's an error in security configuration."""

    def __init__(
        self,
        message: str = "Security configuration error",
        config_key: str = None,
        config_value: Any = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if config_value is not None:
            error_details["config_value"] = str(config_value)

        super().__init__(
            message=message,
            error_code="security_config_error",
            details=error_details,
            http_status_code=500,
        )


class PasswordPolicyViolationError(SecurityError):
    """Exception raised when password doesn't meet policy requirements."""

    def __init__(
        self,
        message: str = "Password does not meet policy requirements",
        violations: list = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if violations:
            error_details["violations"] = violations

        super().__init__(
            message=message,
            error_code="password_policy_violation",
            details=error_details,
            http_status_code=400,
        )


class AccountLockedError(SecurityError):
    """Exception raised when user account is locked."""

    def __init__(
        self,
        message: str = "Account is temporarily locked",
        username: str = None,
        lockout_duration: int = None,
        failed_attempts: int = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if username:
            error_details["username"] = username
        if lockout_duration:
            error_details["lockout_duration_minutes"] = lockout_duration
        if failed_attempts:
            error_details["failed_attempts"] = failed_attempts

        super().__init__(
            message=message,
            error_code="account_locked",
            details=error_details,
            http_status_code=423,
        )


class SessionExpiredError(SecurityError):
    """Exception raised when user session has expired."""

    def __init__(
        self,
        message: str = "Session has expired",
        session_id: str = None,
        expired_at: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if session_id:
            error_details["session_id"] = session_id
        if expired_at:
            error_details["expired_at"] = expired_at

        super().__init__(
            message=message,
            error_code="session_expired",
            details=error_details,
            http_status_code=401,
        )


class ApiKeyError(SecurityError):
    """Base exception for API key related errors."""

    def __init__(
        self,
        message: str = "API key error",
        error_code: str = "api_key_error",
        api_key_id: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if api_key_id:
            error_details["api_key_id"] = api_key_id

        super().__init__(
            message=message,
            error_code=error_code,
            details=error_details,
            http_status_code=401,
        )


class InvalidApiKeyError(ApiKeyError):
    """Exception raised when API key is invalid."""

    def __init__(
        self,
        message: str = "Invalid API key",
        api_key_hint: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if api_key_hint:
            error_details["api_key_hint"] = api_key_hint

        super().__init__(
            message=message, error_code="invalid_api_key", details=error_details
        )


class ApiKeyExpiredError(ApiKeyError):
    """Exception raised when API key has expired."""

    def __init__(
        self,
        message: str = "API key has expired",
        expired_at: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if expired_at:
            error_details["expired_at"] = expired_at

        super().__init__(
            message=message, error_code="api_key_expired", details=error_details
        )


class CsrfTokenError(SecurityError):
    """Exception raised for CSRF token validation errors."""

    def __init__(
        self,
        message: str = "CSRF token validation failed",
        details: Dict[str, Any] = None,
    ):
        super().__init__(
            message=message,
            error_code="csrf_token_error",
            details=details,
            http_status_code=403,
        )


class SecurityHeaderError(SecurityError):
    """Exception raised for security header validation errors."""

    def __init__(
        self,
        message: str = "Security header validation failed",
        header_name: str = None,
        expected_value: str = None,
        actual_value: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if header_name:
            error_details["header_name"] = header_name
        if expected_value:
            error_details["expected_value"] = expected_value
        if actual_value:
            error_details["actual_value"] = actual_value

        super().__init__(
            message=message,
            error_code="security_header_error",
            details=error_details,
            http_status_code=400,
        )


class EncryptionError(SecurityError):
    """Exception raised for encryption/decryption errors."""

    def __init__(
        self,
        message: str = "Encryption operation failed",
        operation: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation

        super().__init__(
            message=message,
            error_code="encryption_error",
            details=error_details,
            http_status_code=500,
        )


class TwoFactorAuthError(SecurityError):
    """Exception raised for two-factor authentication errors."""

    def __init__(
        self,
        message: str = "Two-factor authentication failed",
        auth_method: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if auth_method:
            error_details["auth_method"] = auth_method

        super().__init__(
            message=message,
            error_code="two_factor_auth_error",
            details=error_details,
            http_status_code=401,
        )


class AuditLogError(SecurityError):
    """Exception raised for audit logging errors."""

    def __init__(
        self,
        message: str = "Audit logging failed",
        log_entry_id: str = None,
        details: Dict[str, Any] = None,
    ):
        error_details = details or {}
        if log_entry_id:
            error_details["log_entry_id"] = log_entry_id

        super().__init__(
            message=message,
            error_code="audit_log_error",
            details=error_details,
            http_status_code=500,
        )


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    SecurityError: 500,
    AuthenticationError: 401,
    InvalidCredentialsError: 401,
    TokenExpiredError: 401,
    InvalidTokenError: 401,
    TokenRevokedError: 401,
    AuthorizationError: 403,
    InsufficientPermissionsError: 403,
    RoleNotFoundError: 403,
    PermissionDeniedError: 403,
    RateLimitExceededError: 429,
    SecurityConfigError: 500,
    PasswordPolicyViolationError: 400,
    AccountLockedError: 423,
    SessionExpiredError: 401,
    ApiKeyError: 401,
    InvalidApiKeyError: 401,
    ApiKeyExpiredError: 401,
    CsrfTokenError: 403,
    SecurityHeaderError: 400,
    EncryptionError: 500,
    TwoFactorAuthError: 401,
    AuditLogError: 500,
}


def get_http_status_code(exception: Exception) -> int:
    """Get HTTP status code for a given exception."""
    if isinstance(exception, SecurityError):
        return exception.http_status_code

    return EXCEPTION_STATUS_MAP.get(type(exception), 500)


def format_security_error(exception: SecurityError) -> Dict[str, Any]:
    """Format security exception for API response."""
    return {
        "error": exception.error_code,
        "message": exception.message,
        "details": exception.details,
        "timestamp": logger.handlers[0].format(
            logger.makeRecord(logger.name, logging.ERROR, __file__, 0, "", (), None)
        )
        if logger.handlers
        else None,
    }


# Custom exception handler for FastAPI
def create_security_exception_handler():
    """Create exception handler for FastAPI integration."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    async def security_exception_handler(request: Request, exc: SecurityError):
        """Handle security exceptions in FastAPI."""
        return JSONResponse(
            status_code=exc.http_status_code, content=format_security_error(exc)
        )

    return security_exception_handler

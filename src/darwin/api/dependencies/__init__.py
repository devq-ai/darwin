"""
Darwin API Dependencies Package

This package provides FastAPI dependency functions for the Darwin genetic algorithm
optimization platform. It includes security dependencies, database dependencies,
and other common dependency injections used across API endpoints.

Modules:
- security: Authentication, authorization, and security validation dependencies
"""

from .security import (
    AdminUser,
    AuthenticatedUser,
    OptionalUser,
    ReadOnlyUser,
    RequireAdminAccess,
    RequireAdminRole,
    RequireDeleteAccess,
    RequireReadAccess,
    RequireUserRole,
    RequireWriteAccess,
    WriteUser,
    get_access_control,
    get_auth_manager,
    get_current_token,
    get_current_user,
    get_optional_user,
    rate_limit_check,
    require_permission,
    require_role,
    security_audit_log,
    set_security_managers,
    validate_api_key,
)

__all__ = [
    # Security dependencies
    "get_auth_manager",
    "get_access_control",
    "get_current_token",
    "get_current_user",
    "get_optional_user",
    "require_permission",
    "require_role",
    "validate_api_key",
    "rate_limit_check",
    "security_audit_log",
    "set_security_managers",
    # Common permission dependencies
    "RequireReadAccess",
    "RequireWriteAccess",
    "RequireDeleteAccess",
    "RequireAdminAccess",
    # Common role dependencies
    "RequireAdminRole",
    "RequireUserRole",
    # Type aliases
    "AuthenticatedUser",
    "OptionalUser",
    "AdminUser",
    "ReadOnlyUser",
    "WriteUser",
]

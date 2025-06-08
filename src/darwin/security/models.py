"""
Security Models and Data Structures for Darwin Security System

This module defines Pydantic models and data structures used throughout the Darwin
security system, including authentication tokens, user credentials, security
configuration, and API request/response models.

Features:
- Pydantic models for type safety and validation
- JWT token representation and validation
- User credential models with security constraints
- Security configuration with default values
- API request/response models for authentication endpoints
- Data serialization and deserialization
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field, field_validator
from pydantic.types import SecretStr


class SecurityConfig(BaseModel):
    """Security configuration model."""

    # JWT Configuration
    jwt_secret_key: str = Field(..., description="JWT secret key for signing tokens")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration in days"
    )
    jwt_issuer: str = Field(default="darwin-platform", description="JWT issuer")
    jwt_audience: str = Field(default="darwin-users", description="JWT audience")

    # Password Policy
    password_min_length: int = Field(
        default=8, ge=6, le=128, description="Minimum password length"
    )
    password_require_uppercase: bool = Field(
        default=True, description="Require uppercase letters"
    )
    password_require_lowercase: bool = Field(
        default=True, description="Require lowercase letters"
    )
    password_require_numbers: bool = Field(default=True, description="Require numbers")
    password_require_special_chars: bool = Field(
        default=True, description="Require special characters"
    )
    password_max_age_days: int = Field(
        default=90, description="Maximum password age in days"
    )

    # Session Configuration
    session_max_age_seconds: int = Field(
        default=3600, description="Session maximum age in seconds"
    )
    cookie_secure: bool = Field(default=True, description="Use secure cookies")
    cookie_httponly: bool = Field(default=True, description="Use HTTP-only cookies")
    cookie_samesite: str = Field(default="lax", description="Cookie SameSite policy")

    # Rate Limiting
    rate_limit_default: str = Field(
        default="100/minute", description="Default rate limit"
    )
    rate_limit_auth: str = Field(
        default="10/minute", description="Authentication rate limit"
    )
    rate_limit_burst: str = Field(default="1000/hour", description="Burst rate limit")

    # CORS Configuration
    cors_allow_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods",
    )
    cors_allow_headers: List[str] = Field(
        default=["Authorization", "Content-Type", "X-Requested-With"],
        description="Allowed CORS headers",
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORS credentials"
    )

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v):
        allowed_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        if v not in allowed_algorithms:
            raise ValueError(f"JWT algorithm must be one of: {allowed_algorithms}")
        return v


class UserCredentials(BaseModel):
    """User credentials for authentication."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: SecretStr = Field(
        ..., min_length=8, max_length=128, description="Password"
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        # Username can contain letters, numbers, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "example": {"username": "john_doe", "password": "SecurePass123!"}
        }
    }


class AuthToken(BaseModel):
    """Authentication token model."""

    token: str = Field(..., description="JWT token string")
    token_type: str = Field(default="bearer", description="Token type")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Token issue time",
    )

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def expires_in_seconds(self) -> int:
        """Get seconds until token expires."""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds()))

    model_config = {
        "json_schema_extra": {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_at": "2024-01-01T12:00:00Z",
                "user_id": "user-123",
                "username": "john_doe",
                "roles": ["user"],
                "permissions": ["read", "write"],
            }
        }
    }


class AuthenticationRequest(BaseModel):
    """Authentication request model."""

    username: str = Field(..., description="Username")
    password: SecretStr = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Extended session duration")

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "john_doe",
                "password": "SecurePass123!",
                "remember_me": False,
            }
        }
    }


class AuthenticationResponse(BaseModel):
    """Authentication response model."""

    access_token: AuthToken = Field(..., description="Access token")
    refresh_token: AuthToken = Field(..., description="Refresh token")
    user_info: Dict[str, Any] = Field(..., description="User information")
    expires_in: int = Field(..., description="Access token expiration in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": {
                    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_at": "2024-01-01T12:00:00Z",
                    "user_id": "user-123",
                    "username": "john_doe",
                },
                "refresh_token": {
                    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "refresh",
                    "expires_at": "2024-01-08T12:00:00Z",
                    "user_id": "user-123",
                    "username": "john_doe",
                },
                "user_info": {
                    "id": "user-123",
                    "username": "john_doe",
                    "roles": ["user"],
                    "permissions": ["read", "write"],
                },
                "expires_in": 1800,
            }
        }
    }


class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str = Field(..., description="Refresh token")

    model_config = {
        "json_schema_extra": {
            "example": {"refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}
        }
    }


class TokenRefreshResponse(BaseModel):
    """Token refresh response model."""

    access_token: AuthToken = Field(..., description="New access token")
    refresh_token: AuthToken = Field(..., description="New refresh token")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: SecretStr = Field(..., description="Current password")
    new_password: SecretStr = Field(..., description="New password")
    confirm_password: SecretStr = Field(..., description="Password confirmation")

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "new_password" in info.data:
            new_password = info.data["new_password"]
            if v.get_secret_value() != new_password.get_secret_value():
                raise ValueError("Passwords do not match")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "current_password": "OldPass123!",
                "new_password": "NewSecurePass456!",
                "confirm_password": "NewSecurePass456!",
            }
        }
    }


class UserRegistrationRequest(BaseModel):
    """User registration request model."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: SecretStr = Field(
        ..., min_length=8, max_length=128, description="Password"
    )
    confirm_password: SecretStr = Field(..., description="Password confirmation")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v.lower()

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data:
            password = info.data["password"]
            if v.get_secret_value() != password.get_secret_value():
                raise ValueError("Passwords do not match")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "jane_doe",
                "email": "jane@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "first_name": "Jane",
                "last_name": "Doe",
            }
        }
    }


class UserInfo(BaseModel):
    """User information model."""

    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[EmailStr] = Field(None, description="Email address")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="User active status")
    is_verified: bool = Field(default=False, description="Email verification status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "username": "john_doe",
                "email": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "is_active": True,
                "roles": ["user"],
                "permissions": ["read", "write"],
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T11:00:00Z",
                "last_login": "2024-01-01T11:30:00Z",
            }
        }
    }


class ApiKeyRequest(BaseModel):
    """API key creation request model."""

    name: str = Field(..., max_length=100, description="API key name")
    description: Optional[str] = Field(
        None, max_length=500, description="API key description"
    )
    permissions: List[str] = Field(
        default_factory=list, description="API key permissions"
    )
    expires_at: Optional[datetime] = Field(None, description="API key expiration")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "My API Key",
                "description": "API key for automation scripts",
                "permissions": ["read", "write"],
                "expires_at": "2024-12-31T23:59:59Z",
            }
        }
    }


class ApiKeyResponse(BaseModel):
    """API key creation response model."""

    id: str = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    key: str = Field(..., description="API key value (only shown once)")
    permissions: List[str] = Field(..., description="API key permissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "api-key-123",
                "name": "My API Key",
                "key": "ak_1234567890abcdef",
                "permissions": ["read", "write"],
                "created_at": "2024-01-01T00:00:00Z",
                "expires_at": "2024-12-31T23:59:59Z",
            }
        }
    }


class SecurityAuditLog(BaseModel):
    """Security audit log entry model."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Log entry ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    event_type: str = Field(..., description="Event type")
    action: str = Field(..., description="Action performed")
    resource: Optional[str] = Field(None, description="Resource accessed")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    success: bool = Field(..., description="Whether action was successful")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "audit-123",
                "timestamp": "2024-01-01T12:00:00Z",
                "user_id": "user-123",
                "username": "john_doe",
                "event_type": "authentication",
                "action": "login",
                "resource": "dashboard",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "success": True,
                "details": {"method": "password"},
            }
        }
    }


class SecurityMetrics(BaseModel):
    """Security metrics model."""

    total_users: int = Field(..., description="Total number of users")
    active_sessions: int = Field(..., description="Number of active sessions")
    failed_login_attempts: int = Field(
        ..., description="Failed login attempts in last hour"
    )
    rate_limited_requests: int = Field(
        ..., description="Rate limited requests in last hour"
    )
    security_events: int = Field(..., description="Security events in last 24 hours")
    api_key_usage: int = Field(..., description="API key usage in last hour")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Metrics timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_users": 150,
                "active_sessions": 45,
                "failed_login_attempts": 3,
                "rate_limited_requests": 12,
                "security_events": 8,
                "api_key_usage": 234,
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "authentication_failed",
                "detail": "Invalid username or password",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789",
            }
        }
    }


# Utility models for internal use
class TokenPayload(BaseModel):
    """JWT token payload model for internal use."""

    sub: str = Field(..., description="Subject (user ID)")
    username: str = Field(..., description="Username")
    iat: datetime = Field(..., description="Issued at")
    exp: datetime = Field(..., description="Expires at")
    iss: str = Field(..., description="Issuer")
    aud: str = Field(..., description="Audience")
    type: str = Field(..., description="Token type")
    jti: str = Field(..., description="JWT ID")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")


class SessionData(BaseModel):
    """Session data model."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    is_active: bool = Field(default=True, description="Session active status")
    expires_at: datetime = Field(..., description="Session expiration time")

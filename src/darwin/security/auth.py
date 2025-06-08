"""
Authentication Manager for Darwin Security System

This module provides comprehensive authentication functionality including JWT token
management, password hashing and validation, session management, and user authentication.

Features:
- JWT token generation, validation, and refresh
- Secure password hashing using Argon2
- Session management with secure cookies
- Multi-factor authentication support
- Token blacklisting and revocation
- Password strength validation
- Account lockout protection
- Audit logging for security events
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import jwt
except ImportError:
    jwt = None
from argon2 import PasswordHasher
from argon2.exceptions import HashingError, VerificationError, VerifyMismatchError

from .exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    SecurityConfigError,
    TokenExpiredError,
)
from .models import AuthToken, SecurityConfig, UserCredentials

logger = logging.getLogger(__name__)


class JWTManager:
    """Manages JWT token creation, validation, and refresh."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        issuer: str = "darwin-platform",
        audience: str = "darwin-users",
    ):
        if not secret_key:
            raise SecurityConfigError("JWT secret key is required")

        if len(secret_key) < 32:
            raise SecurityConfigError(
                "JWT secret key must be at least 32 characters long"
            )

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.audience = audience
        self.blacklisted_tokens = set()  # In production, use Redis or database

    def create_access_token(
        self,
        user_id: str,
        username: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        extra_claims: Dict[str, Any] = None,
    ) -> AuthToken:
        """
        Create a new access token.

        Args:
            user_id: Unique user identifier
            username: Username
            roles: List of user roles
            permissions: List of user permissions
            extra_claims: Additional claims to include

        Returns:
            AuthToken object containing token data
        """
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "username": username,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "aud": self.audience,
            "type": "access",
            "jti": secrets.token_urlsafe(32),  # JWT ID for tracking
            "roles": roles or [],
            "permissions": permissions or [],
        }

        if extra_claims:
            payload.update(extra_claims)

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            return AuthToken(
                token=token,
                token_type="bearer",
                expires_at=expire,
                user_id=user_id,
                username=username,
                roles=roles or [],
                permissions=permissions or [],
            )

        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise AuthenticationError("Failed to create access token")

    def create_refresh_token(self, user_id: str, username: str) -> AuthToken:
        """
        Create a new refresh token.

        Args:
            user_id: Unique user identifier
            username: Username

        Returns:
            AuthToken object containing refresh token data
        """
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "username": username,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "aud": self.audience,
            "type": "refresh",
            "jti": secrets.token_urlsafe(32),
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            return AuthToken(
                token=token,
                token_type="refresh",
                expires_at=expire,
                user_id=user_id,
                username=username,
            )

        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise AuthenticationError("Failed to create refresh token")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            TokenExpiredError: If token is expired
            AuthenticationError: If token is invalid
        """
        if token in self.blacklisted_tokens:
            raise AuthenticationError("Token has been revoked")

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
            )

            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(
                timezone.utc
            ):
                raise TokenExpiredError("Token has expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise AuthenticationError("Token validation failed")

    def refresh_access_token(self, refresh_token: str) -> Tuple[AuthToken, AuthToken]:
        """
        Create new access and refresh tokens using a refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        try:
            payload = self.validate_token(refresh_token)

            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type for refresh")

            user_id = payload["sub"]
            username = payload["username"]

            # Blacklist the old refresh token
            self.blacklist_token(refresh_token)

            # Create new tokens
            new_access_token = self.create_access_token(user_id, username)
            new_refresh_token = self.create_refresh_token(user_id, username)

            return new_access_token, new_refresh_token

        except TokenExpiredError:
            raise TokenExpiredError("Refresh token has expired")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Token refresh failed")

    def blacklist_token(self, token: str):
        """Add a token to the blacklist."""
        self.blacklisted_tokens.add(token)
        logger.info("Token added to blacklist")

    def is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted."""
        return token in self.blacklisted_tokens

    def revoke_all_tokens_for_user(self, user_id: str):
        """Revoke all tokens for a specific user."""
        # In production, this would query the database for user tokens
        logger.info(f"All tokens revoked for user: {user_id}")


class PasswordManager:
    """Manages password hashing, validation, and security policies."""

    def __init__(
        self,
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_numbers: bool = True,
        require_special_chars: bool = True,
        max_age_days: int = 90,
    ):
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_numbers = require_numbers
        self.require_special_chars = require_special_chars
        self.max_age_days = max_age_days

        # Initialize Argon2 password hasher
        self.hasher = PasswordHasher(
            time_cost=3,  # Number of iterations
            memory_cost=65536,  # Memory usage in KB
            parallelism=1,  # Number of parallel threads
            hash_len=32,  # Hash output length
            salt_len=16,  # Salt length
        )

        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        try:
            return self.hasher.hash(password)
        except HashingError as e:
            logger.error(f"Password hashing failed: {e}")
            raise AuthenticationError("Password hashing failed")

    def verify_password(self, password, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain text password (str or SecretStr)
            hashed_password: Stored password hash

        Returns:
            True if password matches, False otherwise
        """
        try:
            # Handle SecretStr objects
            if hasattr(password, "get_secret_value"):
                password_str = password.get_secret_value()
            else:
                password_str = str(password)

            self.hasher.verify(hashed_password, password_str)

            # Check if hash needs rehashing (Argon2 upgrade)
            if self.hasher.check_needs_rehash(hashed_password):
                logger.info("Password hash needs upgrade")
                # In production, you would update the stored hash

            return True

        except VerifyMismatchError:
            return False
        except VerificationError as e:
            logger.error(f"Password verification error: {e}")
            return False

    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if len(password) < self.min_length:
            errors.append(
                f"Password must be at least {self.min_length} characters long"
            )

        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if self.require_special_chars and not any(
            c in self.special_chars for c in password
        ):
            errors.append("Password must contain at least one special character")

        # Check for common weak patterns
        if password.lower() in ["password", "123456", "qwerty", "admin"]:
            errors.append("Password is too common")

        # Check for repeated characters
        if len(set(password)) < len(password) / 2:
            errors.append("Password contains too many repeated characters")

        return len(errors) == 0, errors

    def generate_secure_password(self, length: int = 12) -> str:
        """Generate a cryptographically secure random password."""
        import string

        characters = string.ascii_letters + string.digits + self.special_chars
        password = "".join(secrets.choice(characters) for _ in range(length))

        # Ensure it meets requirements
        is_valid, _ = self.validate_password_strength(password)
        if not is_valid:
            return self.generate_secure_password(length)  # Retry

        return password


class SessionManager:
    """Manages user sessions and session security."""

    def __init__(
        self,
        cookie_secure: bool = True,
        cookie_httponly: bool = True,
        cookie_samesite: str = "lax",
        max_age_seconds: int = 3600,
    ):
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
        self.max_age_seconds = max_age_seconds
        self.active_sessions = {}  # In production, use Redis or database

    def create_session(
        self,
        user_id: str,
        username: str,
        ip_address: str = None,
        user_agent: str = None,
    ) -> str:
        """
        Create a new user session.

        Args:
            user_id: User identifier
            username: Username
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)

        session_data = {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "is_active": True,
        }

        self.active_sessions[session_id] = session_data

        logger.info(f"Session created for user: {username}")
        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session and update last activity.

        Args:
            session_id: Session identifier

        Returns:
            Session data if valid, None otherwise
        """
        session_data = self.active_sessions.get(session_id)

        if not session_data or not session_data.get("is_active"):
            return None

        # Check if session has expired
        created_at = session_data["created_at"]
        if datetime.now(timezone.utc) - created_at > timedelta(
            seconds=self.max_age_seconds
        ):
            self.invalidate_session(session_id)
            return None

        # Update last activity
        session_data["last_activity"] = datetime.now(timezone.utc)

        return session_data

    def invalidate_session(self, session_id: str):
        """Invalidate a session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["is_active"] = False
            logger.info(f"Session invalidated: {session_id}")

    def invalidate_all_sessions_for_user(self, user_id: str):
        """Invalidate all sessions for a user."""
        for session_id, session_data in self.active_sessions.items():
            if session_data.get("user_id") == user_id:
                session_data["is_active"] = False

        logger.info(f"All sessions invalidated for user: {user_id}")

    def get_session_cookie_config(self) -> Dict[str, Any]:
        """Get session cookie configuration."""
        return {
            "httponly": self.cookie_httponly,
            "secure": self.cookie_secure,
            "samesite": self.cookie_samesite,
            "max_age": self.max_age_seconds,
        }


class AuthManager:
    """Main authentication manager that coordinates all auth operations."""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

        # Initialize managers
        self.jwt_manager = JWTManager(
            secret_key=self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
            access_token_expire_minutes=self.config.access_token_expire_minutes,
            refresh_token_expire_days=self.config.refresh_token_expire_days,
        )

        self.password_manager = PasswordManager(
            min_length=self.config.password_min_length,
            require_uppercase=self.config.password_require_uppercase,
            require_lowercase=self.config.password_require_lowercase,
            require_numbers=self.config.password_require_numbers,
            require_special_chars=self.config.password_require_special_chars,
        )

        self.session_manager = SessionManager(
            cookie_secure=self.config.cookie_secure,
            cookie_httponly=self.config.cookie_httponly,
            cookie_samesite=self.config.cookie_samesite,
            max_age_seconds=self.config.session_max_age_seconds,
        )

        # Track failed login attempts for account lockout
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15

    def authenticate_user(
        self,
        credentials: UserCredentials,
        ip_address: str = None,
        user_agent: str = None,
    ) -> Tuple[AuthToken, AuthToken]:
        """
        Authenticate a user and return access and refresh tokens.

        Args:
            credentials: User credentials
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Tuple of (access_token, refresh_token)
        """
        username = credentials.username
        password = credentials.password

        # Check for account lockout
        if self._is_account_locked(username):
            raise AuthenticationError(
                "Account is temporarily locked due to too many failed attempts"
            )

        try:
            # In production, this would query the database
            user_data = self._get_user_by_username(username)

            if not user_data:
                self._record_failed_attempt(username)
                raise InvalidCredentialsError("Invalid username or password")

            # Verify password
            if not self.password_manager.verify_password(
                password, user_data["password_hash"]
            ):
                self._record_failed_attempt(username)
                raise InvalidCredentialsError("Invalid username or password")

            # Clear failed attempts on successful login
            self._clear_failed_attempts(username)

            # Create tokens
            access_token = self.jwt_manager.create_access_token(
                user_id=user_data["id"],
                username=username,
                roles=user_data.get("roles", []),
                permissions=user_data.get("permissions", []),
            )

            refresh_token = self.jwt_manager.create_refresh_token(
                user_id=user_data["id"], username=username
            )

            # Create session
            session_id = self.session_manager.create_session(
                user_id=user_data["id"],
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            logger.info(f"User authenticated successfully: {username}")

            return access_token, refresh_token

        except (InvalidCredentialsError, AuthenticationError):
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError("Authentication failed")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate an authentication token."""
        return self.jwt_manager.validate_token(token)

    def refresh_tokens(self, refresh_token: str) -> Tuple[AuthToken, AuthToken]:
        """Refresh access and refresh tokens."""
        return self.jwt_manager.refresh_access_token(refresh_token)

    def logout_user(self, access_token: str, refresh_token: str = None):
        """Logout a user by blacklisting their tokens."""
        self.jwt_manager.blacklist_token(access_token)
        if refresh_token:
            self.jwt_manager.blacklist_token(refresh_token)

        logger.info("User logged out successfully")

    def change_password(
        self, user_id: str, current_password: str, new_password: str
    ) -> bool:
        """
        Change a user's password.

        Args:
            user_id: User identifier
            current_password: Current password
            new_password: New password

        Returns:
            True if successful
        """
        # Validate new password strength
        is_valid, errors = self.password_manager.validate_password_strength(
            new_password
        )
        if not is_valid:
            raise AuthenticationError(
                f"Password validation failed: {', '.join(errors)}"
            )

        # In production, this would update the database
        user_data = self._get_user_by_id(user_id)

        if not self.password_manager.verify_password(
            current_password, user_data["password_hash"]
        ):
            raise InvalidCredentialsError("Current password is incorrect")

        # Hash new password
        new_password_hash = self.password_manager.hash_password(new_password)

        # Update password in database (simulated)
        logger.info(f"Password changed for user: {user_id}")

        # Revoke all existing tokens to force re-authentication
        self.jwt_manager.revoke_all_tokens_for_user(user_id)
        self.session_manager.invalidate_all_sessions_for_user(user_id)

        return True

    def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data by username (mock implementation)."""
        # In production, this would query the database
        mock_users = {
            "admin": {
                "id": "admin-001",
                "username": "admin",
                "password_hash": self.password_manager.hash_password("admin123"),
                "roles": ["admin"],
                "permissions": ["read", "write", "delete", "admin"],
            },
            "user": {
                "id": "user-001",
                "username": "user",
                "password_hash": self.password_manager.hash_password("user123"),
                "roles": ["user"],
                "permissions": ["read", "write"],
            },
        }

        return mock_users.get(username)

    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by user ID (mock implementation)."""
        # In production, this would query the database
        for user in self._get_all_users():
            if user["id"] == user_id:
                return user
        return None

    def _get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users (mock implementation)."""
        return [self._get_user_by_username("admin"), self._get_user_by_username("user")]

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_attempts:
            return False

        attempts_data = self.failed_attempts[username]

        if attempts_data["count"] >= self.max_failed_attempts:
            lockout_time = attempts_data["last_attempt"] + timedelta(
                minutes=self.lockout_duration_minutes
            )
            if datetime.now(timezone.utc) < lockout_time:
                return True
            else:
                # Lockout period expired, clear attempts
                self._clear_failed_attempts(username)

        return False

    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        now = datetime.now(timezone.utc)

        if username in self.failed_attempts:
            self.failed_attempts[username]["count"] += 1
            self.failed_attempts[username]["last_attempt"] = now
        else:
            self.failed_attempts[username] = {"count": 1, "last_attempt": now}

        logger.warning(f"Failed login attempt for user: {username}")

    def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts for a user."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]

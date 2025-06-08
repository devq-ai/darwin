"""
Security Utilities for Darwin Security System

This module provides utility classes and functions for security operations,
including password validation, token validation, security logging, and
general security utilities.

Features:
- Password strength validation
- Token validation utilities
- Security event logging
- Security helper functions
- Rate limiting utilities
- Cryptographic utilities
"""

import hashlib
import hmac
import logging
import re
import secrets
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import logfire

logger = logging.getLogger(__name__)


class PasswordValidator:
    """Utility class for password strength validation."""

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 128,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_numbers: bool = True,
        require_special_chars: bool = True,
        forbidden_patterns: List[str] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_numbers = require_numbers
        self.require_special_chars = require_special_chars
        self.forbidden_patterns = forbidden_patterns or []

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength and return validation result and errors."""
        errors = []

        # Length validation
        if len(password) < self.min_length:
            errors.append(
                f"Password must be at least {self.min_length} characters long"
            )
        if len(password) > self.max_length:
            errors.append(
                f"Password must be no more than {self.max_length} characters long"
            )

        # Character requirements
        if self.require_uppercase and not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        if self.require_lowercase and not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        if self.require_numbers and not re.search(r"\d", password):
            errors.append("Password must contain at least one number")
        if self.require_special_chars and not re.search(
            r'[!@#$%^&*(),.?":{}|<>]', password
        ):
            errors.append("Password must contain at least one special character")

        # Forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                errors.append(f"Password contains forbidden pattern: {pattern}")

        # Common weak passwords
        weak_passwords = [
            "password",
            "123456",
            "qwerty",
            "abc123",
            "password123",
            "admin",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
        ]
        if password.lower() in weak_passwords:
            errors.append("Password is too common and easily guessable")

        return len(errors) == 0, errors

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a cryptographically secure random password."""
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(secrets.choice(characters) for _ in range(length))


class TokenValidator:
    """Utility class for token validation and security checks."""

    @staticmethod
    def is_valid_jwt_format(token: str) -> bool:
        """Check if token has valid JWT format (3 parts separated by dots)."""
        parts = token.split(".")
        return len(parts) == 3 and all(len(part) > 0 for part in parts)

    @staticmethod
    def extract_claims_without_verification(token: str) -> Optional[Dict[str, Any]]:
        """Extract claims from JWT token without signature verification (for inspection only)."""
        import base64
        import json

        try:
            if not TokenValidator.is_valid_jwt_format(token):
                return None

            # Get payload (second part)
            payload_part = token.split(".")[1]
            # Add padding if needed
            payload_part += "=" * (4 - len(payload_part) % 4)

            # Decode base64
            payload_bytes = base64.urlsafe_b64decode(payload_part)
            return json.loads(payload_bytes)
        except Exception:
            return None

    @staticmethod
    def is_token_expired(token: str) -> bool:
        """Check if JWT token is expired (without verification)."""
        claims = TokenValidator.extract_claims_without_verification(token)
        if not claims or "exp" not in claims:
            return True

        exp_timestamp = claims["exp"]
        current_timestamp = datetime.now(timezone.utc).timestamp()
        return current_timestamp >= exp_timestamp


class SecurityLogger:
    """Security-focused logging utility with structured logging."""

    def __init__(self, logger_name: str = "darwin.security"):
        self.logger = logging.getLogger(logger_name)
        logfire.configure(send_to_logfire=False)  # Configure for local logging

    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str = None,
        user_agent: str = None,
        method: str = "password",
        additional_info: Dict[str, Any] = None,
    ):
        """Log authentication attempt with structured data."""
        event_data = {
            "event_type": "authentication",
            "username": username,
            "success": success,
            "method": method,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "additional_info": additional_info or {},
        }

        if success:
            self.logger.info("Authentication successful", extra=event_data)
        else:
            self.logger.warning("Authentication failed", extra=event_data)

    def log_authorization_check(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        ip_address: str = None,
    ):
        """Log authorization check."""
        event_data = {
            "event_type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if success:
            self.logger.info("Authorization granted", extra=event_data)
        else:
            self.logger.warning("Authorization denied", extra=event_data)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        user_id: str = None,
        ip_address: str = None,
        additional_data: Dict[str, Any] = None,
    ):
        """Log general security event."""
        event_data = {
            "event_type": event_type,
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "additional_data": additional_data or {},
        }

        if severity == "critical":
            self.logger.critical(message, extra=event_data)
        elif severity == "high":
            self.logger.error(message, extra=event_data)
        elif severity == "medium":
            self.logger.warning(message, extra=event_data)
        else:
            self.logger.info(message, extra=event_data)


class SecurityUtils:
    """General security utility functions."""

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_api_key(prefix: str = "ak") -> str:
        """Generate a secure API key with prefix."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

    @staticmethod
    def hash_sensitive_data(data: str, salt: str = None) -> Tuple[str, str]:
        """Hash sensitive data with salt for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)

        hash_obj = hashlib.pbkdf2_hmac("sha256", data.encode(), salt.encode(), 100000)
        return hash_obj.hex(), salt

    @staticmethod
    def verify_hash(data: str, hashed_data: str, salt: str) -> bool:
        """Verify hashed data against original."""
        hash_obj = hashlib.pbkdf2_hmac("sha256", data.encode(), salt.encode(), 100000)
        return hmac.compare_digest(hash_obj.hex(), hashed_data)

    @staticmethod
    def mask_sensitive_info(
        data: str, mask_char: str = "*", visible_chars: int = 4
    ) -> str:
        """Mask sensitive information for logging/display."""
        if len(data) <= visible_chars:
            return mask_char * len(data)

        visible_part = data[:visible_chars]
        masked_part = mask_char * (len(data) - visible_chars)
        return visible_part + masked_part

    @staticmethod
    def is_safe_redirect_url(url: str, allowed_hosts: List[str]) -> bool:
        """Check if redirect URL is safe (prevents open redirect attacks)."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            # Allow relative URLs
            if not parsed.netloc:
                return True

            # Check if host is in allowed list
            return parsed.netloc in allowed_hosts
        except Exception:
            return False

    @staticmethod
    def sanitize_user_input(input_str: str) -> str:
        """Basic input sanitization for user-provided strings."""
        if not isinstance(input_str, str):
            return ""

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', "", input_str)
        # Limit length
        sanitized = sanitized[:1000]
        # Strip whitespace
        return sanitized.strip()

    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token for form protection."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return hmac.compare_digest(a, b)


class RateLimitUtils:
    """Utilities for rate limiting operations."""

    @staticmethod
    def parse_rate_limit(rate_string: str) -> Tuple[int, int]:
        """Parse rate limit string like '100/minute' into (requests, seconds)."""
        try:
            parts = rate_string.split("/")
            if len(parts) != 2:
                raise ValueError("Invalid rate limit format")

            requests = int(parts[0])
            time_unit = parts[1].lower()

            time_mapping = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
                "day": 86400,
            }

            if time_unit not in time_mapping:
                raise ValueError(f"Unknown time unit: {time_unit}")

            return requests, time_mapping[time_unit]
        except Exception as e:
            logger.error(f"Failed to parse rate limit '{rate_string}': {e}")
            return 100, 60  # Default fallback

    @staticmethod
    def calculate_reset_time(current_time: datetime, window_seconds: int) -> datetime:
        """Calculate when the rate limit window resets."""
        timestamp = int(current_time.timestamp())
        reset_timestamp = timestamp + window_seconds
        return datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)


# Global instances for convenience
default_password_validator = PasswordValidator()
security_logger = SecurityLogger()
security_utils = SecurityUtils()
rate_limit_utils = RateLimitUtils()

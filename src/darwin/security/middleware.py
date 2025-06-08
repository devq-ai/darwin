"""
Security Middleware for Darwin FastAPI Application

This module provides comprehensive security middleware for the Darwin genetic algorithm
platform, including authentication, authorization, rate limiting, CORS handling,
and security headers enforcement.

Features:
- JWT token authentication middleware
- Role-based authorization middleware
- Rate limiting with configurable limits
- CORS handling with security considerations
- Security headers enforcement (HSTS, CSP, etc.)
- Request/response logging for security auditing
- IP-based access control
- API key authentication support
"""

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .auth import AuthManager
from .exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    TokenExpiredError,
)
from .rbac import AccessControl, PermissionType, ResourceType

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT token authentication."""

    def __init__(
        self,
        app,
        auth_manager: AuthManager,
        excluded_paths: List[str] = None,
        api_key_header: str = "X-API-Key",
        bearer_token_header: str = "Authorization",
    ):
        super().__init__(app)
        self.auth_manager = auth_manager
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
        ]
        self.api_key_header = api_key_header
        self.bearer_token_header = bearer_token_header

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process authentication for incoming requests."""
        path = request.url.path

        # Skip authentication for excluded paths
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return await call_next(request)

        try:
            # Extract and validate authentication token
            user_info = await self._authenticate_request(request)

            # Add user information to request state
            request.state.user_id = user_info["sub"]
            request.state.username = user_info["username"]
            request.state.roles = user_info.get("roles", [])
            request.state.permissions = user_info.get("permissions", [])
            request.state.authenticated = True

            # Log successful authentication
            logger.info(f"Authenticated user: {user_info['username']} for path: {path}")

        except AuthenticationError as e:
            logger.warning(f"Authentication failed for path {path}: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication failed", "detail": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )
        except TokenExpiredError:
            logger.warning(f"Expired token for path {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Token expired",
                    "detail": "Please refresh your token",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Authentication error for path {path}: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal authentication error"},
            )

        # Continue to next middleware/endpoint
        response = await call_next(request)

        # Add authentication headers to response
        response.headers["X-Authenticated-User"] = request.state.username

        return response

    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Extract and validate authentication credentials from request."""
        # Try Bearer token first
        authorization = request.headers.get(self.bearer_token_header)
        if authorization:
            if not authorization.startswith("Bearer "):
                raise AuthenticationError("Invalid authorization header format")

            token = authorization.split(" ", 1)[1]
            return self.auth_manager.validate_token(token)

        # Try API key
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            return await self._validate_api_key(api_key)

        # No authentication provided
        raise AuthenticationError("No authentication credentials provided")

    async def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key (placeholder implementation)."""
        # In production, this would validate against a database
        valid_api_keys = {
            "admin-key-123": {
                "sub": "api-admin",
                "username": "api-admin",
                "roles": ["admin"],
                "permissions": ["admin"],
            },
            "user-key-456": {
                "sub": "api-user",
                "username": "api-user",
                "roles": ["user"],
                "permissions": ["read", "write"],
            },
        }

        if api_key in valid_api_keys:
            return valid_api_keys[api_key]

        raise AuthenticationError("Invalid API key")


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware for role-based authorization."""

    def __init__(
        self,
        app,
        access_control: AccessControl,
        route_permissions: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(app)
        self.access_control = access_control
        self.route_permissions = route_permissions or {}

        # Default route permissions
        self._setup_default_permissions()

    def _setup_default_permissions(self):
        """Setup default route permissions."""
        default_permissions = {
            # Admin routes
            "/api/v1/admin/*": {
                "permission_type": PermissionType.ADMIN,
                "resource_type": ResourceType.SYSTEM,
            },
            "/api/v1/users/*": {
                "permission_type": PermissionType.MANAGE,
                "resource_type": ResourceType.USER,
            },
            # Optimization routes
            "/api/v1/optimizations": {
                "GET": {
                    "permission_type": PermissionType.READ,
                    "resource_type": ResourceType.OPTIMIZATION,
                },
                "POST": {
                    "permission_type": PermissionType.CREATE,
                    "resource_type": ResourceType.OPTIMIZATION,
                },
            },
            "/api/v1/optimizations/*": {
                "GET": {
                    "permission_type": PermissionType.READ,
                    "resource_type": ResourceType.OPTIMIZATION,
                },
                "PUT": {
                    "permission_type": PermissionType.WRITE,
                    "resource_type": ResourceType.OPTIMIZATION,
                },
                "DELETE": {
                    "permission_type": PermissionType.DELETE,
                    "resource_type": ResourceType.OPTIMIZATION,
                },
            },
            # Template routes
            "/api/v1/templates": {
                "GET": {
                    "permission_type": PermissionType.READ,
                    "resource_type": ResourceType.TEMPLATE,
                },
                "POST": {
                    "permission_type": PermissionType.CREATE,
                    "resource_type": ResourceType.TEMPLATE,
                },
            },
            # Dashboard routes
            "/api/v1/dashboard/*": {
                "permission_type": PermissionType.READ,
                "resource_type": ResourceType.DASHBOARD,
            },
        }

        self.route_permissions.update(default_permissions)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process authorization for incoming requests."""
        # Skip if not authenticated
        if not getattr(request.state, "authenticated", False):
            return await call_next(request)

        path = request.url.path
        method = request.method
        user_id = request.state.user_id

        try:
            # Check if route requires specific permissions
            required_permission = self._get_required_permission(path, method)

            if required_permission:
                permission_type = required_permission["permission_type"]
                resource_type = required_permission["resource_type"]
                resource_id = required_permission.get("resource_id")

                # Extract resource ID from path if needed
                if not resource_id:
                    resource_id = self._extract_resource_id(path)

                # Check permission
                if not self.access_control.check_permission(
                    user_id, permission_type, resource_type, resource_id
                ):
                    logger.warning(
                        f"Authorization failed for user {request.state.username} "
                        f"on {method} {path}: insufficient permissions"
                    )

                    # Audit the access attempt
                    self.access_control.audit_access(
                        user_id=user_id,
                        action=f"{method} {path}",
                        resource=f"{resource_type}:{resource_id or 'all'}",
                        granted=False,
                        ip_address=self._get_client_ip(request),
                    )

                    raise InsufficientPermissionsError(
                        f"Insufficient permissions for {method} {path}"
                    )

                # Log successful authorization
                logger.info(
                    f"Authorized user {request.state.username} for {method} {path}"
                )

                # Audit the successful access
                self.access_control.audit_access(
                    user_id=user_id,
                    action=f"{method} {path}",
                    resource=f"{resource_type}:{resource_id or 'all'}",
                    granted=True,
                    ip_address=self._get_client_ip(request),
                )

        except InsufficientPermissionsError as e:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Forbidden", "detail": str(e)},
            )
        except Exception as e:
            logger.error(f"Authorization error for {method} {path}: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal authorization error"},
            )

        return await call_next(request)

    def _get_required_permission(
        self, path: str, method: str
    ) -> Optional[Dict[str, Any]]:
        """Get required permission for a route."""
        # Try exact path match first
        if path in self.route_permissions:
            route_config = self.route_permissions[path]
            if isinstance(route_config, dict) and method in route_config:
                return route_config[method]
            elif not isinstance(route_config, dict):
                return route_config

        # Try wildcard matches
        for route_pattern, config in self.route_permissions.items():
            if route_pattern.endswith("/*"):
                prefix = route_pattern[:-2]
                if path.startswith(prefix):
                    if isinstance(config, dict) and method in config:
                        return config[method]
                    elif not isinstance(config, dict):
                        return config

        return None

    def _extract_resource_id(self, path: str) -> Optional[str]:
        """Extract resource ID from path."""
        # Simple extraction for common patterns like /api/v1/optimizations/{id}
        path_parts = path.strip("/").split("/")

        # Look for UUID-like patterns or numeric IDs
        for part in reversed(path_parts):
            if len(part) > 8 and ("-" in part or part.isdigit()):
                return part

        return None

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(
        self,
        app,
        default_rate: str = "100/minute",
        burst_rate: str = "1000/hour",
        rate_limits: Dict[str, str] = None,
    ):
        super().__init__(app)
        self.default_rate = self._parse_rate(default_rate)
        self.burst_rate = self._parse_rate(burst_rate)
        self.rate_limits = {}

        # Parse custom rate limits
        if rate_limits:
            for pattern, rate in rate_limits.items():
                self.rate_limits[pattern] = self._parse_rate(rate)

        # Storage for rate limit tracking (use Redis in production)
        self.request_counts = {}
        self.last_reset = {}

    def _parse_rate(self, rate_string: str) -> Dict[str, Any]:
        """Parse rate limit string like '100/minute' into dict."""
        parts = rate_string.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid rate format: {rate_string}")

        count = int(parts[0])
        period = parts[1].lower()

        period_seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

        if period not in period_seconds:
            raise ValueError(f"Invalid period: {period}")

        return {"count": count, "period": period, "seconds": period_seconds[period]}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting to requests."""
        client_id = self._get_client_identifier(request)
        path = request.url.path

        # Get applicable rate limit
        rate_limit = self._get_rate_limit_for_path(path)

        try:
            # Check rate limit
            if not self._check_rate_limit(client_id, rate_limit):
                logger.warning(
                    f"Rate limit exceeded for client {client_id} on path {path}"
                )

                # Calculate reset time
                reset_time = self._get_reset_time(client_id, rate_limit)

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Too many requests. Limit: {rate_limit['count']}/{rate_limit['period']}",
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_limit["count"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(rate_limit["seconds"]),
                    },
                )

            # Record the request
            self._record_request(client_id, rate_limit)

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request to proceed on rate limiting errors

        response = await call_next(request)

        # Add rate limit headers to response
        remaining = self._get_remaining_requests(client_id, rate_limit)
        reset_time = self._get_reset_time(client_id, rate_limit)

        response.headers["X-RateLimit-Limit"] = str(rate_limit["count"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Prefer authenticated user ID
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    def _get_rate_limit_for_path(self, path: str) -> Dict[str, Any]:
        """Get applicable rate limit for a path."""
        # Check for specific path limits
        for pattern, rate_limit in self.rate_limits.items():
            if pattern.endswith("/*"):
                if path.startswith(pattern[:-2]):
                    return rate_limit
            elif path == pattern:
                return rate_limit

        # Return default rate limit
        return self.default_rate

    def _check_rate_limit(self, client_id: str, rate_limit: Dict[str, Any]) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_start = now - rate_limit["seconds"]

        # Initialize tracking for new clients
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
            self.last_reset[client_id] = now

        # Clean old requests outside the window
        self.request_counts[client_id] = [
            timestamp
            for timestamp in self.request_counts[client_id]
            if timestamp > window_start
        ]

        # Check if within limit
        return len(self.request_counts[client_id]) < rate_limit["count"]

    def _record_request(self, client_id: str, rate_limit: Dict[str, Any]):
        """Record a request for rate limiting."""
        now = time.time()
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []

        self.request_counts[client_id].append(now)

    def _get_remaining_requests(
        self, client_id: str, rate_limit: Dict[str, Any]
    ) -> int:
        """Get remaining requests for client."""
        if client_id not in self.request_counts:
            return rate_limit["count"]

        return max(0, rate_limit["count"] - len(self.request_counts[client_id]))

    def _get_reset_time(self, client_id: str, rate_limit: Dict[str, Any]) -> int:
        """Get timestamp when rate limit resets."""
        now = time.time()
        return int(now + rate_limit["seconds"])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(self, app, headers: Dict[str, str] = None):
        super().__init__(app)
        self.security_headers = headers or {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        # Add security timestamp
        response.headers["X-Security-Timestamp"] = str(int(time.time()))

        return response


class CustomCORSMiddleware:
    """Custom CORS middleware with enhanced security."""

    def __init__(
        self,
        app,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = False,
        max_age: int = 600,
    ):
        self.app = app
        self.allow_origins = set(allow_origins or ["http://localhost:3000"])
        self.allow_methods = allow_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]
        self.allow_headers = allow_headers or [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-API-Key",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    async def __call__(self, scope, receive, send):
        """CORS middleware handler."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        origin = request.headers.get("origin")

        # Check if origin is allowed
        origin_allowed = False
        if origin:
            origin_allowed = (
                origin in self.allow_origins
                or "*" in self.allow_origins
                or self._check_origin_pattern(origin)
            )

        # Handle preflight requests
        if request.method == "OPTIONS" and origin_allowed:
            response = Response(status_code=200)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.allow_methods
            )
            response.headers["Access-Control-Allow-Headers"] = ", ".join(
                self.allow_headers
            )
            response.headers["Access-Control-Max-Age"] = str(self.max_age)

            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"

            await response(scope, receive, send)
            return

        # Process regular requests
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and origin_allowed:
                headers = dict(message.get("headers", []))

                # Add CORS headers
                headers[b"access-control-allow-origin"] = origin.encode()
                if self.allow_credentials:
                    headers[b"access-control-allow-credentials"] = b"true"

                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _check_origin_pattern(self, origin: str) -> bool:
        """Check if origin matches allowed patterns."""
        parsed = urlparse(origin)

        # Allow localhost with any port for development
        if parsed.hostname in ["localhost", "127.0.0.1"]:
            return True

        return False


def create_security_middleware_stack(
    auth_manager: AuthManager,
    access_control: AccessControl,
    config: Dict[str, Any] = None,
) -> List[Any]:
    """Create a complete security middleware stack."""
    config = config or {}

    middleware_stack = []

    # 1. CORS middleware (first)
    cors_config = config.get("cors", {})
    middleware_stack.append(
        (
            CustomCORSMiddleware,
            {
                "allow_origins": cors_config.get(
                    "allow_origins", ["http://localhost:3000"]
                ),
                "allow_methods": cors_config.get(
                    "allow_methods", ["GET", "POST", "PUT", "DELETE"]
                ),
                "allow_headers": cors_config.get(
                    "allow_headers", ["Authorization", "Content-Type"]
                ),
                "allow_credentials": cors_config.get("allow_credentials", True),
            },
        )
    )

    # 2. Security headers middleware
    security_headers = config.get("security_headers", {})
    middleware_stack.append((SecurityHeadersMiddleware, {"headers": security_headers}))

    # 3. Rate limiting middleware
    rate_limit_config = config.get("rate_limiting", {})
    middleware_stack.append(
        (
            RateLimitMiddleware,
            {
                "default_rate": rate_limit_config.get("default_rate", "100/minute"),
                "burst_rate": rate_limit_config.get("burst_rate", "1000/hour"),
                "rate_limits": rate_limit_config.get("custom_limits", {}),
            },
        )
    )

    # 4. Authentication middleware
    auth_config = config.get("authentication", {})
    middleware_stack.append(
        (
            AuthenticationMiddleware,
            {
                "auth_manager": auth_manager,
                "excluded_paths": auth_config.get("excluded_paths", []),
            },
        )
    )

    # 5. Authorization middleware (last)
    authz_config = config.get("authorization", {})
    middleware_stack.append(
        (
            AuthorizationMiddleware,
            {
                "access_control": access_control,
                "route_permissions": authz_config.get("route_permissions", {}),
            },
        )
    )

    return middleware_stack

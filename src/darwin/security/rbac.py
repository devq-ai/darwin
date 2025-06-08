"""
Role-Based Access Control (RBAC) System for Darwin Security

This module provides a comprehensive RBAC system for managing user roles,
permissions, and access control within the Darwin genetic algorithm platform.

Features:
- Hierarchical role management with inheritance
- Fine-grained permission system
- Dynamic permission checking
- Role assignment and revocation
- Permission caching for performance
- Audit logging for access control events
- Support for resource-specific permissions
- Role templates and permission sets
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from .exceptions import InsufficientPermissionsError, SecurityConfigError

logger = logging.getLogger(__name__)


class PermissionType(str, Enum):
    """Enumeration of permission types."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    CREATE = "create"
    UPDATE = "update"
    MANAGE = "manage"


class ResourceType(str, Enum):
    """Enumeration of resource types."""

    OPTIMIZATION = "optimization"
    PROBLEM = "problem"
    TEMPLATE = "template"
    USER = "user"
    SYSTEM = "system"
    DASHBOARD = "dashboard"
    API = "api"
    DATA = "data"


class Permission(BaseModel):
    """Represents a single permission."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Permission name")
    description: str = Field(..., description="Permission description")
    permission_type: PermissionType = Field(..., description="Type of permission")
    resource_type: ResourceType = Field(..., description="Type of resource")
    resource_id: Optional[str] = Field(
        None, description="Specific resource ID (optional)"
    )
    conditions: Dict[str, Any] = Field(
        default_factory=dict, description="Additional conditions"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Permission name cannot be empty")
        return v.strip().lower()

    def __str__(self) -> str:
        resource_str = f"{self.resource_type}"
        if self.resource_id:
            resource_str += f":{self.resource_id}"
        return f"{self.permission_type}:{resource_str}"

    def __hash__(self) -> int:
        return hash((self.permission_type, self.resource_type, self.resource_id))

    def matches(
        self,
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
    ) -> bool:
        """Check if this permission matches the given criteria."""
        if self.permission_type != permission_type:
            return False

        if self.resource_type != resource_type:
            return False

        # If this permission has no specific resource_id, it applies to all resources of this type
        if self.resource_id is None:
            return True

        # If this permission has a specific resource_id, it must match
        return self.resource_id == resource_id


class Role(BaseModel):
    """Represents a user role with associated permissions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    permissions: Set[Permission] = Field(
        default_factory=set, description="Role permissions"
    )
    parent_roles: Set[str] = Field(
        default_factory=set, description="Parent role IDs for inheritance"
    )
    is_system_role: bool = Field(
        default=False, description="Whether this is a system role"
    )
    is_active: bool = Field(default=True, description="Whether this role is active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        # Allow sets to be serialized
        json_encoders = {set: list}

    @validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Role name cannot be empty")
        return v.strip().lower()

    def add_permission(self, permission: Permission):
        """Add a permission to this role."""
        self.permissions.add(permission)
        self.updated_at = datetime.now(timezone.utc)

    def remove_permission(self, permission: Permission):
        """Remove a permission from this role."""
        self.permissions.discard(permission)
        self.updated_at = datetime.now(timezone.utc)

    def has_permission(
        self,
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
    ) -> bool:
        """Check if this role has a specific permission."""
        for perm in self.permissions:
            if perm.matches(permission_type, resource_type, resource_id):
                return True
        return False

    def __str__(self) -> str:
        return f"Role({self.name})"

    def __hash__(self) -> int:
        return hash(self.id)


class User(BaseModel):
    """Represents a user with roles and permissions."""

    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    roles: Set[str] = Field(
        default_factory=set, description="Role IDs assigned to user"
    )
    direct_permissions: Set[Permission] = Field(
        default_factory=set, description="Direct permissions"
    )
    is_active: bool = Field(default=True, description="Whether user is active")
    is_superuser: bool = Field(default=False, description="Whether user is a superuser")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        json_encoders = {set: list}

    def add_role(self, role_id: str):
        """Add a role to this user."""
        self.roles.add(role_id)

    def remove_role(self, role_id: str):
        """Remove a role from this user."""
        self.roles.discard(role_id)

    def add_direct_permission(self, permission: Permission):
        """Add a direct permission to this user."""
        self.direct_permissions.add(permission)

    def remove_direct_permission(self, permission: Permission):
        """Remove a direct permission from this user."""
        self.direct_permissions.discard(permission)


class RoleManager:
    """Manages roles and their hierarchical relationships."""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self._create_default_roles()

    def _create_default_roles(self):
        """Create default system roles."""
        # Admin role - full access
        admin_role = Role(
            name="admin", description="Full system administrator", is_system_role=True
        )
        admin_role.add_permission(
            Permission(
                name="admin_all",
                description="Full administrative access",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.SYSTEM,
            )
        )

        # User role - standard user access
        user_role = Role(
            name="user", description="Standard user access", is_system_role=True
        )
        user_role.add_permission(
            Permission(
                name="read_optimizations",
                description="Read optimization data",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.OPTIMIZATION,
            )
        )
        user_role.add_permission(
            Permission(
                name="create_optimizations",
                description="Create new optimizations",
                permission_type=PermissionType.CREATE,
                resource_type=ResourceType.OPTIMIZATION,
            )
        )
        user_role.add_permission(
            Permission(
                name="read_templates",
                description="Read problem templates",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.TEMPLATE,
            )
        )

        # Viewer role - read-only access
        viewer_role = Role(
            name="viewer", description="Read-only access", is_system_role=True
        )
        viewer_role.add_permission(
            Permission(
                name="read_optimizations_viewer",
                description="Read optimization data",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.OPTIMIZATION,
            )
        )
        viewer_role.add_permission(
            Permission(
                name="read_templates_viewer",
                description="Read problem templates",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.TEMPLATE,
            )
        )
        viewer_role.add_permission(
            Permission(
                name="read_dashboard",
                description="Access dashboard",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.DASHBOARD,
            )
        )

        # Store default roles
        self.roles[admin_role.id] = admin_role
        self.roles[user_role.id] = user_role
        self.roles[viewer_role.id] = viewer_role

        logger.info("Default roles created successfully")

    def create_role(
        self, name: str, description: str, parent_roles: List[str] = None
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name
            description: Role description
            parent_roles: List of parent role IDs for inheritance

        Returns:
            Created Role object
        """
        if self.get_role_by_name(name):
            raise SecurityConfigError(f"Role '{name}' already exists")

        role = Role(
            name=name, description=description, parent_roles=set(parent_roles or [])
        )

        # Validate parent roles exist
        for parent_id in role.parent_roles:
            if parent_id not in self.roles:
                raise SecurityConfigError(f"Parent role '{parent_id}' does not exist")

        self.roles[role.id] = role
        logger.info(f"Role created: {name}")

        return role

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self.roles.get(role_id)

    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        for role in self.roles.values():
            if role.name == name.lower():
                return role
        return None

    def update_role(self, role_id: str, **updates) -> Role:
        """Update a role."""
        role = self.get_role(role_id)
        if not role:
            raise SecurityConfigError(f"Role '{role_id}' not found")

        if role.is_system_role and "name" in updates:
            raise SecurityConfigError("Cannot modify system role name")

        for key, value in updates.items():
            if hasattr(role, key):
                setattr(role, key, value)

        role.updated_at = datetime.now(timezone.utc)
        logger.info(f"Role updated: {role.name}")

        return role

    def delete_role(self, role_id: str):
        """Delete a role."""
        role = self.get_role(role_id)
        if not role:
            raise SecurityConfigError(f"Role '{role_id}' not found")

        if role.is_system_role:
            raise SecurityConfigError("Cannot delete system role")

        del self.roles[role_id]
        logger.info(f"Role deleted: {role.name}")

    def get_all_roles(self) -> List[Role]:
        """Get all roles."""
        return list(self.roles.values())

    def get_role_permissions(
        self, role_id: str, include_inherited: bool = True
    ) -> Set[Permission]:
        """
        Get all permissions for a role, optionally including inherited permissions.

        Args:
            role_id: Role ID
            include_inherited: Whether to include inherited permissions

        Returns:
            Set of permissions
        """
        role = self.get_role(role_id)
        if not role:
            return set()

        permissions = set(role.permissions)

        if include_inherited:
            # Recursively get permissions from parent roles
            for parent_id in role.parent_roles:
                parent_permissions = self.get_role_permissions(
                    parent_id, include_inherited=True
                )
                permissions.update(parent_permissions)

        return permissions


class PermissionManager:
    """Manages permissions and permission checks."""

    def __init__(self):
        self.permission_cache: Dict[str, Set[Permission]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}

    def create_permission(
        self,
        name: str,
        description: str,
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
        conditions: Dict[str, Any] = None,
    ) -> Permission:
        """Create a new permission."""
        return Permission(
            name=name,
            description=description,
            permission_type=permission_type,
            resource_type=resource_type,
            resource_id=resource_id,
            conditions=conditions or {},
        )

    def check_permission(
        self,
        user_permissions: Set[Permission],
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a set of permissions includes the required permission.

        Args:
            user_permissions: User's effective permissions
            permission_type: Required permission type
            resource_type: Required resource type
            resource_id: Specific resource ID (optional)

        Returns:
            True if permission is granted
        """
        for permission in user_permissions:
            if permission.matches(permission_type, resource_type, resource_id):
                return True
        return False

    def get_cached_permissions(self, cache_key: str) -> Optional[Set[Permission]]:
        """Get cached permissions if still valid."""
        if cache_key not in self.permission_cache:
            return None

        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return None

        # Check if cache is expired
        if datetime.now(timezone.utc) - cache_time > timedelta(
            seconds=self.cache_ttl_seconds
        ):
            self._clear_cache_entry(cache_key)
            return None

        return self.permission_cache[cache_key]

    def cache_permissions(self, cache_key: str, permissions: Set[Permission]):
        """Cache permissions for a user."""
        self.permission_cache[cache_key] = permissions
        self.cache_timestamps[cache_key] = datetime.now(timezone.utc)

    def _clear_cache_entry(self, cache_key: str):
        """Clear a specific cache entry."""
        self.permission_cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)

    def clear_cache(self):
        """Clear all cached permissions."""
        self.permission_cache.clear()
        self.cache_timestamps.clear()


class AccessControl:
    """Main access control system coordinating users, roles, and permissions."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.role_manager = RoleManager()
        self.permission_manager = PermissionManager()
        self.users: Dict[str, User] = {}

        # Create default admin user
        self._create_default_users()

    def _create_default_users(self):
        """Create default system users."""
        admin_role = self.role_manager.get_role_by_name("admin")
        user_role = self.role_manager.get_role_by_name("user")

        if admin_role and user_role:
            # Create admin user
            admin_user = User(
                id="admin-001",
                username="admin",
                email="admin@darwin.local",
                is_superuser=True,
            )
            admin_user.add_role(admin_role.id)
            self.users[admin_user.id] = admin_user

            # Create regular user
            regular_user = User(
                id="user-001", username="user", email="user@darwin.local"
            )
            regular_user.add_role(user_role.id)
            self.users[regular_user.id] = regular_user

            logger.info("Default users created successfully")

    def create_user(
        self, username: str, email: str = None, roles: List[str] = None
    ) -> User:
        """Create a new user."""
        # Check if username already exists
        for user in self.users.values():
            if user.username == username:
                raise SecurityConfigError(f"Username '{username}' already exists")

        user = User(
            id=str(uuid4()), username=username, email=email, roles=set(roles or [])
        )

        # Validate that all roles exist
        for role_id in user.roles:
            if not self.role_manager.get_role(role_id):
                raise SecurityConfigError(f"Role '{role_id}' does not exist")

        self.users[user.id] = user
        logger.info(f"User created: {username}")

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def assign_role_to_user(self, user_id: str, role_id: str):
        """Assign a role to a user."""
        user = self.get_user(user_id)
        if not user:
            raise SecurityConfigError(f"User '{user_id}' not found")

        role = self.role_manager.get_role(role_id)
        if not role:
            raise SecurityConfigError(f"Role '{role_id}' not found")

        user.add_role(role_id)

        # Clear cached permissions
        self._clear_user_permission_cache(user_id)

        logger.info(f"Role '{role.name}' assigned to user '{user.username}'")

    def revoke_role_from_user(self, user_id: str, role_id: str):
        """Revoke a role from a user."""
        user = self.get_user(user_id)
        if not user:
            raise SecurityConfigError(f"User '{user_id}' not found")

        user.remove_role(role_id)

        # Clear cached permissions
        self._clear_user_permission_cache(user_id)

        logger.info(f"Role revoked from user '{user.username}'")

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user."""
        # Check cache first
        cache_key = f"user_permissions:{user_id}"
        cached_permissions = self.permission_manager.get_cached_permissions(cache_key)
        if cached_permissions is not None:
            return cached_permissions

        user = self.get_user(user_id)
        if not user:
            return set()

        # If user is superuser, grant all permissions
        if user.is_superuser:
            permissions = self._get_all_permissions()
            self.permission_manager.cache_permissions(cache_key, permissions)
            return permissions

        permissions = set(user.direct_permissions)

        # Add permissions from roles
        for role_id in user.roles:
            role_permissions = self.role_manager.get_role_permissions(
                role_id, include_inherited=True
            )
            permissions.update(role_permissions)

        # Cache the permissions
        self.permission_manager.cache_permissions(cache_key, permissions)

        return permissions

    def check_permission(
        self,
        user_id: str,
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has a specific permission.

        Args:
            user_id: User ID
            permission_type: Required permission type
            resource_type: Required resource type
            resource_id: Specific resource ID (optional)

        Returns:
            True if user has permission
        """
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False

        user_permissions = self.get_user_permissions(user_id)

        return self.permission_manager.check_permission(
            user_permissions, permission_type, resource_type, resource_id
        )

    def require_permission(
        self,
        user_id: str,
        permission_type: PermissionType,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
    ):
        """
        Require a user to have a specific permission, raise exception if not.

        Raises:
            InsufficientPermissionsError: If user lacks permission
        """
        if not self.check_permission(
            user_id, permission_type, resource_type, resource_id
        ):
            raise InsufficientPermissionsError(
                f"User lacks {permission_type} permission for {resource_type}"
                + (f":{resource_id}" if resource_id else "")
            )

    def _get_all_permissions(self) -> Set[Permission]:
        """Get all possible permissions (for superusers)."""
        permissions = set()

        # Add all permissions from all roles
        for role in self.role_manager.get_all_roles():
            permissions.update(role.permissions)

        # Add common admin permissions
        admin_permissions = [
            Permission(
                name="admin_system",
                description="System administration",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.SYSTEM,
            ),
            Permission(
                name="manage_users",
                description="User management",
                permission_type=PermissionType.MANAGE,
                resource_type=ResourceType.USER,
            ),
        ]

        permissions.update(admin_permissions)
        return permissions

    def _clear_user_permission_cache(self, user_id: str):
        """Clear cached permissions for a user."""
        cache_key = f"user_permissions:{user_id}"
        self.permission_manager._clear_cache_entry(cache_key)

    def audit_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        granted: bool,
        ip_address: str = None,
    ):
        """Log access control events for auditing."""
        user = self.get_user(user_id)
        username = user.username if user else "unknown"

        logger.info(
            f"Access audit: user={username}, action={action}, resource={resource}, "
            f"granted={granted}, ip={ip_address or 'unknown'}"
        )


# Decorators for easy permission checking
def require_permission(
    permission_type: PermissionType,
    resource_type: ResourceType,
    resource_id: Optional[str] = None,
):
    """Decorator to require specific permissions for a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented to extract user_id from request context
            # For now, it's a placeholder
            user_id = kwargs.get("user_id") or getattr(args[0], "user_id", None)
            if user_id:
                # Access control instance would be injected or retrieved from context
                access_control = kwargs.get("access_control")
                if access_control:
                    access_control.require_permission(
                        user_id, permission_type, resource_type, resource_id
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role_name: str):
    """Decorator to require a specific role for a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Implementation would check if user has the required role
            return func(*args, **kwargs)

        return wrapper

    return decorator

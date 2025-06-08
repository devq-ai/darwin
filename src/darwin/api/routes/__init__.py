"""
Darwin API Routes Package

This package contains FastAPI router modules for all API endpoints.
Provides modular organization of API routes for the Darwin genetic algorithm optimization platform.
"""

from .algorithms import router as algorithms_router
from .health import router as health_router
from .metrics import router as metrics_router
from .optimizers import router as optimizers_router
from .templates import router as templates_router

__all__ = [
    "optimizers_router",
    "health_router",
    "templates_router",
    "algorithms_router",
    "metrics_router",
]

# Router configuration
ROUTER_CONFIG = {
    "optimizers": {
        "router": optimizers_router,
        "prefix": "/api/v1",
        "tags": ["optimizers"],
        "description": "Genetic algorithm optimizer management endpoints",
    },
    "health": {
        "router": health_router,
        "prefix": "/api/v1",
        "tags": ["health"],
        "description": "System health check and monitoring endpoints",
    },
    "templates": {
        "router": templates_router,
        "prefix": "/api/v1",
        "tags": ["templates"],
        "description": "Problem template and example endpoints",
    },
    "algorithms": {
        "router": algorithms_router,
        "prefix": "/api/v1",
        "tags": ["algorithms"],
        "description": "Available algorithm information endpoints",
    },
    "metrics": {
        "router": metrics_router,
        "prefix": "/api/v1",
        "tags": ["metrics"],
        "description": "System performance and analytics endpoints",
    },
}

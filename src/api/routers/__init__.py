"""
API routers initialization.
"""

from .health import router as health_router
from .auth import router as auth_router
from .search import router as search_router
from .workers import router as workers_router
from .index import router as index_router
from .performance import router as performance_router

__all__ = ["health_router", "auth_router", "search_router", "workers_router", "index_router", "performance_router"]

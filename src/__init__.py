"""
OpenSearch Worker Search Platform

Production-ready система поиска работников с использованием OpenSearch.
"""

__version__ = "2.0.0"
__author__ = "GitHub Copilot Team"
__description__ = "Enterprise-grade worker search platform powered by OpenSearch"

# Core components
from .core.config import get_config
from .database import get_opensearch_client
from .api.app import create_app

__all__ = [
    "get_config",
    "get_opensearch_client", 
    "create_app"
]

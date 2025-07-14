"""
Database layer initialization.
"""

from .opensearch_client import get_opensearch_client, close_clients

__all__ = ["get_opensearch_client", "close_clients"]

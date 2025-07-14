"""
Health check endpoints for monitoring and readiness probes.
Updated to use unified configuration system and DI container.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from ...core.container import get_opensearch_client, get_config_from_container

router = APIRouter()
logger = logging.getLogger("health")


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "opensearch-worker-search",
        "version": "2.0.0"
    }


@router.get("/readiness")
async def readiness_check(
    opensearch_client=Depends(get_opensearch_client),
    config=Depends(get_config_from_container)
):
    """
    Readiness probe - checks if service is ready to handle requests.
    """
    try:
        # Check OpenSearch connectivity
        health = opensearch_client.cluster.health()
        
        # Check if index exists
        index_exists = opensearch_client.indices.exists(index=config.database.index_name)
        
        if health["status"] in ["green", "yellow"]:
            return {
                "status": "ready",
                "checks": {
                    "opensearch": "healthy",
                    "cluster_status": health["status"],
                    "index_exists": index_exists,
                    "index_name": config.database.index_name
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.warning(f"OpenSearch cluster unhealthy: {health['status']}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "checks": {
                        "opensearch": "unhealthy",
                        "cluster_status": health["status"],
                        "index_exists": index_exists
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {
                    "opensearch": "error",
                    "error": str(e)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/liveness")
async def liveness_check():
    """
    Liveness probe - checks if service is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/config")
async def config_check(config=Depends(get_config_from_container)):
    """
    Configuration check - returns current configuration status.
    """
    return {
        "status": "ok",
        "configuration": {
            "environment": config.environment,
            "debug": config.debug,
            "api_host": config.api.host,
            "api_port": config.api.port,
            "database_host": config.database.host,
            "index_name": config.database.index_name,
            "cache_enabled": config.cache.enabled,
            "cache_backend": config.cache.backend
        },
        "timestamp": datetime.utcnow().isoformat()
    }

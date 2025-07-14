"""
FastAPI application factory and main app configuration.
Updated to use unified configuration system and DI container.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from ..core.config import AppConfig
from ..core.container import get_container
from ..core.shared.exceptions import create_error_handlers
from .routers import health_router, auth_router, search_router, workers_router, index_router, performance_router


logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured FastAPI application
    """
    # Инициализация DI контейнера
    container = get_container()
    
    app = FastAPI(
        title="OpenSearch Worker Search API",
        description="Production-ready API for searching workers using OpenSearch",
        version="2.0.0",
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None,
        openapi_url="/openapi.json" if config.api.enable_docs else None
    )
    
    # Store config and debug mode in app state for error handlers
    app.state.config = config
    app.state.debug = config.debug
    
    # Конфигурация CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Регистрация центральных обработчиков ошибок
    create_error_handlers(app)
    
    # Подключение роутеров
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(auth_router, prefix="/auth", tags=["authentication"])
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(workers_router, prefix="/workers", tags=["workers"])
    app.include_router(index_router, prefix="/index", tags=["index-management"])
    app.include_router(performance_router, prefix="/performance", tags=["performance-monitoring"])

    # События запуска и остановки
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting OpenSearch Worker Search API v2.0.0")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"OpenSearch host: {config.database.host}")
        logger.info(f"Index name: {config.database.index_name}")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down OpenSearch Worker Search API")
        try:
            from ..database.opensearch_client import close_clients
            await close_clients()
        except Exception as e:
            logger.warning(f"Error closing clients: {e}")
    
    return app

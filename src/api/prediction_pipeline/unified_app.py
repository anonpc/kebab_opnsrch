"""
Унифицированное приложение для поиска работников.
Поддерживает различные бэкенды поиска с едиными интерфейсами.
Интегрирует системы безопасности, кэширования и логирования.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Protocol

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Импорты моделей и исключений
from .models import SearchRequest, SearchResponse, WorkerResult
from src.core.shared.exceptions import create_error_handlers, SecurityError, AuthenticationError, AuthorizationError, SearchError

# Импорты систем
from src.core.config import get_config, AppConfig
from src.core.shared.logging_config import setup_logging, get_logger
from src.core.shared.dependency_injection import configure_services, get_container
from src.core.shared.security import get_security_manager, get_current_user, validate_query, User
from src.core.shared.cache import get_cache_manager, cache_search_results
from src.core.shared.metrics import MetricsCollector

# Импорты бэкендов
from .opensearch_engine import OpenSearchEngine
from .query_processor import QueryProcessor

logger = get_logger('api')

class SearchBackend(Protocol):
    """Протокол для бэкендов поиска с поддержкой безопасности и кэширования."""
    
    async def search(self, query: str, user: User, **kwargs) -> Dict[str, Any]:
        """Выполнение поиска с контекстом пользователя."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Расширенная проверка здоровья бэкенда."""
        ...

class OpenSearchBackend:
    """Бэкенд поиска с использованием OpenSearch."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.query_processor = QueryProcessor()
        self._engine = None
        self._initialized = False
        self.cache_manager = get_cache_manager(config)
        self._indexing_strategy = None
    
    async def _ensure_initialized(self):
        """Ленивая инициализация OpenSearch движка с проверкой индексации."""
        if not self._initialized:
            try:
                self._engine = OpenSearchEngine()
                
                # 🔥 НОВОЕ: Инициализация стратегии индексации
                from src.core.shared.indexing_strategies import get_indexing_strategy
                self._indexing_strategy = get_indexing_strategy(self.config)
                
                # 🔥 НОВОЕ: Обеспечение наличия проиндексированных данных
                indexing_success = await self._indexing_strategy.ensure_indexed()
                if not indexing_success:
                    logger.warning("Indexing failed, search functionality may be limited")
                
                self._initialized = True
                logger.info("OpenSearch backend initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenSearch backend: {e}")
                raise SearchError(f"OpenSearch backend initialization failed: {e}")
    
    @cache_search_results(ttl_seconds=3600)
    async def search(self, query: str, user: User, **kwargs) -> Dict[str, Any]:
        """Выполнение поиска с кэшированием."""
        await self._ensure_initialized()
        
        try:
            # Валидация запроса через систему безопасности
            security_manager = get_security_manager()
            clean_query = security_manager.input_validator.validate_search_query(query)
            
            # Логирование запроса
            logger.info(f"Search request from user {user.username}: {query[:100]}")
            
            # Обработка запроса
            processed = self.query_processor.process_query(clean_query)
            
            # Поиск в OpenSearch
            start_time = time.time()
            indices, scores, category_info = self._engine.search(
                clean_query,
                top_k=kwargs.get('top_k', 20)
            )
            search_time = (time.time() - start_time) * 1000
            
            # Логирование производительности
            from src.core.shared.logging_config import log_performance
            log_performance('opensearch_search', search_time, 'ms', 'search')
            
            # Формирование результатов
            results = self._engine.format_results(indices, scores)
            
            return {
                'results': results,
                'query_info': {
                    'original_query': query,
                    'cleaned_query': clean_query,
                    'category': category_info[0] if category_info else None,
                    'confidence': category_info[1] if category_info else 0.0
                },
                'total_found': len(indices),
                'search_time_ms': round(search_time, 2)
            }
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Расширенная проверка здоровья OpenSearch бэкенда с информацией об индексации."""
        try:
            await self._ensure_initialized()
            
            if not self._engine:
                return {
                    'healthy': False,
                    'error': 'Engine not initialized',
                    'cluster_status': 'unknown',
                    'index': {'exists': False, 'document_count': 0, 'has_data': False}
                }
            
            # Проверка подключения к OpenSearch
            health = self._engine.client.cluster.health()
            cluster_healthy = health['status'] in ['green', 'yellow']
            
            # 🔥 НОВОЕ: Проверка состояния индекса
            index_info = await self._check_index_status()
            
            return {
                'healthy': cluster_healthy and index_info['has_data'],
                'cluster_status': health['status'],
                'cluster_name': health.get('cluster_name', 'unknown'),
                'index': index_info,
                'indexing_strategy': type(self._indexing_strategy).__name__ if self._indexing_strategy else None
            }
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'cluster_status': 'unknown',
                'index': {'exists': False, 'document_count': 0, 'has_data': False}
            }
            
    async def _check_index_status(self) -> Dict[str, Any]:
        """Проверка состояния индекса."""
        try:
            index_name = self._engine.index_name
            
            # Проверка существования индекса
            index_exists = self._engine.client.indices.exists(index=index_name)
            
            if not index_exists:
                return {
                    'exists': False,
                    'document_count': 0,
                    'has_data': False,
                    'status': 'missing'
                }
            
            # Получение количества документов
            count_result = self._engine.client.count(index=index_name)
            doc_count = count_result.get('count', 0)
            
            # Получение статистики индекса
            stats = self._engine.client.indices.stats(index=index_name)
            index_size = stats['indices'][index_name]['total']['store']['size_in_bytes']
            
            return {
                'exists': True,
                'document_count': doc_count,
                'has_data': doc_count > 0,
                'size_bytes': index_size,
                'status': 'ready' if doc_count > 0 else 'empty'
            }
            
        except Exception as e:
            logger.error(f"Error checking index status: {e}")
            return {
                'exists': False,
                'document_count': 0,
                'has_data': False,
                'status': 'error',
                'error': str(e)
            }

class SearchService:
    """Сервис поиска с интеграцией всех систем."""
    
    def __init__(self, backend: SearchBackend, config: AppConfig):
        self.backend = backend
        self.config = config
        self.metrics_collector = MetricsCollector()
    
    async def search(self, request: SearchRequest, user: User) -> SearchResponse:
        """Выполнение поиска с полной интеграцией систем."""
        request_id = f"search_{int(time.time() * 1000)}"
        
        try:
            # Валидация запроса
            if not request.query or len(request.query.strip()) < 2:
                raise HTTPException(status_code=400, detail="Query too short")
            
            # Мониторинг производительности
            start_time = time.time()
            
            # Выполнение поиска
            results = await self.backend.search(
                query=request.query,
                user=user,
                top_k=request.top_k,
                filters=request.filters or {}
            )
            
            # Сбор метрик
            execution_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_search_request(
                query=request.query,
                results_count=results.get('total_found', 0),
                execution_time_ms=execution_time,
                user_id=user.id
            )
            
            # Преобразование результатов в модель ответа
            worker_results = []
            for result in results.get('results', []):
                worker_results.append(WorkerResult(
                    id=result.get('id', ''),
                    title=result.get('title', ''),
                    description=result.get('description', ''),
                    category=result.get('category', ''),
                    price=result.get('price', 0),
                    location=result.get('location', ''),
                    score=result.get('score', 0.0)
                ))
            
            return SearchResponse(
                results=worker_results,
                total_found=results.get('total_found', 0),
                query_info=results.get('query_info', {}),
                execution_time_ms=round(execution_time, 2)
            )
            
        except (SecurityError, AuthenticationError, AuthorizationError):
            raise
        except SearchError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search service: {e}")
            raise SearchError(f"Search service error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Комплексная проверка здоровья системы с детальной информацией об индексации."""
        # 🔥 ОБНОВЛЕНО: Получаем расширенную информацию от бэкенда
        backend_health = await self.backend.health_check()
        
        # Проверка кэша
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        
        # Проверка безопасности
        security_manager = get_security_manager()
        
        # 🔥 НОВОЕ: Определяем общий статус системы
        overall_healthy = (
            backend_health.get('healthy', False) and
            backend_health.get('index', {}).get('has_data', False)
        )
        
        return {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'backend': type(self.backend).__name__,
            'opensearch': backend_health,
            'cache': cache_stats,
            'security': {
                'auth_enabled': security_manager.security_config.enable_auth,
                'rate_limiting': security_manager.security_config.rate_limiting_enabled
            },
            'indexing': {
                'strategy': backend_health.get('indexing_strategy', 'unknown'),
                'index_status': backend_health.get('index', {}).get('status', 'unknown'),
                'document_count': backend_health.get('index', {}).get('document_count', 0),
                'has_data': backend_health.get('index', {}).get('has_data', False)
            },
            'timestamp': time.time()
        }

# Фабрика для создания бэкенда
def create_search_backend(config: AppConfig) -> SearchBackend:
    """Создание бэкенда поиска на основе конфигурации."""
    # Пока поддерживаем только OpenSearch
    return OpenSearchBackend(config)

# Глобальные переменные
search_service: Optional[SearchService] = None
app_config: Optional[Config] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения с полной инициализацией."""
    global search_service, app_config
    
    try:
        # Инициализация конфигурации
        app_config = Config()
        
        # Настройка логирования
        setup_logging(app_config)
        logger.info("Starting unified search application")
        
        # Настройка DI контейнера
        configure_services()
        
        # Инициализация систем безопасности
        security_manager = get_security_manager(app_config)
        
        # Создание тестового пользователя для разработки
        if not security_manager.security_config.enable_auth:
            test_user = security_manager.create_test_user()
            logger.info("Test user created for development")
        
        # Инициализация кэша
        cache_manager = get_cache_manager(app_config)
        logger.info(f"Cache initialized: {cache_manager.cache_type}")
        
        # Инициализация поискового сервиса
        backend = create_search_backend(app_config)
        search_service = SearchService(backend, app_config)
        
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Очистка при завершении
        logger.info("Application shutdown")

# Создание приложения
app = FastAPI(
    title="Unified Search API",
    description="Унифицированное API для поиска работников с системами безопасности и кэширования",
    version="3.0.0",
    lifespan=lifespan
)

# Middleware безопасности
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Middleware для безопасности и логирования."""
    security_manager = get_security_manager()
    return await security_manager.security_middleware(request, call_next)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup centralized error handling
app.state.debug = True  # Default debug mode for unified app
create_error_handlers(app)

# Dependency для получения сервиса
async def get_search_service() -> SearchService:
    """Получение инициализированного сервиса поиска."""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    return search_service

# Эндпоинты API
@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о системе."""
    return {
        "message": "Unified Search API", 
        "version": "3.0.0",
        "features": [
            "OpenSearch backend",
            "Security & Authentication",
            "Caching system", 
            "Performance monitoring",
            "Structured logging"
        ]
    }

@app.get("/health")
async def health_check(service: SearchService = Depends(get_search_service)):
    """Комплексная проверка здоровья системы."""
    return await service.health_check()

@app.post("/search/", response_model=SearchResponse)
async def search_workers(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    service: SearchService = Depends(get_search_service)
):
    """
    Поиск работников с полной интеграцией безопасности и мониторинга.
    
    - **query**: Поисковый запрос (минимум 2 символа)
    - **top_k**: Количество результатов (по умолчанию 20)
    - **filters**: Дополнительные фильтры поиска
    """
    return await service.search(request, current_user)

@app.get("/stats/cache")
async def get_cache_stats():
    """Получение статистики кэша."""
    cache_manager = get_cache_manager()
    return cache_manager.get_stats()

@app.get("/stats/performance")
async def get_performance_stats(service: SearchService = Depends(get_search_service)):
    """Получение статистики производительности."""
    return service.metrics_collector.get_stats()

@app.post("/admin/cache/clear")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """Очистка кэша (только для администраторов)."""
    if 'admin' not in current_user.roles:
        raise AuthorizationError("Admin role required")
    
    cache_manager = get_cache_manager()
    cache_manager.clear()
    
    logger.info(f"Cache cleared by admin user {current_user.username}")
    return {"message": "Cache cleared successfully"}

@app.post("/admin/reindex")
async def trigger_reindex(
    force: bool = False,
    current_user: User = Depends(get_current_user),
    service: SearchService = Depends(get_search_service)
):
    """
    🔥 НОВОЕ: Ручной запуск переиндексации (только для администраторов).
    
    Args:
        force: Принудительная переиндексация даже если данные есть
        current_user: Текущий пользователь
        service: Сервис поиска
    """
    if 'admin' not in current_user.roles:
        raise AuthorizationError("Admin role required")
    
    try:
        # Получаем бэкенд и стратегию индексации
        backend = service.backend
        if not hasattr(backend, '_indexing_strategy') or not backend._indexing_strategy:
            return {"error": "Indexing strategy not available"}
        
        # Запускаем переиндексацию
        logger.info(f"Manual reindexing triggered by admin user {current_user.username}, force={force}")
        
        if force:
            # Принудительная переиндексация
            success = await backend._indexing_strategy._run_indexing()
        else:
            # Обычная проверка и индексация при необходимости
            success = await backend._indexing_strategy.ensure_indexed()
        
        if success:
            logger.info("Manual reindexing completed successfully")
            return {
                "message": "Reindexing completed successfully",
                "forced": force,
                "triggered_by": current_user.username
            }
        else:
            logger.error("Manual reindexing failed")
            return {
                "error": "Reindexing failed",
                "forced": force,
                "triggered_by": current_user.username
            }
            
    except Exception as e:
        logger.error(f"Error during manual reindexing: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing error: {e}")

@app.get("/admin/index/status")
async def get_index_status(
    current_user: User = Depends(get_current_user),
    service: SearchService = Depends(get_search_service)
):
    """
    🔥 НОВОЕ: Получение детального статуса индекса (только для администраторов).
    """
    if 'admin' not in current_user.roles:
        raise AuthorizationError("Admin role required")
    
    try:
        # Получаем детальную информацию о здоровье системы
        health_info = await service.health_check()
        
        return {
            "index": health_info.get("indexing", {}),
            "opensearch": health_info.get("opensearch", {}),
            "overall_status": health_info.get("status", "unknown"),
            "timestamp": health_info.get("timestamp", time.time())
        }
        
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "unified_app:app", 
        host="0.0.0.0", 
        port=8005,
        reload=True,
        log_level="info"
    )

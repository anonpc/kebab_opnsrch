"""
Централизованная система обработки исключений и ошибок.
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import structlog
import traceback
from datetime import datetime
from typing import Union

# OpenSearch specific imports
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import NotFoundError as OpenSearchNotFoundError
from opensearchpy.exceptions import RequestError as OpenSearchRequestError
from opensearchpy.exceptions import AuthenticationException as OpenSearchAuthError

logger = structlog.get_logger("error_handler")

class BaseSearchError(Exception):
    """Базовое исключение для ошибок поиска"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class SearchError(BaseSearchError):
    """Общая ошибка поиска"""
    pass

class QueryValidationError(BaseSearchError):
    """Ошибка валидации запроса"""
    pass

class IndexNotFoundError(BaseSearchError):
    """Ошибка - индекс не найден"""
    pass

class SearchBackendError(BaseSearchError):
    """Ошибка бэкенда поиска"""
    pass

class ConfigurationError(BaseSearchError):
    """Ошибка конфигурации"""
    pass

class AuthenticationError(BaseSearchError):
    """Ошибка аутентификации"""
    pass

class AuthorizationError(BaseSearchError):
    """Ошибка авторизации"""
    pass

class RateLimitError(BaseSearchError):
    """Ошибка превышения лимита запросов"""
    pass

class SecurityError(BaseSearchError):
    """Ошибка безопасности"""
    pass

class ModelLoadError(BaseSearchError):
    """Ошибка загрузки модели"""
    pass

class DataProcessingError(BaseSearchError):
    """Ошибка обработки данных"""
    pass

class IndexingError(BaseSearchError):
    """Ошибка индексации"""
    pass

class OpenSearchError(BaseSearchError):
    """Базовая ошибка OpenSearch"""
    pass

class OpenSearchConnectionError(OpenSearchError):
    """Ошибка подключения к OpenSearch"""
    pass

class OpenSearchResourceNotFoundError(OpenSearchError):
    """Ресурс не найден в OpenSearch"""
    pass

class OpenSearchRequestError(OpenSearchError):
    """Ошибка запроса к OpenSearch"""
    pass

class OpenSearchAuthenticationError(OpenSearchError):
    """Ошибка аутентификации OpenSearch"""
    pass


def handle_opensearch_error(e: Exception, operation: str) -> HTTPException:
    """
    Centralized OpenSearch error handling function.
    
    Args:
        e: Exception from OpenSearch
        operation: Description of operation that failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    if isinstance(e, OpenSearchConnectionError):
        logger.error(f"OpenSearch connection failed during {operation}: {str(e)}")
        return HTTPException(
            status_code=503,
            detail=f"OpenSearch недоступен. Невозможно выполнить операцию: {operation}. "
                   f"Проверьте, что OpenSearch запущен"
        )
    elif isinstance(e, OpenSearchNotFoundError):
        logger.error(f"OpenSearch resource not found during {operation}: {str(e)}")
        return HTTPException(
            status_code=404,
            detail=f"Ресурс не найден в OpenSearch: {operation}"
        )
    elif isinstance(e, OpenSearchRequestError):
        logger.error(f"OpenSearch request error during {operation}: {str(e)}")
        return HTTPException(
            status_code=400,
            detail=f"Неверный запрос к OpenSearch: {operation}. {str(e)}"
        )
    elif isinstance(e, OpenSearchAuthError):
        logger.error(f"OpenSearch authentication error during {operation}: {str(e)}")
        return HTTPException(
            status_code=401,
            detail=f"Ошибка аутентификации OpenSearch: {operation}"
        )
    else:
        logger.error(f"OpenSearch error during {operation}: {str(e)}")
        return HTTPException(
            status_code=500,
            detail=f"Ошибка OpenSearch при выполнении операции: {operation}. {str(e)}"
        )


def convert_opensearch_exception(e: Exception) -> BaseSearchError:
    """
    Convert OpenSearch exceptions to our internal exception hierarchy.
    
    Args:
        e: OpenSearch exception
        
    Returns:
        BaseSearchError: Converted internal exception
    """
    if isinstance(e, OpenSearchConnectionError):
        return OpenSearchConnectionError(
            "Failed to connect to OpenSearch",
            details={"original_error": str(e)}
        )
    elif isinstance(e, OpenSearchNotFoundError):
        return OpenSearchResourceNotFoundError(
            "Resource not found in OpenSearch",
            details={"original_error": str(e)}
        )
    elif isinstance(e, OpenSearchRequestError):
        return OpenSearchRequestError(
            "Invalid request to OpenSearch",
            details={"original_error": str(e)}
        )
    elif isinstance(e, OpenSearchAuthError):
        return OpenSearchAuthenticationError(
            "Authentication failed with OpenSearch",
            details={"original_error": str(e)}
        )
    else:
        return OpenSearchError(
            f"OpenSearch error: {str(e)}",
            details={"original_error": str(e), "error_type": type(e).__name__}
        )


# Маппинг исключений на HTTP статус коды
ERROR_STATUS_MAP = {
    QueryValidationError: 400,
    AuthenticationError: 401,
    AuthorizationError: 403,
    RateLimitError: 429,
    IndexNotFoundError: 503,
    SearchBackendError: 503,
    ConfigurationError: 500,
    ModelLoadError: 503,
    DataProcessingError: 500,
    IndexingError: 503,
    SearchError: 500,
    OpenSearchConnectionError: 503,
    OpenSearchResourceNotFoundError: 404,
    OpenSearchRequestError: 400,
    OpenSearchAuthenticationError: 401,
    OpenSearchError: 500,
    BaseSearchError: 500,
}

async def search_error_handler(request: Request, exc: BaseSearchError) -> JSONResponse:
    """Обработчик ошибок поиска"""
    status_code = ERROR_STATUS_MAP.get(type(exc), 500)
    
    # Логирование ошибки
    logger.error(
        "Search error occurred",
        error_type=type(exc).__name__,
        error_message=str(exc),
        error_details=exc.details,
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        status_code=status_code
    )
    
    # Формирование ответа
    error_response = {
        "error": type(exc).__name__,
        "message": str(exc),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": request.url.path
    }
    
    # Добавляем детали только в режиме отладки
    if exc.details and hasattr(request.app.state, 'debug') and request.app.state.debug:
        error_response["details"] = exc.details
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )

async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP ошибок"""
    logger.warning(
        "HTTP error occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    return await http_exception_handler(request, exc)

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик общих исключений"""
    # Логирование с полным трейсом
    logger.error(
        "Unexpected error occurred",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        traceback=traceback.format_exc()
    )
    
    # В продакшене не показываем детали
    debug_mode = hasattr(request.app.state, 'debug') and request.app.state.debug
    
    error_response = {
        "error": "InternalServerError",
        "message": "An unexpected error occurred" if not debug_mode else str(exc),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": request.url.path
    }
    
    if debug_mode:
        error_response["traceback"] = traceback.format_exc().split('\n')
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )

async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    """Обработчик ошибок валидации Pydantic"""
    logger.warning(
        "Validation error occurred",
        errors=exc.errors() if hasattr(exc, 'errors') else str(exc),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    error_response = {
        "error": "ValidationError",
        "message": "Request validation failed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": request.url.path,
        "details": exc.errors() if hasattr(exc, 'errors') else [{"msg": str(exc)}]
    }
    
    return JSONResponse(
        status_code=422,
        content=error_response
    )

def create_error_handlers(app):
    """Регистрация обработчиков ошибок в FastAPI приложении"""
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError
    
    # Наши кастомные исключения
    app.add_exception_handler(BaseSearchError, search_error_handler)
    app.add_exception_handler(SearchError, search_error_handler)
    app.add_exception_handler(IndexNotFoundError, search_error_handler)
    app.add_exception_handler(QueryValidationError, search_error_handler)
    app.add_exception_handler(SearchBackendError, search_error_handler)
    app.add_exception_handler(ConfigurationError, search_error_handler)
    app.add_exception_handler(ModelLoadError, search_error_handler)
    app.add_exception_handler(DataProcessingError, search_error_handler)
    app.add_exception_handler(IndexingError, search_error_handler)
    app.add_exception_handler(OpenSearchError, search_error_handler)
    app.add_exception_handler(OpenSearchConnectionError, search_error_handler)
    app.add_exception_handler(OpenSearchResourceNotFoundError, search_error_handler)
    app.add_exception_handler(OpenSearchRequestError, search_error_handler)
    app.add_exception_handler(OpenSearchAuthenticationError, search_error_handler)
    
    # Стандартные исключения
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Общий обработчик
    app.add_exception_handler(Exception, general_exception_handler)

# Декораторы для обработки ошибок
def handle_errors(default_error_type: type = SearchError):
    """Декоратор для обработки ошибок в функциях"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except BaseSearchError:
                raise  # Пропускаем наши исключения как есть
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise default_error_type(f"Error in {func.__name__}: {str(e)}")
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseSearchError:
                raise  # Пропускаем наши исключения как есть
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise default_error_type(f"Error in {func.__name__}: {str(e)}")
        
        # Определяем, асинхронная ли функция
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def handle_opensearch_errors(operation: str = "OpenSearch operation"):
    """Декоратор для обработки ошибок OpenSearch"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (OpenSearchConnectionError, OpenSearchNotFoundError, 
                    OpenSearchRequestError, OpenSearchAuthError) as e:
                # Convert to our internal exceptions
                converted_exc = convert_opensearch_exception(e)
                logger.error(f"OpenSearch error in {func.__name__} during {operation}: {e}")
                raise converted_exc
            except BaseSearchError:
                raise  # Пропускаем наши исключения как есть
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__} during {operation}: {e}")
                raise OpenSearchError(f"Unexpected error during {operation}: {str(e)}")
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (OpenSearchConnectionError, OpenSearchNotFoundError, 
                    OpenSearchRequestError, OpenSearchAuthError) as e:
                # Convert to our internal exceptions
                converted_exc = convert_opensearch_exception(e)
                logger.error(f"OpenSearch error in {func.__name__} during {operation}: {e}")
                raise converted_exc
            except BaseSearchError:
                raise  # Пропускаем наши исключения как есть
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__} during {operation}: {e}")
                raise OpenSearchError(f"Unexpected error during {operation}: {str(e)}")
        
        # Определяем, асинхронная ли функция
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

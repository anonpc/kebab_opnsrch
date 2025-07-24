"""
Search endpoints for worker search functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import structlog
import time

from ...core.shared.security import SecurityManager, User, get_current_user
from ...core.shared.caching import CacheManager
from ...database.opensearch_client import get_opensearch_client
from ...api.prediction_pipeline.opensearch_engine import OpenSearchEngine
from ...api.prediction_pipeline.models import SearchRequest, SearchResponse, QueryInfo, WorkerResult

router = APIRouter()
logger = structlog.get_logger("search")


@router.post("/", response_model=SearchResponse)
async def search_workers(
    request: SearchRequest,
    http_request: Request
):
    """
    Search for workers using OpenSearch.
    
    Args:
        request: Search parameters
        
    Returns:
        Search results with metadata
    """
    try:
        # Initialize dependencies
        security_manager = SecurityManager()
        cache_manager = CacheManager()
        opensearch_client = get_opensearch_client()
        
        # Get current user (if auth enabled)
        current_user = None
        if security_manager.security_config.enable_auth:
            # Extract token from Authorization header
            auth_header = http_request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Authentication required")
            
            token = auth_header.split(" ")[1]
            payload = security_manager.auth_manager.verify_token(token)
            current_user = security_manager.auth_manager.get_user_by_username(payload.get("sub"))
        
        # Rate limiting check
        client_ip = security_manager.get_client_ip(http_request)
        if not security_manager.rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded", ip=client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Input validation
        security_manager.input_validator.validate_search_query(request.query)
        
        # Create search engine instance
        engine = OpenSearchEngine(
            client=opensearch_client,
            cache_manager=cache_manager
        )
        
        start_time = time.time()
        
        # Perform search - используем прямой запрос к OpenSearch вместо проблемного метода search
        try:
            # Выполняем простой текстовый поиск без векторов
            search_query = {
                "query": {
                    "match": {
                        "combined_text": request.query
                    }
                },
                "size": request.top_k
            }
            
            response = engine.client.search(
                index=engine.index_name,
                body=search_query
            )
            
            # Обрабатываем результаты
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "id": hit['_id'],
                    "title": source.get('title', ''),
                    "description": source.get('description', ''),
                    "category": source.get('category', ''),
                    "price": source.get('price', 0),
                    "location": source.get('location', ''),
                    "photo_urls": source.get('photo_urls', []),
                    "executor_telegram_id": source.get('executor_telegram_id'),
                    "score": hit['_score']
                })
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            results = []
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Преобразуем результаты в формат WorkerResult
        worker_results = []
        for result in results:
            worker_results.append(
                WorkerResult(
                    id=result.get("id"),
                    title=result.get("title", "Неизвестно"),
                    description=result.get("description"),
                    category=result.get("category"),
                    price=result.get("price"),
                    location=result.get("location"),
                    photo_urls=result.get("photo_urls", []),
                    executor_telegram_id=result.get("executor_telegram_id"),
                    score=result.get("score")
                )
            )
        
        # Создаем информацию о запросе
        query_info = QueryInfo(
            original_query=request.query,
            cleaned_query=request.query,  # В реальности здесь должен быть очищенный запрос
            category=None,
            confidence=0.0,
            processing_time_ms=processing_time_ms
        )
        
        # Формируем ответ
        response = SearchResponse(
            results=worker_results,
            query_info=query_info,
            total_found=len(worker_results)
        )
        
        username = current_user.username if current_user else "anonymous"
        logger.info(
            "Search performed",
            user=username,
            query=request.query[:50],
            results_count=len(worker_results)
        )
        
        return response
        
    except Exception as e:
        username = current_user.username if current_user else "anonymous"
        logger.error(
            "Search error",
            user=username,
            query=request.query[:50],
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Internal search error"
        )


@router.get("/analytics")
async def get_search_analytics(
    current_user: User = Depends(get_current_user),
    opensearch_client = Depends(get_opensearch_client)
):
    """
    Get search analytics (admin only).
    """
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    try:
        engine = OpenSearchEngine(client=opensearch_client)
        analytics = await engine.get_analytics()
        
        return analytics
        
    except Exception as e:
        logger.error("Analytics error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analytics"
        )


@router.delete("/cache")
async def clear_cache(
    current_user: User = Depends(get_current_user)
):
    """
    Clear search cache (admin only).
    """
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    try:
        cache_manager = CacheManager()
        cache_manager.clear()
        logger.info("Cache cleared", user=current_user.username)
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error("Cache clear error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to clear cache"
        )

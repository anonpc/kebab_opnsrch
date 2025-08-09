"""
Workers management endpoints for adding, updating and deleting workers.
Updated to use DI container and simplified architecture.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import logging
from datetime import datetime, timezone
import uuid

from ...core.container import get_opensearch_client, get_cache_manager, get_config_from_container
from ...api.prediction_pipeline.models import (
    WorkerCreate, WorkerUpdate, WorkerDeleteRequest,
    WorkerResponse, OperationResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


class WorkerService:
    """Service for worker-related operations."""
    
    def __init__(self, opensearch_client, cache_manager, config):
        self.opensearch_client = opensearch_client
        self.cache_manager = cache_manager
        self.config = config
        self.index_name = config.database.index_name
    
    def create_worker(self, worker_data: WorkerCreate) -> OperationResponse:
        """Create a new worker."""
        try:
            # Генерация уникального ID для работника
            worker_id = str(uuid.uuid4())
            
            # Создание объединенного текста для поиска
            combined_text = f"{worker_data.title} {worker_data.category} {worker_data.description} {worker_data.location}"
            
            # Подготовка документа для индексации
            doc = {
                "id": worker_id,
                "title": worker_data.title,
                "description": worker_data.description,
                "category": worker_data.category,
                "seller_name": worker_data.seller_name,
                "price": worker_data.price,
                "location": worker_data.location,
                "url": worker_data.url,
                "photo_urls": worker_data.photo_urls,
                "executor_telegram_id": worker_data.executor_telegram_id,
                "rating": worker_data.rating,
                "combined_text": combined_text,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Индексация документа
            response = self.opensearch_client.index(
                index=self.index_name,
                id=worker_id,
                body=doc,
                refresh=True
            )
            
            if response.get('result') not in ['created', 'updated']:
                raise HTTPException(status_code=500, detail="Failed to index worker")
            
            # Очистка кэша для обеспечения свежих результатов поиска
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()
            
            # Создание ответа
            worker_response = WorkerResponse(
                id=worker_id,
                title=worker_data.title,
                description=worker_data.description,
                category=worker_data.category,
                seller_name=worker_data.seller_name,
                price=worker_data.price,
                location=worker_data.location,
                url=worker_data.url,
                photo_urls=worker_data.photo_urls,
                executor_telegram_id=worker_data.executor_telegram_id,
                rating=worker_data.rating,
                indexed_at=doc["indexed_at"]
            )

            logger.info(f"Worker added successfully. ID: {worker_id}, Title: {worker_data.title}")

            return OperationResponse(
                success=True,
                message=f"Worker '{worker_data.title}' successfully added",
                worker=worker_response,
                affected_count=1
            )

        except Exception as e:
            logger.error(f"Error adding worker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error adding worker: {str(e)}")
    
    def get_worker(self, worker_id: str) -> WorkerResponse:
        """Get worker by ID."""
        try:
            response = self.opensearch_client.get(index=self.index_name, id=worker_id)
            source = response["_source"]
        except Exception:
            raise HTTPException(status_code=404, detail="Worker not found")

        return WorkerResponse(
            id=worker_id,
            title=source.get("title", ""),
            description=source.get("description"),
            category=source.get("category"),
            seller_name=source.get("seller_name"),
            price=source.get("price"),
            location=source.get("location"),
            url=source.get("url"),
            photo_urls=source.get("photo_urls", []),
            executor_telegram_id=source.get("executor_telegram_id"),
            rating=source.get("rating"),
            indexed_at=source.get("indexed_at")
        )
    
    def update_worker(self, worker_id: str, worker_data: WorkerUpdate) -> OperationResponse:
        """Update existing worker."""
        try:
            # Проверка существования работника
            try:
                current_doc = self.opensearch_client.get(index=self.index_name, id=worker_id)
                current_source = current_doc["_source"]
            except Exception:
                raise HTTPException(status_code=404, detail="Worker not found")

            # Подготовка данных обновления (включить только не-None поля)
            update_data = {}
            if worker_data.title is not None:
                update_data["title"] = worker_data.title
            if worker_data.description is not None:
                update_data["description"] = worker_data.description
            if worker_data.category is not None:
                update_data["category"] = worker_data.category
            if worker_data.seller_name is not None:
                update_data["seller_name"] = worker_data.seller_name
            if worker_data.location is not None:
                update_data["location"] = worker_data.location
            if worker_data.price is not None:
                update_data["price"] = worker_data.price
            if worker_data.url is not None:
                update_data["url"] = worker_data.url
            if worker_data.photo_urls is not None:
                update_data["photo_urls"] = worker_data.photo_urls
            if worker_data.executor_telegram_id is not None:
                update_data["executor_telegram_id"] = worker_data.executor_telegram_id
            if worker_data.rating is not None:
                update_data["rating"] = worker_data.rating

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            # Обновление combined_text если соответствующие поля изменились
            if any(field in update_data for field in ['title', 'category', 'description', 'location']):
                title = update_data.get("title", current_source.get("title", ""))
                category = update_data.get("category", current_source.get("category", ""))
                description = update_data.get("description", current_source.get("description", ""))
                location = update_data.get("location", current_source.get("location", ""))

                update_data["combined_text"] = f"{title} {category} {description} {location}".strip()

            # Добавление отметки времени обновления
            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Обновление документа
            self.opensearch_client.update(
                index=self.index_name,
                id=worker_id,
                body={"doc": update_data},
                refresh=True
            )

            # Очистка кэша
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()

            # Получение обновленного документа для ответа
            updated_doc = self.opensearch_client.get(index=self.index_name, id=worker_id)
            updated_source = updated_doc["_source"]

            worker_response = WorkerResponse(
                id=worker_id,
                title=updated_source.get("title", ""),
                description=updated_source.get("description"),
                category=updated_source.get("category"),
                seller_name=updated_source.get("seller_name"),
                price=updated_source.get("price"),
                location=updated_source.get("location"),
                url=updated_source.get("url"),
                photo_urls=updated_source.get("photo_urls", []),
                executor_telegram_id=updated_source.get("executor_telegram_id"),
                rating=updated_source.get("rating"),
                indexed_at=updated_source.get("indexed_at")
            )

            logger.info(f"Worker updated successfully. ID: {worker_id}")

            return OperationResponse(
                success=True,
                message=f"Worker with ID '{worker_id}' successfully updated",
                worker=worker_response,
                affected_count=1
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating worker {worker_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating worker: {str(e)}")
    
    def delete_worker(self, delete_request: WorkerDeleteRequest) -> OperationResponse:
        """Delete worker(s) by specified criteria."""
        try:
            affected_count = 0
            message = ""

            if delete_request.delete_by == "id":
                # Прямое удаление по ID
                try:
                    self.opensearch_client.delete(
                        index=self.index_name,
                        id=delete_request.value,
                        refresh=True
                    )
                    affected_count = 1
                    message = f"Worker with ID '{delete_request.value}' successfully deleted"
                except Exception as e:
                    if "not_found" in str(e).lower():
                        raise HTTPException(status_code=404, detail="Worker not found")
                    raise HTTPException(status_code=500, detail=f"Error deleting worker: {str(e)}")

            else:
                # Поиск и удаление по значению поля
                field_mapping = {
                    "seller_name": "seller_name.keyword",
                    "category": "category",
                    "location": "location"
                }
                
                search_field = field_mapping.get(delete_request.delete_by)
                if not search_field:
                    raise HTTPException(status_code=400, detail="Invalid delete_by field")
                
                # Поиск соответствующих работников
                search_query = {
                    "query": {
                        "term": {search_field: delete_request.value}
                    }
                }
                
                search_response = self.opensearch_client.search(
                    index=self.index_name,
                    body=search_query,
                    size=100  # Ограничение для предотвращения случайного массового удаления
                )
                
                # Удаление найденных работников
                hits = search_response.get('hits', {}).get('hits', [])
                if not hits:
                    raise HTTPException(status_code=404, detail="No workers found matching criteria")
                
                affected_count = len(hits)
                for hit in hits:
                    try:
                        self.opensearch_client.delete(
                            index=self.index_name,
                            id=hit['_id'],
                            refresh=True
                        )
                    except Exception as e:
                        logger.error(f"Error deleting worker {hit['_id']}: {str(e)}")

            # Очистка кэша
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()

            logger.info(f"Workers deleted successfully. Delete by: {delete_request.delete_by}, Value: {delete_request.value}, Count: {affected_count}")

            return OperationResponse(
                success=True,
                message=message,
                worker=None,
                affected_count=affected_count
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting worker. Delete by: {delete_request.delete_by}, Value: {delete_request.value}, Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting worker: {str(e)}")
    
    def list_workers(self, limit: int = 10, offset: int = 0, category: Optional[str] = None, 
                    location: Optional[str] = None) -> List[WorkerResponse]:
        """Get list of workers with filtering."""
        try:
            if limit > 100:
                limit = 100

            # Построение запроса с фильтрами
            query = {"match_all": {}}

            filters = []
            if category:
                filters.append({"term": {"category.keyword": category}})
            if location:
                filters.append({"term": {"location.keyword": location}})

            if filters:
                query = {
                    "bool": {
                        "must": query,
                        "filter": filters
                    }
                }

            search_body = {
                "query": query,
                "from": offset,
                "size": limit,
                "sort": [{"indexed_at": {"order": "desc"}}]
            }

            response = self.opensearch_client.search(index=self.index_name, body=search_body)

            workers = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit["_source"]
                worker = WorkerResponse(
                    id=hit["_id"],
                    title=source.get("title", ""),
                    description=source.get("description"),
                    category=source.get("category"),
                    seller_name=source.get("seller_name"),
                    price=source.get("price"),
                    location=source.get("location"),
                    url=source.get("url"),
                    photo_urls=source.get("photo_urls", []),
                    executor_telegram_id=source.get("executor_telegram_id"),
                    rating=source.get("rating"),
                    indexed_at=source.get("indexed_at")
                )
                workers.append(worker)

            return workers

        except Exception as e:
            logger.error(f"Error listing workers: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting workers list: {str(e)}")


def get_worker_service() -> WorkerService:
    """Get worker service with dependencies."""
    opensearch_client = get_opensearch_client()
    cache_manager = get_cache_manager()
    config = get_config_from_container()
    return WorkerService(opensearch_client, cache_manager, config)


@router.post("/", response_model=OperationResponse)
async def add_worker(
    worker_data: WorkerCreate,
    worker_service: WorkerService = Depends(get_worker_service)
):
    """Add new worker to index."""
    return worker_service.create_worker(worker_data)


@router.put("/{worker_id}", response_model=OperationResponse)
async def update_worker(
    worker_id: str,
    worker_data: WorkerUpdate,
    worker_service: WorkerService = Depends(get_worker_service)
):
    """Update existing worker."""
    return worker_service.update_worker(worker_id, worker_data)


@router.delete("/", response_model=OperationResponse)
async def delete_worker(
    delete_request: WorkerDeleteRequest,
    worker_service: WorkerService = Depends(get_worker_service)
):
    """Delete worker(s) by specified criteria."""
    return worker_service.delete_worker(delete_request)


@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    worker_id: str,
    worker_service: WorkerService = Depends(get_worker_service)
):
    """Get worker information by ID."""
    return worker_service.get_worker(worker_id)


@router.get("/", response_model=List[WorkerResponse])
async def list_workers(
    limit: int = 10,
    offset: int = 0,
    category: Optional[str] = None,
    location: Optional[str] = None,
    worker_service: WorkerService = Depends(get_worker_service)
):
    """Get list of workers with filtering."""
    return worker_service.list_workers(limit, offset, category, location)

"""
Index management endpoints for OpenSearch index operations.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timezone
import asyncio
from opensearchpy.exceptions import ConnectionError, NotFoundError

from ...core.shared.security import SecurityManager, get_current_user, User
from ...core.shared.caching import CacheManager
from ...core.shared.exceptions import handle_opensearch_error
from ...database.opensearch_client import get_opensearch_client
from ...core.config import get_config, AppConfig

router = APIRouter()
logger = logging.getLogger(__name__)


class IndexStatusResponse(BaseModel):
    """Модель ответа статуса индекса"""
    index_name: str = Field(..., description="Название индекса")
    exists: bool = Field(..., description="Существует ли индекс")
    status: Optional[str] = Field(None, description="Статус индекса (green/yellow/red)")
    document_count: Optional[int] = Field(None, description="Количество документов")
    size_in_bytes: Optional[int] = Field(None, description="Размер индекса в байтах")
    created_at: Optional[str] = Field(None, description="Время создания индекса")
    settings: Optional[Dict[str, Any]] = Field(None, description="Настройки индекса")
    mappings: Optional[Dict[str, Any]] = Field(None, description="Маппинги полей")


class IndexOperationResponse(BaseModel):
    """Модель ответа операции с индексом"""
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")
    index_name: str = Field(..., description="Название индекса")
    operation: str = Field(..., description="Тип операции")
    timestamp: str = Field(..., description="Время выполнения операции")


class IndexCreateRequest(BaseModel):
    """Модель запроса создания индекса"""
    force_recreate: bool = Field(default=False, description="Пересоздать индекс если он существует")
    load_sample_data: bool = Field(default=True, description="Загрузить тестовые данные")
    sample_size: int = Field(default=1000, ge=1, le=100000, description="Количество документов для загрузки")


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status(
    include_mappings: bool = False,
    include_settings: bool = False
):
    """
    Получить статус индекса OpenSearch.

    Args:
        include_mappings: Включить маппинги полей в ответ
        include_settings: Включить настройки индекса в ответ

    Returns:
        Подробная информация о статусе индекса
    """
    try:
        opensearch_client = get_opensearch_client()
        config = get_config()
        index_name = config.database.index_name

        # Проверяем существование индекса
        index_exists = opensearch_client.indices.exists(index=index_name)

        if not index_exists:
            return IndexStatusResponse(
                index_name=index_name,
                exists=False
            )

        # Получаем информацию о индексе
        index_stats = opensearch_client.indices.stats(index=index_name)
        index_info = opensearch_client.indices.get(index=index_name)

        # Извлекаем статистику
        stats = index_stats.get('indices', {}).get(index_name, {})
        total_stats = stats.get('total', {})
        
        document_count = total_stats.get('docs', {}).get('count', 0)
        size_in_bytes = total_stats.get('store', {}).get('size_in_bytes', 0)

        # Получаем статус кластера для конкретного индекса
        cluster_health = opensearch_client.cluster.health(index=index_name)
        index_status = cluster_health.get('status', 'unknown')

        # Получаем время создания индекса
        creation_date = index_info.get(index_name, {}).get('settings', {}).get('index', {}).get('creation_date')
        created_at = None
        if creation_date:
            try:
                created_at = datetime.fromtimestamp(int(creation_date) / 1000, tz=timezone.utc).isoformat()
            except (ValueError, TypeError):
                created_at = None

        # Опционально включаем маппинги и настройки
        mappings = None
        settings = None

        if include_mappings:
            mappings = index_info.get(index_name, {}).get('mappings', {})

        if include_settings:
            settings = index_info.get(index_name, {}).get('settings', {})

        logger.info(f"Index status retrieved. Index: {index_name}, Documents: {document_count}, Status: {index_status}")

        return IndexStatusResponse(
            index_name=index_name,
            exists=True,
            status=index_status,
            document_count=document_count,
            size_in_bytes=size_in_bytes,
            created_at=created_at,
            settings=settings,
            mappings=mappings
        )

    except Exception as e:
        logger.error(f"Error getting index status: {str(e)}")
        raise handle_opensearch_error(e, "получения статуса индекса")


@router.delete("/", response_model=IndexOperationResponse)
async def delete_index(
    confirm: bool = False,
    current_user: User = Depends(get_current_user)
):
    """
    Удалить индекс OpenSearch.

    Args:
        confirm: Подтверждение удаления (обязательно для безопасности)
        current_user: Текущий пользователь (требуется admin роль)

    Returns:
        Результат операции удаления
    """
    try:
        # Проверка прав доступа
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Для удаления индекса требуются права администратора"
            )

        if not confirm:
            raise HTTPException(
                status_code=400,
                            detail="Требуется подтверждение удаления. Установите параметр confirm=true"
        )

        opensearch_client = get_opensearch_client()
        config = get_config()
        index_name = config.database.index_name

        # Проверяем существование индекса
        index_exists = opensearch_client.indices.exists(index=index_name)

        if not index_exists:
            raise HTTPException(status_code=404, detail=f"Индекс '{index_name}' не существует")

        # Удаляем индекс
        response = opensearch_client.indices.delete(index=index_name)

        # Очищаем кэш
        cache_manager = CacheManager()
        if cache_manager.enabled:
            try:
                cache_manager.clear()
            except AttributeError:
                pass

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"Index deleted successfully. Index: {index_name}, User: {current_user.username}")

        return IndexOperationResponse(
            success=True,
            message=f"Индекс '{index_name}' успешно удален",
            index_name=index_name,
            operation="delete",
            timestamp=timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        raise handle_opensearch_error(e, "удаления индекса")


@router.post("/create", response_model=IndexOperationResponse)
async def create_index(
    request: IndexCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Создать новый индекс OpenSearch.

    Args:
        request: Параметры создания индекса
        current_user: Текущий пользователь (требуется admin роль)

    Returns:
        Результат операции создания
    """
    try:
                # Проверка прав доступа
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Для создания индекса требуются права администратора"
            )

        opensearch_client = get_opensearch_client()
        config = get_config()
        index_name = config.database.index_name

        # Проверяем существование индекса
        index_exists = opensearch_client.indices.exists(index=index_name)

        if index_exists and not request.force_recreate:
            raise HTTPException(
                status_code=400,
                detail=f"Индекс '{index_name}' уже существует. Используйте force_recreate=true для пересоздания"
            )

        # Удаляем существующий индекс если требуется пересоздание
        if index_exists and request.force_recreate:
            opensearch_client.indices.delete(index=index_name)
            logger.info(f"Existing index deleted for recreation: {index_name}")

        # Создаем индекс с настройками
        index_settings = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "analysis": {
                        "analyzer": {
                            "russian_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "russian_stop",
                                    "russian_stemmer"
                                ]
                            }
                        },
                        "filter": {
                            "russian_stop": {
                                "type": "stop",
                                "stopwords": "_russian_"
                            },
                            "russian_stemmer": {
                                "type": "stemmer",
                                "language": "russian"
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "category": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "russian_analyzer"
                            }
                        }
                    },
                    "seller_name": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "location": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "russian_analyzer"
                            }
                        }
                    },
                    "price": {
                        "type": "float"
                    },
                    "url": {
                        "type": "keyword",
                        "index": False
                    },
                    "combined_text": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "combined_text_vector": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 200,
                                "m": 16
                            }
                        }
                    },
                    "indexed_at": {
                        "type": "date"
                    },
                    "updated_at": {
                        "type": "date"
                    }
                }
            }
        }

        # Создаем индекс
        response = opensearch_client.indices.create(
            index=index_name,
            body=index_settings
        )

        logger.info(f"Index created successfully: {index_name}")

        # Загружаем тестовые данные если требуется
        if request.load_sample_data:
            await _load_sample_data(opensearch_client, index_name, request.sample_size)

        # Очищаем кэш
        cache_manager = CacheManager()
        if cache_manager.enabled:
            try:
                cache_manager.clear()
            except AttributeError:
                pass

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"Index created and configured. Index: {index_name}, User: {current_user.username}")

        return IndexOperationResponse(
            success=True,
            message=f"Индекс '{index_name}' успешно создан" + 
                   (f" и загружено {request.sample_size} тестовых документов" if request.load_sample_data else ""),
            index_name=index_name,
            operation="create",
            timestamp=timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise handle_opensearch_error(e, "создания индекса")


@router.post("/reindex", response_model=IndexOperationResponse)
async def reindex_data(
    sample_size: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """
    Переиндексировать данные в существующий индекс.

    Args:
        sample_size: Количество документов для загрузки
        current_user: Текущий пользователь (требуется admin роль)

    Returns:
        Результат операции переиндексации
    """
    try:
                # Проверка прав доступа
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Для переиндексации требуются права администратора"
            )

        opensearch_client = get_opensearch_client()
        config = get_config()
        index_name = config.database.index_name

        # Проверяем существование индекса
        index_exists = opensearch_client.indices.exists(index=index_name)

        if not index_exists:
            raise HTTPException(status_code=404, detail=f"Индекс '{index_name}' не существует")

        # Загружаем данные
        await _load_sample_data(opensearch_client, index_name, sample_size)

        # Очищаем кэш
        cache_manager = CacheManager()
        if cache_manager.enabled:
            try:
                cache_manager.clear()
            except AttributeError:
                pass

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"Data reindexed successfully. Index: {index_name}, Documents: {sample_size}, User: {current_user.username}")

        return IndexOperationResponse(
            success=True,
            message=f"Данные успешно переиндексированы. Загружено {sample_size} документов",
            index_name=index_name,
            operation="reindex",
            timestamp=timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing data: {str(e)}")
        raise handle_opensearch_error(e, "переиндексации данных")


async def _load_sample_data(opensearch_client, index_name: str, sample_size: int):
    """
    Загрузить тестовые данные в индекс.

    Args:
        opensearch_client: Клиент OpenSearch
        index_name: Название индекса
        sample_size: Количество документов для загрузки
    """
    try:
        import pandas as pd
        import glob
        import os
        import uuid

        # Путь к данным
        data_path = 'data/raw'
        
        # Ищем CSV файлы
        target_file = os.path.join(data_path, 'workers.csv')
        if not os.path.exists(target_file):
            csv_files = glob.glob(os.path.join(data_path, '*.csv'))
            if not csv_files:
                logger.warning("No CSV files found for sample data loading")
                return
            
            # Берем первый файл с данными
            for file in csv_files:
                file_size = os.path.getsize(file)
                if file_size > 100:
                    target_file = file
                    break

        if not os.path.exists(target_file) or os.path.getsize(target_file) <= 100:
            logger.warning("No suitable CSV files found for sample data loading - creating minimal sample data")
            # Создаем минимальные тестовые данные
            await _create_minimal_sample_data(opensearch_client, index_name, sample_size)
            return

        # Читаем данные
        df = pd.read_csv(target_file, nrows=sample_size)
        logger.info(f"Loaded {len(df)} records for indexing")

        # Подготовка данных
        df = df.fillna('')
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['combined_text'] = df.apply(
            lambda row: f"{row['title']} {row['category']} {row['description']} {row['location']}", 
            axis=1
        )

        # Индексация данных пакетами
        from opensearchpy.helpers import bulk
        
        actions = []
        for i, row in df.iterrows():
            doc = {
                'title': row['title'],
                'description': row['description'],
                'category': row['category'],
                'seller_name': row.get('seller_name', ''),
                'location': row['location'],
                'price': row['price'],
                'url': row.get('url', ''),
                'combined_text': row['combined_text'],
                'indexed_at': datetime.now(timezone.utc).isoformat()
            }

            actions.append({
                '_index': index_name,
                '_id': str(uuid.uuid4()),
                '_source': doc
            })

            # Отправляем пакет каждые 100 документов
            if len(actions) >= 100:
                success, failed = bulk(
                    opensearch_client,
                    actions,
                    refresh=True,
                    request_timeout=60
                )
                logger.info(f"Indexed batch: {success} documents, {len(failed)} failed")
                actions = []

        # Индексируем оставшиеся документы
        if actions:
            success, failed = bulk(
                opensearch_client,
                actions,
                refresh=True,
                request_timeout=60
            )
            logger.info(f"Indexed final batch: {success} documents, {len(failed)} failed")

        logger.info(f"Sample data loading completed. Total documents: {len(df)}")

    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        raise handle_opensearch_error(e, "загрузки тестовых данных")


async def _create_minimal_sample_data(opensearch_client, index_name: str, sample_size: int):
    """
    Создать минимальные тестовые данные, если CSV файлы недоступны.
    
    Args:
        opensearch_client: Клиент OpenSearch
        index_name: Название индекса
        sample_size: Количество документов для создания
    """
    try:
        import uuid
        from opensearchpy.helpers import bulk
        
        # Создаем минимальные тестовые данные
        categories = ["Строительство", "IT", "Доставка", "Уборка", "Ремонт"]
        locations = ["Москва", "СПб", "Новосибирск", "Екатеринбург", "Казань"]
        
        actions = []
        for i in range(min(sample_size, 10)):  # Максимум 10 документов
            doc = {
                'title': f'Тестовая работа {i+1}',
                'description': f'Описание тестовой работы номер {i+1}',
                'category': categories[i % len(categories)],
                'seller_name': f'Работодатель {i+1}',
                'location': locations[i % len(locations)],
                'price': (i + 1) * 1000,
                'url': f'https://example.com/job/{i+1}',
                'combined_text': f'Тестовая работа {i+1} {categories[i % len(categories)]} Описание тестовой работы номер {i+1} {locations[i % len(locations)]}',
                'indexed_at': datetime.now(timezone.utc).isoformat()
            }

            actions.append({
                '_index': index_name,
                '_id': str(uuid.uuid4()),
                '_source': doc
            })

        # Индексируем документы
        success, failed = bulk(
            opensearch_client,
            actions,
            refresh=True,
            request_timeout=60
        )
        
        logger.info(f"Minimal sample data created: {success} documents, {len(failed)} failed")

    except Exception as e:
        logger.error(f"Error creating minimal sample data: {str(e)}")
        # Не бросаем исключение, чтобы не блокировать создание индекса
        pass


@router.get("/health", response_model=Dict[str, Any])
async def get_index_health():
    """
    Получить информацию о здоровье индекса и кластера OpenSearch.

    Returns:
        Детальная информация о состоянии кластера и индекса
    """
    try:
        opensearch_client = get_opensearch_client()
        config = get_config()
        index_name = config.database.index_name

        # Получаем информацию о кластере
        cluster_health = opensearch_client.cluster.health()
        cluster_stats = opensearch_client.cluster.stats()

        # Получаем информацию об индексах
        index_health = opensearch_client.cluster.health(index=index_name) if opensearch_client.indices.exists(index=index_name) else None

        # Получаем информацию о нодах
        nodes_info = opensearch_client.nodes.info()

        return {
            "cluster": {
                "name": cluster_health.get("cluster_name"),
                "status": cluster_health.get("status"),
                "number_of_nodes": cluster_health.get("number_of_nodes"),
                "active_primary_shards": cluster_health.get("active_primary_shards"),
                "active_shards": cluster_health.get("active_shards"),
                "relocating_shards": cluster_health.get("relocating_shards"),
                "initializing_shards": cluster_health.get("initializing_shards"),
                "unassigned_shards": cluster_health.get("unassigned_shards")
            },
            "index": {
                "name": index_name,
                "exists": opensearch_client.indices.exists(index=index_name),
                "status": index_health.get("status") if index_health else None,
                "number_of_shards": index_health.get("active_primary_shards") if index_health else None,
                "number_of_replicas": index_health.get("number_of_replicas") if index_health else None
            },
            "nodes": {
                "total": len(nodes_info.get("nodes", {})),
                "details": {
                    node_id: {
                        "name": node_info.get("name"),
                        "version": node_info.get("version"),
                        "roles": node_info.get("roles", [])
                    }
                    for node_id, node_info in nodes_info.get("nodes", {}).items()
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting index health: {str(e)}")
        raise handle_opensearch_error(e, "получения информации о здоровье")
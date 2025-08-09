import logging
import numpy as np
from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, ConnectionError as OSConnectionError
import pandas as pd
import time
import json
import re
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache

from ...core.config import get_config, AppConfig
from ...core.shared.caching import CacheManager
from ...core.shared.models import EmbeddingModel, SearchResult

logger = logging.getLogger('prediction')

class OpenSearchEngine:
    """
    OpenSearch engine для гибридного поиска, заменяющий BM25 + FAISS.
    Поддерживает лексический (BM25) и векторный (k-NN) поиск в едином индексе.
    """
    
    def __init__(self, client: Optional[OpenSearch] = None, cache_manager: Optional[CacheManager] = None):
        """
        Инициализация OpenSearch движка.
        
        Args:
            client: Готовый клиент OpenSearch
            cache_manager: Менеджер кэширования
        """
        self.config = get_config()
        
        # Настройки подключения из конфигурации
        if isinstance(client, dict):
            # Если передаём конфигурацию как словарь
            config_dict = client
            opensearch_config = config_dict.get('opensearch', {})
            self.host = opensearch_config.get('host', 'localhost:9200')
            self.index_name = opensearch_config.get('index_name', 'workers')
            self.timeout = opensearch_config.get('timeout', 30)
            self.client = None  # Будет инициализирован позже
        else:
            # Если передан готовый клиент или используем конфигурацию из Config
            try:
                self.host = self.config.database.host
                self.index_name = self.config.database.index_name
                self.timeout = self.config.database.timeout
            except AttributeError:
                # Используем значения по умолчанию если атрибуты недоступны
                self.host = 'localhost:9200'
                self.index_name = 'workers'
                self.timeout = 30
        self.max_retries = 3
        self.embedding_dimension = 768
        
        # Фиксированные веса для точного режима
        self.text_weight = 0.8
        self.vector_weight = 0.2
        
        # Настройки поиска - упрощенные для точного режима
        self.minimum_should_match = 0.5
        self.category_direct_match_boost = 0.4
        
        # Инициализация клиента OpenSearch
        self.client = client
        self.cache_manager = cache_manager
        if not self.client:
            self._init_client()
        
        # Инициализация модели эмбеддингов
        try:
            from ...core.shared.models import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Модель эмбеддингов загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            self.embedding_model = None
        
        # Кэш для категорий и запросов
        self._category_cache = {}
        self._setup_performance_monitoring()
        
    def _init_client(self):
        """Инициализация клиента OpenSearch."""
        try:
            # Use centralized client creation instead of duplicated logic
            from ...database.opensearch_client import create_opensearch_client
            from ...core.config import AppConfig
            
            # Create a custom config with the engine's specific settings if needed
            if hasattr(self, 'host') and self.host != 'localhost:9200':
                # Create custom config for non-standard host
                config = AppConfig()
                config.database_host = self.host
                config.database_index_name = self.index_name
                config.database_timeout = self.timeout
                config.database_max_retries = self.max_retries
                config.database_use_ssl = False
                config.database_verify_certs = False
                config.database_ssl_show_warn = False
                
                self.client = create_opensearch_client(config)
            else:
                # Use global config
                self.client = create_opensearch_client()
            
            # Проверка подключения
            info = self.client.info()
            logger.info(f"Подключение к OpenSearch успешно: {info['version']['number']}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к OpenSearch: {e}")
            self.client = None
            
    def _setup_performance_monitoring(self):
        """Настройка мониторинга производительности."""
        self._performance_stats = {
            'total_searches': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'last_search_time': 0.0
        }
        
    def create_index(self):
        """
        Создание индекса в OpenSearch с настройками для гибридного поиска.
        """
        if not self.client:
            logger.error("Клиент OpenSearch не инициализирован")
            return False
            
        try:
            # Use centralized index creation function
            from ...database.opensearch_client import create_index_with_settings
            
            result = create_index_with_settings(
                client=self.client,
                index_name=self.index_name,
                recreate_if_exists=False
            )
            
            if result["success"]:
                logger.info(f"Index operation result: {result['message']}")
                return True
            else:
                logger.error(f"Index creation failed: {result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка создания индекса: {e}")
            return False
            
    def index_data(self, data: pd.DataFrame, batch_size: int = 100):
        """
        Индексация данных в OpenSearch.
        
        Args:
            data (pd.DataFrame): Данные для индексации
            batch_size (int): Размер батча для bulk операций
        """
        if not self.client:
            logger.error("Клиент OpenSearch не инициализирован")
            return False
            
        if self.embedding_model is None:
            logger.error("Модель эмбеддингов не загружена")
            return False
            
        try:
            logger.info(f"Начинаем индексацию {len(data)} записей")
            
            # Создаем индекс если не существует
            self.create_index()
            
            # Генерируем эмбеддинги для всех текстов
            combined_texts = data['combined_text'].fillna('').tolist()
            logger.info("Генерируем эмбеддинги...")
            
            embeddings = self.embedding_model.encode(
                combined_texts,
                batch_size=32,
                show_progress=True
            )
            
            # Подготовка документов для bulk индексации
            actions = []
            timestamp = time.time()
            
            for idx, (_, row) in enumerate(data.iterrows()):
                doc = {
                    "_index": self.index_name,
                    "_id": str(idx),
                    "_source": {
                        "title": str(row.get('title', '')),
                        "description": str(row.get('description', '')),
                        "category": str(row.get('category', '')),
                        "seller_name": str(row.get('seller_name', '')),
                        "location": str(row.get('location', '')),
                        "price": float(row.get('price', 0.0)),
                        "url": str(row.get('url', '')),
                        "combined_text": str(row.get('combined_text', '')),
                        "combined_text_vector": embeddings[idx].tolist(),
                        "indexed_at": timestamp
                    }
                }
                actions.append(doc)
                
                # Выполняем bulk операцию когда достигли размера батча
                if len(actions) >= batch_size:
                    self._bulk_index(actions)
                    actions = []
                    
            # Индексируем оставшиеся документы
            if actions:
                self._bulk_index(actions)
                
            logger.info(f"Индексация завершена. Проиндексировано {len(data)} документов")
            
            # Принудительное обновление индекса
            self.client.indices.refresh(index=self.index_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка индексации данных: {e}")
            return False
            
    def _bulk_index(self, actions: List[Dict]):
        """Выполнение bulk индексации."""
        try:
            from opensearchpy.helpers import bulk
            
            response = bulk(
                self.client,
                actions,
                index=self.index_name,
                refresh=False,
                request_timeout=60
            )
            
            logger.debug(f"Проиндексировано {len(actions)} документов")
            
        except Exception as e:
            logger.error(f"Ошибка bulk индексации: {e}")
            
    def search(self, query: str, top_k: int = 10, 
               use_categories: bool = True, filters: Dict = None) -> Tuple[List[int], List[float], Optional[Tuple]]:
        """
        Выполнение гибридного поиска в OpenSearch с точным режимом.
        
        Args:
            query (str): Поисковый запрос
            top_k (int): Количество результатов для возврата
            use_categories (bool): Использовать ли категоризацию запроса
            filters (Dict): Дополнительные фильтры
            
        Returns:
            Tuple[List[int], List[float], Optional[Tuple]]: Индексы, оценки и информация о категории
        """
        if not self.client:
            logger.error("Клиент OpenSearch не инициализирован")
            return [], [], None
            
        start_time = time.time()
        
        try:
            # Предобработка запроса
            cleaned_query = self._clean_query(query)
            
            # Определение категории запроса
            query_category, category_confidence = None, 0.0
            if use_categories:
                query_category, category_confidence = self._categorize_query(query, cleaned_query)
                
            # Генерация эмбеддинга запроса
            query_embedding = None
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([cleaned_query])[0]
                
            # Определение весов - фиксированные для точного режима
            text_weight, vector_weight = 0.6, 0.4
            
            # Построение поискового запроса
            search_body = self._build_search_query(
                query=cleaned_query,
                query_embedding=query_embedding,
                text_weight=text_weight,
                vector_weight=vector_weight,
                top_k=top_k,
                query_category=query_category,
                category_confidence=category_confidence,
                filters=filters
            )
            
            # Выполнение поиска
            response = self.client.search(
                index=self.index_name,
                body=search_body,
                timeout=f"{self.timeout}s"
            )
            
            # Обработка результатов
            indices, scores = self._process_search_results(response, query_category, category_confidence)
            
            # Обновление статистики производительности
            search_time = time.time() - start_time
            self._update_performance_stats(search_time)
            
            logger.debug(f"Поиск выполнен за {search_time:.3f}с, найдено {len(indices)} результатов")
            
            # Информация о категории для возврата
            category_info = (query_category, category_confidence) if query_category else None
            
            return indices, scores, category_info
            
        except Exception as e:
            logger.error(f"Ошибка выполнения поиска: {e}")
            return [], [], None
            
    def _clean_query(self, query: str) -> str:
        """Очистка и нормализация запроса."""
        # Удаление лишних пробелов и приведение к нижнему регистру
        cleaned = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Удаление специальных символов, кроме важных
        cleaned = re.sub(r'[^\w\s\-\+\.\#\&]', ' ', cleaned)
        
        return cleaned
        
    @lru_cache(maxsize=256)
    def _categorize_query(self, query: str, cleaned_query: str) -> Tuple[Optional[str], float]:
        """
        Определение категории запроса с использованием OpenSearch.
        
        Args:
            query (str): Оригинальный запрос
            cleaned_query (str): Очищенный запрос
            
        Returns:
            Tuple[Optional[str], float]: Категория и уровень уверенности
        """
        try:
            # Простая проверка на прямое совпадение с категориями
            category_search = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "category": {
                                        "query": cleaned_query,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "term": {
                                    "category.keyword": {
                                        "value": cleaned_query,
                                        "boost": 3.0
                                    }
                                }
                            }
                        ]
                    }
                },
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "category.keyword",
                            "size": 5
                        }
                    }
                },
                "size": 0
            }
            
            response = self.client.search(
                index=self.index_name,
                body=category_search
            )
            
            # Анализ агрегации категорий
            categories = response.get('aggregations', {}).get('categories', {}).get('buckets', [])
            
            if categories:
                top_category = categories[0]
                category_name = top_category['key']
                doc_count = top_category['doc_count']
                
                # Вычисляем уверенность на основе количества документов и релевантности
                total_docs = sum(cat['doc_count'] for cat in categories)
                confidence = min(doc_count / max(total_docs, 1), 0.9)
                
                if confidence > 0.3:  # Минимальный порог уверенности
                    logger.debug(f"Определена категория: {category_name} (уверенность: {confidence:.2f})")
                    return category_name, confidence
                    
        except Exception as e:
            logger.error(f"Ошибка категоризации запроса: {e}")
            
        return None, 0.0
        
    def _build_search_query(self, query: str, query_embedding: Optional[np.ndarray],
                           text_weight: float, vector_weight: float, top_k: int,
                           query_category: Optional[str] = None,
                           category_confidence: float = 0.0, filters: Dict = None) -> Dict:
        """
        Построение запроса для гибридного поиска в OpenSearch с точным режимом.
        """
        # Базовый гибридный запрос
        hybrid_queries = []
        
        # Текстовый компонент (BM25) - точный режим без нечеткого поиска
        text_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",
                    "description^2", 
                    "category.text^2",
                    "combined_text",
                    "seller_name",
                    "location.text"
                ],
                "type": "best_fields",
                "fuzziness": "0",
                "minimum_should_match": f"{int(self.minimum_should_match * 100)}%"
            }
        }
        
        if text_weight > 0:
            hybrid_queries.append({
                "bool": {
                    "should": [text_query],
                    "boost": text_weight
                }
            })
        
        # Векторный компонент (k-NN)
        if query_embedding is not None and vector_weight > 0:
            knn_query = {
                "knn": {
                    "combined_text_vector": {
                        "vector": query_embedding.tolist(),
                        "k": top_k * 3,  # Получаем больше кандидатов для лучшего ранжирования
                        "boost": vector_weight
                    }
                }
            }
            hybrid_queries.append(knn_query)
        
        # Основной запрос
        main_query = {
            "bool": {
                "should": hybrid_queries,
                "minimum_should_match": 1
            }
        }
        
        # Добавление фильтров
        filters_list = []
        
        # Фильтр по категории если определена с высокой уверенностью - всегда применяем в точном режиме
        if query_category and category_confidence > 0.7:
            filters_list.append({
                "term": {
                    "category.keyword": query_category
                }
            })
        
        # Дополнительные фильтры
        if filters:
            if filters.get('price_min') is not None:
                filters_list.append({
                    "range": {
                        "price": {"gte": filters['price_min']}
                    }
                })
            if filters.get('price_max') is not None:
                filters_list.append({
                    "range": {
                        "price": {"lte": filters['price_max']}
                    }
                })
            if filters.get('location'):
                filters_list.append({
                    "term": {
                        "location.keyword": filters['location']
                    }
                })
            if filters.get('category'):
                filters_list.append({
                    "term": {
                        "category.keyword": filters['category']
                    }
                })
        
        # Применение фильтров
        if filters_list:
            main_query = {
                "bool": {
                    "must": [main_query],
                    "filter": filters_list
                }
            }
        
        # Полный поисковый запрос
        search_body = {
            "query": main_query,
            "size": top_k,
            "_source": ["title", "description", "category", "seller_name", "location", "price", "url", "photo_urls", "executor_telegram_id", "rating"],
            "highlight": {
                "fields": {
                    "title": {},
                    "description": {},
                    "category.text": {}
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        }
        
        # Добавление буста для категории если определена
        if query_category and category_confidence > 0.5:
            search_body["query"] = {
                "function_score": {
                    "query": search_body["query"],
                    "functions": [{
                        "filter": {
                            "term": {
                                "category.keyword": query_category
                            }
                        },
                        "boost": 1.0 + self.category_direct_match_boost
                    }],
                    "boost_mode": "multiply",
                    "score_mode": "max"
                }
            }
        
        return search_body
        
    def _process_search_results(self, response: Dict, query_category: Optional[str] = None,
                               category_confidence: float = 0.0) -> Tuple[List[int], List[float]]:
        """
        Обработка результатов поиска из OpenSearch.
        
        Args:
            response (Dict): Ответ от OpenSearch
            query_category (Optional[str]): Определенная категория запроса
            category_confidence (float): Уверенность в категории
            
        Returns:
            Tuple[List[int], List[float]]: Индексы и оценки результатов
        """
        hits = response.get('hits', {}).get('hits', [])
        
        indices = []
        scores = []
        
        for hit in hits:
            try:
                # Извлекаем индекс документа
                doc_id = int(hit['_id'])
                score = float(hit['_score'])
                
                indices.append(doc_id)
                scores.append(score)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Ошибка обработки результата поиска: {e}")
                continue
                
        return indices, scores
        
    def _update_performance_stats(self, search_time: float):
        """Обновление статистики производительности."""
        self._performance_stats['total_searches'] += 1
        self._performance_stats['total_time'] += search_time
        self._performance_stats['last_search_time'] = search_time
        
        if self._performance_stats['total_searches'] > 0:
            self._performance_stats['avg_time'] = (
                self._performance_stats['total_time'] / self._performance_stats['total_searches']
            )
            
    def get_performance_stats(self) -> Dict:
        """Получение статистики производительности."""
        return self._performance_stats.copy()
        
    def get_index_stats(self) -> Dict:
        """Получение статистики индекса."""
        if not self.client:
            return {}
            
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return {
                'document_count': stats['_all']['total']['docs']['count'],
                'index_size': stats['_all']['total']['store']['size_in_bytes'],
                'search_time': stats['_all']['total']['search']['query_time_in_millis'],
                'search_count': stats['_all']['total']['search']['query_total']
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики индекса: {e}")
            return {}
            
    def delete_index(self):
        """Удаление индекса."""
        if not self.client:
            return False
            
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Индекс {self.index_name} удален")
                return True
            else:
                logger.info(f"Индекс {self.index_name} не существует")
                return True
        except Exception as e:
            logger.error(f"Ошибка удаления индекса: {e}")
            return False

    def format_results(self, indices: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """
        Форматирование результатов поиска для возврата клиенту.
        
        Args:
            indices (List[int]): Индексы найденных документов
            scores (List[float]): Оценки релевантности
            
        Returns:
            List[Dict[str, Any]]: Отформатированные результаты
        """
        if not self.client or not indices:
            return []
        
        try:
            # Получаем документы по индексам
            formatted_results = []
            
            for i, (doc_id, score) in enumerate(zip(indices, scores)):
                try:
                    # Получаем документ из OpenSearch
                    response = self.client.get(
                        index=self.index_name,
                        id=str(doc_id)
                    )
                    
                    source = response.get('_source', {})
                    
                    # Форматируем результат
                    result = {
                        'id': str(doc_id),
                        'title': source.get('title', ''),
                        'description': source.get('description', ''),
                        'category': source.get('category', ''),
                        'seller_name': source.get('seller_name', ''),
                        'location': source.get('location', ''),
                        'price': float(source.get('price', 0.0)),
                        'url': source.get('url', ''),
                        'photo_urls': source.get('photo_urls', []),
                        'executor_telegram_id': source.get('executor_telegram_id'),
                        'rating': source.get('rating'),
                        'score': float(score),
                        'rank': i + 1
                    }
                    
                    formatted_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Ошибка получения документа {doc_id}: {e}")
                    continue
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка форматирования результатов: {e}")
            return []

    def health_check(self) -> bool:
        """
        Проверка здоровья OpenSearch соединения.
        
        Returns:
            bool: True если OpenSearch доступен и работает
        """
        try:
            if not self.client:
                return False
            
            # Проверяем соединение
            if not self.client.ping():
                return False
            
            # Проверяем существование индекса
            if not self.client.indices.exists(index=self.index_name):
                logger.warning(f"Индекс {self.index_name} не существует")
                return False
            
            # Проверяем статистику индекса
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['_all']['total']['docs']['count']
            
            if doc_count == 0:
                logger.warning(f"Индекс {self.index_name} пуст")
                return False
            
            logger.debug(f"Health check passed: {doc_count} documents in index")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение документа по ID.
        
        Args:
            doc_id (str): ID документа
            
        Returns:
            Optional[Dict[str, Any]]: Документ или None
        """
        if not self.client:
            return None
        
        try:
            response = self.client.get(
                index=self.index_name,
                id=doc_id
            )
            
            return response.get('_source')
            
        except NotFoundError:
            logger.warning(f"Документ {doc_id} не найден")
            return None
        except Exception as e:
            logger.error(f"Ошибка получения документа {doc_id}: {e}")
            return None

    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        Обновление документа в индексе.
        
        Args:
            doc_id (str): ID документа
            updates (Dict[str, Any]): Обновления для применения
            
        Returns:
            bool: True если обновление прошло успешно
        """
        if not self.client:
            return False
        
        try:
            # Если обновляется combined_text, нужно пересчитать эмбеддинг
            if 'combined_text' in updates and self.embedding_model:
                new_text = updates['combined_text']
                new_embedding = self.embedding_model.encode([new_text])[0]
                updates['combined_text_vector'] = new_embedding.tolist()
            
            response = self.client.update(
                index=self.index_name,
                id=doc_id,
                body={'doc': updates}
            )
            
            logger.debug(f"Документ {doc_id} обновлен успешно")
            return response.get('result') == 'updated'
            
        except Exception as e:
            logger.error(f"Ошибка обновления документа {doc_id}: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """
        Удаление документа из индекса.
        
        Args:
            doc_id (str): ID документа
            
        Returns:
            bool: True если удаление прошло успешно
        """
        if not self.client:
            return False
        
        try:
            response = self.client.delete(
                index=self.index_name,
                id=doc_id
            )
            
            logger.debug(f"Документ {doc_id} удален успешно")
            return response.get('result') == 'deleted'
            
        except NotFoundError:
            logger.warning(f"Документ {doc_id} не найден для удаления")
            return False
        except Exception as e:
            logger.error(f"Ошибка удаления документа {doc_id}: {e}")
            return False

    def suggest_queries(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Автодополнение поисковых запросов.
        
        Args:
            partial_query (str): Частичный запрос
            max_suggestions (int): Максимальное количество предложений
            
        Returns:
            List[str]: Список предложений
        """
        if not self.client or len(partial_query) < 2:
            return []
        
        try:
            # Поиск по началам слов в названиях и категориях
            suggest_body = {
                "suggest": {
                    "title_suggest": {
                        "prefix": partial_query,
                        "completion": {
                            "field": "title.suggest",
                            "size": max_suggestions
                        }
                    },
                    "category_suggest": {
                        "prefix": partial_query,
                        "completion": {
                            "field": "category.suggest", 
                            "size": max_suggestions
                        }
                    }
                }
            }
            
            response = self.client.search(
                index=self.index_name,
                body=suggest_body
            )
            
            suggestions = set()
            
            # Извлекаем предложения из ответа
            for suggest_type in ['title_suggest', 'category_suggest']:
                for option in response.get('suggest', {}).get(suggest_type, []):
                    for suggestion in option.get('options', []):
                        suggestions.add(suggestion['text'])
            
            return list(suggestions)[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Ошибка автодополнения: {e}")
            return []

    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получение популярных поисковых запросов.
        
        Args:
            limit (int): Количество запросов
            
        Returns:
            List[Dict[str, Any]]: Популярные запросы с метриками
        """
        # В реальном приложении здесь был бы анализ логов поиска
        # Пока возвращаем статичный список популярных категорий
        try:
            agg_body = {
                "aggs": {
                    "popular_categories": {
                        "terms": {
                            "field": "category.keyword",
                            "size": limit
                        }
                    },
                    "popular_locations": {
                        "terms": {
                            "field": "location.keyword", 
                            "size": limit
                        }
                    }
                },
                "size": 0
            }
            
            response = self.client.search(
                index=self.index_name,
                body=agg_body
            )
            
            popular_queries = []
            
            # Добавляем популярные категории
            categories = response.get('aggregations', {}).get('popular_categories', {}).get('buckets', [])
            for cat in categories:
                popular_queries.append({
                    'query': cat['key'],
                    'type': 'category',
                    'doc_count': cat['doc_count']
                })
            
            # Добавляем популярные локации
            locations = response.get('aggregations', {}).get('popular_locations', {}).get('buckets', [])
            for loc in locations:
                popular_queries.append({
                    'query': loc['key'],
                    'type': 'location', 
                    'doc_count': loc['doc_count']
                })
            
            # Сортируем по количеству документов
            popular_queries.sort(key=lambda x: x['doc_count'], reverse=True)
            
            return popular_queries[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка получения популярных запросов: {e}")
            return []

    def optimize_index(self) -> bool:
        """
        Оптимизация индекса для улучшения производительности поиска.
        
        Returns:
            bool: True если оптимизация прошла успешно
        """
        if not self.client:
            return False
        
        try:
            logger.info("Начинаем оптимизацию индекса...")
            
            # Принудительное слияние сегментов
            self.client.indices.forcemerge(
                index=self.index_name,
                max_num_segments=1,
                wait_for_completion=True
            )
            
            # Обновление настроек для оптимизации поиска
            self.client.indices.put_settings(
                index=self.index_name,
                body={
                    "index": {
                        "refresh_interval": "1s",
                        "number_of_replicas": 0,
                        "max_result_window": 10000
                    }
                }
            )
            
            # Принудительное обновление
            self.client.indices.refresh(index=self.index_name)
            
            logger.info("Оптимизация индекса завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации индекса: {e}")
            return False

    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Получение аналитики по поиску.
        
        Returns:
            Dict[str, Any]: Аналитические данные
        """
        try:
            analytics = {
                'performance': self.get_performance_stats(),
                'index': self.get_index_stats(),
                'health': self.health_check()
            }
            
            # Добавляем статистику по категориям
            if self.client:
                cat_stats = self.client.search(
                    index=self.index_name,
                    body={
                        "aggs": {
                            "categories": {
                                "terms": {
                                    "field": "category.keyword",
                                    "size": 20
                                }
                            },
                            "price_stats": {
                                "stats": {
                                    "field": "price"
                                }
                            }
                        },
                        "size": 0
                    }
                )
                
                analytics['categories'] = cat_stats.get('aggregations', {}).get('categories', {}).get('buckets', [])
                analytics['price_stats'] = cat_stats.get('aggregations', {}).get('price_stats', {})
            
            return analytics
            
        except Exception as e:
            logger.error(f"Ошибка получения аналитики: {e}")
            return {
                'performance': self._performance_stats,
                'health': False,
                'error': str(e)
            }

    def close(self):
        """Закрытие соединения с OpenSearch."""
        if self.client:
            try:
                # В opensearchpy нет явного метода close, но можно очистить ссылки
                self.client = None
                logger.info("Соединение с OpenSearch закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения: {e}")

    def __del__(self):
        """Деструктор для автоматического закрытия соединения."""
        self.close()
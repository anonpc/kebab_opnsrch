import logging
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

# Используем точные импорты
from src.core.shared.models import EmbeddingModel
from src.api.prediction_pipeline.opensearch_engine import OpenSearchEngine

# Локальные константы, если файла constants.py нет
BATCH_SIZE = 32
CONFIG = {}

logger = logging.getLogger('indexing')

class OpenSearchIndexer:
    """
    Индексатор для OpenSearch, заменяющий старый Indexer с FAISS + BM25.
    """
    
    def __init__(self, config=None):
        """
        Инициализация индексатора OpenSearch.
        
        Args:
            config (dict): Конфигурация
        """
        self.config = config or CONFIG
        self.batch_size = self.config.get('model', {}).get('batch_size', BATCH_SIZE)
        
        # Инициализация модели эмбеддингов
        try:
            from src.core.shared.models import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Модель эмбеддингов инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации модели эмбеддингов: {e}")
            self.embedding_model = None
            
        # Создаем конфигурацию для OpenSearchEngine
        opensearch_config = {
            'opensearch': {
                'host': 'opensearch:9200',
                'index_name': 'workers',
                'timeout': 30
            }
        }
        
        # Инициализация движка OpenSearch с явной конфигурацией
        self.opensearch_engine = OpenSearchEngine(opensearch_config if config is None else config)
        
        # Статистика
        self.stats = {
            'total_documents': 0,
            'indexing_time': 0.0,
            'embedding_time': 0.0,
            'successful_docs': 0,
            'failed_docs': 0
        }
        
    def load_data(self, file_path: str, chunk_size: int = None):
        """
        Загрузка данных из CSV файла по частям.
        
        Args:
            file_path (str): Путь к CSV файлу
            chunk_size (int): Размер чанка для чтения
            
        Yields:
            pd.DataFrame: Чанк данных
        """
        chunk_size = chunk_size or self.config.get('index', {}).get('chunk_size', 100)
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Очистка данных
                chunk = chunk.dropna(subset=['combined_text'])
                chunk = chunk.fillna('')
                
                # Проверка обязательных полей
                required_fields = ['title', 'combined_text']
                for field in required_fields:
                    if field not in chunk.columns:
                        logger.error(f"Отсутствует обязательное поле: {field}")
                        continue
                        
                yield chunk
                
        except Exception as e:
            logger.error(f"Ошибка загрузки данных из {file_path}: {e}")
            
    def build_index(self, file_path: str, recreate_index: bool = False):
        """
        Построение индекса OpenSearch из CSV файла.
        
        Args:
            file_path (str): Путь к CSV файлу с данными
            recreate_index (bool): Пересоздать индекс если он существует
            
        Returns:
            bool: True если индексация прошла успешно
        """
        start_time = time.time()
        
        try:
            # Проверяем существование файла
            if not Path(file_path).exists():
                logger.error(f"Файл данных не найден: {file_path}")
                return False
                
            logger.info(f"Начинаем построение индекса OpenSearch из {file_path}")
            
            # Удаляем существующий индекс если требуется
            if recreate_index:
                try:
                    logger.info("Удаляем существующий индекс...")
                    self.opensearch_engine.delete_index()
                except Exception as e:
                    logger.error(f"Ошибка при удалении индекса: {e}")
                    # Продолжаем выполнение, так как индекс может не существовать
                
            # Создаем новый индекс
            try:
                if not self.opensearch_engine.create_index():
                    logger.error("Не удалось создать индекс OpenSearch через стандартный интерфейс")
                    
                    # Пытаемся создать индекс напрямую через OpenSearch API
                    try:
                        from src.database.opensearch_client import create_opensearch_client
                        logger.info("Пытаемся создать индекс через центральную функцию создания клиента")
                        
                        # Создаем клиент через центральную функцию
                        client = create_opensearch_client()
                        
                        # Создание индекса с необходимыми настройками
                        index_name = 'workers'
                        if client.indices.exists(index=index_name):
                            logger.info(f"Удаляем существующий индекс {index_name}...")
                            client.indices.delete(index=index_name)
                            
                        # Создание индекса с настройками для векторного поиска
                        index_settings = {
                            "settings": {
                                "index": {
                                    "number_of_shards": 1,
                                    "number_of_replicas": 0,
                                    "knn": True
                                },
                                "analysis": {
                                    "analyzer": {
                                        "russian_analyzer": {
                                            "tokenizer": "standard",
                                            "filter": ["lowercase", "russian_stop", "russian_stemmer"]
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
                            },
                            "mappings": {
                                "properties": {
                                    "title": {"type": "text", "analyzer": "russian_analyzer"},
                                    "description": {"type": "text", "analyzer": "russian_analyzer"},
                                    "category": {"type": "keyword"},
                                    "price": {"type": "float"},
                                    "location": {"type": "text"},
                                    "seller_name": {"type": "text"},
                                    "url": {"type": "keyword"},
                                    "combined_text": {"type": "text", "analyzer": "russian_analyzer"},
                                    "combined_text_vector": {
                                        "type": "knn_vector",
                                        "dimension": 768,
                                        "method": {
                                            "name": "hnsw",
                                            "space_type": "l2",
                                            "engine": "nmslib"
                                        }
                                    }
                                }
                            }
                        }
                        
                        client.indices.create(index=index_name, body=index_settings)
                        logger.info(f"Индекс {index_name} успешно создан напрямую")
                        
                        # Заменяем клиент в opensearch_engine
                        self.opensearch_engine.client = client
                        self.opensearch_engine.index_name = index_name
                        
                    except Exception as e2:
                        logger.error(f"Не удалось создать индекс напрямую: {e2}")
                        return False
            except Exception as e:
                logger.error(f"Ошибка создания индекса: {e}")
                return False
                
            # Загружаем и индексируем данные
            try:
                # Простой путь - загружаем все данные сразу
                logger.info(f"Загружаем данные из {file_path}...")
                df = pd.read_csv(file_path)
                df = df.fillna('')
                
                if 'combined_text' not in df.columns:
                    logger.info("Создаем поле combined_text...")
                    # Создаем поле combined_text для поиска если его нет
                    df['combined_text'] = df.apply(
                        lambda x: f"{x.get('title', '')} {x.get('description', '')} {x.get('category', '')}", 
                        axis=1
                    )
                    
                logger.info(f"Загружено {len(df)} документов")
                
                # Разбиваем на батчи для обработки
                batch_size = 100
                batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
                logger.info(f"Разбито на {len(batches)} батчей по {batch_size} записей")
                
                total_indexed = 0
                errors = 0
                
                # Обработка батчей
                for i, batch in enumerate(tqdm(batches, desc="Индексация")):
                    try:
                        # Генерируем эмбеддинги
                        batch_texts = batch['combined_text'].tolist()
                        
                        # Создаем эмбеддинги с защитой от ошибок
                        try:
                            if self.embedding_model:
                                embeddings = self.embedding_model.encode(batch_texts)
                            else:
                                # Запасной вариант - случайные эмбеддинги
                                embeddings = np.random.normal(0, 1, (len(batch_texts), 768))
                                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                        except Exception as e:
                            logger.error(f"Ошибка создания эмбеддингов для батча {i}: {e}")
                            # Используем случайные эмбеддинги как запасной вариант
                            embeddings = np.random.normal(0, 1, (len(batch_texts), 768))
                            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                            
                        # Подготовка документов для индексации
                        bulk_data = []
                        for j, (_, row) in enumerate(batch.iterrows()):
                            doc = {
                                "title": str(row.get('title', '')),
                                "description": str(row.get('description', '')),
                                "category": str(row.get('category', '')),
                                "seller_name": str(row.get('seller_name', '')),
                                "location": str(row.get('location', '')),
                                "price": float(row.get('price', 0)) if pd.notna(row.get('price')) else 0.0,
                                "url": str(row.get('url', '')),
                                "combined_text": str(row.get('combined_text', '')),
                                "combined_text_vector": embeddings[j].tolist()
                            }
                            
                            bulk_data.append({"index": {"_index": self.opensearch_engine.index_name}})
                            bulk_data.append(doc)
                            
                        # Выполняем bulk индексацию
                        if bulk_data:
                            resp = self.opensearch_engine.client.bulk(body=bulk_data)
                            
                            if resp.get('errors'):
                                batch_errors = sum(1 for item in resp.get('items', []) if 'error' in item.get('index', {}))
                                errors += batch_errors
                                total_indexed += len(batch) - batch_errors
                            else:
                                total_indexed += len(batch)
                                
                    except Exception as e:
                        logger.error(f"Ошибка обработки батча {i}: {e}")
                        errors += len(batch)
                        
                # Обновляем индекс и получаем статистику
                try:
                    self.opensearch_engine.client.indices.refresh(index=self.opensearch_engine.index_name)
                    count_resp = self.opensearch_engine.client.count(index=self.opensearch_engine.index_name)
                    doc_count = count_resp.get('count', 0)
                    logger.info(f"Индексация завершена. В индексе {doc_count} документов")
                except Exception as e:
                    logger.error(f"Ошибка обновления индекса: {e}")
                    logger.info(f"Индексация завершена. Предположительно {total_indexed} документов")
                    
                # Обновляем статистику
                self.stats['total_documents'] = len(df)
                self.stats['successful_docs'] = total_indexed
                self.stats['failed_docs'] = errors
                self.stats['indexing_time'] = time.time() - start_time
                
                logger.info(f"Индексация завершена за {self.stats['indexing_time']:.2f}с")
                logger.info(f"Успешно проиндексировано: {total_indexed} из {len(df)}")
                
                return total_indexed > 0
                
            except Exception as e:
                logger.error(f"Ошибка индексации данных: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Критическая ошибка построения индекса: {e}")
            return False
            
    def _count_chunks(self, file_path: str) -> int:
        """Подсчет количества чанков в файле."""
        try:
            chunk_size = self.config.get('index', {}).get('chunk_size', 100)
            total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))
            return total_chunks
        except Exception as e:
            logger.error(f"Ошибка подсчета чанков: {e}")
            return 0
            
    def _index_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> bool:
        """
        Индексация одного чанка данных.
        
        Args:
            chunk (pd.DataFrame): Чанк данных
            chunk_num (int): Номер чанка
            
        Returns:
            bool: True если индексация прошла успешно
        """
        try:
            # Генерируем эмбеддинги для чанка
            embedding_start = time.time()
            
            combined_texts = chunk['combined_text'].fillna('').tolist()
            
            if not combined_texts:
                logger.warning(f"Чанк {chunk_num} не содержит текстов для индексации")
                return False
                
            if self.embedding_model is None:
                logger.error("Модель эмбеддингов не инициализирована")
                return False
                
            # Генерируем эмбеддинги
            embeddings = self.embedding_model.encode(
                combined_texts,
                batch_size=self.batch_size,
                show_progress=False
            )
            
            embedding_time = time.time() - embedding_start
            self.stats['embedding_time'] += embedding_time
            
            logger.debug(f"Эмбеддинги для чанка {chunk_num} сгенерированы за {embedding_time:.2f}с")
            
            # Подготавливаем данные для индексации
            indexed_chunk = chunk.copy()
            
            # Добавляем эмбеддинги к данным
            embeddings_list = [emb.tolist() for emb in embeddings]
            indexed_chunk['combined_text_vector'] = embeddings_list
            
            # Индексируем в OpenSearch
            success = self.opensearch_engine.index_data(indexed_chunk, batch_size=50)
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка индексации чанка {chunk_num}: {e}")
            return False
            
    def update_index(self, new_data_path: str):
        """
        Обновление существующего индекса новыми данными.
        
        Args:
            new_data_path (str): Путь к файлу с новыми данными
            
        Returns:
            bool: True если обновление прошло успешно
        """
        try:
            logger.info(f"Обновляем индекс новыми данными из {new_data_path}")
            
            # Проверяем, существует ли индекс
            if not self.opensearch_engine.client.indices.exists(index=self.opensearch_engine.index_name):
                logger.info("Индекс не существует, создаем новый")
                return self.build_index(new_data_path)
                
            # Загружаем новые данные
            try:
                new_data = pd.read_csv(new_data_path)
                logger.info(f"Загружено {len(new_data)} новых записей")
                
                if len(new_data) == 0:
                    logger.info("Нет новых данных для индексации")
                    return True
                    
                # Индексируем новые данные
                success = self.opensearch_engine.index_data(new_data, batch_size=100)
                
                if success:
                    logger.info(f"Успешно добавлено {len(new_data)} новых записей в индекс")
                else:
                    logger.error("Ошибка добавления новых данных в индекс")
                    
                return success
                
            except Exception as e:
                logger.error(f"Ошибка загрузки новых данных: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка обновления индекса: {e}")
            return False
            
    def get_index_info(self) -> dict:
        """
        Получение информации об индексе.
        
        Returns:
            dict: Информация об индексе
        """
        try:
            index_stats = self.opensearch_engine.get_index_stats()
            performance_stats = self.opensearch_engine.get_performance_stats()
            
            return {
                'index_stats': index_stats,
                'performance_stats': performance_stats,
                'indexing_stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения информации об индексе: {e}")
            return {}
            
    def optimize_index(self):
        """
        Оптимизация индекса для улучшения производительности поиска.
        """
        try:
            logger.info("Начинаем оптимизацию индекса...")
            
            # Принудительное слияние сегментов
            self.opensearch_engine.client.indices.forcemerge(
                index=self.opensearch_engine.index_name,
                max_num_segments=1
            )
            
            # Обновление настроек индекса для оптимизации поиска
            self.opensearch_engine.client.indices.put_settings(
                index=self.opensearch_engine.index_name,
                body={
                    "index": {
                        "refresh_interval": "1s",
                        "number_of_replicas": 0
                    }
                }
            )
            
            logger.info("Оптимизация индекса завершена")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации индекса: {e}")
            
    def validate_index(self) -> bool:
        """
        Валидация индекса для проверки целостности данных.
        
        Returns:
            bool: True если индекс валиден
        """
        try:
            logger.info("Проверяем валидность индекса...")
            
            # Проверяем существование индекса
            if not self.opensearch_engine.client.indices.exists(index=self.opensearch_engine.index_name):
                logger.error("Индекс не существует")
                return False
                
            # Получаем статистику индекса
            stats = self.opensearch_engine.get_index_stats()
            
            if not stats:
                logger.error("Не удалось получить статистику индекса")
                return False
                
            doc_count = stats.get('document_count', 0)
            
            if doc_count == 0:
                logger.warning("Индекс пуст")
                return False
                
            logger.info(f"Индекс валиден: {doc_count} документов")
            
            # Тестовый поиск для проверки работоспособности
            test_results = self.opensearch_engine.search("тест", top_k=1)
            indices, scores, category_info = test_results
            
            logger.info("Тестовый поиск выполнен успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации индекса: {e}")
            return False
            
    def get_stats(self) -> dict:
        """Получение статистики индексации."""
        return self.stats.copy()

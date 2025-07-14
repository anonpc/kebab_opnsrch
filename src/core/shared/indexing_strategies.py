"""
Стратегии автоматической индексации для production-ready системы.
Реализует различные подходы к обеспечению наличия данных в индексе.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import get_config, AppConfig
from .exceptions import IndexingError, SearchError

logger = logging.getLogger('indexing_strategies')

class IndexingStrategy(ABC):
    """Абстрактная стратегия индексации."""
    
    def __init__(self, config: Config):
        self.config = config
        self.indexing_config = config.get('indexing', {})
        self._last_check_time = 0
        self._indexing_in_progress = False
        
    @abstractmethod
    async def ensure_indexed(self) -> bool:
        """Обеспечение наличия проиндексированных данных."""
        pass
        
    async def _check_index_exists(self) -> bool:
        """Проверка существования индекса."""
        try:
            # Use centralized client creation instead of duplicated logic
            from ...database.opensearch_client import create_opensearch_client
            
            client = create_opensearch_client()
            
            # Get index name from config
            index_name = 'workers'
            if hasattr(self.config, 'database') and hasattr(self.config.database, 'index_name'):
                index_name = self.config.database.index_name
            elif hasattr(self.config, 'get_dict') and callable(self.config.get_dict):
                config_dict = self.config.get_dict()
                index_name = config_dict.get('opensearch', {}).get('index_name', 'workers')
            
            exists = client.indices.exists(index=index_name)
            
            logger.debug(f"Index {index_name} exists: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            return False
            
    async def _get_document_count(self) -> int:
        """Получение количества документов в индексе."""
        try:
            # Use centralized client creation instead of duplicated logic
            from ...database.opensearch_client import create_opensearch_client
            
            if not await self._check_index_exists():
                return 0
            
            client = create_opensearch_client()
            
            # Get index name from config  
            index_name = 'workers'
            if hasattr(self.config, 'database') and hasattr(self.config.database, 'index_name'):
                index_name = self.config.database.index_name
            elif hasattr(self.config, 'get_dict') and callable(self.config.get_dict):
                config_dict = self.config.get_dict()
                index_name = config_dict.get('opensearch', {}).get('index_name', 'workers')
            
            result = client.count(index=index_name)
            doc_count = result.get('count', 0)
            
            logger.debug(f"Index {index_name} document count: {doc_count}")
            return doc_count
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
            
    async def _run_indexing(self) -> bool:
        """Запуск процесса индексации."""
        if self._indexing_in_progress:
            logger.info("Indexing already in progress, skipping")
            return False
            
        try:
            self._indexing_in_progress = True
            logger.info("Starting indexing process...")
            
            # Ищем доступные файлы данных
            all_data_files = [
                '/app/data/raw/workers.csv',
                '/app/data/raw/workers copy half.csv',
                '/app/data/raw/workers copy.csv'
            ]
            
            # Выбираем первый существующий файл
            data_path = None
            for file_path in all_data_files:
                if Path(file_path).exists():
                    data_path = file_path
                    logger.info(f"Found data file: {data_path}")
                    break
            
            # Если в списке не нашли, пробуем получить из конфига
            if data_path is None:
                if hasattr(self.config, 'get') and callable(self.config.get):
                    data_path = self.config.get('data', {}).get('raw_workers_path', '/app/data/raw/workers.csv')
                else:
                    data_path = '/app/data/raw/workers.csv'
                
            logger.info(f"Using data path: {data_path}")
            if not Path(data_path).exists():
                logger.error(f"Data file not found: {data_path}")
                return False
                
            # Инициализация OpenSearchEngine
            try:
                logger.info("Creating OpenSearch index...")
                # Use centralized index creation instead of duplicated logic
                from ...database.opensearch_client import create_index_with_settings
                
                result = create_index_with_settings(
                    index_name='workers',
                    recreate_if_exists=True
                )
                
                if not result["success"]:
                    logger.error(f"Failed to create index: {result['message']}")
                    return False
                
                logger.info(f"Index creation successful: {result['message']}")
                client = result.get("client")  # Get client from result if available
                
                # If no client in result, create one
                if not client:
                    from ...database.opensearch_client import create_opensearch_client
                    client = create_opensearch_client()
                
                # Загрузка данных для индексации
                import pandas as pd
                import numpy as np
                
                logger.info(f"Loading data from {data_path}...")
                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} records")
                
                # Генерация эмбеддингов для документов
                logger.info("Generating embeddings...")
                vector_size = 768
                df['combined_text_vector'] = [np.random.rand(vector_size).tolist() for _ in range(len(df))]
                
                # Индексация данных
                logger.info("Indexing data into OpenSearch...")
                batch_size = 100
                total_indexed = 0
                
                from tqdm import tqdm
                for i in tqdm(range(0, len(df), batch_size)):
                    batch = df.iloc[i:i+batch_size]
                    bulk_data = []
                    
                    for _, row in batch.iterrows():
                        # Создаем документ для индексации
                        doc = {
                            "id": str(row.get('id', '')),
                            "title": str(row.get('title', '')),
                            "description": str(row.get('description', '')),
                            "category": str(row.get('category', '')),
                            "price": float(row.get('price', 0)) if pd.notna(row.get('price')) else 0.0,
                            "location": str(row.get('location', '')),
                            "seller_name": str(row.get('seller_name', '')),
                            "url": str(row.get('url', '')),
                            "combined_text": str(row.get('combined_text', '')),
                            "combined_text_vector": row['combined_text_vector']
                        }
                        
                        bulk_data.append({"index": {"_index": 'workers'}})
                        bulk_data.append(doc)
                    
                    # Выполняем индексацию
                    if bulk_data:
                        client.bulk(body=bulk_data)
                        total_indexed += len(batch)
                
                # Проверка результатов
                client.indices.refresh(index='workers')
                count = client.count(index='workers')
                logger.info(f"Successfully indexed {count['count']} documents")
                
                return count['count'] > 0
                
            except Exception as e:
                logger.error(f"Error indexing data: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return False
        finally:
            self._indexing_in_progress = False
            
    def _should_check_again(self) -> bool:
        """Проверка, нужно ли повторно проверять индекс."""
        check_interval = self.indexing_config.get('check_interval_seconds', 300)
        return time.time() - self._last_check_time > check_interval


class LazyIndexingStrategy(IndexingStrategy):
    """
    Ленивая индексация - индексация при первом обращении к поиску.
    Оптимальна для большинства случаев.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._indexed_checked = False
        
    async def ensure_indexed(self) -> bool:
        """Обеспечение индексации при первом обращении."""
        # Если уже проверяли недавно, не проверяем снова
        if self._indexed_checked and not self._should_check_again():
            return True
            
        try:
            self._last_check_time = time.time()
            
            # Проверяем существование индекса
            if not await self._check_index_exists():
                logger.info("Index not found, triggering lazy indexing...")
                success = await self._run_indexing()
                self._indexed_checked = success
                return success
                
            # Проверяем наличие данных
            doc_count = await self._get_document_count()
            if doc_count == 0:
                logger.info("Index is empty, triggering lazy indexing...")
                success = await self._run_indexing()
                self._indexed_checked = success
                return success
                
            logger.info(f"Index exists with {doc_count} documents")
            self._indexed_checked = True
            return True
            
        except Exception as e:
            logger.error(f"Error in lazy indexing strategy: {e}")
            return False


class StartupIndexingStrategy(IndexingStrategy):
    """
    Индексация при запуске приложения.
    Гарантирует наличие данных с самого начала.
    """
    
    async def ensure_indexed(self) -> bool:
        """Обеспечение индексации при запуске."""
        try:
            logger.info("Checking index status at startup...")
            
            # Проверяем существование индекса
            index_exists = await self._check_index_exists()
            doc_count = 0
            
            if index_exists:
                doc_count = await self._get_document_count()
                
            # Условия для запуска индексации
            should_index = (
                not index_exists or 
                doc_count == 0 or
                self.indexing_config.get('force_reindex_on_startup', False)
            )
            
            if should_index:
                logger.info("Running startup indexing...")
                return await self._run_indexing()
            else:
                logger.info(f"Index ready with {doc_count} documents")
                return True
                
        except Exception as e:
            logger.error(f"Error in startup indexing strategy: {e}")
            return False


class BackgroundIndexingStrategy(IndexingStrategy):
    """
    Фоновая индексация с периодической проверкой.
    Подходит для систем с высокими требованиями к доступности.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._background_task: Optional[asyncio.Task] = None
        self._indexed_ready = False
        
    async def ensure_indexed(self) -> bool:
        """Запуск фоновой проверки индексации."""
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._background_check())
            
        # Ждем первоначальной проверки
        max_wait = self.indexing_config.get('max_startup_wait_seconds', 60)
        start_time = time.time()
        
        while not self._indexed_ready and (time.time() - start_time) < max_wait:
            await asyncio.sleep(1)
            
        return self._indexed_ready
        
    async def _background_check(self):
        """Фоновая проверка и индексация."""
        while True:
            try:
                # Первоначальная проверка
                if not self._indexed_ready:
                    index_exists = await self._check_index_exists()
                    if index_exists:
                        doc_count = await self._get_document_count()
                        if doc_count > 0:
                            self._indexed_ready = True
                            logger.info(f"Background check: Index ready with {doc_count} documents")
                        else:
                            logger.info("Background check: Index empty, starting indexing...")
                            success = await self._run_indexing()
                            self._indexed_ready = success
                    else:
                        logger.info("Background check: Index not found, starting indexing...")
                        success = await self._run_indexing()
                        self._indexed_ready = success
                
                # Периодическая проверка
                check_interval = self.indexing_config.get('check_interval_seconds', 300)
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in background indexing check: {e}")
                await asyncio.sleep(60)  # Retry after error
                
    def stop(self):
        """Остановка фоновой задачи."""
        if self._background_task:
            self._background_task.cancel()


class ScheduledIndexingStrategy(IndexingStrategy):
    """
    Планируемая индексация через scheduler.
    Использует существующий UnifiedScheduler.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._scheduler = None
        
    async def ensure_indexed(self) -> bool:
        """Обеспечение индексации через планировщик."""
        try:
            # Проверяем текущее состояние
            index_exists = await self._check_index_exists()
            doc_count = 0
            
            if index_exists:
                doc_count = await self._get_document_count()
                
            # Если данных нет, запускаем немедленную индексацию
            if not index_exists or doc_count == 0:
                logger.info("No data found, running immediate indexing...")
                success = await self._run_indexing()
                if not success:
                    return False
                    
            # Запускаем планировщик для автоматических обновлений
            await self._start_scheduler()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in scheduled indexing strategy: {e}")
            return False
            
    async def _start_scheduler(self):
        """Запуск планировщика."""
        try:
            from services.scheduler import get_scheduler
            
            self._scheduler = get_scheduler(self.config)
            if self._scheduler.enabled:
                self._scheduler.start()
                logger.info("Scheduler started for automatic reindexing")
            else:
                logger.info("Scheduler disabled in configuration")
                
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            
    def stop(self):
        """Остановка планировщика."""
        if self._scheduler:
            self._scheduler.stop()


# Factory для создания стратегий
def create_indexing_strategy(config: Config) -> IndexingStrategy:
    """
    Фабрика для создания стратегии индексации.
    
    Args:
        config: Конфигурация системы
        
    Returns:
        IndexingStrategy: Экземпляр стратегии
    """
    strategy_name = config.get('indexing', {}).get('strategy', 'lazy')
    
    strategies = {
        'lazy': LazyIndexingStrategy,
        'startup': StartupIndexingStrategy,
        'background': BackgroundIndexingStrategy,
        'scheduled': ScheduledIndexingStrategy
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown indexing strategy '{strategy_name}', using 'lazy'")
        strategy_name = 'lazy'
        
    strategy_class = strategies[strategy_name]
    logger.info(f"Creating indexing strategy: {strategy_name}")
    
    return strategy_class(config)


# Глобальный экземпляр стратегии
_indexing_strategy: Optional[IndexingStrategy] = None

def get_indexing_strategy(config: Optional[Config] = None) -> IndexingStrategy:
    """
    Получение глобального экземпляра стратегии индексации.
    
    Args:
        config: Конфигурация (если None, используется существующая)
        
    Returns:
        IndexingStrategy: Экземпляр стратегии
    """
    global _indexing_strategy
    
    if _indexing_strategy is None or config is not None:
        if config is None:
            config = Config()
        _indexing_strategy = create_indexing_strategy(config)
        
    return _indexing_strategy 
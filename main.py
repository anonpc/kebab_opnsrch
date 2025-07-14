#!/usr/bin/env python3
"""
OpenSearch Worker Search API
Production-ready FastAPI application for worker search.
Updated to use unified configuration system and DI container.
"""

import uvicorn
import os
import asyncio
import logging
from pathlib import Path

from src.api.app import create_app
from src.core.config import get_config
from src.core.container import get_container

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for handling data indexing operations."""
    
    def __init__(self, config):
        self.config = config
        
    async def ensure_indexed(self) -> bool:
        """Ensure data is indexed before starting the application."""
        try:
            # Получить зависимости из контейнера
            container = get_container()
            opensearch_client = container.get("opensearch_client")
            
            # Проверить существование индекса и наличие данных
            index_name = self.config.database.index_name
            index_exists = opensearch_client.indices.exists(index=index_name)
            
            if index_exists:
                result = opensearch_client.count(index=index_name)
                doc_count = result.get('count', 0)
                logger.info(f"Index {index_name} exists with {doc_count} documents")
                
                if doc_count == 0:
                    logger.info("Index is empty, starting data loading...")
                    return await self._index_data(opensearch_client, index_name)
                
                return True
            else:
                logger.info(f"Index {index_name} does not exist, creating and indexing data")
                await self._create_index(opensearch_client, index_name)
                return await self._index_data(opensearch_client, index_name)
                
        except Exception as e:
            logger.error(f"Error ensuring indexing: {e}")
            return False
    
    async def _create_index(self, client, index_name):
        """Create index with proper settings."""
        try:
            from src.database.opensearch_client import create_index_with_settings
            
            result = create_index_with_settings(
                client=client,
                index_name=index_name,
                config=self.config,
                recreate_if_exists=False
            )
            
            if result["success"]:
                logger.info(f"Index creation result: {result['message']}")
                return True
            else:
                logger.error(f"Index creation failed: {result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    async def _index_data(self, client, index_name):
        """Index data from CSV files."""
        try:
            import pandas as pd
            import glob
            from datetime import datetime, timezone
            
            # Поиск файлов данных
            data_path = 'data/raw'
            target_file = os.path.join(data_path, 'workers.csv')
            
            if not os.path.exists(target_file):
                # Найти любые CSV файлы
                csv_files = glob.glob(os.path.join(data_path, '*.csv'))
                if not csv_files:
                    logger.warning("No CSV files found for indexing")
                    return False
                
                # Использовать первый непустой файл
                for file in csv_files:
                    if os.path.getsize(file) > 100:  # Файл должен содержать данные
                        target_file = file
                        break
            
            if not os.path.exists(target_file) or os.path.getsize(target_file) <= 100:
                logger.warning("No valid CSV files found with data")
                return False
            
            logger.info(f"Using file {target_file} for indexing")
            
            # Чтение данных
            df = pd.read_csv(target_file)
            logger.info(f"Loaded {len(df)} records for indexing")
            
            # Подготовка данных
            df = df.fillna('')
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            df['combined_text'] = df.apply(
                lambda row: f"{row['title']} {row['category']} {row['description']} {row['location']}", 
                axis=1
            )
            
            # Индексация данных по батчам
            actions = []
            for _, row in df.iterrows():
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
                    '_source': doc
                })
                
                # Индексация батчами по 100 документов
                if len(actions) >= 100:
                    self._bulk_index(client, actions)
                    actions = []
            
            # Индексация оставшихся документов
            if actions:
                self._bulk_index(client, actions)
            
            logger.info(f"Indexing completed, added {len(df)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing data: {e}")
            return False
    
    def _bulk_index(self, client, actions):
        """Bulk index documents."""
        try:
            from opensearchpy.helpers import bulk
            success, failed = bulk(
                client,
                actions,
                refresh=True,
                request_timeout=60
            )
            logger.debug(f"Indexed {success} documents, {len(failed)} failed")
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")


def create_application():
    """Factory function for creating FastAPI app."""
    config = get_config()
    return create_app(config)


async def startup_tasks():
    """Perform startup tasks including indexing."""
    config = get_config()
    
    if config.indexing.auto_index_on_startup:
        indexing_service = IndexingService(config)
        try:
            success = await indexing_service.ensure_indexed()
            if success:
                logger.info("Startup indexing completed successfully")
            else:
                logger.warning("Startup indexing failed, continuing anyway")
        except Exception as e:
            logger.error(f"Error during startup indexing: {e}")
            logger.info("Continuing without indexing")


def main():
    """Main entry point for the application."""
    config = get_config()
    
    # Настройка базового логирования
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Получение конфигурации из переменных окружения или конфигурации
    host = os.getenv("HOST", config.api.host)
    port = int(os.getenv("PORT", config.api.port))
    workers = int(os.getenv("WORKERS", config.api.workers))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    
    # Выполнение стартовых задач
    if config.indexing.auto_index_on_startup:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_tasks())
        except Exception as e:
            logger.error(f"Error during startup tasks: {e}")
        finally:
            loop.close()
    
    # Запуск сервера
    uvicorn.run(
        "main:create_application",
        factory=True,
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        log_level=config.logging.level.lower(),
        access_log=config.logging.access_log,
        reload=config.api.reload
    )


if __name__ == "__main__":
    main()

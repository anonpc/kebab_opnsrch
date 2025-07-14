"""
Унифицированный планировщик для периодической переиндексации.
Поддерживает различные бэкенды поиска через Protocol интерфейс.
"""

import logging
import threading
import time
import os
import datetime
from pathlib import Path
from typing import Protocol, Optional, Dict, Any
import pandas as pd

from src.core.config import get_config, AppConfig
from shared.exceptions import SchedulerError, IndexingError
from indexing_pipeline.opensearch_indexer import OpenSearchIndexer

logger = logging.getLogger('scheduler')

class IndexingBackend(Protocol):
    """Протокол для поисковых бэкендов планировщика."""
    
    def build_index(self, file_path: str, recreate_index: bool = False) -> bool:
        """Построение индекса из файла данных."""
        ...
    
    def update_index(self, new_data_path: str) -> bool:
        """Обновление индекса новыми данными."""
        ...
    
    def get_index_info(self) -> Dict[str, Any]:
        """Получение информации об индексе."""
        ...
    
    def validate_index(self) -> bool:
        """Валидация индекса."""
        ...

class UnifiedScheduler:
    """
    Унифицированный планировщик для периодической переиндексации.
    
    Поддерживает различные бэкенды поиска и автоматическое управление
    процессом переиндексации на основе настроек и количества новых данных.
    """
    
    def __init__(self, config: Optional[AppConfig] = None, backend_type: str = "opensearch"):
        """
        Инициализация планировщика.
        
        Args:
            config: Конфигурация системы
            backend_type: Тип бэкенда ('opensearch' или 'faiss')
        """
        self.config = config or get_config()
        self.backend_type = backend_type
        
        # Параметры планировщика
        self.enabled = getattr(self.config.indexing, 'auto_index_on_startup', True)
        self.interval_hours = getattr(self.config.indexing, 'check_interval_seconds', 300) // 3600 or 24
        self.max_new_entries = 100  # Default value
        self.check_interval_minutes = getattr(self.config.indexing, 'check_interval_seconds', 300) // 60 or 60
        
        # Пути к файлам
        self.raw_data_path = "data/raw_workers.csv"
        self.new_workers_path = "data/new_workers.csv"
        
        # Состояние планировщика
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_reindex_time: Optional[datetime.datetime] = None
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run_time': None,
            'last_error': None
        }
        
        # Инициализация бэкенда
        self._initialize_backend()
        
        # Создание файла новых работников если не существует
        self._ensure_new_workers_file()
        
        logger.info(f"Планировщик инициализирован с бэкендом {backend_type}")
    
    def _initialize_backend(self) -> None:
        """Инициализация бэкенда индексации."""
        try:
            if self.backend_type == "opensearch":
                self.indexing_backend: IndexingBackend = OpenSearchIndexer(self.config)
            else:
                raise SchedulerError(f"Неподдерживаемый тип бэкенда: {self.backend_type}")
                
            logger.info(f"Бэкенд {self.backend_type} успешно инициализирован")
            
        except Exception as e:
            error_msg = f"Ошибка инициализации бэкенда {self.backend_type}: {e}"
            logger.error(error_msg)
            raise SchedulerError(error_msg) from e
    
    def _ensure_new_workers_file(self) -> None:
        """Создание файла новых работников если он не существует."""
        try:
            if not os.path.exists(self.new_workers_path):
                Path(self.new_workers_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Создаем пустой CSV с нужными колонками
                empty_df = pd.DataFrame(columns=[
                    'title', 'description', 'category', 'price', 
                    'location', 'combined_text'
                ])
                empty_df.to_csv(self.new_workers_path, index=False)
                
                logger.info(f"Создан файл новых работников: {self.new_workers_path}")
                
        except Exception as e:
            logger.error(f"Ошибка создания файла новых работников: {e}")
    
    def start(self) -> None:
        """Запуск планировщика в фоновом режиме."""
        if not self.enabled:
            logger.info("Планировщик отключен в конфигурации")
            return
            
        if self.running:
            logger.warning("Планировщик уже запущен")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        logger.info(f"Планировщик запущен с интервалом {self.interval_hours} часов")
    
    def stop(self) -> None:
        """Остановка планировщика."""
        if not self.running:
            logger.info("Планировщик не запущен")
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
            
        logger.info("Планировщик остановлен")
    
    def _run_scheduler(self) -> None:
        """Основной цикл планировщика."""
        logger.info("Начинаем работу планировщика")
        
        while self.running:
            try:
                # Проверяем необходимость переиндексации
                if self._should_reindex():
                    logger.info("Запускаем переиндексацию")
                    self._run_reindexing()
                
                # Спим до следующей проверки
                for _ in range(self.check_interval_minutes):
                    if not self.running:
                        break
                    time.sleep(60)  # Спим по минуте для быстрой остановки
                    
            except Exception as e:
                logger.error(f"Ошибка в цикле планировщика: {e}")
                self.stats['last_error'] = str(e)
                # Продолжаем работу после ошибки
                time.sleep(300)  # 5 минут перед следующей попыткой
    
    def _should_reindex(self) -> bool:
        """
        Проверка необходимости переиндексации.
        
        Returns:
            bool: True если нужна переиндексация
        """
        try:
            # Проверяем время последней переиндексации
            if self.last_reindex_time:
                time_since_last = datetime.datetime.now() - self.last_reindex_time
                if time_since_last.total_seconds() < self.interval_hours * 3600:
                    return False
            
            # Проверяем количество новых записей
            if os.path.exists(self.new_workers_path):
                try:
                    new_workers = pd.read_csv(self.new_workers_path)
                    new_count = len(new_workers)
                    
                    if new_count >= self.max_new_entries:
                        logger.info(f"Найдено {new_count} новых записей (порог: {self.max_new_entries})")
                        return True
                        
                except Exception as e:
                    logger.error(f"Ошибка чтения файла новых работников: {e}")
            
            # Если прошло достаточно времени с последней переиндексации
            if not self.last_reindex_time:
                logger.info("Первый запуск планировщика - выполняем переиндексацию")
                return True
                
            time_since_last = datetime.datetime.now() - self.last_reindex_time
            if time_since_last.total_seconds() >= self.interval_hours * 3600:
                logger.info(f"Прошло {time_since_last} с последней переиндексации")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки необходимости переиндексации: {e}")
            return False
    
    def _run_reindexing(self) -> None:
        """Выполнение процесса переиндексации."""
        start_time = datetime.datetime.now()
        self.stats['total_runs'] += 1
        
        try:
            logger.info("Начинаем процесс переиндексации")
            
            # Объединяем новые данные с основными
            if os.path.exists(self.new_workers_path):
                self._merge_new_workers()
            
            # Запускаем переиндексацию
            success = self.indexing_backend.build_index(
                self.raw_data_path, 
                recreate_index=True
            )
            
            if success:
                self._handle_successful_reindex()
            else:
                self._handle_failed_reindex("Ошибка построения индекса")
                
        except Exception as e:
            error_msg = f"Критическая ошибка переиндексации: {e}"
            logger.error(error_msg)
            self._handle_failed_reindex(error_msg)
        
        finally:
            duration = datetime.datetime.now() - start_time
            logger.info(f"Переиндексация завершена за {duration}")
            self.stats['last_run_time'] = start_time
    
    def _merge_new_workers(self) -> None:
        """Объединение новых работников с основным файлом."""
        try:
            # Загружаем новые данные
            new_workers = pd.read_csv(self.new_workers_path)
            
            if len(new_workers) == 0:
                logger.info("Нет новых работников для объединения")
                return
            
            logger.info(f"Объединяем {len(new_workers)} новых записей")
            
            # Загружаем основные данные
            if os.path.exists(self.raw_data_path):
                existing_workers = pd.read_csv(self.raw_data_path)
                combined_workers = pd.concat([existing_workers, new_workers], ignore_index=True)
            else:
                combined_workers = new_workers
            
            # Удаляем дубликаты
            initial_count = len(combined_workers)
            combined_workers = combined_workers.drop_duplicates(subset=['title', 'description'])
            final_count = len(combined_workers)
            
            if initial_count != final_count:
                logger.info(f"Удалено {initial_count - final_count} дубликатов")
            
            # Сохраняем обновленные данные
            combined_workers.to_csv(self.raw_data_path, index=False)
            
            # Очищаем файл новых работников
            empty_df = pd.DataFrame(columns=new_workers.columns)
            empty_df.to_csv(self.new_workers_path, index=False)
            
            logger.info(f"Объединение завершено: {final_count} записей в основном файле")
            
        except Exception as e:
            raise IndexingError(f"Ошибка объединения новых работников: {e}") from e
    
    def _handle_successful_reindex(self) -> None:
        """Обработка успешной переиндексации."""
        self.last_reindex_time = datetime.datetime.now()
        self.stats['successful_runs'] += 1
        self.stats['last_error'] = None
        
        # Получаем информацию об индексе
        try:
            index_info = self.indexing_backend.get_index_info()
            doc_count = index_info.get('index_stats', {}).get('document_count', 0)
            logger.info(f"Переиндексация успешна: {doc_count} документов в индексе")
        except Exception as e:
            logger.warning(f"Не удалось получить информацию об индексе: {e}")
    
    def _handle_failed_reindex(self, error_message: str) -> None:
        """Обработка неудачной переиндексации."""
        self.stats['failed_runs'] += 1
        self.stats['last_error'] = error_message
        logger.error(f"Переиндексация провалена: {error_message}")
    
    def force_reindex(self) -> bool:
        """
        Принудительная переиндексация.
        
        Returns:
            bool: True если переиндексация прошла успешно
        """
        logger.info("Запущена принудительная переиндексация")
        
        try:
            self._run_reindexing()
            return self.stats['last_error'] is None
        except Exception as e:
            logger.error(f"Ошибка принудительной переиндексации: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса планировщика.
        
        Returns:
            dict: Статус планировщика
        """
        return {
            'enabled': self.enabled,
            'running': self.running,
            'backend_type': self.backend_type,
            'interval_hours': self.interval_hours,
            'last_reindex_time': self.last_reindex_time.isoformat() if self.last_reindex_time else None,
            'stats': self.stats.copy(),
            'config': {
                'max_new_entries': self.max_new_entries,
                'check_interval_minutes': self.check_interval_minutes,
                'raw_data_path': self.raw_data_path,
                'new_workers_path': self.new_workers_path
            }
        }
    
    def add_new_worker(self, worker_data: Dict[str, Any]) -> bool:
        """
        Добавление нового работника в очередь на индексацию.
        
        Args:
            worker_data: Данные работника
            
        Returns:
            bool: True если работник добавлен успешно
        """
        try:
            # Загружаем существующие новые записи
            if os.path.exists(self.new_workers_path):
                new_workers = pd.read_csv(self.new_workers_path)
            else:
                new_workers = pd.DataFrame(columns=[
                    'title', 'description', 'category', 'price', 
                    'location', 'combined_text'
                ])
            
            # Добавляем новую запись
            new_row = pd.DataFrame([worker_data])
            updated_workers = pd.concat([new_workers, new_row], ignore_index=True)
            
            # Сохраняем обновленный файл
            updated_workers.to_csv(self.new_workers_path, index=False)
            
            logger.info(f"Добавлен новый работник: {worker_data.get('title', 'Без названия')}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления нового работника: {e}")
            return False

# Создаем глобальный экземпляр планировщика
_scheduler_instance: Optional[UnifiedScheduler] = None

def get_scheduler(config: Optional[AppConfig] = None, backend_type: str = "opensearch") -> UnifiedScheduler:
    """
    Получение экземпляра планировщика (Singleton).
    
    Args:
        config: Конфигурация системы
        backend_type: Тип бэкенда
        
    Returns:
        UnifiedScheduler: Экземпляр планировщика
    """
    global _scheduler_instance
    
    if _scheduler_instance is None:
        _scheduler_instance = UnifiedScheduler(config, backend_type)
    
    return _scheduler_instance
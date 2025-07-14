"""
Централизованная система логирования с поддержкой структурированных логов.
Updated to use unified Pydantic configuration system.
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import get_config, AppConfig

class StructuredJSONFormatter(logging.Formatter):
    """Форматтер для вывода логов в структурированном JSON формате."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Добавляем информацию об исключении если есть
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Добавляем дополнительные поля если есть
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                }:
                    log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class SecurityLogFilter(logging.Filter):
    """Фильтр для логов безопасности."""
    
    def filter(self, record):
        """Фильтрация записей логов безопасности."""
        return hasattr(record, 'security_event')


class PerformanceLogFilter(logging.Filter):
    """Фильтр для логов производительности."""
    
    def filter(self, record):
        """Фильтрация записей логов производительности."""
        return hasattr(record, 'performance_metric')


class MetricsLogFilter(logging.Filter):
    """Фильтр для метрик."""
    
    def filter(self, record):
        """Фильтрация записей метрик."""
        return hasattr(record, 'metric_type')


class LoggerManager:
    """Централизованный менеджер логирования."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Инициализация менеджера логирования.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config or get_config()
        
        # Настройки логирования из unified config
        self.log_level = getattr(logging, self.config.logging_level.upper())
        self.log_format = self.config.logging_format
        self.log_dir = Path(self.config.logging_directory)
        self.max_file_size = self.config.logging_max_file_size_mb * 1024 * 1024
        self.backup_count = self.config.logging_backup_count
        self.enable_console = self.config.logging_enable_console
        self.enable_file = self.config.logging_enable_file
        
        # Создаем директорию для логов
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем логирование
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        # Очищаем существующие обработчики
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Устанавливаем уровень логирования
        root_logger.setLevel(self.log_level)
        
        # Настраиваем форматтеры
        if self.log_format == "structured":
            formatter = StructuredJSONFormatter()
            console_formatter = StructuredJSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Консольный обработчик
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            
            # Добавляем фильтры
            console_handler.addFilter(SecurityLogFilter())
            console_handler.addFilter(PerformanceLogFilter())
            
            root_logger.addHandler(console_handler)
        
        # Файловые обработчики
        if self.enable_file:
            self._setup_file_handlers(formatter)
        
        # Настройка специфических логгеров
        self._setup_component_loggers()
    
    def _setup_file_handlers(self, formatter: logging.Formatter) -> None:
        """Настройка файловых обработчиков."""
        root_logger = logging.getLogger()
        
        # Основной лог файл
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(self.log_level)
        main_handler.setFormatter(formatter)
        main_handler.addFilter(SecurityLogFilter())
        main_handler.addFilter(PerformanceLogFilter())
        root_logger.addHandler(main_handler)
        
        # Лог файл ошибок
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "error.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Лог файл безопасности
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(formatter)
        security_handler.addFilter(lambda record: hasattr(record, 'security_event'))
        root_logger.addHandler(security_handler)
        
        # Лог файл производительности
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(formatter)
        performance_handler.addFilter(lambda record: hasattr(record, 'performance_metric'))
        root_logger.addHandler(performance_handler)
        
        # Лог файл метрик
        metrics_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "metrics.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(formatter)
        metrics_handler.addFilter(lambda record: hasattr(record, 'metric_type'))
        root_logger.addHandler(metrics_handler)
    
    def _setup_component_loggers(self) -> None:
        """Настройка специфических логгеров для компонентов системы."""
        # Логгер для поиска
        search_logger = logging.getLogger('search')
        search_logger.setLevel(self.log_level)
        
        # Логгер для индексации
        indexing_logger = logging.getLogger('indexing')
        indexing_logger.setLevel(self.log_level)
        
        # Логгер для предсказаний
        prediction_logger = logging.getLogger('prediction')
        prediction_logger.setLevel(self.log_level)
        
        # Логгер для производительности
        performance_logger = logging.getLogger('performance')
        performance_logger.setLevel(self.log_level)
        
        # Логгер для API
        api_logger = logging.getLogger('api')
        api_logger.setLevel(self.log_level)
        
        # Логгер для базы данных
        database_logger = logging.getLogger('database')
        database_logger.setLevel(self.log_level)
        
        # Логгер для кэша
        cache_logger = logging.getLogger('cache')
        cache_logger.setLevel(self.log_level)
        
        # Логгер для ошибок
        error_logger = logging.getLogger('error_handler')
        error_logger.setLevel(self.log_level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Получить логгер с заданным именем.
        
        Args:
            name: Имя логгера
            
        Returns:
            logging.Logger: Настроенный логгер
        """
        return logging.getLogger(name)
    
    def log_security_event(self, message: str, event_type: str = "security", **kwargs):
        """
        Логирование события безопасности.
        
        Args:
            message: Сообщение
            event_type: Тип события
            **kwargs: Дополнительные параметры
        """
        logger = self.get_logger('security')
        record = logger.makeRecord(
            logger.name, logging.WARNING, __file__, 0, message, (), None
        )
        record.security_event = event_type
        record.extra_data = kwargs
        logger.handle(record)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """
        Логирование метрики производительности.
        
        Args:
            metric_name: Название метрики
            value: Значение
            unit: Единица измерения
            **kwargs: Дополнительные параметры
        """
        logger = self.get_logger('performance')
        message = f"Metric {metric_name}: {value} {unit}"
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.performance_metric = metric_name
        record.metric_value = value
        record.metric_unit = unit
        record.extra_data = kwargs
        logger.handle(record)
    
    def log_custom_metric(self, metric_type: str, data: Dict[str, Any]):
        """
        Логирование кастомной метрики.
        
        Args:
            metric_type: Тип метрики
            data: Данные метрики
        """
        logger = self.get_logger('metrics')
        message = f"Custom metric {metric_type}: {data}"
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.metric_type = metric_type
        record.metric_data = data
        logger.handle(record)


# Глобальный экземпляр менеджера логирования
_logger_manager: Optional[LoggerManager] = None

def setup_logging(config: Optional[AppConfig] = None) -> LoggerManager:
    """
    Настройка системы логирования.
    
    Args:
        config: Конфигурация системы
        
    Returns:
        LoggerManager: Менеджер логирования
    """
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = LoggerManager(config)
    
    return _logger_manager

def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер с заданным именем.
    
    Args:
        name: Имя логгера
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    if _logger_manager is None:
        setup_logging()
    
    return _logger_manager.get_logger(name)

def log_security_event(message: str, event_type: str = "security", **kwargs):
    """
    Логирование события безопасности.
    
    Args:
        message: Сообщение
        event_type: Тип события
        **kwargs: Дополнительные параметры
    """
    if _logger_manager is None:
        setup_logging()
    
    _logger_manager.log_security_event(message, event_type, **kwargs)

def log_performance(metric_name: str, value: float, unit: str = "", **kwargs):
    """
    Логирование метрики производительности.
    
    Args:
        metric_name: Название метрики
        value: Значение
        unit: Единица измерения
        **kwargs: Дополнительные параметры
    """
    if _logger_manager is None:
        setup_logging()
    
    _logger_manager.log_performance_metric(metric_name, value, unit, **kwargs)

def log_metric(metric_type: str, data: Dict[str, Any]):
    """
    Логирование кастомной метрики.
    
    Args:
        metric_type: Тип метрики
        data: Данные метрики
    """
    if _logger_manager is None:
        setup_logging()
    
    _logger_manager.log_custom_metric(metric_type, data)

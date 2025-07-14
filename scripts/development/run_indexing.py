#!/usr/bin/env python3
"""
Скрипт для ручного запуска индексации данных.
Использует новую систему стратегий индексации.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_config, AppConfig
from src.core.shared.indexing_strategies import get_indexing_strategy
from src.core.shared.logging_config import setup_logging

async def main():
    """Основная функция для запуска индексации."""
    try:
        # Инициализация конфигурации
        config = get_config()
        
        # Настройка логирования
        setup_logging(config)
        logger = logging.getLogger('manual_indexing')
        
        logger.info("Начало процесса ручной индексации...")
        
        # Получение стратегии индексации
        indexing_strategy = get_indexing_strategy(config)
        logger.info(f"Используется стратегия индексации: {type(indexing_strategy).__name__}")
        
        # Запуск индексации
        success = await indexing_strategy.ensure_indexed()
        
        if success:
            logger.info("Индексация завершена успешно!")
            print("Индексация завершена успешно!")
            return 0
        else:
            logger.error("Индексация завершилась с ошибкой!")
            print("Индексация завершилась с ошибкой!")
            return 1
            
    except Exception as e:
        logger.error(f"Ошибка во время индексации: {e}")
        print(f"Ошибка во время индексации: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
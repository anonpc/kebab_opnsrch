# import logging

# def process_query(query, model):
#     logger = logging.getLogger('prediction')
#     logger.info(f"Processing query: {query}")
#     embedding = model.encode([query], batch_size=1)[0]
#     return embedding


from razdel import tokenize
from stop_words import get_stop_words
import torch
import logging
import functools
from shared.models import EmbeddingModel
from shared.monitoring import timer

logger = logging.getLogger('prediction')

class QueryProcessor:
    def __init__(self, model_name=None):
        """Инициализация обработчика запросов с указанной моделью.

        Args:
            model_name (str, optional): Название модели для генерации эмбеддингов.
                                        Если None, используется модель по умолчанию из конфигурации.
        """
        self.embedding_model = EmbeddingModel.get_instance(model_name)
        self.stop_words = get_stop_words('russian')
        
        # Кэш для токенизации частых запросов
        self.tokenization_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000  # Ограничиваем размер кэша для экономии памяти
        
        logger.debug("QueryProcessor инициализирован с кэшированием")

    @timer("tokenize_query")
    def tokenize_query(self, query):
        """Токенизация запроса с кэшированием результатов для повторяющихся запросов.
        
        Args:
            query (str): Строка запроса для токенизации.
            
        Returns:
            str: Очищенный запрос после токенизации и удаления стоп-слов.
        """
        # Проверяем кэш
        if query in self.tokenization_cache:
            self.cache_hits += 1
            return self.tokenization_cache[query]
        
        self.cache_misses += 1
        tokens = [token.text.lower() for token in tokenize(query) if token.text.lower() not in self.stop_words]
        cleaned_query = ' '.join(tokens)
        
        # Ограничиваем размер кэша
        if len(self.tokenization_cache) >= self.max_cache_size:
            # Удаляем первый элемент (примитивная стратегия LRU)
            self.tokenization_cache.pop(next(iter(self.tokenization_cache)))
        
        # Сохраняем результат в кэш
        self.tokenization_cache[query] = cleaned_query
        
        return cleaned_query

    @timer("process_query")
    def process_query(self, query):
        """Обработка запроса: токенизация, удаление стоп-слов и генерация эмбеддинга.

        Args:
            query (str): Строка запроса для обработки.

        Returns:
            dict: Словарь, содержащий исходный запрос, очищенный запрос и его эмбеддинг.
        """
        logger.debug(f"Обработка запроса: {query}")
        
        # Токенизация с кэшированием
        cleaned_query = self.tokenize_query(query)
        
        # Генерация эмбеддинга с оптимизированными настройками
        with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
            embedding = self.embedding_model.encode([cleaned_query], show_progress=False)[0]
        
        # Упаковываем результат в словарь для более гибкой передачи данных
        result = {
            'query': query,
            'cleaned_query': cleaned_query,
            'embedding': embedding
        }
        
        logger.debug(f"Очищенный запрос: {cleaned_query}")
        
        # Статистика кэша для диагностики
        if (self.cache_hits + self.cache_misses) % 100 == 0:
            hit_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            logger.info(f"Статистика кэша токенизации: {hit_ratio:.1f}% попаданий, размер кэша: {len(self.tokenization_cache)}")
        
        return result
        
    def get_cache_stats(self):
        """Возвращает статистику использования кэша.
        
        Returns:
            dict: Словарь со статистикой кэша.
        """
        total = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total * 100 if total > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_ratio': hit_ratio,
            'cache_size': len(self.tokenization_cache),
            'max_cache_size': self.max_cache_size
        }
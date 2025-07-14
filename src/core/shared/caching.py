"""
Система кэширования для улучшения производительности поиска.
Поддерживает различные бэкенды кэширования.
Updated to use the new unified configuration system.
"""

import logging
import time
import json
import hashlib
from typing import Any, Optional, Dict, Union, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import AppConfig

logger = logging.getLogger('cache')

@dataclass
class CacheEntry:
    """Запись в кэше."""
    
    value: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Проверка истечения времени жизни записи."""
        if self.ttl_seconds <= 0:
            return False  # Бесконечное время жизни
        
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self) -> None:
        """Обновление времени последнего доступа."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

class CacheBackend(Protocol):
    """Протокол для бэкендов кэширования."""
    
    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        ...
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Сохранение значения в кэш."""
        ...
    
    def delete(self, key: str) -> None:
        """Удаление значения из кэша."""
        ...
    
    def clear(self) -> None:
        """Очистка кэша."""
        ...
    
    def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        ...

class MemoryCacheBackend:
    """Бэкенд кэширования в памяти."""
    
    def __init__(self, max_size: int = 1000):
        """
        Инициализация кэша в памяти.
        
        Args:
            max_size: Максимальное количество записей
        """
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        entry = self.cache[key]
        
        # Проверяем истечение времени
        if entry.is_expired():
            del self.cache[key]
            self.stats['misses'] += 1
            return None
        
        # Обновляем статистику доступа
        entry.touch()
        self.stats['hits'] += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Сохранение значения в кэш."""
        # Проверяем лимит размера
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        entry = CacheEntry(
            value=value,
            created_at=datetime.utcnow(),
            ttl_seconds=ttl_seconds
        )
        
        self.cache[key] = entry
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> None:
        """Удаление значения из кэша."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()
        logger.info("Memory cache cleared")
    
    def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            return False
        
        return True
    
    def _evict_lru(self) -> None:
        """Удаление наименее используемой записи (LRU)."""
        if not self.cache:
            return
        
        # Находим запись с наименьшим временем последнего доступа
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed or self.cache[k].created_at
        )
        
        del self.cache[lru_key]
        self.stats['evictions'] += 1
        logger.debug(f"Evicted cache entry: {lru_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'type': 'memory',
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate_percent': round(hit_rate, 2),
            **self.stats
        }
    
    def cleanup_expired(self) -> int:
        """Очистка истекших записей."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)

class RedisCacheBackend:
    """Бэкенд кэширования с использованием Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Инициализация Redis кэша.
        
        Args:
            redis_url: URL для подключения к Redis
        """
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()  # Проверяем соединение
            logger.info(f"Connected to Redis: {redis_url}")
        except ImportError:
            raise ImportError("Redis library not installed. Run: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из Redis."""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Сохранение значения в Redis."""
        try:
            data = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl_seconds, data)
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")

    def delete(self, key: str) -> None:
        """Удаление значения из Redis."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")

    def clear(self) -> None:
        """Очистка Redis кэша."""
        try:
            self.redis_client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")

    def exists(self, key: str) -> bool:
        """Проверка существования ключа в Redis."""
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking Redis key existence: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики Redis."""
        try:
            info = self.redis_client.info()
            return {
                'type': 'redis',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {'type': 'redis', 'error': str(e)}


class CacheManager:
    """Менеджер кэширования с поддержкой различных бэкендов."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Инициализация менеджера кэширования.
        
        Args:
            config: Конфигурация приложения
        """
        if config is None:
            from ..config import get_config
            config = get_config()
            
        self.config = config
        self.enabled = config.cache.enabled
        self.default_ttl = config.cache.ttl_seconds
        
        if self.enabled:
            self.backend = self._create_backend()
        else:
            self.backend = DummyCacheBackend()
        
        logger.info(f"Cache manager initialized: {type(self.backend).__name__}")
    
    def _create_backend(self) -> CacheBackend:
        """Создание бэкенда кэширования."""
        backend_type = self.config.cache.backend.lower()
        
        if backend_type == "redis":
            if self.config.cache.redis_url:
                return RedisCacheBackend(self.config.cache.redis_url)
            else:
                logger.warning("Redis backend requested but no redis_url provided, falling back to memory")
                return MemoryCacheBackend(self.config.cache.max_size)
        elif backend_type == "memory":
            return MemoryCacheBackend(self.config.cache.max_size)
        else:
            logger.warning(f"Unknown cache backend: {backend_type}, using memory")
            return MemoryCacheBackend(self.config.cache.max_size)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Генерация ключа кэша на основе аргументов."""
        # Создаем уникальный ключ из аргументов
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"cache:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.
        
        Args:
            key: Ключ кэша
            
        Returns:
            Значение из кэша или None
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        result = self.backend.get(key)
        end_time = time.time()
        
        logger.debug(f"Cache get: {key} -> {'HIT' if result is not None else 'MISS'} ({(end_time - start_time)*1000:.2f}ms)")
        
        return result
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Сохранение значения в кэш.
        
        Args:
            key: Ключ кэша
            value: Значение для сохранения
            ttl_seconds: Время жизни в секундах
        """
        if not self.enabled:
            return
        
        ttl = ttl_seconds or self.default_ttl
        
        start_time = time.time()
        self.backend.set(key, value, ttl)
        end_time = time.time()
        
        logger.debug(f"Cache set: {key} (TTL: {ttl}s) ({(end_time - start_time)*1000:.2f}ms)")
    
    def delete(self, key: str) -> None:
        """Удаление значения из кэша."""
        if not self.enabled:
            return
        
        self.backend.delete(key)
        logger.debug(f"Cache delete: {key}")
    
    def clear(self) -> None:
        """Очистка кэша."""
        if not self.enabled:
            return
        
        self.backend.clear()
    
    def cache_result(self, ttl_seconds: Optional[int] = None, key_prefix: str = ""):
        """
        Декоратор для кэширования результатов функций.
        
        Args:
            ttl_seconds: Время жизни кэша
            key_prefix: Префикс для ключа кэша
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Генерируем ключ кэша
                cache_key = f"{key_prefix}:{func.__name__}:{self._generate_cache_key(*args, **kwargs)}"
                
                # Проверяем кэш
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Выполняем функцию и кэшируем результат
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl_seconds)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        if hasattr(self.backend, 'get_stats'):
            return self.backend.get_stats()
        return {'type': 'unknown', 'enabled': self.enabled}


class DummyCacheBackend:
    """Заглушка для отключенного кэша."""
    
    def get(self, key: str) -> Optional[Any]:
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        pass
    
    def delete(self, key: str) -> None:
        pass
    
    def clear(self) -> None:
        pass
    
    def exists(self, key: str) -> bool:
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {'type': 'disabled', 'enabled': False}


def get_cache_manager(config: Optional[AppConfig] = None) -> CacheManager:
    """
    Получение глобального менеджера кэширования.
    
    Args:
        config: Конфигурация приложения
        
    Returns:
        CacheManager: Менеджер кэширования
    """
    try:
        from ..container import get_cache_manager as get_from_container
        return get_from_container()
    except ImportError:
        # Fallback to direct creation
        return CacheManager(config)


def cache_search_results(ttl_seconds: int = 3600):
    """Декоратор для кэширования результатов поиска."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            return cache_manager.cache_result(ttl_seconds, "search")(func)(*args, **kwargs)
        return wrapper
    return decorator


def cache_embeddings(ttl_seconds: int = 7200):
    """Декоратор для кэширования эмбеддингов."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            return cache_manager.cache_result(ttl_seconds, "embeddings")(func)(*args, **kwargs)
        return wrapper
    return decorator

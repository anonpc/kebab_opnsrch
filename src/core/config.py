"""
Унифицированная система управления конфигурацией с использованием Pydantic BaseSettings.
Заменяет все дублирующие системы конфигурации в проекте.
Поддерживает загрузку из YAML файлов и переменных окружения.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache
import yaml

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic.v1 import BaseSettings, Field
    except ImportError:
        # Создаём простую fallback BaseSettings
        from pydantic import BaseModel, Field
        class BaseSettings(BaseModel):
            class Config:
                env_file = ".env"


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, searches for default files.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Search for configuration files in order of preference
        possible_paths = [
            "src/config/production.yaml",
            "src/config/config.yaml", 
            "config/production.yaml",
            "config/config.yaml",
            "production.yaml",
            "config.yaml"
        ]
        
        config_path = None
        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
                
        if config_path is None:
            return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
            
        # Flatten nested config for compatibility with environment variables
        flattened = {}
        
        # API settings
        if 'api' in config_data:
            api_config = config_data['api']
            flattened.update({
                'API_HOST': api_config.get('host'),
                'API_PORT': api_config.get('port'),
                'API_WORKERS': api_config.get('workers'),
                'API_RELOAD': api_config.get('reload'),
                'API_ENABLE_DOCS': api_config.get('enable_docs')
            })
        
        # OpenSearch/Database settings
        if 'opensearch' in config_data:
            os_config = config_data['opensearch'] 
            flattened.update({
                'OPENSEARCH_HOST': os_config.get('host'),
                'OPENSEARCH_INDEX_NAME': os_config.get('index_name'),
                'OPENSEARCH_USERNAME': os_config.get('username'),
                'OPENSEARCH_PASSWORD': os_config.get('password'),
                'OPENSEARCH_TIMEOUT': os_config.get('timeout'),
                'OPENSEARCH_MAX_RETRIES': os_config.get('max_retries'),
                'OPENSEARCH_USE_SSL': os_config.get('use_ssl'),
                'OPENSEARCH_VERIFY_CERTS': os_config.get('verify_certs'),
                'OPENSEARCH_SSL_SHOW_WARN': os_config.get('ssl_show_warn'),
                'OPENSEARCH_EMBEDDING_DIMENSION': os_config.get('embedding_dimension')
            })
        
        # Cache settings
        if 'cache' in config_data:
            cache_config = config_data['cache']
            flattened.update({
                'CACHE_ENABLED': cache_config.get('enabled'),
                'CACHE_BACKEND': cache_config.get('backend'),
                'CACHE_TTL_SECONDS': cache_config.get('ttl_seconds'),
                'CACHE_MAX_SIZE': cache_config.get('max_size'),
                'CACHE_REDIS_URL': cache_config.get('redis_url')
            })
        
        # Logging settings
        if 'logging' in config_data:
            log_config = config_data['logging']
            flattened.update({
                'LOG_LEVEL': log_config.get('level'),
                'LOG_DIRECTORY': log_config.get('directory'),
                'LOG_MAX_FILE_SIZE_MB': log_config.get('max_file_size_mb'),
                'LOG_BACKUP_COUNT': log_config.get('backup_count'),
                'LOG_ACCESS_LOG': log_config.get('access_log'),
                'LOG_FORMAT': log_config.get('format'),
                'LOG_ENABLE_CONSOLE': log_config.get('enable_console'),
                'LOG_ENABLE_FILE': log_config.get('enable_file')
            })
        
        # Model settings
        if 'model' in config_data:
            model_config = config_data['model']
            flattened.update({
                'MODEL_NAME': model_config.get('name'),
                'MODEL_DEVICE': model_config.get('device'),
                'MODEL_BATCH_SIZE': model_config.get('batch_size'),
                'MODEL_CACHE_DIR': model_config.get('cache_dir')
            })
        
        # Indexing settings
        if 'indexing' in config_data:
            idx_config = config_data['indexing']
            flattened.update({
                'INDEXING_STRATEGY': idx_config.get('strategy'),
                'INDEXING_AUTO_INDEX_ON_STARTUP': idx_config.get('auto_index_on_startup'),
                'INDEXING_CHECK_INTERVAL_SECONDS': idx_config.get('check_interval_seconds'),
                'INDEXING_MAX_STARTUP_WAIT_SECONDS': idx_config.get('max_startup_wait_seconds'),
                'INDEXING_FORCE_REINDEX_ON_STARTUP': idx_config.get('force_reindex_on_startup')
            })
        
        # Processing settings
        if 'processing' in config_data:
            proc_config = config_data['processing']
            flattened.update({
                'PROCESSING_MAX_WORKERS': proc_config.get('max_workers'),
                'PROCESSING_TORCH_NUM_THREADS': proc_config.get('torch_num_threads')
            })
        
        # Search settings
        if 'search' in config_data:
            search_config = config_data['search']
            flattened.update({
                'SEARCH_DEFAULT_TOP_K': search_config.get('default_top_k'),
                'SEARCH_MINIMUM_SHOULD_MATCH': search_config.get('minimum_should_match'),
                'SEARCH_CATEGORY_DIRECT_MATCH_BOOST': search_config.get('category_direct_match_boost'),
                'SEARCH_BACKEND': 'opensearch'  # Default backend
            })
        
        # Security settings
        if 'security' in config_data:
            sec_config = config_data['security']
            flattened.update({
                'SECURITY_ENABLE_AUTH': sec_config.get('enable_auth'),
                'SECURITY_JWT_SECRET': sec_config.get('jwt_secret'),
                'SECURITY_JWT_EXPIRATION_HOURS': sec_config.get('jwt_expiration_hours'),
                'SECURITY_RATE_LIMITING_ENABLED': sec_config.get('rate_limiting_enabled'),
                'SECURITY_REQUESTS_PER_MINUTE': sec_config.get('requests_per_minute'),
                'SECURITY_BURST_LIMIT': sec_config.get('burst_limit'),
                'SECURITY_MAX_QUERY_LENGTH': sec_config.get('max_query_length'),
                'SECURITY_MAX_FILE_SIZE_MB': sec_config.get('max_file_size_mb')
            })
        
        # Performance Analyzer settings (if not already present)
        flattened.setdefault('PA_ENABLED', True)
        flattened.setdefault('PA_HOST', 'localhost')
        flattened.setdefault('PA_PORT', 9600)
        
        # Remove None values
        return {k: v for k, v in flattened.items() if v is not None}
        
    except Exception as e:
        print(f"Warning: Could not load YAML config from {config_path}: {e}")
        return {}


class AppConfig(BaseSettings):
    """Основная конфигурация приложения со всеми настройками в одном месте."""
    
    # Отладка и окружение
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Настройки API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8005, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    api_enable_docs: bool = Field(default=True, env="API_ENABLE_DOCS")
    
    # Настройки базы данных
    database_host: str = Field(default="localhost:9200", env="OPENSEARCH_HOST")
    database_index_name: str = Field(default="workers", env="OPENSEARCH_INDEX_NAME")
    database_username: Optional[str] = Field(default=None, env="OPENSEARCH_USERNAME")
    database_password: Optional[str] = Field(default=None, env="OPENSEARCH_PASSWORD")
    database_timeout: int = Field(default=30, env="OPENSEARCH_TIMEOUT")
    database_max_retries: int = Field(default=3, env="OPENSEARCH_MAX_RETRIES")
    database_use_ssl: bool = Field(default=False, env="OPENSEARCH_USE_SSL")
    database_verify_certs: bool = Field(default=False, env="OPENSEARCH_VERIFY_CERTS")
    database_ssl_show_warn: bool = Field(default=False, env="OPENSEARCH_SSL_SHOW_WARN")
    database_embedding_dimension: int = Field(default=768, env="OPENSEARCH_EMBEDDING_DIMENSION")
    
    # Настройки кэша
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_backend: str = Field(default="memory", env="CACHE_BACKEND")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    cache_redis_url: Optional[str] = Field(default=None, env="CACHE_REDIS_URL")
    
    # Настройки логирования
    logging_level: str = Field(default="INFO", env="LOG_LEVEL")
    logging_directory: str = Field(default="logs", env="LOG_DIRECTORY")
    logging_max_file_size_mb: int = Field(default=100, env="LOG_MAX_FILE_SIZE_MB")
    logging_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    logging_access_log: bool = Field(default=True, env="LOG_ACCESS_LOG")
    logging_format: str = Field(default="structured", env="LOG_FORMAT")
    logging_enable_console: bool = Field(default=True, env="LOG_ENABLE_CONSOLE")
    logging_enable_file: bool = Field(default=True, env="LOG_ENABLE_FILE")
    
    # Настройки модели
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="MODEL_NAME")
    model_device: str = Field(default="auto", env="MODEL_DEVICE")
    model_batch_size: int = Field(default=32, env="MODEL_BATCH_SIZE")
    model_cache_dir: Optional[str] = Field(default=None, env="MODEL_CACHE_DIR")
    
    # Настройки индексации
    indexing_strategy: str = Field(default="lazy", env="INDEXING_STRATEGY")
    indexing_auto_index_on_startup: bool = Field(default=True, env="INDEXING_AUTO_INDEX_ON_STARTUP")
    indexing_check_interval_seconds: int = Field(default=300, env="INDEXING_CHECK_INTERVAL_SECONDS")
    indexing_max_startup_wait_seconds: int = Field(default=60, env="INDEXING_MAX_STARTUP_WAIT_SECONDS")
    indexing_force_reindex_on_startup: bool = Field(default=False, env="INDEXING_FORCE_REINDEX_ON_STARTUP")
    
    # Настройки обработки
    processing_max_workers: int = Field(default=12, env="PROCESSING_MAX_WORKERS")
    processing_torch_num_threads: int = Field(default=4, env="PROCESSING_TORCH_NUM_THREADS")
    
    # Настройки поиска
    search_default_top_k: int = Field(default=20, env="SEARCH_DEFAULT_TOP_K")
    search_minimum_should_match: float = Field(default=0.5, env="SEARCH_MINIMUM_SHOULD_MATCH")
    search_category_direct_match_boost: float = Field(default=0.4, env="SEARCH_CATEGORY_DIRECT_MATCH_BOOST")
    search_backend: str = Field(default="opensearch", env="SEARCH_BACKEND")
    
    # Настройки безопасности
    security_enable_auth: bool = Field(default=False, env="SECURITY_ENABLE_AUTH")
    security_jwt_secret: str = Field(default="development-secret-key", env="SECURITY_JWT_SECRET")
    security_jwt_expiration_hours: int = Field(default=24, env="SECURITY_JWT_EXPIRATION_HOURS")
    security_rate_limiting_enabled: bool = Field(default=True, env="SECURITY_RATE_LIMITING_ENABLED")
    security_requests_per_minute: int = Field(default=100, env="SECURITY_REQUESTS_PER_MINUTE")
    security_burst_limit: int = Field(default=200, env="SECURITY_BURST_LIMIT")
    security_max_query_length: int = Field(default=1000, env="SECURITY_MAX_QUERY_LENGTH")
    security_max_file_size_mb: int = Field(default=50, env="SECURITY_MAX_FILE_SIZE_MB")
    
    # Настройки Performance Analyzer
    performance_analyzer_enabled: bool = Field(default=True, env="PA_ENABLED")
    performance_analyzer_host: str = Field(default="localhost", env="PA_HOST")
    performance_analyzer_port: int = Field(default=9600, env="PA_PORT")
    performance_analyzer_timeout: int = Field(default=10, env="PA_TIMEOUT")
    performance_analyzer_metrics_interval: int = Field(default=60, env="PA_METRICS_INTERVAL")
    performance_analyzer_retention_period: int = Field(default=7, env="PA_RETENTION_DAYS")
    performance_analyzer_node_metrics: bool = Field(default=True, env="PA_NODE_METRICS")
    performance_analyzer_cluster_metrics: bool = Field(default=True, env="PA_CLUSTER_METRICS")
    performance_analyzer_index_metrics: bool = Field(default=True, env="PA_INDEX_METRICS")
    performance_analyzer_thread_metrics: bool = Field(default=True, env="PA_THREAD_METRICS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Игнорировать дополнительные поля из окружения

    def __init__(self, yaml_config_path: Optional[str] = None, **kwargs):
        # Load YAML config first
        yaml_config = load_yaml_config(yaml_config_path)
        
        # Merge with provided kwargs (kwargs take precedence)
        merged_config = {**yaml_config, **kwargs}
        
        super().__init__(**merged_config)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Создать необходимые директории."""
        directories = [
            self.logging_directory,
            'data/raw',
            'data/processed', 
            'data/index'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    # Удобные свойства для обратной совместимости
    @property
    def api(self):
        """Конфигурация API как пространство имён."""
        return type('APIConfig', (), {
            'host': self.api_host,
            'port': self.api_port,
            'workers': self.api_workers,
            'reload': self.api_reload,
            'enable_docs': self.api_enable_docs,
            'allowed_origins': ["*"]
        })()

    @property
    def database(self):
        """Конфигурация базы данных как пространство имён."""
        return type('DatabaseConfig', (), {
            'host': self.database_host,
            'index_name': self.database_index_name,
            'username': self.database_username,
            'password': self.database_password,
            'timeout': self.database_timeout,
            'max_retries': self.database_max_retries,
            'use_ssl': self.database_use_ssl,
            'verify_certs': self.database_verify_certs,
            'ssl_show_warn': self.database_ssl_show_warn,
            'embedding_dimension': self.database_embedding_dimension
        })()

    @property
    def cache(self):
        """Конфигурация кэша как пространство имён."""
        return type('CacheConfig', (), {
            'enabled': self.cache_enabled,
            'backend': self.cache_backend,
            'ttl_seconds': self.cache_ttl_seconds,
            'max_size': self.cache_max_size,
            'redis_url': self.cache_redis_url
        })()

    @property
    def logging(self):
        """Конфигурация логирования как пространство имён."""
        return type('LoggingConfig', (), {
            'level': self.logging_level,
            'directory': self.logging_directory,
            'max_file_size_mb': self.logging_max_file_size_mb,
            'backup_count': self.logging_backup_count,
            'access_log': self.logging_access_log,
            'format': self.logging_format,
            'enable_console': self.logging_enable_console,
            'enable_file': self.logging_enable_file
        })()

    @property
    def model(self):
        """Конфигурация модели как пространство имён."""
        return type('ModelConfig', (), {
            'name': self.model_name,
            'device': self.model_device,
            'batch_size': self.model_batch_size,
            'cache_dir': self.model_cache_dir
        })()

    @property
    def indexing(self):
        """Конфигурация индексации как пространство имён."""
        return type('IndexingConfig', (), {
            'strategy': self.indexing_strategy,
            'auto_index_on_startup': self.indexing_auto_index_on_startup,
            'check_interval_seconds': self.indexing_check_interval_seconds,
            'max_startup_wait_seconds': self.indexing_max_startup_wait_seconds,
            'force_reindex_on_startup': self.indexing_force_reindex_on_startup
        })()

    @property
    def processing(self):
        """Конфигурация обработки как пространство имён."""
        return type('ProcessingConfig', (), {
            'max_workers': self.processing_max_workers,
            'torch_num_threads': self.processing_torch_num_threads
        })()

    @property
    def search(self):
        """Конфигурация поиска как пространство имён."""
        return type('SearchConfig', (), {
            'default_top_k': self.search_default_top_k,
            'minimum_should_match': self.search_minimum_should_match,
            'category_direct_match_boost': self.search_category_direct_match_boost,
            'backend': self.search_backend
        })()

    @property
    def security(self):
        """Конфигурация безопасности как пространство имён."""
        return type('SecurityConfig', (), {
            'enable_auth': self.security_enable_auth,
            'jwt_secret': self.security_jwt_secret,
            'jwt_expiration_hours': self.security_jwt_expiration_hours,
            'rate_limiting_enabled': self.security_rate_limiting_enabled,
            'requests_per_minute': self.security_requests_per_minute,
            'burst_limit': self.security_burst_limit,
            'max_query_length': self.security_max_query_length,
            'max_file_size_mb': self.security_max_file_size_mb
        })()

    @property
    def performance_analyzer(self):
        """Конфигурация Performance Analyzer как пространство имён."""
        return type('PerformanceAnalyzerConfig', (), {
            'enabled': self.performance_analyzer_enabled,
            'host': self.performance_analyzer_host,
            'port': self.performance_analyzer_port,
            'timeout': self.performance_analyzer_timeout,
            'metrics_interval': self.performance_analyzer_metrics_interval,
            'retention_period': self.performance_analyzer_retention_period,
            'node_metrics': self.performance_analyzer_node_metrics,
            'cluster_metrics': self.performance_analyzer_cluster_metrics,
            'index_metrics': self.performance_analyzer_index_metrics,
            'thread_metrics': self.performance_analyzer_thread_metrics,
            'base_url': f"http://{self.performance_analyzer_host}:{self.performance_analyzer_port}"
        })()

    def get_dict(self) -> Dict[str, Any]:
        """Получить конфигурацию как словарь для обратной совместимости."""
        return {
            'api': {
                'host': self.api_host,
                'port': self.api_port,
                'workers': self.api_workers,
                'reload': self.api_reload,
                'enable_docs': self.api_enable_docs,
                'allowed_origins': ["*"]
            },
            'database': {
                'host': self.database_host,
                'index_name': self.database_index_name,
                'username': self.database_username,
                'password': self.database_password,
                'timeout': self.database_timeout,
                'max_retries': self.database_max_retries,
                'use_ssl': self.database_use_ssl,
                'verify_certs': self.database_verify_certs,
                'ssl_show_warn': self.database_ssl_show_warn,
                'embedding_dimension': self.database_embedding_dimension
            },
            'opensearch': {  # Псевдоним для обратной совместимости
                'host': self.database_host,
                'index_name': self.database_index_name,
                'username': self.database_username,
                'password': self.database_password,
                'timeout': self.database_timeout,
                'max_retries': self.database_max_retries,
                'use_ssl': self.database_use_ssl,
                'verify_certs': self.database_verify_certs,
                'ssl_show_warn': self.database_ssl_show_warn,
                'embedding_dimension': self.database_embedding_dimension
            },
            'cache': {
                'enabled': self.cache_enabled,
                'backend': self.cache_backend,
                'ttl_seconds': self.cache_ttl_seconds,
                'max_size': self.cache_max_size,
                'redis_url': self.cache_redis_url
            },
            'logging': {
                'level': self.logging_level,
                'directory': self.logging_directory,
                'max_file_size_mb': self.logging_max_file_size_mb,
                'backup_count': self.logging_backup_count,
                'access_log': self.logging_access_log,
                'format': self.logging_format,
                'enable_console': self.logging_enable_console,
                'enable_file': self.logging_enable_file
            },
            'model': {
                'name': self.model_name,
                'device': self.model_device,
                'batch_size': self.model_batch_size,
                'cache_dir': self.model_cache_dir
            },
            'indexing': {
                'strategy': self.indexing_strategy,
                'auto_index_on_startup': self.indexing_auto_index_on_startup,
                'check_interval_seconds': self.indexing_check_interval_seconds,
                'max_startup_wait_seconds': self.indexing_max_startup_wait_seconds,
                'force_reindex_on_startup': self.indexing_force_reindex_on_startup
            },
            'processing': {
                'max_workers': self.processing_max_workers,
                'torch_num_threads': self.processing_torch_num_threads
            },
            'search': {
                'default_top_k': self.search_default_top_k,
                'minimum_should_match': self.search_minimum_should_match,
                'category_direct_match_boost': self.search_category_direct_match_boost,
                'backend': self.search_backend
            },
            'security': {
                'enable_auth': self.security_enable_auth,
                'jwt_secret': self.security_jwt_secret,
                'jwt_expiration_hours': self.security_jwt_expiration_hours,
                'rate_limiting_enabled': self.security_rate_limiting_enabled,
                'requests_per_minute': self.security_requests_per_minute,
                'burst_limit': self.security_burst_limit,
                'max_query_length': self.security_max_query_length,
                'max_file_size_mb': self.security_max_file_size_mb
            },
            'performance_analyzer': {
                'enabled': self.performance_analyzer_enabled,
                'host': self.performance_analyzer_host,
                'port': self.performance_analyzer_port,
                'timeout': self.performance_analyzer_timeout,
                'metrics_interval': self.performance_analyzer_metrics_interval,
                'retention_period': self.performance_analyzer_retention_period,
                'node_metrics': self.performance_analyzer_node_metrics,
                'cluster_metrics': self.performance_analyzer_cluster_metrics,
                'index_metrics': self.performance_analyzer_index_metrics,
                'thread_metrics': self.performance_analyzer_thread_metrics,
                'base_url': f"http://{self.performance_analyzer_host}:{self.performance_analyzer_port}"
            },
            'debug': self.debug,
            'environment': self.environment
        }


# Singleton паттерн для глобальной конфигурации
_config_instance: Optional[AppConfig] = None


@lru_cache()
def get_config(yaml_config_path: Optional[str] = None) -> AppConfig:
    """
    Получить экземпляр глобальной конфигурации.
    
    Args:
        yaml_config_path: Optional path to YAML config file
    
    Returns:
        AppConfig: Экземпляр глобальной конфигурации
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = AppConfig(yaml_config_path=yaml_config_path)
    
    return _config_instance


def reload_config(yaml_config_path: Optional[str] = None) -> AppConfig:
    """
    Перезагрузить конфигурацию из окружения и YAML файлов.
    
    Args:
        yaml_config_path: Optional path to YAML config file
        
    Returns:
        AppConfig: Свежий экземпляр конфигурации
    """
    global _config_instance
    _config_instance = AppConfig(yaml_config_path=yaml_config_path)
    return _config_instance


# Экспорт для удобства
__all__ = ["AppConfig", "get_config", "reload_config"]

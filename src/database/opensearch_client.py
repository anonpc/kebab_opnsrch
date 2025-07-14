"""
OpenSearch client configuration and connection management.
Updated to use the unified configuration system and DI container exclusively.
"""

from opensearchpy import OpenSearch
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timezone

from ..core.config import get_config, AppConfig

logger = logging.getLogger("database")


def create_opensearch_client(config: Optional[AppConfig] = None) -> OpenSearch:
    """
    Create OpenSearch client instance using unified configuration.
    
    Args:
        config: Optional configuration instance. If None, uses global config.
        
    Returns:
        Configured OpenSearch client
    """
    if config is None:
        config = get_config()
    
    client_params = {
        'hosts': [config.database.host],
        'use_ssl': config.database.use_ssl,
        'verify_certs': config.database.verify_certs,
        'ssl_show_warn': config.database.ssl_show_warn,
        'timeout': config.database.timeout,
        'max_retries': config.database.max_retries,
        'retry_on_timeout': True
    }
    
    # Only add http_auth if username is provided
    if config.database.username:
        client_params['http_auth'] = (
            config.database.username,
            config.database.password
        )
    
    client = OpenSearch(**client_params)
    logger.info(f"OpenSearch client created for host: {config.database.host}")
    return client


def create_index_with_settings(
    client: Optional[OpenSearch] = None,
    index_name: Optional[str] = None,
    config: Optional[AppConfig] = None,
    recreate_if_exists: bool = False
) -> Dict[str, Any]:
    """
    Create OpenSearch index with standardized settings for worker search.
    
    Args:
        client: OpenSearch client instance. If None, creates new client.
        index_name: Index name. If None, uses config default.
        config: Configuration instance. If None, uses global config.
        recreate_if_exists: Whether to recreate index if it already exists.
        
    Returns:
        Dict with operation result details
    """
    if config is None:
        config = get_config()
    
    if client is None:
        client = create_opensearch_client(config)
    
    if index_name is None:
        index_name = config.database.index_name
    
    try:
        # Check if index exists
        index_exists = client.indices.exists(index=index_name)
        
        if index_exists:
            if recreate_if_exists:
                logger.info(f"Deleting existing index {index_name} for recreation")
                client.indices.delete(index=index_name)
            else:
                logger.info(f"Index {index_name} already exists")
                return {
                    "success": True,
                    "message": f"Index {index_name} already exists",
                    "created": False,
                    "index_name": index_name
                }
        
        # Define standardized index settings
        index_settings = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "analysis": {
                        "analyzer": {
                            "russian_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "russian_stop",
                                    "russian_stemmer"
                                ]
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
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "category": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "russian_analyzer"
                            }
                        }
                    },
                    "seller_name": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "location": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "russian_analyzer"
                            }
                        }
                    },
                    "price": {
                        "type": "float"
                    },
                    "url": {
                        "type": "keyword",
                        "index": False
                    },
                    "combined_text": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "combined_text_vector": {
                        "type": "knn_vector",
                        "dimension": config.database.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 200,
                                "m": 16
                            }
                        }
                    },
                    "indexed_at": {
                        "type": "date"
                    }
                }
            }
        }
        
        # Create the index
        client.indices.create(index=index_name, body=index_settings)
        
        logger.info(f"Successfully created index {index_name} with Russian language support and vector search")
        
        return {
            "success": True,
            "message": f"Index {index_name} created successfully",
            "created": True,
            "index_name": index_name,
            "settings": index_settings
        }
        
    except Exception as e:
        error_msg = f"Failed to create index {index_name}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "created": False,
            "index_name": index_name,
            "error": str(e)
        }


def get_opensearch_client() -> OpenSearch:
    """
    Get OpenSearch client instance using DI container.
    
    Returns:
        Configured OpenSearch client
    """
    try:
        from ..core.container import get_opensearch_client as get_client_from_container
        return get_client_from_container()
    except ImportError:
        # Fallback to direct creation if container is not available
        logger.warning("DI container not available, creating client directly")
        return create_opensearch_client()


async def close_clients():
    """Close all OpenSearch client connections."""
    # OpenSearch clients don't require explicit closing
    # but we can clear the container cache if needed
    try:
        from ..core.container import reset_container
        reset_container()
        logger.info("OpenSearch clients closed and container reset")
    except ImportError:
        logger.info("OpenSearch clients closed")

"""
Модели данных и утилиты для обработки эмбеддингов.
Enhanced with centralized model management through DI container.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger('models')


class EmbeddingModel:
    """
    Модель для создания эмбеддингов с централизованным управлением.
    """
    
    _instance = None
    
    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Название модели
            device: Устройство для вычислений
        """
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.dimension = 768
        self.is_loaded = False
        self._model = None
        
        # Attempt to load real model if available
        self._load_model()
        
    def _load_model(self):
        """Загрузка реальной модели."""
        try:
            # Try to load sentence-transformers model
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self._model.get_sentence_embedding_dimension()
            self.is_loaded = True
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback model")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.is_loaded = False
    
    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, device: str = "auto") -> 'EmbeddingModel':
        """
        DEPRECATED: Use create_embedding_model() or DI container instead.
        
        Получить единственный экземпляр модели.
        """
        logger.warning(
            "EmbeddingModel.get_instance() is deprecated. "
            "Use create_embedding_model() or DI container instead."
        )
        
        if cls._instance is None:
            cls._instance = cls(model_name, device)
        return cls._instance
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Создать эмбеддинги для текстов.
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Массив эмбеддингов
        """
        if not texts:
            return np.empty((0, self.dimension))
        
        if self.is_loaded and self._model:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e:
                logger.error(f"Error encoding with real model: {e}")
                # Fall back to simple hashing
        
        # Fallback implementation using hashing
        embeddings = []
        for text in texts:
            # Простое хеширование для демонстрации
            hash_value = hash(text.lower())
            # Создаем вектор фиксированной размерности
            vector = np.random.RandomState(abs(hash_value)).normal(0, 1, self.dimension)
            # Нормализуем
            vector = vector / np.linalg.norm(vector)
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Создать эмбеддинг для одного текста.
        
        Args:
            text: Текст для обработки
            
        Returns:
            Вектор эмбеддинга
        """
        return self.encode([text])[0]


def create_embedding_model(config=None) -> EmbeddingModel:
    """
    Factory function for creating embedding models.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        EmbeddingModel: Configured embedding model instance
    """
    if config is None:
        try:
            from ..config import get_config
            config = get_config()
        except ImportError:
            # Fallback if config is not available
            return EmbeddingModel()
    
    # Extract model configuration
    try:
        model_name = config.model.name if hasattr(config, 'model') else config.model_name
        device = config.model.device if hasattr(config, 'model') else config.model_device
    except AttributeError:
        # Fallback to defaults
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "auto"
    
    model = EmbeddingModel(model_name=model_name, device=device)
    logger.info(f"Created embedding model: {model_name} on {device}")
    return model


def get_embedding_model() -> EmbeddingModel:
    """
    Get embedding model from DI container.
    
    Returns:
        EmbeddingModel: Model instance from container
    """
    try:
        from ..container import get_container
        container = get_container()
        
        # Try to get from container first
        try:
            return container.get("embedding_model")
        except KeyError:
            # Register and return if not found
            model = create_embedding_model()
            container.register_instance("embedding_model", model)
            return model
    except ImportError:
        # Fallback if container is not available
        logger.warning("DI container not available, creating model directly")
        return create_embedding_model()


class SearchResult:
    """Результат поиска с метаданными."""
    
    def __init__(self, doc_id: str, score: float, source: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.source = source
        
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'id': self.doc_id,
            'score': self.score,
            'source': self.source
        }

"""
Модели данных для API с валидацией и типизацией.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import re

class SearchRequest(BaseModel):
    """Модель запроса на поиск"""
    query: str = Field(..., min_length=2, max_length=1000, description="Поисковый запрос")
    top_k: int = Field(default=20, ge=1, le=100, description="Количество результатов")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Фильтры поиска")
    search_type: str = Field(default="hybrid", description="Тип поиска: text, vector, hybrid")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Валидация и очистка поискового запроса"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        
        # Удаляем потенциально опасные символы
        cleaned = re.sub(r'[<>"\';{}]', '', v.strip())
        
        if len(cleaned) < 2:
            raise ValueError('Query too short after cleaning')
        
        return cleaned
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v):
        """Валидация фильтров"""
        if not isinstance(v, dict):
            raise ValueError('Filters must be a dictionary')
        
        # Ограничиваем глубину фильтров
        if len(str(v)) > 1000:
            raise ValueError('Filters too complex')
        
        return v

class WorkerResult(BaseModel):
    """Модель результата - информация о работнике"""
    id: Optional[str] = Field(None, description="ID работника")
    title: str = Field(..., description="Название/специальность")
    description: Optional[str] = Field(None, description="Описание услуг")
    category: Optional[str] = Field(None, description="Категория")
    price: Optional[float] = Field(None, ge=0, description="Цена")
    location: Optional[str] = Field(None, description="Местоположение")
    score: Optional[float] = Field(None, ge=0, description="Релевантность")
    photo_urls: List[str] = Field(default_factory=list, description="URLs фотографий")
    executor_telegram_id: Optional[int] = Field(None, description="Telegram ID исполнителя")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Рейтинг")

class QueryInfo(BaseModel):
    """Информация о запросе"""
    original_query: str = Field(..., description="Исходный запрос")
    cleaned_query: str = Field(..., description="Очищенный запрос")
    category: Optional[str] = Field(None, description="Предсказанная категория")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Уверенность в категории")
    processing_time_ms: Optional[float] = Field(None, description="Время обработки в мс")

class SearchResponse(BaseModel):
    """Модель ответа поиска"""
    results: List[WorkerResult] = Field(default_factory=list, description="Результаты поиска")
    query_info: QueryInfo = Field(..., description="Информация о запросе")
    total_found: int = Field(default=0, ge=0, description="Общее количество найденных")
    
    class Config:
        json_encoders = {
            # Кастомные энкодеры при необходимости
        }

class HealthResponse(BaseModel):
    """Модель ответа проверки здоровья"""
    status: str = Field(..., description="Статус системы")
    backend: str = Field(..., description="Тип используемого бэкенда")
    backend_healthy: bool = Field(..., description="Состояние бэкенда")
    uptime_seconds: Optional[float] = Field(None, description="Время работы в секундах")

class PerformanceStats(BaseModel):
    """Модель статистики производительности"""
    operation: str = Field(..., description="Название операции")
    count: int = Field(..., ge=0, description="Количество вызовов")
    avg_time_ms: float = Field(..., ge=0, description="Среднее время выполнения")
    min_time_ms: float = Field(..., ge=0, description="Минимальное время")
    max_time_ms: float = Field(..., ge=0, description="Максимальное время")
    last_execution: Optional[str] = Field(None, description="Время последнего выполнения")

class ErrorResponse(BaseModel):
    """Модель ответа об ошибке"""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Описание ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")
    timestamp: str = Field(..., description="Время возникновения ошибки")
    path: Optional[str] = Field(None, description="Путь запроса")

class WorkerCreate(BaseModel):
    """Модель для создания нового работника"""
    title: str = Field(..., min_length=2, max_length=200, description="Название/специальность")
    description: Optional[str] = Field(None, max_length=2000, description="Описание услуг")
    category: Optional[str] = Field(None, max_length=100, description="Категория")
    seller_name: Optional[str] = Field(None, max_length=100, description="Имя продавца/работника")
    price: Optional[float] = Field(None, ge=0, description="Цена")
    location: Optional[str] = Field(None, max_length=200, description="Местоположение")
    url: Optional[str] = Field(None, max_length=500, description="URL профиля")
    photo_urls: List[str] = Field(default_factory=list, description="URLs фотографий")
    executor_telegram_id: Optional[int] = Field(None, description="Telegram ID исполнителя")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Рейтинг")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v:
            return v.strip()
        return v

    @field_validator('seller_name')
    @classmethod
    def validate_seller_name(cls, v):
        if v:
            return v.strip()
        return v

    @field_validator('photo_urls')
    @classmethod
    def validate_photo_urls(cls, v):
        if v:
            # Проверяем что все элементы массива - строки
            for url in v:
                if not isinstance(url, str):
                    raise ValueError('All photo URLs must be strings')
                if len(url) > 500:
                    raise ValueError('Photo URL too long')
        return v

    @field_validator('executor_telegram_id')
    @classmethod
    def validate_executor_telegram_id(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Telegram ID must be positive integer')
        return v

class WorkerUpdate(BaseModel):
    """Модель для обновления работника"""
    title: Optional[str] = Field(None, min_length=2, max_length=200, description="Название/специальность")
    description: Optional[str] = Field(None, max_length=2000, description="Описание услуг")
    category: Optional[str] = Field(None, max_length=100, description="Категория")
    seller_name: Optional[str] = Field(None, max_length=100, description="Имя продавца/работника")
    price: Optional[float] = Field(None, ge=0, description="Цена")
    location: Optional[str] = Field(None, max_length=200, description="Местоположение")
    url: Optional[str] = Field(None, max_length=500, description="URL профиля")
    photo_urls: Optional[List[str]] = Field(None, description="URLs фотографий")
    executor_telegram_id: Optional[int] = Field(None, description="Telegram ID исполнителя")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Рейтинг")

    @field_validator('photo_urls')
    @classmethod
    def validate_photo_urls(cls, v):
        if v is not None:
            # Проверяем что все элементы массива - строки
            for url in v:
                if not isinstance(url, str):
                    raise ValueError('All photo URLs must be strings')
                if len(url) > 500:
                    raise ValueError('Photo URL too long')
        return v

    @field_validator('executor_telegram_id')
    @classmethod
    def validate_executor_telegram_id(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Telegram ID must be positive integer')
        return v

class WorkerDeleteRequest(BaseModel):
    """Модель запроса на удаление работника"""
    delete_by: str = Field(..., description="Поле для поиска: 'id', 'seller_name', 'title'")
    value: str = Field(..., min_length=1, description="Значение для поиска")

    @field_validator('delete_by')
    @classmethod
    def validate_delete_by(cls, v):
        allowed_fields = ['id', 'seller_name', 'title']
        if v not in allowed_fields:
            raise ValueError(f'delete_by must be one of: {allowed_fields}')
        return v

class WorkerResponse(BaseModel):
    """Модель ответа с информацией о работнике"""
    id: str = Field(..., description="ID работника")
    title: str = Field(..., description="Название/специальность")
    description: Optional[str] = Field(None, description="Описание услуг")
    category: Optional[str] = Field(None, description="Категория")
    seller_name: Optional[str] = Field(None, description="Имя продавца/работника")
    price: Optional[float] = Field(None, description="Цена")
    location: Optional[str] = Field(None, description="Местоположение")
    url: Optional[str] = Field(None, description="URL профиля")
    indexed_at: Optional[str] = Field(None, description="Время индексации")
    photo_urls: List[str] = Field(default_factory=list, description="URLs фотографий")
    executor_telegram_id: Optional[int] = Field(None, description="Telegram ID исполнителя")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Рейтинг")

class OperationResponse(BaseModel):
    """Модель ответа операции"""
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")
    worker: Optional[WorkerResponse] = Field(None, description="Данные работника")
    affected_count: Optional[int] = Field(None, description="Количество затронутых записей")

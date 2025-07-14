"""
Unified Dependency Injection container for managing application dependencies.
This replaces global variables and provides proper lifecycle management.
Enhanced with advanced DI features like scopes, auto-injection, and factory support.
"""
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable, Protocol
from abc import ABC, abstractmethod
import logging
import inspect
from enum import Enum
from functools import wraps
from opensearchpy import OpenSearch

from .config import get_config, AppConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceScope(Enum):
    """Service scope enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceRegistration:
    """Service registration information."""
    
    def __init__(self, 
                 service_type: Type,
                 implementation: Type,
                 scope: ServiceScope = ServiceScope.TRANSIENT,
                 factory: Optional[Callable] = None,
                 instance: Optional[Any] = None):
        self.service_type = service_type
        self.implementation = implementation
        self.scope = scope
        self.factory = factory
        self.instance = instance


class Container:
    """Enhanced dependency injection container with advanced features."""
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._is_scoped = False
        
        # Legacy support for simple string-based registration
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._singletons: Dict[str, Any] = {}
        
    # Legacy string-based methods for backward compatibility
    def register_factory(self, name: str, factory: callable) -> None:
        """Register a factory function for a dependency (legacy method)."""
        self._factories[name] = factory
        
    def register_singleton(self, name: str, factory: callable) -> None:
        """Register a singleton factory for a dependency (legacy method)."""
        self._factories[name] = factory
        self._singletons[name] = None
        
    def register_instance(self, name: str, instance: Any) -> None:
        """Register a concrete instance (legacy method)."""
        self._instances[name] = instance
        
    def get(self, name: str) -> Any:
        """Get a dependency by name (legacy method)."""
        # Check if we have a concrete instance
        if name in self._instances:
            return self._instances[name]
            
        # Check if it's a singleton
        if name in self._singletons:
            if self._singletons[name] is None:
                self._singletons[name] = self._factories[name]()
            return self._singletons[name]
            
        # Create new instance from factory
        if name in self._factories:
            return self._factories[name]()
            
        raise KeyError(f"No dependency registered for '{name}'")
        
    def has(self, name: str) -> bool:
        """Check if a dependency is registered (legacy method)."""
        return name in self._instances or name in self._factories
    
    # Enhanced type-based methods
    def register_type_singleton(self, service_type: Type[T], implementation: Type[T]) -> 'Container':
        """
        Register a service as singleton.
        
        Args:
            service_type: Abstract service type
            implementation: Concrete implementation
            
        Returns:
            Container: Current container for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type, implementation, ServiceScope.SINGLETON
        )
        logger.debug(f"Registered singleton: {service_type.__name__} -> {implementation.__name__}")
        return self
    
    def register_type_transient(self, service_type: Type[T], implementation: Type[T]) -> 'Container':
        """
        Register a transient service (new instance each time).
        
        Args:
            service_type: Abstract service type
            implementation: Concrete implementation
            
        Returns:
            Container: Current container for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type, implementation, ServiceScope.TRANSIENT
        )
        logger.debug(f"Registered transient: {service_type.__name__} -> {implementation.__name__}")
        return self
    
    def register_type_scoped(self, service_type: Type[T], implementation: Type[T]) -> 'Container':
        """
        Register a scoped service (one instance per scope).
        
        Args:
            service_type: Abstract service type
            implementation: Concrete implementation
            
        Returns:
            Container: Current container for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type, implementation, ServiceScope.SCOPED
        )
        logger.debug(f"Registered scoped: {service_type.__name__} -> {implementation.__name__}")
        return self
    
    def register_type_factory(self, service_type: Type[T], factory: Callable[[], T]) -> 'Container':
        """
        Register a factory method for creating services.
        
        Args:
            service_type: Service type
            factory: Factory method
            
        Returns:
            Container: Current container for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type, service_type, ServiceScope.TRANSIENT, factory
        )
        logger.debug(f"Registered factory: {service_type.__name__}")
        return self
    
    def register_type_instance(self, service_type: Type[T], instance: T) -> 'Container':
        """
        Register a concrete instance.
        
        Args:
            service_type: Service type
            instance: Concrete instance
            
        Returns:
            Container: Current container for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type, type(instance), ServiceScope.SINGLETON, None, instance
        )
        logger.debug(f"Registered instance: {service_type.__name__}")
        return self
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service by type.
        
        Args:
            service_type: Type of the service to retrieve
            
        Returns:
            Service instance
        """
        if service_type not in self._services:
            raise KeyError(f"Service type {service_type.__name__} is not registered")
        
        registration = self._services[service_type]
        
        # Return existing instance if available
        if registration.instance is not None:
            return registration.instance
        
        # Handle singleton scope
        if registration.scope == ServiceScope.SINGLETON:
            if registration.instance is None:
                registration.instance = self._create_instance(registration)
            return registration.instance
        
        # Handle scoped services
        if registration.scope == ServiceScope.SCOPED:
            if self._is_scoped and service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(registration)
            if self._is_scoped:
                self._scoped_instances[service_type] = instance
            return instance
        
        # Handle transient services
        return self._create_instance(registration)
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create an instance from registration."""
        if registration.factory:
            return registration.factory()
        
        try:
            return self._create_with_injection(registration.implementation)
        except Exception as e:
            logger.error(f"Failed to create instance of {registration.implementation.__name__}: {e}")
            # Fallback to simple instantiation
            return registration.implementation()
    
    def _create_with_injection(self, implementation_type: Type) -> Any:
        """Create instance with automatic dependency injection."""
        constructor = implementation_type.__init__
        sig = inspect.signature(constructor)
        
        args = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation == inspect.Parameter.empty:
                continue
                
            # Try to resolve dependency
            try:
                args[param_name] = self.get_service(param.annotation)
            except KeyError:
                # If dependency not found and parameter has default, skip it
                if param.default != inspect.Parameter.empty:
                    continue
                else:
                    logger.warning(f"Could not resolve dependency {param.annotation.__name__} for {implementation_type.__name__}")
                    continue
        
        return implementation_type(**args)
    
    def create_scope(self) -> 'ScopedContainer':
        """Create a new scope for scoped services."""
        return ScopedContainer(self)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services
    
    def get_registrations(self) -> Dict[Type, ServiceRegistration]:
        """Get all service registrations."""
        return self._services.copy()


class ScopedContainer:
    """Scoped container context manager."""
    
    def __init__(self, parent_container: Container):
        self.parent_container = parent_container
    
    def __enter__(self) -> Container:
        self.parent_container._is_scoped = True
        return self.parent_container
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent_container._is_scoped = False
        self.parent_container._scoped_instances.clear()


def inject(service_type: Type[T]) -> Callable:
    """
    Decorator for automatic dependency injection.
    
    Args:
        service_type: Type of service to inject
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            service = container.get_service(service_type)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


def create_opensearch_client_for_container() -> OpenSearch:
    """Factory function for creating OpenSearch clients in the DI container."""
    from ..database.opensearch_client import create_opensearch_client
    return create_opensearch_client()


class CacheManagerFactory:
    """Factory for creating cache managers."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
    def create_cache_manager(self):
        """Create cache manager based on configuration."""
        from ..core.shared.caching import CacheManager
        return CacheManager(self.config)


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get or create the global DI container."""
    global _container
    
    if _container is None:
        _container = _setup_container()
    
    return _container


def _setup_container() -> Container:
    """Setup the DI container with all dependencies."""
    container = Container()
    config = get_config()
    
    # Register configuration
    container.register_instance("config", config)
    
    # Register OpenSearch client factory using centralized logic
    container.register_singleton("opensearch_client", create_opensearch_client_for_container)
    
    # Register cache manager factory
    cache_factory = CacheManagerFactory(config)
    container.register_singleton("cache_manager", cache_factory.create_cache_manager)
    
    # Register embedding model factory
    def create_embedding_model_for_container():
        """Factory for embedding model in container."""
        from ..core.shared.models import create_embedding_model
        return create_embedding_model(config)
    
    container.register_singleton("embedding_model", create_embedding_model_for_container)
    
    logger.info("DI container setup completed")
    return container


def reset_container() -> None:
    """Reset the global container."""
    global _container
    _container = None


# Convenience functions for getting dependencies
def get_opensearch_client() -> OpenSearch:
    """Get OpenSearch client from DI container."""
    return get_container().get("opensearch_client")


def get_cache_manager():
    """Get cache manager from DI container."""
    return get_container().get("cache_manager")


def get_embedding_model():
    """Get embedding model from DI container."""
    return get_container().get("embedding_model")


def get_config_from_container() -> AppConfig:
    """Get configuration from DI container."""
    return get_container().get("config")


# Type-based convenience functions
def get_service(service_type: Type[T]) -> T:
    """Get service by type from global container."""
    return get_container().get_service(service_type)


def register_singleton(service_type: Type[T], implementation: Type[T]) -> None:
    """Register singleton service in global container."""
    get_container().register_type_singleton(service_type, implementation)


def register_transient(service_type: Type[T], implementation: Type[T]) -> None:
    """Register transient service in global container."""
    get_container().register_type_transient(service_type, implementation)


def register_scoped(service_type: Type[T], implementation: Type[T]) -> None:
    """Register scoped service in global container."""
    get_container().register_type_scoped(service_type, implementation) 
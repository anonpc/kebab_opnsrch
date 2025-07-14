"""
DEPRECATED: This module has been consolidated into src.core.container

This module has been merged with the main container module to provide
a unified dependency injection system with enhanced features.

Migration Guide:
===============

OLD (this module):
```python
from src.core.shared.dependency_injection import get_container, configure_services

container = get_container()
container.register_singleton(IService, ServiceImpl)
service = container.get_service(IService)
```

NEW (unified container):
```python
from src.core.container import get_container, register_singleton, get_service

# Legacy string-based registration (still supported)
container = get_container()
container.register_singleton("service", factory_function)
service = container.get("service")

# Enhanced type-based registration (recommended)
register_singleton(IService, ServiceImpl)
service = get_service(IService)

# Or using container directly
container = get_container()
container.register_type_singleton(IService, ServiceImpl)
service = container.get_service(IService)
```

Key Changes:
- All functionality moved to src.core.container
- Both legacy string-based and new type-based APIs supported
- Enhanced features: auto-injection, scoped services, factory methods
- Backward compatibility maintained for existing code

For detailed migration instructions, see src/config/README.md
"""

import warnings
from typing import Type, TypeVar

# Import from the unified container for backward compatibility
from ..container import (
    get_container as _get_container,
    Container,
    ServiceScope,
    ServiceRegistration,
    ScopedContainer,
    inject,
    get_service as _get_service,
    register_singleton as _register_singleton,
    register_transient as _register_transient,
    register_scoped as _register_scoped
)

T = TypeVar('T')

def _deprecated_warning(old_function: str, new_function: str):
    """Issue deprecation warning."""
    warnings.warn(
        f"{old_function} is deprecated. Use {new_function} instead. "
        f"See src.core.container for the unified DI system.",
        DeprecationWarning,
        stacklevel=3
    )

# Backward compatibility functions with deprecation warnings
def get_container():
    """DEPRECATED: Use src.core.container.get_container instead."""
    _deprecated_warning(
        "src.core.shared.dependency_injection.get_container",
        "src.core.container.get_container"
    )
    return _get_container()

def configure_services():
    """DEPRECATED: Services are auto-configured in the unified container."""
    _deprecated_warning(
        "src.core.shared.dependency_injection.configure_services",
        "src.core.container.get_container (auto-configured)"
    )
    return _get_container()

def get_service(service_type: Type[T]) -> T:
    """DEPRECATED: Use src.core.container.get_service instead."""
    _deprecated_warning(
        "src.core.shared.dependency_injection.get_service",
        "src.core.container.get_service"
    )
    return _get_service(service_type)

# Export for backward compatibility
DIContainer = Container

# Legacy class aliases for backward compatibility
class ISearchEngine:
    """DEPRECATED: Define your own interfaces."""
    pass

class IIndexer:
    """DEPRECATED: Define your own interfaces."""
    pass

class IScheduler:
    """DEPRECATED: Define your own interfaces."""
    pass

class IEmbeddingModel:
    """DEPRECATED: Define your own interfaces."""
    pass

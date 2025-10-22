"""
Dependency injection container for the DocsReview RAG application.
Provides centralized dependency management and loose coupling.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Local imports
from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceProvider(ABC):
    """Abstract base class for service providers."""
    
    @abstractmethod
    def provide(self, container: 'DIContainer') -> Any:
        """Provide a service instance."""
        pass


@dataclass
class ServiceDefinition:
    """Service definition with provider and configuration."""
    provider: ServiceProvider
    singleton: bool = True
    dependencies: Optional[Dict[str, str]] = None


class DIContainer:
    """
    Dependency injection container for managing service dependencies.
    Implements singleton pattern and lazy initialization.
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized = False
    
    def register_singleton(
        self,
        service_name: str,
        provider: ServiceProvider,
        dependencies: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a singleton service.
        
        Args:
            service_name: Name of the service
            provider: Service provider
            dependencies: Service dependencies
        """
        self._services[service_name] = ServiceDefinition(
            provider=provider,
            singleton=True,
            dependencies=dependencies
        )
        logger.debug(f"Registered singleton service: {service_name}")
    
    def register_transient(
        self,
        service_name: str,
        provider: ServiceProvider,
        dependencies: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a transient service.
        
        Args:
            service_name: Name of the service
            provider: Service provider
            dependencies: Service dependencies
        """
        self._services[service_name] = ServiceDefinition(
            provider=provider,
            singleton=False,
            dependencies=dependencies
        )
        logger.debug(f"Registered transient service: {service_name}")
    
    def register_factory(
        self,
        service_name: str,
        factory: Callable[[], Any]
    ) -> None:
        """
        Register a factory function for service creation.
        
        Args:
            service_name: Name of the service
            factory: Factory function
        """
        self._factories[service_name] = factory
        logger.debug(f"Registered factory for service: {service_name}")
    
    def get(self, service_name: str) -> Any:
        """
        Get a service instance.
        
        Args:
            service_name: Name of the service
        
        Returns:
            Service instance
        
        Raises:
            ConfigurationError: If service is not registered
        """
        if service_name in self._factories:
            return self._factories[service_name]()
        
        if service_name not in self._services:
            raise ConfigurationError(f"Service not registered: {service_name}")
        
        service_def = self._services[service_name]
        
        # Return existing singleton instance
        if service_def.singleton and service_name in self._instances:
            return self._instances[service_name]
        
        # Create new instance
        try:
            instance = service_def.provider.provide(self)
            
            # Store singleton instance
            if service_def.singleton:
                self._instances[service_name] = instance
            
            logger.debug(f"Created instance for service: {service_name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create service {service_name}: {e}")
            raise ConfigurationError(f"Failed to create service {service_name}: {e}")
    
    def get_typed(self, service_name: str, service_type: Type[T]) -> T:
        """
        Get a service instance with type checking.
        
        Args:
            service_name: Name of the service
            service_type: Expected service type
        
        Returns:
            Typed service instance
        """
        instance = self.get(service_name)
        if not isinstance(instance, service_type):
            raise ConfigurationError(
                f"Service {service_name} is not of type {service_type.__name__}"
            )
        return instance
    
    def is_registered(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._services or service_name in self._factories
    
    def clear(self) -> None:
        """Clear all services and instances."""
        self._services.clear()
        self._instances.clear()
        self._factories.clear()
        self._initialized = False
        logger.info("Container cleared")
    
    def initialize(self) -> None:
        """Initialize the container and register default services."""
        if self._initialized:
            return
        
        try:
            # Register core services
            self._register_core_services()
            self._initialized = True
            logger.info("Container initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize container: {e}")
            raise ConfigurationError(f"Container initialization failed: {e}")
    
    def _register_core_services(self) -> None:
        """Register core application services."""
        # This will be implemented with actual service providers
        pass


# Service Providers
class ConfigurationProvider(ServiceProvider):
    """Provider for configuration service."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide configuration instance."""
        from .config.settings import get_settings
        return get_settings()


class LoggingProvider(ServiceProvider):
    """Provider for logging service."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide logging instance."""
        from .logging_config import get_logger
        return get_logger(__name__)


class RAGSystemProvider(ServiceProvider):
    """Provider for RAG system."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide RAG system instance."""
        from .rag.unified_rag import UnifiedRAGSystem
        config = container.get('config')
        
        return UnifiedRAGSystem(
            embedding_model=config.rag.embedding_model,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            top_k=config.rag.top_k,
            min_score=config.rag.min_score,
            vector_store_path=config.rag.vector_store_path
        )


class LLMManagerProvider(ServiceProvider):
    """Provider for LLM manager."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide LLM manager instance."""
        from .llm.llm_manager import LLMManager, LLMConfig, LLMProvider
        
        config = container.get('config')
        
        # Create LLM configurations
        llm_configs = []
        
        if config.llm.primary_provider == "ollama":
            llm_configs.append(LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=config.llm.primary_model,
                base_url=config.llm.primary_base_url,
                api_key=config.llm.ollama_api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            ))
        
        # Add fallback configuration
        llm_configs.append(LLMConfig(
            provider=LLMProvider.FALLBACK,
            model="fallback",
            temperature=0.1,
            max_tokens=1000
        ))
        
        return LLMManager(llm_configs)


class DatabaseProvider(ServiceProvider):
    """Provider for database service."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide database instance."""
        from ..infrastructure.database.database_service import DatabaseService
        config = container.get('config')
        
        return DatabaseService(
            url=config.database.url,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow
        )


class CacheProvider(ServiceProvider):
    """Provider for cache service."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide cache instance."""
        from ..infrastructure.cache.cache_service import CacheService
        config = container.get('config')
        
        return CacheService(
            redis_url=config.cache.redis_url,
            enable_cache=config.cache.enable_cache,
            ttl=config.cache.cache_ttl
        )


class InputValidatorProvider(ServiceProvider):
    """Provider for input validator."""
    
    def provide(self, container: DIContainer) -> Any:
        """Provide input validator instance."""
        from .validation.input_validator import InputValidator
        config = container.get('config')
        
        return InputValidator({
            "max_text_length": config.security.max_input_length,
            "min_text_length": 1,
            "max_file_size_mb": config.security.max_file_size_mb,
            "allowed_file_types": config.security.allowed_file_types,
            "max_url_length": config.security.max_url_length
        })


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = DIContainer()
        _container.initialize()
    return _container


def configure_container() -> DIContainer:
    """Configure and return the container with all services."""
    container = get_container()
    
    # Register all services
    container.register_singleton('config', ConfigurationProvider())
    container.register_singleton('logger', LoggingProvider())
    container.register_singleton('rag_system', RAGSystemProvider())
    container.register_singleton('llm_manager', LLMManagerProvider())
    container.register_singleton('database', DatabaseProvider())
    container.register_singleton('cache', CacheProvider())
    container.register_singleton('input_validator', InputValidatorProvider())
    
    logger.info("Container configured with all services")
    return container


@lru_cache(maxsize=128)
def get_service(service_name: str) -> Any:
    """
    Get a service from the global container.
    
    Args:
        service_name: Name of the service
    
    Returns:
        Service instance
    """
    container = get_container()
    return container.get(service_name)


def get_typed_service(service_name: str, service_type: Type[T]) -> T:
    """
    Get a typed service from the global container.
    
    Args:
        service_name: Name of the service
        service_type: Expected service type
    
    Returns:
        Typed service instance
    """
    container = get_container()
    return container.get_typed(service_name, service_type)


# Convenience functions for common services
def get_rag_system():
    """Get the RAG system service."""
    return get_service('rag_system')


def get_llm_manager():
    """Get the LLM manager service."""
    return get_service('llm_manager')


def get_database():
    """Get the database service."""
    return get_service('database')


def get_cache():
    """Get the cache service."""
    return get_service('cache')


def get_input_validator():
    """Get the input validator service."""
    return get_service('input_validator')


def get_config():
    """Get the configuration service."""
    return get_service('config')


def get_logger():
    """Get the logger service."""
    return get_service('logger')


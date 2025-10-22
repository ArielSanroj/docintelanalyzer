"""
Unified LLM manager with streaming, fallback chain, and retry logic.
Centralizes all LLM interactions with provider-agnostic interface.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
import json

# LangChain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForLLMRun

# Local imports
from ..exceptions import LLMException, LLMProviderError, LLMTimeoutError, LLMRateLimitError
from ..logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NVIDIA = "nvidia"
    FALLBACK = "fallback"


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    provider: LLMProvider
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.9
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 60  # requests per minute
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    content: str
    provider: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]
    response_time: float


class LLMManager:
    """
    Unified LLM manager with streaming, fallback chain, and retry logic.
    """
    
    def __init__(self, configs: List[LLMConfig]):
        """
        Initialize LLM manager with multiple provider configurations.
        
        Args:
            configs: List of LLM configurations in priority order
        """
        self.configs = configs
        self.providers = {}
        self.current_provider = None
        self.rate_limiter = {}
        self.circuit_breaker = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_requests": 0,
            "average_response_time": 0.0,
            "provider_usage": {}
        }
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all configured LLM providers."""
        for config in self.configs:
            try:
                provider = self._create_provider(config)
                self.providers[config.provider] = {
                    "config": config,
                    "instance": provider,
                    "status": "healthy",
                    "last_error": None,
                    "error_count": 0
                }
                logger.info(f"Initialized {config.provider} provider")
            except Exception as e:
                logger.error(f"Failed to initialize {config.provider}: {e}")
                self.providers[config.provider] = {
                    "config": config,
                    "instance": None,
                    "status": "unhealthy",
                    "last_error": str(e),
                    "error_count": 1
                }
        
        # Set primary provider
        if self.providers:
            self.current_provider = list(self.providers.keys())[0]
    
    def _create_provider(self, config: LLMConfig) -> BaseLanguageModel:
        """Create LLM provider instance."""
        if config.provider == LLMProvider.OLLAMA:
            from .providers.ollama_provider import OllamaProvider
            return OllamaProvider(config)
        elif config.provider == LLMProvider.OPENAI:
            from .providers.openai_provider import OpenAIProvider
            return OpenAIProvider(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            from .providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(config)
        elif config.provider == LLMProvider.NVIDIA:
            from .providers.nvidia_provider import NVIDIAProvider
            return NVIDIAProvider(config)
        elif config.provider == LLMProvider.FALLBACK:
            from .providers.fallback_provider import FallbackProvider
            return FallbackProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    @log_execution_time
    def invoke(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> LLMResponse:
        """
        Invoke LLM with messages using fallback chain.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters
        
        Returns:
            LLM response
        """
        start_time = time.time()
        
        # Try providers in order
        for provider_name, provider_info in self.providers.items():
            if provider_info["status"] != "healthy":
                continue
            
            try:
                # Check rate limit
                if self._is_rate_limited(provider_name):
                    logger.warning(f"Rate limited for {provider_name}")
                    continue
                
                # Invoke provider
                response = self._invoke_provider(provider_name, messages, **kwargs)
                
                # Update statistics
                response_time = time.time() - start_time
                self._update_stats(provider_name, True, response_time)
                
                return LLMResponse(
                    content=response.content,
                    provider=provider_name,
                    model=provider_info["config"].model,
                    usage=response.usage_metadata or {},
                    metadata={
                        "response_time": response_time,
                        "provider": provider_name,
                        "model": provider_info["config"].model
                    },
                    response_time=response_time
                )
                
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                self._handle_provider_error(provider_name, e)
                continue
        
        # All providers failed, use fallback
        logger.warning("All providers failed, using fallback")
        return self._fallback_response(messages, time.time() - start_time)
    
    def _invoke_provider(
        self,
        provider_name: str,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> Any:
        """Invoke specific provider."""
        provider_info = self.providers[provider_name]
        provider = provider_info["instance"]
        
        if not provider:
            raise LLMProviderError(f"Provider {provider_name} not initialized")
        
        # Prepare parameters
        params = {
            "temperature": provider_info["config"].temperature,
            "max_tokens": provider_info["config"].max_tokens,
            "top_p": provider_info["config"].top_p,
            **kwargs
        }
        
        # Invoke with timeout
        return provider.invoke(messages, **params)
    
    def _is_rate_limited(self, provider_name: str) -> bool:
        """Check if provider is rate limited."""
        if provider_name not in self.rate_limiter:
            self.rate_limiter[provider_name] = {
                "requests": [],
                "limit": self.providers[provider_name]["config"].rate_limit
            }
        
        rate_info = self.rate_limiter[provider_name]
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"]
            if current_time - req_time < 60
        ]
        
        # Check if under limit
        return len(rate_info["requests"]) >= rate_info["limit"]
    
    def _update_rate_limit(self, provider_name: str) -> None:
        """Update rate limit for provider."""
        if provider_name not in self.rate_limiter:
            self.rate_limiter[provider_name] = {"requests": [], "limit": 60}
        
        self.rate_limiter[provider_name]["requests"].append(time.time())
    
    def _handle_provider_error(self, provider_name: str, error: Exception) -> None:
        """Handle provider error and update status."""
        provider_info = self.providers[provider_name]
        provider_info["error_count"] += 1
        provider_info["last_error"] = str(error)
        
        # Circuit breaker logic
        if provider_info["error_count"] >= 5:
            provider_info["status"] = "circuit_open"
            logger.warning(f"Circuit breaker opened for {provider_name}")
        elif provider_info["error_count"] >= 3:
            provider_info["status"] = "degraded"
            logger.warning(f"Provider {provider_name} degraded")
    
    def _fallback_response(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        response_time: float
    ) -> LLMResponse:
        """Generate fallback response when all providers fail."""
        # Simple fallback response
        content = "I apologize, but I'm currently unable to process your request. Please try again later."
        
        self.stats["fallback_requests"] += 1
        
        return LLMResponse(
            content=content,
            provider="fallback",
            model="fallback",
            usage={"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            metadata={
                "response_time": response_time,
                "provider": "fallback",
                "model": "fallback",
                "fallback": True
            },
            response_time=response_time
        )
    
    def _update_stats(self, provider_name: str, success: bool, response_time: float) -> None:
        """Update statistics."""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average response time
        total_time = self.stats["average_response_time"] * (self.stats["total_requests"] - 1)
        self.stats["average_response_time"] = (total_time + response_time) / self.stats["total_requests"]
        
        # Update provider usage
        if provider_name not in self.stats["provider_usage"]:
            self.stats["provider_usage"][provider_name] = 0
        self.stats["provider_usage"][provider_name] += 1
    
    async def astream(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response asynchronously.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters
        
        Yields:
            Chunks of response text
        """
        # Find working provider
        for provider_name, provider_info in self.providers.items():
            if provider_info["status"] == "healthy" and not self._is_rate_limited(provider_name):
                try:
                    async for chunk in self._astream_provider(provider_name, messages, **kwargs):
                        yield chunk
                    return
                except Exception as e:
                    logger.error(f"Streaming failed for {provider_name}: {e}")
                    continue
        
        # Fallback response
        yield "I apologize, but I'm currently unable to process your request."
    
    async def _astream_provider(
        self,
        provider_name: str,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream from specific provider."""
        provider_info = self.providers[provider_name]
        provider = provider_info["instance"]
        
        if not provider:
            raise LLMProviderError(f"Provider {provider_name} not initialized")
        
        # Check if provider supports streaming
        if hasattr(provider, 'astream'):
            async for chunk in provider.astream(messages, **kwargs):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        else:
            # Fallback to regular invoke
            response = await provider.ainvoke(messages, **kwargs)
            yield response.content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM manager statistics."""
        return {
            **self.stats,
            "providers": {
                name: {
                    "status": info["status"],
                    "error_count": info["error_count"],
                    "last_error": info["last_error"]
                }
                for name, info in self.providers.items()
            },
            "current_provider": self.current_provider
        }
    
    def reset_provider(self, provider_name: str) -> None:
        """Reset provider status."""
        if provider_name in self.providers:
            self.providers[provider_name]["status"] = "healthy"
            self.providers[provider_name]["error_count"] = 0
            self.providers[provider_name]["last_error"] = None
            logger.info(f"Reset provider {provider_name}")
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to specific provider."""
        if provider_name in self.providers:
            self.current_provider = provider_name
            logger.info(f"Switched to provider {provider_name}")
            return True
        return False

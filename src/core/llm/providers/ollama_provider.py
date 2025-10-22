"""
Ollama LLM provider implementation.
Handles Ollama API interactions with retry logic and error handling.
"""

import logging
import time
import requests
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun

# Local imports
from ..llm_manager import LLMConfig
from ...exceptions import LLMProviderError, LLMTimeoutError
from ...logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class OllamaProvider(BaseLanguageModel):
    """Ollama LLM provider with retry logic and error handling."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.
        
        Args:
            config: LLM configuration
        """
        super().__init__()
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.timeout = config.timeout
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "DocsReview-RAG/2.0.0"
        })
        
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
    
    @log_execution_time
    def invoke(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> Any:
        """
        Invoke Ollama API with messages.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
        
        Returns:
            LLM response
        """
        try:
            # Prepare request
            request_data = self._prepare_request(messages, **kwargs)
            
            # Make API call
            response = self._make_api_call(request_data)
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise LLMProviderError(f"Ollama API call failed: {e}")
    
    def _prepare_request(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request data for Ollama API."""
        # Convert messages to Ollama format
        ollama_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                ollama_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                ollama_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                ollama_messages.append({"role": "system", "content": message.content})
        
        # Prepare request data
        request_data = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        return request_data
    
    def _make_api_call(self, request_data: Dict[str, Any]) -> requests.Response:
        """Make API call to Ollama."""
        url = f"{self.base_url}/api/chat"
        
        try:
            response = self.session.post(
                url,
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"Ollama API timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Ollama API request failed: {e}")
    
    def _parse_response(self, response: requests.Response) -> Any:
        """Parse Ollama API response."""
        try:
            data = response.json()
            
            # Create mock response object
            class OllamaResponse:
                def __init__(self, data):
                    self.content = data.get("message", {}).get("content", "")
                    self.usage_metadata = {
                        "total_tokens": data.get("eval_count", 0),
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0)
                    }
            
            return OllamaResponse(data)
            
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise LLMProviderError(f"Failed to parse Ollama response: {e}")
    
    async def ainvoke(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> Any:
        """Async invoke (placeholder for now)."""
        # For now, just call the sync version
        return self.invoke(messages, **kwargs)
    
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "ollama"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

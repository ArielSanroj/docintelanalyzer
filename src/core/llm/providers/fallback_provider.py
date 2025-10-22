"""
Fallback LLM provider implementation.
Provides basic responses when all other providers fail.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun

# Local imports
from ..llm_manager import LLMConfig
from ...exceptions import LLMProviderError
from ...logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class FallbackProvider(BaseLanguageModel):
    """Fallback LLM provider for when all other providers fail."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize fallback provider.
        
        Args:
            config: LLM configuration
        """
        super().__init__()
        self.config = config
        self.model = "fallback"
        
        # Simple response templates
        self.response_templates = {
            "greeting": [
                "Hello! I'm here to help you with document analysis and questions.",
                "Hi there! I can assist you with analyzing documents and answering questions.",
                "Greetings! I'm ready to help you with document processing and analysis."
            ],
            "question": [
                "I understand you have a question. Let me help you find the information you need.",
                "That's an interesting question. Let me search through the available documents for you.",
                "I'll do my best to answer your question based on the available information."
            ],
            "error": [
                "I apologize, but I'm currently unable to process your request. Please try again later.",
                "I'm experiencing some technical difficulties. Please try again in a moment.",
                "I'm sorry, but I can't process that request right now. Please try again."
            ],
            "general": [
                "I'm here to help you with document analysis and questions. What would you like to know?",
                "I can assist you with analyzing documents and answering questions. How can I help?",
                "I'm ready to help you with document processing. What do you need assistance with?"
            ]
        }
    
    @log_execution_time
    def invoke(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> Any:
        """
        Generate fallback response.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
        
        Returns:
            Fallback response
        """
        try:
            # Analyze the last message to determine response type
            if not messages:
                response_type = "general"
            else:
                last_message = messages[-1]
                if isinstance(last_message, HumanMessage):
                    content = last_message.content.lower()
                    if any(word in content for word in ["hello", "hi", "hey", "greetings"]):
                        response_type = "greeting"
                    elif "?" in content or any(word in content for word in ["what", "how", "why", "when", "where"]):
                        response_type = "question"
                    else:
                        response_type = "general"
                else:
                    response_type = "general"
            
            # Generate response
            response = self._generate_response(response_type)
            
            # Create mock response object
            class FallbackResponse:
                def __init__(self, content):
                    self.content = content
                    self.usage_metadata = {
                        "total_tokens": len(content.split()),
                        "prompt_tokens": 0,
                        "completion_tokens": len(content.split())
                    }
            
            return FallbackResponse(response)
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            # Return a basic error response
            class ErrorResponse:
                def __init__(self):
                    self.content = "I apologize, but I'm currently unable to process your request."
                    self.usage_metadata = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            
            return ErrorResponse()
    
    def _generate_response(self, response_type: str) -> str:
        """Generate response based on type."""
        import random
        
        templates = self.response_templates.get(response_type, self.response_templates["general"])
        return random.choice(templates)
    
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
        return "fallback"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "provider": "fallback"
        }

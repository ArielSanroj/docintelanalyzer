"""
Generalized LLM Helper for DocsReview RAG System
Provides a flexible, provider-agnostic LLM integration pattern with retry logic,
structured prompts, and easy provider switching.
"""

import os
import logging
import time
import json
from typing import Any, Dict, List, Optional, Union
from functools import wraps
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# Provider imports
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Import existing Ollama implementation as fallback
from llm_fallback import OllamaLLM, FallbackLLM

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.9

def retry_on_failure(max_attempts: int = 3, delay: float = 2.0):
    """Decorator for retrying LLM calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class LLMHelper:
    """
    Generalized LLM helper with provider-agnostic interface.
    Supports NVIDIA, OpenAI, Anthropic, Ollama, and custom implementations.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize LLM helper with specified configuration.
        
        Args:
            config: LLMConfig object with provider settings
            **kwargs: Additional configuration parameters
        """
        self.config = config or self._get_default_config()
        self.llm = self._initialize_llm()
        self.conversation_history: List[Dict] = []
        
        logger.info(f"LLMHelper initialized with provider: {self.config.provider}")
    
    def _get_default_config(self) -> LLMConfig:
        """Get default configuration based on environment variables"""
        # Check for Ollama API key first (primary choice)
        ollama_key = os.getenv('OLLAMA_API_KEY')
        if ollama_key:
            return LLMConfig(
                provider='ollama',
                model='llama3.1:8b',
                base_url='http://localhost:11434',
                api_key=ollama_key
            )
        
        # Check for NVIDIA API key as fallback
        nvidia_key = os.getenv('NVIDIA_API_KEY')
        if nvidia_key and NVIDIA_AVAILABLE:
            return LLMConfig(
                provider='nvidia',
                model='meta/llama-3.1-8b-instruct',
                api_key=nvidia_key
            )
        
        # Check for OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and OPENAI_AVAILABLE:
            return LLMConfig(
                provider='openai',
                model='gpt-4',
                api_key=openai_key
            )
        
        # Check for Anthropic API key
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and ANTHROPIC_AVAILABLE:
            return LLMConfig(
                provider='anthropic',
                model='claude-3-sonnet-20240229',
                api_key=anthropic_key
            )
        
        # Final fallback to Ollama without API key
        return LLMConfig(
            provider='ollama',
            model='llama3.1:8b',
            base_url='http://localhost:11434'
        )
    
    def _initialize_llm(self) -> Any:
        """Initialize the LLM based on configuration"""
        try:
            if self.config.provider == 'nvidia' and NVIDIA_AVAILABLE:
                return ChatNVIDIA(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    temperature=self.config.temperature
                )
            
            elif self.config.provider == 'openai' and OPENAI_AVAILABLE:
                return ChatOpenAI(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            
            elif self.config.provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                return ChatAnthropic(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            
            elif self.config.provider == 'ollama' and OLLAMA_AVAILABLE:
                return ChatOllama(
                    model=self.config.model,
                    base_url=self.config.base_url or "http://localhost:11434",
                    temperature=self.config.temperature
                )
            
            else:
                # Use custom Ollama implementation as fallback
                logger.warning(f"Provider {self.config.provider} not available, using custom Ollama implementation")
                return OllamaLLM(
                    model=self.config.model,
                    base_url=self.config.base_url or "http://localhost:11434"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.provider} LLM: {e}")
            logger.info("Falling back to simple fallback LLM")
            return FallbackLLM()
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def invoke_llm(self, prompt_str: str, variables: Dict[str, Any] = None) -> str:
        """
        Invoke LLM with a prompt template and variables.
        
        Args:
            prompt_str: Prompt template string
            variables: Variables to substitute in the template
            
        Returns:
            Generated response text
        """
        variables = variables or {}
        
        try:
            # Create prompt template if variables provided
            if variables:
                prompt = PromptTemplate(
                    input_variables=list(variables.keys()),
                    template=prompt_str
                )
                formatted_prompt = prompt.format(**variables)
            else:
                formatted_prompt = prompt_str
            
            # Invoke LLM
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                return response.content.strip() if hasattr(response, 'content') else str(response)
            else:
                # Fallback for custom implementations
                response = self.llm.invoke(formatted_prompt)
                return response.content.strip() if hasattr(response, 'content') else str(response)
                
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise
    
    def extract_entities(self, text: str, entities: List[str] = None) -> Dict[str, str]:
        """
        Extract entities from text using structured prompts.
        
        Args:
            text: Input text to analyze
            entities: List of entity types to extract
            
        Returns:
            Dictionary of extracted entities
        """
        entities = entities or ["company", "person", "title", "date", "location", "amount"]
        entity_list = "\n".join([f"- {e}: {e} if mentioned" for e in entities])
        
        prompt_str = f"""Extract the following entities from this text:

Text: {{text}}

Extract:
{entity_list}

Return ONLY a JSON object with the format: {{{", ".join([f'"{e}": "value or null"' for e in entities])}}}

Example: {{"company": "Microsoft", "person": "John Doe", "date": "2024-01-15", "location": "Seattle", "amount": "500000", "title": "CTO"}}"""
        
        try:
            response = self.invoke_llm(prompt_str, {"text": text})
            # Clean response and try to parse JSON
            response = response.strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]  # Remove outer quotes
            if response.startswith("'") and response.endswith("'"):
                response = response[1:-1]  # Remove outer quotes
            
            # Try to parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.warning("No JSON found in response, returning raw text")
                    return {"raw_response": response}
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"error": str(e)}
    
    def generate_content(self, template: str, variables: Dict[str, Any] = None) -> str:
        """
        Generate content using a template and variables.
        
        Args:
            template: Content template
            variables: Variables to substitute
            
        Returns:
            Generated content
        """
        variables = variables or {}
        prompt_str = template + "\n\nOutput:"
        return self.invoke_llm(prompt_str, variables)
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        prompt_str = """Analyze the sentiment of this text: {text}

Return: positive, negative, or neutral with a brief explanation."""
        
        return self.invoke_llm(prompt_str, {"text": text})
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text to specified length.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Text summary
        """
        prompt_str = f"""Summarize the following text in approximately {max_length} words:

Text: {{text}}

Summary:"""
        
        return self.invoke_llm(prompt_str, {"text": text})
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: Question to answer
            context: Context to base answer on
            
        Returns:
            Answer to the question
        """
        prompt_str = """Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:"""
        
        return self.invoke_llm(prompt_str, {"question": question, "context": context})
    
    def generate_rag_response(self, query: str, retrieved_chunks: List[str], 
                            conversation_context: List[Dict] = None) -> str:
        """
        Generate RAG response using retrieved chunks and conversation context.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved document chunks
            conversation_context: Previous conversation context
            
        Returns:
            Generated RAG response
        """
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[Chunk {i+1}]\n{chunk}\n")
        
        context = "\n".join(context_parts)
        
        # Prepare conversation context
        conversation_str = ""
        if conversation_context:
            recent_messages = conversation_context[-3:]  # Last 3 interactions
            conversation_str = "\n\nCONVERSATION CONTEXT:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_str += f"{role}: {msg['content']}\n"
        
        prompt_str = f"""You are a document analysis assistant. Answer the user's question based ONLY on the provided document chunks.

DOCUMENT CHUNKS:
{context}

USER QUESTION: {query}
{conversation_str}

RULES:
1. Answer based ONLY on the provided document chunks
2. If information is not in the chunks, say "I don't have that information in the provided document"
3. Be specific and cite relevant chunks when possible
4. Consider the conversation context but prioritize document information

Answer:"""
        
        return self.invoke_llm(prompt_str, {"query": query, "context": context})
    
    def switch_provider(self, new_provider: str, model: str = None, **kwargs) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            new_provider: New provider name
            model: Model name for the new provider
            **kwargs: Additional configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            old_provider = self.config.provider
            self.config.provider = new_provider
            if model:
                self.config.model = model
            
            # Update other config parameters
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Reinitialize LLM
            self.llm = self._initialize_llm()
            
            logger.info(f"Successfully switched from {old_provider} to {new_provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to {new_provider}: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "available_providers": self._get_available_providers()
        }
    
    def _get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        providers = []
        if NVIDIA_AVAILABLE:
            providers.append("nvidia")
        if OPENAI_AVAILABLE:
            providers.append("openai")
        if ANTHROPIC_AVAILABLE:
            providers.append("anthropic")
        if OLLAMA_AVAILABLE:
            providers.append("ollama")
        providers.append("custom_ollama")  # Always available
        providers.append("fallback")  # Always available
        return providers
    
    def update_conversation_history(self, user_input: str, assistant_response: str):
        """Update conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": time.time()
        })
        
        # Keep only last 20 interactions
        if len(self.conversation_history) > 40:  # 20 pairs
            self.conversation_history = self.conversation_history[-40:]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history."""
        if not self.conversation_history:
            return {"status": "No conversation history"}
        
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        return {
            "total_interactions": len(user_messages),
            "last_user_message": user_messages[-1]["content"] if user_messages else None,
            "last_assistant_message": assistant_messages[-1]["content"] if assistant_messages else None,
            "conversation_length": len(self.conversation_history)
        }

# Factory function for easy initialization
def create_llm_helper(provider: str = None, model: str = None, **kwargs) -> LLMHelper:
    """
    Factory function to create LLMHelper with specific configuration.
    
    Args:
        provider: LLM provider name
        model: Model name
        **kwargs: Additional configuration
        
    Returns:
        Configured LLMHelper instance
    """
    config = None
    if provider or model:
        config = LLMConfig(
            provider=provider or "auto",
            model=model or "default",
            **kwargs
        )
    
    return LLMHelper(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the LLM helper
    helper = create_llm_helper()
    
    # Test basic functionality
    print("Provider info:", helper.get_provider_info())
    
    # Test entity extraction
    text = "Microsoft hired Sarah Johnson as Chief Technology Officer on January 15, 2024 in Seattle, Washington for $500,000 annually."
    entities = helper.extract_entities(text)
    print("Extracted entities:", entities)
    
    # Test content generation
    summary = helper.summarize_text(text, max_length=50)
    print("Summary:", summary)
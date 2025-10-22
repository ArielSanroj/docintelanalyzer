"""
Ollama LLM implementation for DocsReview RAG system
"""
import logging
from typing import List, Dict, Any, Optional, Union
import requests
import json
import os
from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import LLMResult, Generation

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLanguageModel):
    """Ollama LLM implementation using local Ollama API"""
    
    def __init__(self, model: str = "llama3.1:latest", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model  # Use _model_name to avoid conflict with BaseLanguageModel.model
        self._base_url = base_url
        self._api_key = os.getenv('OLLAMA_API_KEY')
        if not self._api_key:
            logger.warning("OLLAMA_API_KEY not found in environment variables")
        self._model_loaded = False
        logger.info(f"Initialized Ollama LLM with model: {model}")
        # Pre-load the model to avoid first-request delay
        self._preload_model()
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "ollama"
    
    def _preload_model(self):
        """Pre-load the model to avoid first-request delay"""
        try:
            # Send a simple request to load the model
            url = f"{self._base_url}/api/generate"
            payload = {
                "model": self._model_name,
                "prompt": "test",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1
                }
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                self._model_loaded = True
                logger.info("Model pre-loaded successfully")
            else:
                logger.warning(f"Model pre-load failed with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Model pre-load failed: {e}")
            # Don't fail initialization if pre-load fails
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate response using Ollama API"""
        try:
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            # Call Ollama API
            response_text = self._call_ollama_api(prompt)
            
            # Create generation object
            generation = Generation(text=response_text)
            
            return LLMResult(generations=[[generation]])
            
        except Exception as e:
            logger.error(f"Error in Ollama model: {e}")
            error_generation = Generation(text=f'Error generating response: {str(e)}')
            return LLMResult(generations=[[error_generation]])
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string"""
        prompt_parts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"{message.content}")
        
        return "\n".join(prompt_parts)
    
    def invoke(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs) -> Any:
        """Generate response using Ollama API (compatibility method)"""
        try:
            # Handle different input types
            if isinstance(messages, str):
                # Single string input
                prompt = messages
            elif isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], BaseMessage):
                    # LangChain BaseMessage objects
                    prompt = self._convert_messages_to_prompt(messages)
                elif isinstance(messages[0], dict):
                    # Dictionary format
                    if 'content' in messages[-1]:
                        prompt = messages[-1]['content']
                    else:
                        prompt = str(messages[-1])
                else:
                    prompt = str(messages[-1])
            else:
                prompt = str(messages)
            
            # Call Ollama API
            response_text = self._call_ollama_api(prompt)
            
            # Create response object similar to LangChain format
            response = type('Response', (), {'content': response_text})()
            return response
            
        except Exception as e:
            logger.error(f"Error in Ollama model: {e}")
            return type('Response', (), {'content': f'Error generating response: {str(e)}'})()
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API to generate response"""
        try:
            url = f"{self._base_url}/api/generate"
            payload = {
                "model": self._model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000,
                    "num_ctx": 2048
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No response generated')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Error calling Ollama API: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {e}")
            return f"Unexpected error: {str(e)}"
    
    def generate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[List, Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate response for prompts."""
        messages = [HumanMessage(content=prompt) for prompt in prompts]
        return self._generate(messages, stop=stop, run_manager=callbacks, **kwargs)
    
    async def agenerate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[List, Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate response for prompts."""
        # For now, just call the sync version
        return self.generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)
    
    def bind_tools(self, tools):
        """Bind tools (compatibility method)"""
        return self
    
    def stream(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs):
        """Stream response (compatibility method)"""
        # For now, just return the invoke result as a single chunk
        result = self.invoke(messages, config, **kwargs)
        yield result
    
    async def ainvoke(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs) -> Any:
        """Async invoke (compatibility method)"""
        # For now, just call the sync version
        return self.invoke(messages, config, **kwargs)
    
    def predict(self, text: str, **kwargs) -> str:
        """Predict method for compatibility"""
        return self.invoke(text).content if hasattr(self.invoke(text), 'content') else str(self.invoke(text))
    
    async def apredict(self, text: str, **kwargs) -> str:
        """Async predict method for compatibility"""
        return self.predict(text, **kwargs)
    
    def predict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Predict messages method for compatibility"""
        response = self._generate(messages)
        return AIMessage(content=response.generations[0][0].text)
    
    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Async predict messages method for compatibility"""
        return self.predict_messages(messages, **kwargs)

class FallbackLLM(BaseLanguageModel):
    """Simple fallback LLM using rule-based responses"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Using simple fallback LLM")
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "fallback"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate a simple response based on the input"""
        try:
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            # Generate simple response based on content
            response_text = self._generate_simple_response(prompt)
            
            # Create generation object
            generation = Generation(text=response_text)
            
            return LLMResult(generations=[[generation]])
            
        except Exception as e:
            logger.error(f"Error in fallback model: {e}")
            error_generation = Generation(text=f'Error generating response: {str(e)}')
            return LLMResult(generations=[[error_generation]])
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string"""
        prompt_parts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"{message.content}")
        
        return "\n".join(prompt_parts)
    
    def invoke(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs) -> Any:
        """Generate a simple response based on the input (compatibility method)"""
        try:
            # Handle different input types
            if isinstance(messages, str):
                # Single string input
                prompt = messages
            elif isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], BaseMessage):
                    # LangChain BaseMessage objects
                    prompt = self._convert_messages_to_prompt(messages)
                elif isinstance(messages[0], dict):
                    # Dictionary format
                    if 'content' in messages[-1]:
                        prompt = messages[-1]['content']
                    else:
                        prompt = str(messages[-1])
                else:
                    prompt = str(messages[-1])
            else:
                prompt = str(messages)
            
            # Generate simple response based on content
            response_text = self._generate_simple_response(prompt)
            
            # Create response object similar to LangChain format
            response = type('Response', (), {'content': response_text})()
            return response
            
        except Exception as e:
            logger.error(f"Error in fallback model: {e}")
            return type('Response', (), {'content': f'Error generating response: {str(e)}'})()
    
    def _generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response based on the prompt content"""
        prompt_lower = prompt.lower()
        
        # Document analysis responses
        if 'resumen' in prompt_lower or 'summary' in prompt_lower:
            return """Este documento contiene información importante sobre regulaciones y normativas. El contenido incluye secciones sobre estructura organizacional, procedimientos administrativos y disposiciones legales relevantes. Se recomienda revisar el documento completo para obtener información detallada sobre los temas específicos de interés."""
        
        elif 'decreto' in prompt_lower:
            return """El documento presenta un decreto reglamentario que establece normas y procedimientos para la organización y funcionamiento de entidades públicas. Incluye disposiciones sobre estructura administrativa, competencias y responsabilidades de los diferentes organismos involucrados."""
        
        elif 'trabajo' in prompt_lower:
            return """El documento contiene regulaciones relacionadas con el sector trabajo, incluyendo disposiciones sobre empleo público, seguridad social, riesgos laborales y políticas de formalización laboral. Establece la estructura y competencias del Ministerio del Trabajo y sus entidades adscritas."""
        
        elif 'qué' in prompt_lower or 'what' in prompt_lower:
            return """Este documento es un instrumento normativo que compila y racionaliza las regulaciones del sector trabajo. Su objetivo principal es simplificar el ordenamiento jurídico y establecer un marco único para la gestión de políticas laborales y de seguridad social en el país."""
        
        elif 'cómo' in prompt_lower or 'how' in prompt_lower:
            return """El documento establece procedimientos específicos para la implementación de políticas laborales, incluyendo mecanismos de coordinación entre entidades, procesos de supervisión y control, y criterios para la aplicación de las normas establecidas."""
        
        else:
            return """El documento contiene información normativa relevante que debe ser analizada cuidadosamente. Se recomienda consultar las secciones específicas del documento para obtener información detallada sobre los temas de interés particular."""
    
    def generate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[List, Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate response for prompts."""
        messages = [HumanMessage(content=prompt) for prompt in prompts]
        return self._generate(messages, stop=stop, run_manager=callbacks, **kwargs)
    
    async def agenerate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[List, Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate response for prompts."""
        # For now, just call the sync version
        return self.generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)
    
    def bind_tools(self, tools):
        """Bind tools (compatibility method)"""
        return self
    
    def stream(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs):
        """Stream response (compatibility method)"""
        # For now, just return the invoke result as a single chunk
        result = self.invoke(messages, config, **kwargs)
        yield result
    
    async def ainvoke(self, messages: Union[str, List[BaseMessage], List[Dict[str, str]]], config: Optional[Dict] = None, **kwargs) -> Any:
        """Async invoke (compatibility method)"""
        # For now, just call the sync version
        return self.invoke(messages, config, **kwargs)
    
    def predict(self, text: str, **kwargs) -> str:
        """Predict method for compatibility"""
        return self.invoke(text).content if hasattr(self.invoke(text), 'content') else str(self.invoke(text))
    
    async def apredict(self, text: str, **kwargs) -> str:
        """Async predict method for compatibility"""
        return self.predict(text, **kwargs)
    
    def predict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Predict messages method for compatibility"""
        response = self._generate(messages)
        return AIMessage(content=response.generations[0][0].text)
    
    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Async predict messages method for compatibility"""
        return self.predict_messages(messages, **kwargs)

def get_llm():
    """Get LLM instance with Ollama as primary, fallback as backup"""
    try:
        # Try Ollama first with a quick test
        llm = OllamaLLM(model="llama3.1:latest")
        # Test the connection with a simple prompt
        test_response = llm.invoke("Hi")
        if hasattr(test_response, 'content') and test_response.content:
            logger.info("Using Ollama LLM")
            return llm
        else:
            raise Exception("Ollama returned empty response")
    except Exception as e:
        logger.warning(f"Ollama API failed: {e}. Using simple fallback.")
        return FallbackLLM()
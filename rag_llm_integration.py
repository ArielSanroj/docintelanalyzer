"""
Módulo de integración RAG + LLM para mejorar las respuestas del sistema.
Combina recuperación inteligente de información con generación contextualizada.
Integra RAG-Anything para capacidades multimodales avanzadas.
Ahora incluye LLMHelper para integración generalizada y robusta.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np

from rag_system import RAGSystem, RetrievalResult
from llm_helper import LLMHelper, create_llm_helper

# RAG-Anything imports
try:
    from rag_anything_integration import get_rag_anything
    RAG_ANYTHING_AVAILABLE = True
except ImportError:
    RAG_ANYTHING_AVAILABLE = False
    print("Warning: RAG-Anything not available. Using fallback embeddings.")

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Respuesta generada por el sistema RAG + LLM."""
    answer: str
    relevant_chunks: List[str]
    confidence_score: float
    retrieval_info: Dict
    llm_metadata: Dict

class RAGLLMIntegration:
    """Integración avanzada entre RAG y LLM con soporte multimodal."""
    
    def __init__(self, llm: Any, rag_system: RAGSystem, use_rag_anything: bool = True, 
                 use_llm_helper: bool = True, llm_config: Optional[Dict] = None):
        """
        Inicializa la integración RAG + LLM.
        
        Args:
            llm: Instancia del LLM configurado (legacy support)
            rag_system: Sistema RAG configurado
            use_rag_anything: Si usar RAG-Anything para capacidades multimodales
            use_llm_helper: Si usar el nuevo LLMHelper para integración robusta
            llm_config: Configuración para LLMHelper si se usa
        """
        self.rag_system = rag_system
        self.conversation_history: List[Dict] = []
        self.use_rag_anything = use_rag_anything and RAG_ANYTHING_AVAILABLE
        self.use_llm_helper = use_llm_helper
        
        # Initialize LLM - use LLMHelper if enabled, otherwise legacy LLM
        if use_llm_helper:
            try:
                # Create LLMHelper with provided config or auto-detect
                if llm_config:
                    self.llm_helper = create_llm_helper(**llm_config)
                else:
                    self.llm_helper = create_llm_helper()
                self.llm = self.llm_helper.llm  # Keep legacy interface
                logger.info("Using LLMHelper for enhanced LLM integration")
            except Exception as e:
                logger.warning(f"Failed to initialize LLMHelper: {e}. Falling back to legacy LLM.")
                self.llm = llm
                self.llm_helper = None
                self.use_llm_helper = False
        else:
            self.llm = llm
            self.llm_helper = None
        
        # Initialize RAG-Anything if available
        if self.use_rag_anything and RAG_ANYTHING_AVAILABLE:
            try:
                self.rag_anything = get_rag_anything()
                if self.rag_anything.is_available():
                    logger.info("RAG-Anything initialized successfully")
                else:
                    logger.warning("RAG-Anything not available")
                    self.use_rag_anything = False
            except Exception as e:
                logger.warning(f"Failed to initialize RAG-Anything: {e}")
                self.use_rag_anything = False
        else:
            self.use_rag_anything = False
        
        # Initialize local embeddings as fallback
        self.local_embeddings = self._initialize_local_embeddings()
    
    def _initialize_local_embeddings(self) -> Optional[SentenceTransformer]:
        """Initialize local embeddings as fallback."""
        try:
            # Use a lightweight, multilingual model
            embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Local embeddings initialized successfully")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to initialize local embeddings: {e}")
            return None
    
    def _get_enhanced_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings using RAG-Anything or local fallback."""
        if self.use_rag_anything and hasattr(self, 'rag_anything'):
            try:
                # RAG-Anything doesn't have a direct get_embeddings method
                # Use local embeddings as fallback
                logger.info("Using local embeddings (RAG-Anything doesn't expose embeddings directly)")
            except Exception as e:
                logger.warning(f"RAG-Anything embeddings failed: {e}")
        
        # Use local embeddings
        if self.local_embeddings:
            try:
                return self.local_embeddings.encode(text)
            except Exception as e:
                logger.warning(f"Local embeddings failed: {e}")
        
        return None
    
    def _multimodal_retrieval(self, query: str, document_path: Optional[str] = None) -> RetrievalResult:
        """Enhanced retrieval with multimodal capabilities."""
        if self.use_rag_anything and hasattr(self, 'rag_anything') and self.rag_anything.is_available():
            try:
                # Use RAG-Anything for multimodal retrieval
                if document_path:
                    result = self.rag_anything.query_document(query, document_path)
                    if "error" not in result:
                        # Convert RAG-Anything result to RetrievalResult format
                        chunks = result.get('chunks', [])
                        scores = result.get('scores', [1.0] * len(chunks))
                        metadata = result.get('metadata', {})
                        return RetrievalResult(
                            chunks=chunks,
                            scores=scores,
                            metadata=metadata,
                            retrieval_method="rag_anything_multimodal"
                        )
            except Exception as e:
                logger.warning(f"RAG-Anything retrieval failed: {e}")
        
        # Fallback to standard RAG system
        return self.rag_system.retrieve_relevant_chunks(query)
    
    def generate_enhanced_prompt(self, query: str, retrieval_result: RetrievalResult, 
                               conversation_context: Optional[List[Dict]] = None, query_intent: Dict = None) -> str:
        """
        Genera un prompt mejorado que integra información recuperada por RAG con few-shot adaptativo.
        
        Args:
            query: Consulta del usuario
            retrieval_result: Resultado de la recuperación RAG
            conversation_context: Contexto de conversación previa
            query_intent: Análisis de la intención de la consulta
            
        Returns:
            Prompt optimizado para el LLM
        """
        # Preparar contexto recuperado
        retrieved_context = self.rag_system.get_context_for_llm(retrieval_result)
        
        # Preparar contexto de conversación si existe
        conversation_context_str = ""
        if conversation_context:
            recent_messages = conversation_context[-3:]  # Últimas 3 interacciones
            conversation_context_str = "\n\nCONTEXTO DE CONVERSACIÓN RECIENTE:\n"
            for msg in recent_messages:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                conversation_context_str += f"{role}: {msg['content']}\n"
        
        # Few-shot dinámico basado en intent
        few_shot = ""
        if query_intent and query_intent["query_type"] == "definicion":
            few_shot = """
EJEMPLO GENÉRICO: Para '¿de qué trata?', usa chunks overview: 'Este documento trata de [tema principal del chunk 0], con secciones sobre [detalles relevantes]. Todo del documento.'"""
        elif query_intent and query_intent["query_type"] == "proceso":
            few_shot = """
EJEMPLO: Para 'cómo hacer X', lista pasos de chunks secuenciales: 'Paso 1: [de chunk1]. Paso 2: [de chunk2]. Del documento.'"""
        elif query_intent and query_intent["query_type"] == "temporal":
            few_shot = """
EJEMPLO: Para 'cuándo/cronología', busca fechas en chunks: 'Según el documento, [evento] ocurrió en [fecha específica del chunk].'"""
        elif query_intent and query_intent["query_type"] == "causal":
            few_shot = """
EJEMPLO: Para 'por qué/razón', conecta causas y efectos: 'El documento indica que [causa del chunk X] resulta en [efecto del chunk Y].'"""
        
        # Construir prompt mejorado
        enhanced_prompt = f"""Eres un asistente especializado en análisis de documentos. Tu función es responder preguntas basándote ÚNICAMENTE en la información específica del documento proporcionado.

DOCUMENTO ACTUAL:
Estás analizando un documento específico que el usuario ha cargado. Solo debes responder basándote en la información de este documento.

INFORMACIÓN RECUPERADA DEL DOCUMENTO:
{retrieved_context}

CONSULTA DEL USUARIO: {query}
{conversation_context_str}
{few_shot}

REGLAS ESTRICTAS:
1. **SOLO información del documento**: Responde ÚNICAMENTE basándote en la información recuperada del documento actual
2. **NO conocimiento general**: NO uses conocimiento general externo a menos que sea absolutamente necesario para explicar algo del documento
3. **Indica claramente las fuentes**: Menciona específicamente qué información viene del documento
4. **Si no hay información**: Si no encuentras información relevante en el documento, di claramente "No encontré información sobre esto en el documento"
5. **Sé específico**: Proporciona detalles concretos y citas exactas del documento cuando sea posible
6. **Contexto conversacional**: Considera el contexto de la conversación pero siempre prioriza la información del documento
7. **Prioriza chunks overview**: Para preguntas generales ("¿de qué trata?"), prioriza chunks con metadata "type: overview"

FORMATO DE RESPUESTA:
- Estructura con Título/Tema si definición; Pasos si proceso
- Responde de manera clara y estructurada
- Usa viñetas cuando sea apropiado
- Incluye citas específicas del documento entre comillas
- Si no encuentras información específica, indícalo claramente
- Al final, menciona qué información viene del documento vs conocimiento general
- Cita chunks específicos cuando sea relevante

RESPUESTA:"""
        
        return enhanced_prompt
    
    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """
        Analiza la intención de la consulta para optimizar la recuperación.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Diccionario con análisis de la intención
        """
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        query_type = "general"
        if any(word in query_lower for word in ["qué", "qué es", "definir", "significa"]):
            query_type = "definicion"
        elif any(word in query_lower for word in ["cómo", "proceso", "pasos", "método"]):
            query_type = "proceso"
        elif any(word in query_lower for word in ["cuándo", "fecha", "tiempo", "cronología"]):
            query_type = "temporal"
        elif any(word in query_lower for word in ["dónde", "ubicación", "lugar"]):
            query_type = "ubicacion"
        elif any(word in query_lower for word in ["quién", "persona", "autor", "responsable"]):
            query_type = "persona"
        elif any(word in query_lower for word in ["por qué", "razón", "causa", "motivo"]):
            query_type = "causal"
        
        # Detectar palabras clave importantes
        important_words = []
        stop_words = {"el", "la", "de", "del", "en", "con", "por", "para", "un", "una", "es", "son", "se", "lo", "las", "los"}
        words = query.split()
        for word in words:
            if len(word) > 3 and word.lower() not in stop_words:
                important_words.append(word.lower())
        
        return {
            "query_type": query_type,
            "important_words": important_words,
            "query_length": len(query),
            "has_specific_terms": len(important_words) > 0
        }
    
    def optimize_retrieval_params(self, query_intent: Dict) -> Dict:
        """
        Optimiza los parámetros de recuperación basado en la intención de la consulta.
        
        Args:
            query_intent: Análisis de la intención de la consulta
            
        Returns:
            Parámetros optimizados para la recuperación
        """
        base_params = {"top_k": 5, "min_score": 0.3}
        
        query_type = query_intent["query_type"]
        
        if query_type == "definicion":
            # Para definiciones, necesitamos menos chunks pero más específicos
            base_params.update({"top_k": 3, "min_score": 0.4})
        elif query_type == "proceso":
            # Para procesos, necesitamos más chunks para capturar secuencias
            base_params.update({"top_k": 7, "min_score": 0.25})
        elif query_type == "temporal":
            # Para información temporal, priorizamos chunks con fechas
            base_params.update({"top_k": 4, "min_score": 0.35})
        elif query_type == "causal":
            # Para relaciones causales, necesitamos contexto amplio
            base_params.update({"top_k": 6, "min_score": 0.3})
        
        return base_params
    
    def generate_response(self, query: str, conversation_context: Optional[List[Dict]] = None, 
                         document_path: Optional[str] = None) -> RAGResponse:
        """
        Genera una respuesta completa usando RAG + LLM con capacidades multimodales.
        
        Args:
            query: Consulta del usuario
            conversation_context: Contexto de conversación
            document_path: Ruta del documento para análisis multimodal
            
        Returns:
            Respuesta completa del sistema
        """
        logger.info(f"Generando respuesta para: '{query}'")
        
        # Analizar intención de la consulta
        query_intent = self.analyze_query_intent(query)
        logger.debug(f"Intención detectada: {query_intent}")
        
        # Optimizar parámetros de recuperación
        retrieval_params = self.optimize_retrieval_params(query_intent)
        logger.debug(f"Parámetros de recuperación: {retrieval_params}")
        
        # Recuperar información relevante (multimodal si está disponible)
        retrieval_result = self._multimodal_retrieval(query, document_path)
        
        # Ajustar parámetros si es necesario
        if retrieval_result.retrieval_method == "rag_anything_multimodal":
            # RAG-Anything ya optimiza internamente
            logger.info("Usando RAG-Anything para recuperación multimodal")
        else:
            # Usar parámetros optimizados para RAG estándar
            retrieval_result = self.rag_system.retrieve_relevant_chunks(
                query, 
                top_k=retrieval_params["top_k"],
                min_score=retrieval_params["min_score"]
            )
        
        # Generar prompt mejorado
        enhanced_prompt = self.generate_enhanced_prompt(query, retrieval_result, conversation_context, query_intent)
        
        # Generar respuesta con LLM usando LLMHelper si está disponible
        try:
            if self.use_llm_helper and self.llm_helper:
                # Use LLMHelper for robust response generation
                answer = self.llm_helper.invoke_llm(enhanced_prompt)
            else:
                # Use legacy LLM interface
                response = self.llm.invoke([HumanMessage(content=enhanced_prompt)])
                answer = response.content if response.content else "No pude generar una respuesta."
        except Exception as e:
            logger.error(f"Error generando respuesta LLM: {e}")
            answer = f"Error al generar respuesta: {str(e)}"
        
        # Calcular score de confianza basado en la calidad de la recuperación
        confidence_score = self._calculate_confidence_score(retrieval_result, query_intent)
        
        # Preparar información de recuperación
        retrieval_info = {
            "chunks_found": len(retrieval_result.chunks),
            "total_chunks_searched": retrieval_result.total_chunks_searched,
            "avg_relevance_score": sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0,
            "query_type": query_intent["query_type"],
            "retrieval_params": retrieval_params
        }
        
        # Preparar metadatos del LLM
        llm_metadata = {
            "prompt_length": len(enhanced_prompt),
            "response_length": len(answer),
            "model_used": "meta/llama-3.1-8b-instruct"
        }
        
        # Extraer chunks relevantes para mostrar
        relevant_chunks = [chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text 
                          for chunk in retrieval_result.chunks]
        
        return RAGResponse(
            answer=answer,
            relevant_chunks=relevant_chunks,
            confidence_score=confidence_score,
            retrieval_info=retrieval_info,
            llm_metadata=llm_metadata
        )
    
    def generate_response_with_llm_helper(self, query: str, conversation_context: Optional[List[Dict]] = None) -> RAGResponse:
        """
        Genera respuesta usando LLMHelper para mayor robustez y funcionalidades avanzadas.
        
        Args:
            query: Consulta del usuario
            conversation_context: Contexto de conversación
            
        Returns:
            Respuesta generada con LLMHelper
        """
        if not self.use_llm_helper or not self.llm_helper:
            logger.warning("LLMHelper not available, falling back to standard method")
            return self.generate_response(query, conversation_context)
        
        logger.info(f"Generando respuesta con LLMHelper para: '{query}'")
        
        # Analizar intención de la consulta
        query_intent = self.analyze_query_intent(query)
        logger.debug(f"Intención detectada: {query_intent}")
        
        # Optimizar parámetros de recuperación
        retrieval_params = self.optimize_retrieval_params(query_intent)
        logger.debug(f"Parámetros de recuperación: {retrieval_params}")
        
        # Recuperar información relevante
        retrieval_result = self.rag_system.retrieve_relevant_chunks(
            query, 
            top_k=retrieval_params["top_k"],
            min_score=retrieval_params["min_score"]
        )
        
        # Extraer chunks como texto para LLMHelper
        chunk_texts = [chunk.text for chunk in retrieval_result.chunks]
        
        # Usar LLMHelper para generar respuesta RAG
        try:
            answer = self.llm_helper.generate_rag_response(
                query=query,
                retrieved_chunks=chunk_texts,
                conversation_context=conversation_context
            )
        except Exception as e:
            logger.error(f"Error en LLMHelper: {e}")
            answer = f"Error al generar respuesta: {str(e)}"
        
        # Calcular score de confianza
        confidence_score = self._calculate_confidence_score(retrieval_result, query_intent)
        
        # Preparar información de recuperación
        retrieval_info = {
            "chunks_found": len(retrieval_result.chunks),
            "total_chunks_searched": retrieval_result.total_chunks_searched,
            "avg_relevance_score": sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0,
            "query_type": query_intent["query_type"],
            "retrieval_params": retrieval_params,
            "llm_helper_used": True
        }
        
        # Preparar metadatos del LLM
        llm_metadata = {
            "provider": self.llm_helper.get_provider_info()["provider"] if self.llm_helper else "legacy",
            "model": self.llm_helper.get_provider_info()["model"] if self.llm_helper else "unknown",
            "response_length": len(answer)
        }
        
        # Extraer chunks relevantes para mostrar
        relevant_chunks = [chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text 
                          for chunk in retrieval_result.chunks]
        
        return RAGResponse(
            answer=answer,
            relevant_chunks=relevant_chunks,
            confidence_score=confidence_score,
            retrieval_info=retrieval_info,
            llm_metadata=llm_metadata
        )
    
    def _calculate_confidence_score(self, retrieval_result: RetrievalResult, query_intent: Dict) -> float:
        """
        Calcula un score de confianza basado en la calidad de la recuperación.
        
        Args:
            retrieval_result: Resultado de la recuperación
            query_intent: Análisis de la intención
            
        Returns:
            Score de confianza entre 0 y 1
        """
        if not retrieval_result.scores:
            return 0.0
        
        # Score base basado en la relevancia promedio
        avg_relevance = sum(retrieval_result.scores) / len(retrieval_result.scores)
        
        # Bonus por cantidad de chunks relevantes encontrados
        coverage_bonus = min(len(retrieval_result.chunks) / 5, 0.2)  # Máximo 20% bonus
        
        # Penalty si no se encontraron chunks muy relevantes
        max_relevance = max(retrieval_result.scores)
        relevance_penalty = 0.1 if max_relevance < 0.5 else 0
        
        # Bonus por consultas específicas
        specificity_bonus = 0.1 if query_intent["has_specific_terms"] else 0
        
        confidence = avg_relevance + coverage_bonus - relevance_penalty + specificity_bonus
        
        return min(max(confidence, 0.0), 1.0)
    
    def update_conversation_history(self, user_query: str, response: RAGResponse):
        """
        Actualiza el historial de conversación.
        
        Args:
            user_query: Consulta del usuario
            response: Respuesta generada
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": None  # Se puede agregar timestamp si es necesario
        })
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": response.answer,
            "confidence": response.confidence_score,
            "retrieval_info": response.retrieval_info
        })
        
        # Mantener solo las últimas 10 interacciones
        if len(self.conversation_history) > 20:  # 10 pares de user/assistant
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_summary(self) -> Dict:
        """Obtiene un resumen del historial de conversación."""
        if not self.conversation_history:
            return {"status": "No conversation history"}
        
        user_queries = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        avg_confidence = sum(msg.get("confidence", 0) for msg in self.conversation_history if msg["role"] == "assistant") / len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        
        summary = {
            "total_interactions": len(self.conversation_history) // 2,
            "user_queries": user_queries,
            "average_confidence": avg_confidence,
            "last_query": user_queries[-1] if user_queries else None
        }
        
        # Add LLMHelper summary if available
        if self.use_llm_helper and self.llm_helper:
            summary["llm_helper_summary"] = self.llm_helper.get_conversation_summary()
            summary["llm_provider_info"] = self.llm_helper.get_provider_info()
        
        return summary
    
    def extract_document_entities(self, text: str, entity_types: List[str] = None) -> Dict[str, str]:
        """
        Extrae entidades del documento usando LLMHelper.
        
        Args:
            text: Texto del documento
            entity_types: Tipos de entidades a extraer
            
        Returns:
            Diccionario de entidades extraídas
        """
        if not self.use_llm_helper or not self.llm_helper:
            logger.warning("LLMHelper not available for entity extraction")
            return {"error": "LLMHelper not available"}
        
        entity_types = entity_types or ["person", "organization", "date", "location", "amount", "concept"]
        return self.llm_helper.extract_entities(text, entity_types)
    
    def analyze_document_sentiment(self, text: str) -> str:
        """
        Analiza el sentimiento del documento usando LLMHelper.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Análisis de sentimiento
        """
        if not self.use_llm_helper or not self.llm_helper:
            logger.warning("LLMHelper not available for sentiment analysis")
            return "LLMHelper not available"
        
        return self.llm_helper.analyze_sentiment(text)
    
    def summarize_document_section(self, text: str, max_length: int = 200) -> str:
        """
        Resume una sección del documento usando LLMHelper.
        
        Args:
            text: Texto a resumir
            max_length: Longitud máxima del resumen
            
        Returns:
            Resumen del texto
        """
        if not self.use_llm_helper or not self.llm_helper:
            logger.warning("LLMHelper not available for summarization")
            return "LLMHelper not available"
        
        return self.llm_helper.summarize_text(text, max_length)
    
    def switch_llm_provider(self, provider: str, model: str = None, **kwargs) -> bool:
        """
        Cambia el proveedor de LLM usando LLMHelper.
        
        Args:
            provider: Nuevo proveedor
            model: Modelo a usar
            **kwargs: Configuración adicional
            
        Returns:
            True si fue exitoso, False en caso contrario
        """
        if not self.use_llm_helper or not self.llm_helper:
            logger.warning("LLMHelper not available for provider switching")
            return False
        
        success = self.llm_helper.switch_provider(provider, model, **kwargs)
        if success:
            # Update the legacy LLM reference
            self.llm = self.llm_helper.llm
            logger.info(f"Successfully switched to {provider} provider")
        
        return success
    
    def get_llm_capabilities(self) -> Dict[str, Any]:
        """Obtiene información sobre las capacidades del LLM actual."""
        if self.use_llm_helper and self.llm_helper:
            return {
                "provider_info": self.llm_helper.get_provider_info(),
                "available_providers": self.llm_helper._get_available_providers(),
                "llm_helper_enabled": True
            }
        else:
            return {
                "provider_info": {"provider": "legacy", "model": "unknown"},
                "available_providers": ["legacy"],
                "llm_helper_enabled": False
            }
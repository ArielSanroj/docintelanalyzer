"""
Módulo de integración RAG + LLM para mejorar las respuestas del sistema.
Combina recuperación inteligente de información con generación contextualizada.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from rag_system import RAGSystem, RetrievalResult

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
    """Integración avanzada entre RAG y LLM."""
    
    def __init__(self, llm: ChatNVIDIA, rag_system: RAGSystem):
        """
        Inicializa la integración RAG + LLM.
        
        Args:
            llm: Instancia del LLM configurado
            rag_system: Sistema RAG configurado
        """
        self.llm = llm
        self.rag_system = rag_system
        self.conversation_history: List[Dict] = []
    
    def generate_enhanced_prompt(self, query: str, retrieval_result: RetrievalResult, 
                               conversation_context: Optional[List[Dict]] = None) -> str:
        """
        Genera un prompt mejorado que integra información recuperada por RAG.
        
        Args:
            query: Consulta del usuario
            retrieval_result: Resultado de la recuperación RAG
            conversation_context: Contexto de conversación previa
            
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
        
        # Construir prompt mejorado
        enhanced_prompt = f"""Eres un asistente especializado en análisis de documentos. Tu función es responder preguntas basándote ÚNICAMENTE en la información específica del documento proporcionado.

DOCUMENTO ACTUAL:
Estás analizando un documento específico que el usuario ha cargado. Solo debes responder basándote en la información de este documento.

INFORMACIÓN RECUPERADA DEL DOCUMENTO:
{retrieved_context}

CONSULTA DEL USUARIO: {query}
{conversation_context_str}

REGLAS ESTRICTAS:
1. **SOLO información del documento**: Responde ÚNICAMENTE basándote en la información recuperada del documento actual
2. **NO conocimiento general**: NO uses conocimiento general externo a menos que sea absolutamente necesario para explicar algo del documento
3. **Indica claramente las fuentes**: Menciona específicamente qué información viene del documento
4. **Si no hay información**: Si no encuentras información relevante en el documento, di claramente "No encontré información sobre esto en el documento"
5. **Sé específico**: Proporciona detalles concretos y citas exactas del documento cuando sea posible
6. **Contexto conversacional**: Considera el contexto de la conversación pero siempre prioriza la información del documento

FORMATO DE RESPUESTA:
- Responde de manera clara y estructurada
- Usa viñetas cuando sea apropiado
- Incluye citas específicas del documento entre comillas
- Si no encuentras información específica, indícalo claramente
- Al final, menciona qué información viene del documento vs conocimiento general

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
    
    def generate_response(self, query: str, conversation_context: Optional[List[Dict]] = None) -> RAGResponse:
        """
        Genera una respuesta completa usando RAG + LLM.
        
        Args:
            query: Consulta del usuario
            conversation_context: Contexto de conversación
            
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
        
        # Recuperar información relevante
        retrieval_result = self.rag_system.retrieve_relevant_chunks(
            query, 
            top_k=retrieval_params["top_k"],
            min_score=retrieval_params["min_score"]
        )
        
        # Generar prompt mejorado
        enhanced_prompt = self.generate_enhanced_prompt(query, retrieval_result, conversation_context)
        
        # Generar respuesta con LLM
        try:
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
        
        return {
            "total_interactions": len(self.conversation_history) // 2,
            "user_queries": user_queries,
            "average_confidence": avg_confidence,
            "last_query": user_queries[-1] if user_queries else None
        }
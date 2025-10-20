"""
Integración avanzada RAG + LLM con agente ReAct y reranking.
Basado en las mejores prácticas de NVIDIA Nemotron.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from advanced_rag_system import AdvancedRAGSystem, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class AdvancedRAGResponse:
    """Respuesta generada por el sistema RAG avanzado."""
    answer: str
    relevant_chunks: List[str]
    confidence_score: float
    retrieval_info: Dict
    agent_reasoning: Optional[str] = None
    sources_cited: List[str] = None

class AdvancedRAGLLMIntegration:
    """Integración avanzada entre RAG y LLM con agente ReAct."""
    
    def __init__(self, llm: Any, rag_system: AdvancedRAGSystem):
        """
        Inicializa la integración RAG avanzada.
        
        Args:
            llm: Instancia del LLM configurado
            rag_system: Sistema RAG avanzado configurado
        """
        self.llm = llm
        self.rag_system = rag_system
        self.conversation_history: List[Dict] = []
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Inicializa el agente ReAct."""
        try:
            self.agent = self.rag_system.create_react_agent()
            logger.info("Agente ReAct inicializado")
        except Exception as e:
            logger.error(f"Error inicializando agente ReAct: {e}")
            self.agent = None
    
    def generate_response_with_agent(self, query: str, conversation_context: Optional[List[Dict]] = None) -> AdvancedRAGResponse:
        """
        Genera respuesta usando el agente ReAct.
        
        Args:
            query: Consulta del usuario
            conversation_context: Contexto de conversación previa
            
        Returns:
            AdvancedRAGResponse con respuesta del agente
        """
        if self.agent is None:
            logger.warning("Agente no disponible, usando método tradicional")
            return self.generate_response_traditional(query, conversation_context)
        
        try:
            logger.info(f"Generando respuesta con agente ReAct para: '{query[:50]}...'")
            
            # Preparar contexto de conversación
            if conversation_context:
                # Convertir contexto a formato de mensajes
                messages = []
                for msg in conversation_context[-6:]:  # Últimos 6 mensajes
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(SystemMessage(content=msg["content"]))
                
                # Agregar mensaje actual
                messages.append(HumanMessage(content=query))
            else:
                messages = [HumanMessage(content=query)]
            
            # Ejecutar agente
            result = self.agent.invoke({"messages": messages})
            
            # Extraer respuesta
            answer = "No se pudo generar respuesta"
            if "messages" in result and result["messages"]:
                # Buscar el último mensaje del asistente
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content and not msg.content.startswith("I need to"):
                        answer = msg.content
                        break
                    elif hasattr(msg, 'content') and msg.content:
                        answer = msg.content
                        break
                
                # Si no encontramos contenido útil, usar el último mensaje
                if answer == "No se pudo generar respuesta":
                    last_message = result["messages"][-1]
                    answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                answer = str(result)
            
            # Limpiar respuesta si es muy técnica
            if answer.startswith("I need to") or answer.startswith("I'll search"):
                answer = "El agente está procesando la consulta. Por favor, reformula tu pregunta de manera más específica."
            
            # Extraer información de herramientas usadas
            tool_calls = []
            sources_cited = []
            agent_reasoning = []
            
            if "messages" in result:
                for msg in result["messages"]:
                    # Extraer razonamiento del agente
                    if hasattr(msg, 'content') and msg.content:
                        if msg.content.startswith("I need to") or msg.content.startswith("I'll search") or msg.content.startswith("Let me"):
                            agent_reasoning.append(msg.content)
                        elif getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "document_search":
                            sources_cited.append(msg.content if len(msg.content) < 400 else msg.content[:400] + "...")
                    
                    # Extraer llamadas de herramientas
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                            tool_result = tool_call.get("result") if isinstance(tool_call, dict) else getattr(tool_call, "result", "")

                            tool_calls.append({
                                "tool": tool_name or "unknown",
                                "args": tool_args or {},
                                "result": tool_result or ""
                            })
                            
                            if tool_name == "document_search" and tool_result:
                                sources_cited.append(tool_result[:400] + "..." if len(tool_result) > 400 else tool_result)
            
            # Calcular score de confianza basado en herramientas usadas
            confidence_score = 0.8 if tool_calls else 0.5
            
            return AdvancedRAGResponse(
                answer=answer,
                relevant_chunks=sources_cited,
                confidence_score=confidence_score,
                retrieval_info={
                    "tool_calls": len(tool_calls),
                    "agent_used": True,
                    "search_type": "agent_guided"
                },
                agent_reasoning=f"Agente ejecutó {len(tool_calls)} herramientas",
                sources_cited=sources_cited
            )
            
        except Exception as e:
            logger.error(f"Error en agente ReAct: {e}")
            # Fallback al método tradicional
            return self.generate_response_traditional(query, conversation_context)
    
    def generate_response_traditional(self, query: str, conversation_context: Optional[List[Dict]] = None) -> AdvancedRAGResponse:
        """
        Genera respuesta usando el método tradicional mejorado.
        
        Args:
            query: Consulta del usuario
            conversation_context: Contexto de conversación previa
            
        Returns:
            AdvancedRAGResponse con respuesta tradicional mejorada
        """
        logger.info(f"Generando respuesta tradicional mejorada para: '{query[:50]}...'")
        
        try:
            # Usar búsqueda híbrida
            retrieval_result = self.rag_system.retrieve_documents(
                query, 
                search_type="hybrid", 
                top_k=6
            )
            
            # Rerankear si está disponible
            if self.rag_system.reranker is not None:
                retrieval_result = self.rag_system.rerank_documents(query, retrieval_result)
            
            # Generar prompt mejorado
            enhanced_prompt = self._generate_enhanced_prompt(
                query, retrieval_result, conversation_context
            )
            
            # Generar respuesta
            response = self.llm.invoke(enhanced_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calcular score de confianza
            confidence_score = self._calculate_confidence_score(retrieval_result)
            
            # Preparar chunks relevantes
            relevant_chunks = [chunk.text for chunk in retrieval_result.chunks[:3]]
            
            return AdvancedRAGResponse(
                answer=answer,
                relevant_chunks=relevant_chunks,
                confidence_score=confidence_score,
                retrieval_info={
                    "chunks_found": len(retrieval_result.chunks),
                    "retrieval_type": retrieval_result.retrieval_type,
                    "total_searched": retrieval_result.total_chunks_searched,
                    "avg_relevance_score": np.mean(retrieval_result.scores) if retrieval_result.scores else 0.0,
                    "agent_used": False
                },
                sources_cited=relevant_chunks
            )
            
        except Exception as e:
            logger.error(f"Error en respuesta tradicional: {e}")
            return AdvancedRAGResponse(
                answer=f"Error al procesar la consulta: {str(e)}",
                relevant_chunks=[],
                confidence_score=0.0,
                retrieval_info={"error": str(e)},
                sources_cited=[]
            )
    
    def _generate_enhanced_prompt(self, query: str, retrieval_result: RetrievalResult, 
                                conversation_context: Optional[List[Dict]] = None) -> str:
        """
        Genera un prompt mejorado basado en las mejores prácticas de NVIDIA.
        
        Args:
            query: Consulta del usuario
            retrieval_result: Resultado de la recuperación
            conversation_context: Contexto de conversación
            
        Returns:
            Prompt mejorado
        """
        # Construir contexto de documentos
        document_context = ""
        for i, chunk in enumerate(retrieval_result.chunks[:3]):  # Top 3 chunks
            document_context += f"\n[Doc {i+1}] {chunk.text}\n"
        
        # Construir contexto de conversación
        conversation_context_str = ""
        if conversation_context:
            conversation_context_str = "\nContexto de conversación previa:\n"
            for msg in conversation_context[-3:]:  # Últimos 3 mensajes
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                conversation_context_str += f"{role}: {msg['content']}\n"
        
        # Prompt optimizado basado en NVIDIA
        enhanced_prompt = f"""Eres un asistente experto en análisis de documentos. Tu función es proporcionar respuestas precisas basadas únicamente en la información de los documentos proporcionados.

INFORMACIÓN DE DOCUMENTOS:
{document_context}

CONSULTA DEL USUARIO: {query}
{conversation_context_str}

INSTRUCCIONES:
1. Responde ÚNICAMENTE basándote en la información de los documentos proporcionados
2. Si encuentras información relevante, responde con esa información específica
3. Si NO encuentras información relevante, di claramente "No encontré información sobre esto en los documentos"
4. Cita las fuentes usando [Doc 1], [Doc 2], etc.
5. Mantén las respuestas concisas y precisas
6. Responde en español a menos que se especifique lo contrario
7. Si la información es incompleta, explica qué información falta

RESPUESTA:"""

        return enhanced_prompt
    
    def _calculate_confidence_score(self, retrieval_result: RetrievalResult) -> float:
        """
        Calcula el score de confianza basado en la calidad de la recuperación.
        
        Args:
            retrieval_result: Resultado de la recuperación
            
        Returns:
            Score de confianza entre 0 y 1
        """
        if not retrieval_result.scores:
            return 0.0
        
        # Factor 1: Score promedio de relevancia
        avg_score = np.mean(retrieval_result.scores)
        
        # Factor 2: Número de documentos encontrados
        num_docs = len(retrieval_result.chunks)
        doc_factor = min(num_docs / 3.0, 1.0)  # Normalizar a máximo 1.0
        
        # Factor 3: Tipo de búsqueda (híbrida es mejor)
        search_factor = 1.0 if "hybrid" in retrieval_result.retrieval_type else 0.8
        
        # Combinar factores
        confidence = (avg_score * 0.5 + doc_factor * 0.3 + search_factor * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    def update_conversation_history(self, query: str, response: AdvancedRAGResponse):
        """
        Actualiza el historial de conversación.
        
        Args:
            query: Consulta del usuario
            response: Respuesta generada
        """
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": self._get_timestamp()
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response.answer,
            "confidence_score": response.confidence_score,
            "retrieval_info": response.retrieval_info,
            "timestamp": self._get_timestamp()
        })
        
        # Mantener solo los últimos 20 mensajes
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_summary(self) -> Dict:
        """Obtiene un resumen de la conversación."""
        if not self.conversation_history:
            return {"total_messages": 0, "avg_confidence": 0.0}
        
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        avg_confidence = 0.0
        if assistant_messages:
            confidences = [msg.get("confidence_score", 0.0) for msg in assistant_messages]
            avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_confidence": avg_confidence,
            "agent_available": self.agent is not None
        }
    
    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual."""
        from datetime import datetime
        return datetime.now().isoformat()
"""
Integración VLM-Enhanced RAG basada en RAG-Anything.
Implementa capacidades de análisis multimodal con modelos de visión-lenguaje.
"""

import logging
import base64
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io

from multimodal_rag_system import MultimodalRAGSystem, MultimodalRetrievalResult, MultimodalChunk
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

@dataclass
class VLMEnhancedResponse:
    """Respuesta mejorada con capacidades VLM."""
    answer: str
    relevant_chunks: List[MultimodalChunk]
    confidence_score: float
    visual_analysis: Optional[Dict] = None
    cross_modal_insights: List[Dict] = None
    retrieval_info: Dict = None

class VLMEnhancedRAG:
    """Sistema RAG mejorado con capacidades VLM."""
    
    def __init__(self, 
                 multimodal_rag: MultimodalRAGSystem,
                 vlm_model: Any = None,
                 enable_visual_analysis: bool = True):
        """
        Inicializa el sistema VLM-Enhanced RAG.
        
        Args:
            multimodal_rag: Sistema RAG multimodal
            vlm_model: Modelo de visión-lenguaje
            enable_visual_analysis: Habilitar análisis visual
        """
        self.multimodal_rag = multimodal_rag
        self.vlm_model = vlm_model
        self.enable_visual_analysis = enable_visual_analysis
        
        if not self.vlm_model and self.enable_visual_analysis:
            self._initialize_vlm()
    
    def _initialize_vlm(self):
        """Inicializa el modelo VLM."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self.vlm_model = ChatNVIDIA(
                model="nvidia/llama-3.2-vision-instruct",
                temperature=0.3,
                max_tokens=4096
            )
            logger.info("Modelo VLM inicializado para análisis multimodal")
        except Exception as e:
            logger.warning(f"No se pudo inicializar VLM: {e}")
            self.enable_visual_analysis = False
    
    def process_query_with_vision(self, query: str, top_k: int = 5) -> VLMEnhancedResponse:
        """
        Procesa una consulta con capacidades de visión mejoradas.
        
        Args:
            query: Consulta del usuario
            top_k: Número de chunks a recuperar
            
        Returns:
            Respuesta mejorada con análisis visual
        """
        logger.info(f"Procesando consulta VLM: '{query}'")
        
        # Recuperación multimodal estándar
        retrieval_result = self.multimodal_rag.retrieve_multimodal(query, top_k)
        
        # Análisis visual mejorado
        visual_analysis = None
        cross_modal_insights = []
        
        if self.enable_visual_analysis and retrieval_result.cross_modal_matches:
            visual_analysis = self._analyze_visual_content(retrieval_result.chunks, query)
            cross_modal_insights = self._extract_cross_modal_insights(retrieval_result)
        
        # Generar respuesta contextualizada
        answer = self._generate_contextualized_response(query, retrieval_result, visual_analysis)
        
        # Calcular score de confianza
        confidence_score = self._calculate_enhanced_confidence(retrieval_result, visual_analysis)
        
        return VLMEnhancedResponse(
            answer=answer,
            relevant_chunks=retrieval_result.chunks,
            confidence_score=confidence_score,
            visual_analysis=visual_analysis,
            cross_modal_insights=cross_modal_insights,
            retrieval_info={
                "total_chunks": len(retrieval_result.chunks),
                "cross_modal_matches": len(retrieval_result.cross_modal_matches),
                "visual_analysis_available": visual_analysis is not None
            }
        )
    
    def _analyze_visual_content(self, chunks: List[MultimodalChunk], query: str) -> Dict:
        """Analiza contenido visual usando VLM."""
        if not self.vlm_model:
            return None
        
        visual_chunks = [chunk for chunk in chunks if chunk.content_type in ["image", "chart", "diagram"]]
        
        if not visual_chunks:
            return None
        
        try:
            # Preparar contenido visual para análisis
            visual_analysis = {
                "visual_elements": [],
                "query_relevance": [],
                "cross_modal_connections": []
            }
            
            for chunk in visual_chunks:
                if chunk.metadata.get("image_data"):
                    # Analizar imagen con VLM
                    img_analysis = self._analyze_image_with_vlm(chunk, query)
                    visual_analysis["visual_elements"].append(img_analysis)
                    
                    # Evaluar relevancia para la consulta
                    relevance = self._assess_visual_relevance(img_analysis, query)
                    visual_analysis["query_relevance"].append({
                        "chunk_id": chunk.chunk_id,
                        "relevance_score": relevance,
                        "content_type": chunk.content_type
                    })
            
            return visual_analysis
            
        except Exception as e:
            logger.error(f"Error en análisis visual: {e}")
            return None
    
    def _analyze_image_with_vlm(self, chunk: MultimodalChunk, query: str) -> Dict:
        """Analiza una imagen específica con VLM."""
        try:
            if not chunk.metadata.get("image_data"):
                return {"error": "No image data available"}
            
            # Preparar prompt contextualizado
            prompt = f"""
            Analiza esta imagen en el contexto de la consulta: "{query}"
            
            Proporciona:
            1. Descripción detallada del contenido visual
            2. Elementos relevantes para la consulta
            3. Texto visible en la imagen
            4. Objetos, diagramas o gráficos identificados
            5. Relación con la consulta del usuario
            
            Responde en español de manera clara y estructurada.
            """
            
            # Usar VLM para análisis
            response = self.vlm_model.invoke([
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chunk.metadata['image_data']}"}}
            ])
            
            return {
                "chunk_id": chunk.chunk_id,
                "analysis": response.content,
                "content_type": chunk.content_type,
                "query_context": query
            }
            
        except Exception as e:
            logger.error(f"Error analizando imagen con VLM: {e}")
            return {"error": str(e), "chunk_id": chunk.chunk_id}
    
    def _assess_visual_relevance(self, img_analysis: Dict, query: str) -> float:
        """Evalúa la relevancia visual para la consulta."""
        if "error" in img_analysis:
            return 0.0
        
        analysis_text = img_analysis.get("analysis", "").lower()
        query_terms = query.lower().split()
        
        # Contar coincidencias de términos
        matches = sum(1 for term in query_terms if term in analysis_text)
        relevance = min(matches / len(query_terms), 1.0) if query_terms else 0.0
        
        return relevance
    
    def _extract_cross_modal_insights(self, retrieval_result: MultimodalRetrievalResult) -> List[Dict]:
        """Extrae insights cross-modal de los resultados."""
        insights = []
        
        # Agrupar chunks por tipo de contenido
        content_groups = {}
        for chunk in retrieval_result.chunks:
            content_type = chunk.content_type
            if content_type not in content_groups:
                content_groups[content_type] = []
            content_groups[content_type].append(chunk)
        
        # Generar insights cross-modal
        for content_type, chunks in content_groups.items():
            if len(chunks) > 1:
                insights.append({
                    "type": "content_coherence",
                    "content_type": content_type,
                    "chunk_count": len(chunks),
                    "description": f"Múltiples elementos de {content_type} encontrados"
                })
        
        # Detectar relaciones entre diferentes tipos de contenido
        if "text" in content_groups and "image" in content_groups:
            insights.append({
                "type": "text_image_connection",
                "description": "Conexión entre texto e imágenes detectada",
                "text_chunks": len(content_groups["text"]),
                "image_chunks": len(content_groups["image"])
            })
        
        if "table" in content_groups and "text" in content_groups:
            insights.append({
                "type": "table_text_connection",
                "description": "Conexión entre tablas y texto detectada",
                "table_chunks": len(content_groups["table"]),
                "text_chunks": len(content_groups["text"])
            })
        
        return insights
    
    def _generate_contextualized_response(self, query: str, retrieval_result: MultimodalRetrievalResult, visual_analysis: Optional[Dict]) -> str:
        """Genera una respuesta contextualizada integrando información multimodal."""
        
        # Preparar contexto textual
        text_context = []
        for chunk in retrieval_result.chunks:
            if chunk.content_type == "text":
                text_context.append(f"[Texto] {chunk.content}")
            elif chunk.content_type == "table":
                text_context.append(f"[Tabla] {chunk.content}")
            elif chunk.content_type == "image":
                text_context.append(f"[Imagen] {chunk.content}")
        
        context = "\n\n".join(text_context)
        
        # Preparar análisis visual si está disponible
        visual_context = ""
        if visual_analysis and visual_analysis.get("visual_elements"):
            visual_context = "\n\nANÁLISIS VISUAL:\n"
            for element in visual_analysis["visual_elements"]:
                if "analysis" in element:
                    visual_context += f"- {element['analysis']}\n"
        
        # Generar respuesta
        response_prompt = f"""
        Eres un asistente experto en análisis de documentos multimodales. Responde la consulta del usuario basándote ÚNICAMENTE en la información proporcionada.

        CONSULTA: {query}

        INFORMACIÓN DEL DOCUMENTO:
        {context}
        {visual_context}

        INSTRUCCIONES:
        1. Responde basándote únicamente en la información proporcionada
        2. Si hay contenido visual relevante, menciona específicamente qué información visual es importante
        3. Si hay tablas, explica los datos más relevantes
        4. Integra información de diferentes tipos de contenido cuando sea relevante
        5. Si no encuentras información suficiente, indícalo claramente
        6. Responde en español de manera clara y estructurada

        RESPUESTA:
        """
        
        try:
            # Usar el LLM del sistema multimodal para generar respuesta
            if hasattr(self.multimodal_rag, 'vlm_model') and self.multimodal_rag.vlm_model:
                response = self.multimodal_rag.vlm_model.invoke([HumanMessage(content=response_prompt)])
                return response.content
            else:
                # Fallback a respuesta simple
                return self._generate_simple_response(query, context, visual_context)
        except Exception as e:
            logger.error(f"Error generando respuesta contextualizada: {e}")
            return self._generate_simple_response(query, context, visual_context)
    
    def _generate_simple_response(self, query: str, context: str, visual_context: str) -> str:
        """Genera una respuesta simple como fallback."""
        response = f"Basándome en el documento analizado:\n\n"
        
        if context:
            response += f"Información textual encontrada:\n{context[:1000]}...\n\n"
        
        if visual_context:
            response += f"Análisis visual:\n{visual_context}\n\n"
        
        response += "Nota: Para un análisis más detallado, se requiere un modelo de lenguaje avanzado."
        
        return response
    
    def _calculate_enhanced_confidence(self, retrieval_result: MultimodalRetrievalResult, visual_analysis: Optional[Dict]) -> float:
        """Calcula un score de confianza mejorado considerando análisis visual."""
        base_confidence = np.mean(retrieval_result.scores) if retrieval_result.scores else 0.0
        
        # Bonus por análisis visual exitoso
        visual_bonus = 0.0
        if visual_analysis and visual_analysis.get("visual_elements"):
            visual_bonus = 0.1  # 10% bonus por análisis visual
        
        # Bonus por diversidad de tipos de contenido
        content_types = set(chunk.content_type for chunk in retrieval_result.chunks)
        diversity_bonus = min(len(content_types) * 0.05, 0.15)  # Máximo 15% bonus
        
        # Bonus por coincidencias cross-modal
        cross_modal_bonus = 0.0
        if retrieval_result.cross_modal_matches:
            cross_modal_bonus = min(len(retrieval_result.cross_modal_matches) * 0.05, 0.1)  # Máximo 10% bonus
        
        enhanced_confidence = base_confidence + visual_bonus + diversity_bonus + cross_modal_bonus
        
        return min(max(enhanced_confidence, 0.0), 1.0)
    
    def get_visual_summary(self) -> Dict:
        """Obtiene un resumen de las capacidades visuales del sistema."""
        return {
            "vlm_enabled": self.enable_visual_analysis,
            "vlm_model_available": self.vlm_model is not None,
            "multimodal_chunks": len(self.multimodal_rag.multimodal_chunks),
            "visual_chunks": len([c for c in self.multimodal_rag.multimodal_chunks if c.content_type in ["image", "chart", "diagram"]]),
            "table_chunks": len([c for c in self.multimodal_rag.multimodal_chunks if c.content_type == "table"]),
            "text_chunks": len([c for c in self.multimodal_rag.multimodal_chunks if c.content_type == "text"])
        }
"""
Módulo de procesamiento consciente del contexto basado en RAG-Anything.
Implementa integración inteligente de información contextual relevante.
"""

import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class ContextualInfo:
    """Información contextual extraída."""
    context_type: str  # "temporal", "spatial", "causal", "hierarchical", "semantic"
    content: str
    confidence: float
    source_chunk_id: int
    metadata: Dict = None

@dataclass
class ContextualAnalysis:
    """Análisis contextual completo."""
    temporal_context: List[ContextualInfo]
    spatial_context: List[ContextualInfo]
    causal_context: List[ContextualInfo]
    hierarchical_context: List[ContextualInfo]
    semantic_context: List[ContextualInfo]
    context_connections: List[Dict]

class ContextAwareProcessor:
    """Procesador consciente del contexto para RAG."""
    
    def __init__(self):
        """Inicializa el procesador contextual."""
        self.temporal_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Fechas
            r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
            r'\b(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b',
            r'\b(?:antes|después|durante|mientras|cuando|entonces|ahora|antes de|después de)\b'
        ]
        
        self.spatial_patterns = [
            r'\b(?:en|sobre|bajo|dentro de|fuera de|cerca de|lejos de|al lado de|frente a|detrás de)\b',
            r'\b(?:norte|sur|este|oeste|arriba|abajo|izquierda|derecha)\b',
            r'\b(?:país|ciudad|región|zona|área|territorio|lugar|ubicación)\b'
        ]
        
        self.causal_patterns = [
            r'\b(?:porque|ya que|debido a|a causa de|gracias a|por|por lo tanto|por consiguiente|así que|entonces)\b',
            r'\b(?:causa|efecto|resultado|consecuencia|motivo|razón|origen)\b',
            r'\b(?:si|cuando|mientras|aunque|a pesar de|sin embargo|no obstante)\b'
        ]
        
        self.hierarchical_patterns = [
            r'\b(?:capítulo|sección|artículo|párrafo|apartado|inciso|literal)\b',
            r'\b(?:primero|segundo|tercero|último|siguiente|anterior)\b',
            r'\b(?:nivel|nivel 1|nivel 2|nivel 3|principal|secundario|terciario)\b'
        ]
        
        self.semantic_patterns = [
            r'\b(?:definición|concepto|término|significado|explicación)\b',
            r'\b(?:ejemplo|caso|instancia|muestra|ilustración)\b',
            r'\b(?:característica|propiedad|atributo|aspecto|elemento)\b'
        ]
    
    def analyze_context(self, chunks: List[Any], query: str) -> ContextualAnalysis:
        """
        Analiza el contexto de los chunks para extraer información relevante.
        
        Args:
            chunks: Lista de chunks multimodales
            query: Consulta del usuario
            
        Returns:
            Análisis contextual completo
        """
        logger.info("Analizando contexto de los chunks...")
        
        # Extraer diferentes tipos de contexto
        temporal_context = self._extract_temporal_context(chunks, query)
        spatial_context = self._extract_spatial_context(chunks, query)
        causal_context = self._extract_causal_context(chunks, query)
        hierarchical_context = self._extract_hierarchical_context(chunks, query)
        semantic_context = self._extract_semantic_context(chunks, query)
        
        # Identificar conexiones contextuales
        context_connections = self._identify_context_connections(
            temporal_context, spatial_context, causal_context, 
            hierarchical_context, semantic_context
        )
        
        return ContextualAnalysis(
            temporal_context=temporal_context,
            spatial_context=spatial_context,
            causal_context=causal_context,
            hierarchical_context=hierarchical_context,
            semantic_context=semantic_context,
            context_connections=context_connections
        )
    
    def _extract_temporal_context(self, chunks: List[Any], query: str) -> List[ContextualInfo]:
        """Extrae información temporal del contexto."""
        temporal_info = []
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            for pattern in self.temporal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_pattern_confidence(match, pattern, content)
                    temporal_info.append(ContextualInfo(
                        context_type="temporal",
                        content=match,
                        confidence=confidence,
                        source_chunk_id=getattr(chunk, 'chunk_id', 0),
                        metadata={"pattern": pattern, "full_content": content[:200]}
                    ))
        
        return temporal_info
    
    def _extract_spatial_context(self, chunks: List[Any], query: str) -> List[ContextualInfo]:
        """Extrae información espacial del contexto."""
        spatial_info = []
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            for pattern in self.spatial_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_pattern_confidence(match, pattern, content)
                    spatial_info.append(ContextualInfo(
                        context_type="spatial",
                        content=match,
                        confidence=confidence,
                        source_chunk_id=getattr(chunk, 'chunk_id', 0),
                        metadata={"pattern": pattern, "full_content": content[:200]}
                    ))
        
        return spatial_info
    
    def _extract_causal_context(self, chunks: List[Any], query: str) -> List[ContextualInfo]:
        """Extrae información causal del contexto."""
        causal_info = []
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            for pattern in self.causal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_pattern_confidence(match, pattern, content)
                    causal_info.append(ContextualInfo(
                        context_type="causal",
                        content=match,
                        confidence=confidence,
                        source_chunk_id=getattr(chunk, 'chunk_id', 0),
                        metadata={"pattern": pattern, "full_content": content[:200]}
                    ))
        
        return causal_info
    
    def _extract_hierarchical_context(self, chunks: List[Any], query: str) -> List[ContextualInfo]:
        """Extrae información jerárquica del contexto."""
        hierarchical_info = []
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            for pattern in self.hierarchical_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_pattern_confidence(match, pattern, content)
                    hierarchical_info.append(ContextualInfo(
                        context_type="hierarchical",
                        content=match,
                        confidence=confidence,
                        source_chunk_id=getattr(chunk, 'chunk_id', 0),
                        metadata={"pattern": pattern, "full_content": content[:200]}
                    ))
        
        return hierarchical_info
    
    def _extract_semantic_context(self, chunks: List[Any], query: str) -> List[ContextualInfo]:
        """Extrae información semántica del contexto."""
        semantic_info = []
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            for pattern in self.semantic_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_pattern_confidence(match, pattern, content)
                    semantic_info.append(ContextualInfo(
                        context_type="semantic",
                        content=match,
                        confidence=confidence,
                        source_chunk_id=getattr(chunk, 'chunk_id', 0),
                        metadata={"pattern": pattern, "full_content": content[:200]}
                    ))
        
        return semantic_info
    
    def _calculate_pattern_confidence(self, match: str, pattern: str, content: str) -> float:
        """Calcula la confianza de un patrón encontrado."""
        # Confianza base basada en la longitud del match
        base_confidence = min(len(match) / 20, 1.0)
        
        # Bonus por frecuencia en el contenido
        frequency = content.lower().count(match.lower())
        frequency_bonus = min(frequency * 0.1, 0.3)
        
        # Bonus por posición en el contenido (más cerca del inicio = más importante)
        position = content.lower().find(match.lower())
        position_bonus = max(0, (len(content) - position) / len(content)) * 0.2
        
        return min(base_confidence + frequency_bonus + position_bonus, 1.0)
    
    def _identify_context_connections(self, *context_lists) -> List[Dict]:
        """Identifica conexiones entre diferentes tipos de contexto."""
        connections = []
        
        # Agrupar por chunk_id
        chunk_contexts = defaultdict(list)
        for context_list in context_lists:
            for context_info in context_list:
                chunk_contexts[context_info.source_chunk_id].append(context_info)
        
        # Identificar chunks con múltiples tipos de contexto
        for chunk_id, contexts in chunk_contexts.items():
            if len(contexts) > 1:
                context_types = [c.context_type for c in contexts]
                connections.append({
                    "chunk_id": chunk_id,
                    "context_types": context_types,
                    "connection_strength": len(set(context_types)) / len(context_types),
                    "contexts": contexts
                })
        
        # Identificar patrones de co-ocurrencia
        co_occurrence_patterns = self._find_co_occurrence_patterns(chunk_contexts)
        connections.extend(co_occurrence_patterns)
        
        return connections
    
    def _find_co_occurrence_patterns(self, chunk_contexts: Dict) -> List[Dict]:
        """Encuentra patrones de co-ocurrencia en el contexto."""
        patterns = []
        
        # Contar combinaciones de tipos de contexto
        type_combinations = Counter()
        for contexts in chunk_contexts.values():
            if len(contexts) > 1:
                types = tuple(sorted([c.context_type for c in contexts]))
                type_combinations[types] += 1
        
        # Identificar patrones frecuentes
        for combination, count in type_combinations.items():
            if count > 1:  # Patrón que aparece en múltiples chunks
                patterns.append({
                    "type": "co_occurrence_pattern",
                    "context_types": list(combination),
                    "frequency": count,
                    "description": f"Patrón frecuente: {', '.join(combination)}"
                })
        
        return patterns
    
    def enhance_query_with_context(self, query: str, contextual_analysis: ContextualAnalysis) -> str:
        """
        Mejora la consulta con información contextual relevante.
        
        Args:
            query: Consulta original
            contextual_analysis: Análisis contextual
            
        Returns:
            Consulta mejorada con contexto
        """
        enhanced_query = query
        
        # Agregar contexto temporal si es relevante
        if contextual_analysis.temporal_context:
            temporal_terms = [ctx.content for ctx in contextual_analysis.temporal_context[:3]]
            enhanced_query += f" (contexto temporal: {', '.join(temporal_terms)})"
        
        # Agregar contexto espacial si es relevante
        if contextual_analysis.spatial_context:
            spatial_terms = [ctx.content for ctx in contextual_analysis.spatial_context[:3]]
            enhanced_query += f" (contexto espacial: {', '.join(spatial_terms)})"
        
        # Agregar contexto causal si es relevante
        if contextual_analysis.causal_context:
            causal_terms = [ctx.content for ctx in contextual_analysis.causal_context[:3]]
            enhanced_query += f" (contexto causal: {', '.join(causal_terms)})"
        
        return enhanced_query
    
    def generate_contextual_summary(self, contextual_analysis: ContextualAnalysis) -> str:
        """Genera un resumen contextual de la información extraída."""
        summary_parts = []
        
        if contextual_analysis.temporal_context:
            temporal_count = len(contextual_analysis.temporal_context)
            summary_parts.append(f"Información temporal: {temporal_count} elementos encontrados")
        
        if contextual_analysis.spatial_context:
            spatial_count = len(contextual_analysis.spatial_context)
            summary_parts.append(f"Información espacial: {spatial_count} elementos encontrados")
        
        if contextual_analysis.causal_context:
            causal_count = len(contextual_analysis.causal_context)
            summary_parts.append(f"Información causal: {causal_count} elementos encontrados")
        
        if contextual_analysis.hierarchical_context:
            hierarchical_count = len(contextual_analysis.hierarchical_context)
            summary_parts.append(f"Información jerárquica: {hierarchical_count} elementos encontrados")
        
        if contextual_analysis.semantic_context:
            semantic_count = len(contextual_analysis.semantic_context)
            summary_parts.append(f"Información semántica: {semantic_count} elementos encontrados")
        
        if contextual_analysis.context_connections:
            connection_count = len(contextual_analysis.context_connections)
            summary_parts.append(f"Conexiones contextuales: {connection_count} identificadas")
        
        return " | ".join(summary_parts) if summary_parts else "No se encontró información contextual relevante"
    
    def get_context_statistics(self, contextual_analysis: ContextualAnalysis) -> Dict:
        """Obtiene estadísticas del análisis contextual."""
        return {
            "total_context_elements": sum([
                len(contextual_analysis.temporal_context),
                len(contextual_analysis.spatial_context),
                len(contextual_analysis.causal_context),
                len(contextual_analysis.hierarchical_context),
                len(contextual_analysis.semantic_context)
            ]),
            "temporal_elements": len(contextual_analysis.temporal_context),
            "spatial_elements": len(contextual_analysis.spatial_context),
            "causal_elements": len(contextual_analysis.causal_context),
            "hierarchical_elements": len(contextual_analysis.hierarchical_context),
            "semantic_elements": len(contextual_analysis.semantic_context),
            "context_connections": len(contextual_analysis.context_connections),
            "avg_confidence": np.mean([
                ctx.confidence for ctx_list in [
                    contextual_analysis.temporal_context,
                    contextual_analysis.spatial_context,
                    contextual_analysis.causal_context,
                    contextual_analysis.hierarchical_context,
                    contextual_analysis.semantic_context
                ] for ctx in ctx_list
            ]) if any([
                contextual_analysis.temporal_context,
                contextual_analysis.spatial_context,
                contextual_analysis.causal_context,
                contextual_analysis.hierarchical_context,
                contextual_analysis.semantic_context
            ]) else 0.0
        }
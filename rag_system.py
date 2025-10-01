"""
Sistema RAG (Retrieval-Augmented Generation) para mejorar la integración
entre recuperación de información y generación de respuestas con LLM.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Representa un fragmento del documento con metadatos."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    metadata: Dict = None

@dataclass
class RetrievalResult:
    """Resultado de la recuperación de información."""
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    total_chunks_searched: int

class RAGSystem:
    """Sistema RAG para recuperación inteligente de información."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el sistema RAG.
        
        Args:
            model_name: Nombre del modelo de embeddings a usar
        """
        self.model_name = model_name
        self.embedding_model = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_embeddings: np.ndarray = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Carga el modelo de embeddings."""
        try:
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("Modelo de embeddings cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        Divide el documento en chunks inteligentes.
        
        Args:
            text: Texto del documento
            chunk_size: Tamaño máximo de cada chunk
            overlap: Solapamiento entre chunks
            
        Returns:
            Lista de DocumentChunk
        """
        logger.info(f"Dividiendo documento en chunks (tamaño: {chunk_size}, solapamiento: {overlap})")
        
        # Limpiar texto
        text = self._clean_text(text)
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Intentar cortar en límites de oración
            if end < len(text):
                # Buscar el último punto, exclamación o interrogación
                last_sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if last_sentence_end > start + chunk_size // 2:
                    end = last_sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = DocumentChunk(
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    chunk_id=chunk_id,
                    metadata={
                        'length': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'has_numbers': bool(re.search(r'\d', chunk_text)),
                        'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk_text))
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Mover el inicio con solapamiento
            start = max(start + chunk_size - overlap, end)
        
        logger.info(f"Documento dividido en {len(chunks)} chunks")
        self.document_chunks = chunks
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Limpia el texto para mejor procesamiento."""
        # Remover caracteres de control
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalizar espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        
        # Remover líneas muy cortas (probablemente ruido)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return ' '.join(cleaned_lines)
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Genera embeddings para todos los chunks del documento.
        
        Returns:
            Array de embeddings
        """
        if not self.document_chunks:
            raise ValueError("No hay chunks de documento disponibles. Ejecute chunk_document() primero.")
        
        logger.info(f"Generando embeddings para {len(self.document_chunks)} chunks")
        
        chunk_texts = [chunk.text for chunk in self.document_chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        self.chunk_embeddings = embeddings
        logger.info(f"Embeddings generados: {embeddings.shape}")
        
        return embeddings
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, min_score: float = 0.3) -> RetrievalResult:
        """
        Recupera los chunks más relevantes para una consulta.
        
        Args:
            query: Consulta del usuario
            top_k: Número máximo de chunks a recuperar
            min_score: Puntuación mínima de similitud
            
        Returns:
            RetrievalResult con chunks relevantes
        """
        if self.chunk_embeddings is None:
            raise ValueError("No hay embeddings disponibles. Ejecute generate_embeddings() primero.")
        
        logger.info(f"Recuperando chunks relevantes para: '{query}'")
        
        # Generar embedding para la consulta
        query_embedding = self.embedding_model.encode([query])
        
        # Calcular similitudes
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Obtener índices ordenados por similitud
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filtrar por puntuación mínima y tomar top_k
        relevant_chunks = []
        relevant_scores = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_score and len(relevant_chunks) < top_k:
                relevant_chunks.append(self.document_chunks[idx])
                relevant_scores.append(float(similarities[idx]))
        
        logger.info(f"Recuperados {len(relevant_chunks)} chunks relevantes (de {len(self.document_chunks)} total)")
        
        return RetrievalResult(
            chunks=relevant_chunks,
            scores=relevant_scores,
            query=query,
            total_chunks_searched=len(self.document_chunks)
        )
    
    def get_context_for_llm(self, retrieval_result: RetrievalResult, max_context_length: int = 3000) -> str:
        """
        Prepara el contexto para el LLM basado en los chunks recuperados.
        
        Args:
            retrieval_result: Resultado de la recuperación
            max_context_length: Longitud máxima del contexto
            
        Returns:
            Contexto formateado para el LLM
        """
        context_parts = []
        current_length = 0
        
        for i, (chunk, score) in enumerate(zip(retrieval_result.chunks, retrieval_result.scores)):
            # Formatear chunk con metadatos
            chunk_text = f"[Chunk {i+1}, Relevancia: {score:.3f}]\n{chunk.text}\n"
            
            if current_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                current_length += len(chunk_text)
            else:
                break
        
        context = "\n".join(context_parts)
        
        # Agregar información sobre la recuperación
        context += f"\n\n[Información de recuperación: {len(retrieval_result.chunks)} chunks relevantes encontrados de {retrieval_result.total_chunks_searched} chunks totales]"
        
        return context
    
    def process_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> None:
        """
        Procesa un documento completo: chunking + embeddings.
        
        Args:
            text: Texto del documento
            chunk_size: Tamaño de chunks
            overlap: Solapamiento entre chunks
        """
        logger.info("Procesando documento completo")
        
        # Chunking
        self.chunk_document(text, chunk_size, overlap)
        
        # Generar embeddings
        self.generate_embeddings()
        
        logger.info("Documento procesado exitosamente")
    
    def get_document_stats(self) -> Dict:
        """Obtiene estadísticas del documento procesado."""
        if not self.document_chunks:
            return {"status": "No document processed"}
        
        total_chars = sum(len(chunk.text) for chunk in self.document_chunks)
        total_words = sum(chunk.metadata['word_count'] for chunk in self.document_chunks)
        
        return {
            "total_chunks": len(self.document_chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_length": total_chars / len(self.document_chunks),
            "avg_chunk_words": total_words / len(self.document_chunks),
            "has_embeddings": self.chunk_embeddings is not None
        }
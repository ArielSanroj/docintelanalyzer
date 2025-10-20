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

# BM25 for hybrid search (install with: pip install rank_bm25)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available. Install with: pip install rank_bm25 for hybrid search")

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
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el sistema RAG.
        
        Args:
            model_name: Nombre del modelo de embeddings a usar
                      Opciones recomendadas:
                      - "all-MiniLM-L6-v2" (rápido, ~90MB)
                      - "paraphrase-multilingual-MiniLM-L12-v2" (multilingüe, ~120MB)
                      - "intfloat/multilingual-e5-large" (mejor calidad, ~1GB)
        """
        self.model_name = model_name
        self.embedding_model = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_embeddings: np.ndarray = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Carga el modelo de embeddings con caché."""
        try:
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            # Usar caché para evitar descargas repetidas
            cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            self.embedding_model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_folder
            )
            logger.info("Modelo de embeddings cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def chunk_document(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[DocumentChunk]:
        """
        Divide el documento en chunks adaptativos con detección de estructura.
        
        Args:
            text: Texto del documento
            chunk_size: Tamaño base de cada chunk
            overlap: Solapamiento entre chunks
            
        Returns:
            Lista de DocumentChunk
        """
        logger.info(f"Chunking adaptativo: size={chunk_size}, overlap={overlap}")
        text = self._clean_text(text)
        
        if not text or len(text.strip()) == 0:
            logger.warning("Texto vacío, chunks vacíos")
            self.document_chunks = []
            return []
        
        # Detecta separators genéricos via regex (headings comunes)
        heading_pattern = r'\n[A-ZÀ-Ú]{3,}[a-z]*\s*\d*\.?\s*'  # Ej: "ARTÍCULO 1", "SECTION 2", "CAPÍTULO III"
        separators = re.split(heading_pattern, text)  # Divide en secciones grandes
        granular_chunks = []
        
        for i, section in enumerate(separators):
            if len(section) > chunk_size // 2:  # Solo si sección grande
                # Chunk granular en section
                start = 0
                while start < len(section):
                    end = min(start + chunk_size, len(section))
                    # Corta en oraciones o sub-headings
                    last_end = re.search(r'\.\s+[A-Z]', section[start:end])  # Fin oración + mayús
                    if last_end:
                        end = start + last_end.start() + 1
                    chunk_text = section[start:end].strip()
                    if chunk_text:
                        metadata = {
                            'type': 'detail', 
                            'section_id': i,
                            'length': len(chunk_text),
                            'word_count': len(chunk_text.split()),
                            'has_numbers': bool(re.search(r'\d', chunk_text)),
                            'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk_text))
                        }
                        granular_chunks.append(DocumentChunk(
                            text=chunk_text, 
                            start_pos=start, 
                            end_pos=end, 
                            chunk_id=len(granular_chunks), 
                            metadata=metadata
                        ))
                    start = end - overlap
        
        # Overview: Primer 20% o hasta primer heading
        first_heading = re.search(heading_pattern, text)
        intro_end = first_heading.start() if first_heading else min(int(len(text) * 0.2), 2000)
        overview_text = text[:intro_end].strip()
        
        overview = DocumentChunk(
            text=overview_text, 
            start_pos=0, 
            end_pos=intro_end, 
            chunk_id=-1, 
            metadata={
                'type': 'overview',
                'length': len(overview_text),
                'word_count': len(overview_text.split()),
                'has_numbers': bool(re.search(r'\d', overview_text)),
                'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', overview_text))
            }
        )
        
        chunks = [overview] + granular_chunks
        self.document_chunks = chunks
        logger.info(f"Chunks adaptativos: {len(chunks)} total ({1} overview + {len(granular_chunks)} details)")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Limpia mínimo para preservar títulos cortos en HTML/legal."""
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Filtra solo líneas muy cortas (<3 chars, puro ruido)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        return ' '.join(cleaned_lines)
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Genera embeddings para todos los chunks del documento.
        
        Returns:
            Array de embeddings
        """
        if not self.document_chunks:
            logger.warning("No chunks, embeddings vacíos")
            self.chunk_embeddings = np.empty((0, 384))  # Dim default; genérico para MiniLM
            return self.chunk_embeddings
        
        logger.info(f"Generando embeddings para {len(self.document_chunks)} chunks")
        
        try:
            chunk_texts = [chunk.text for chunk in self.document_chunks]
            embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
            self.chunk_embeddings = embeddings
            logger.info(f"Embeddings generados: {embeddings.shape}")
            return embeddings
        except Exception as e:
            if "slice" in str(e).lower():
                logger.error(f"Error slice en encode (empty batch?): {e}. Usando empty array.")
                self.chunk_embeddings = np.empty((0, 384))
            else:
                raise
            return self.chunk_embeddings
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, min_score: float = 0.25) -> RetrievalResult:
        """
        Recupera los chunks más relevantes usando búsqueda híbrida (semántica + BM25).
        
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
        
        # Calcular similitudes semánticas
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Hybrid genérico: BM25 para keywords (cualquier query)
        if BM25_AVAILABLE and not hasattr(self, 'bm25_index'):
            tokenized = [chunk.text.lower().split() for chunk in self.document_chunks]
            self.bm25_index = BM25Okapi(tokenized)
        
        if BM25_AVAILABLE:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Boost overview si query general (ej. len(query)<10 palabras)
            overview_bonus = 0.2 if len(query.split()) < 10 else 0
            hybrid_scores = 0.7 * similarities + 0.3 * bm25_scores
            
            # Boost overview chunk for general queries
            if self.document_chunks and self.document_chunks[0].metadata.get('type') == 'overview':
                hybrid_scores[0] += overview_bonus
            
            # Normalize BM25 scores to [0,1] range
            if len(bm25_scores) > 0:
                bm25_max = np.max(bm25_scores)
                if bm25_max > 0:
                    bm25_scores = bm25_scores / bm25_max
                hybrid_scores = 0.7 * similarities + 0.3 * bm25_scores
        else:
            hybrid_scores = similarities
        
        # Obtener índices ordenados por puntuación híbrida
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        
        # Filtrar por puntuación mínima y tomar top_k
        relevant_chunks = []
        relevant_scores = []
        
        for idx in sorted_indices:
            if hybrid_scores[idx] >= min_score and len(relevant_chunks) < top_k:
                relevant_chunks.append(self.document_chunks[idx])
                relevant_scores.append(float(hybrid_scores[idx]))

        # Fallback: ensure at least one chunk is returned for the LLM context
        if not relevant_chunks and len(self.document_chunks) > 0 and len(sorted_indices) > 0:
            fallback_idx = sorted_indices[0]
            logger.warning(
                "No chunks met the min_score %.2f; falling back to top-scored chunk (score=%.3f)",
                min_score,
                float(hybrid_scores[fallback_idx])
            )
            relevant_chunks.append(self.document_chunks[fallback_idx])
            relevant_scores.append(float(hybrid_scores[fallback_idx]))
        
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
    
    def process_document(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> None:
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
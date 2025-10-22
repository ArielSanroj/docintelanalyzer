"""
Retrieval strategies for the unified RAG system.
Implements semantic, keyword, and hybrid search strategies.
"""

import logging
import numpy as np
import re
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from .base_rag import RAGStrategy, DocumentChunk
from ..exceptions import RetrievalError
from ..logging_config import get_logger, log_execution_time

# Optional imports
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = get_logger(__name__)


class SemanticStrategy(RAGStrategy):
    """Semantic similarity-based retrieval strategy."""
    
    def __init__(self, embedding_model):
        """
        Initialize semantic strategy.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedding_model = embedding_model
        self.cache = {}
    
    @log_execution_time
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Retrieve chunks using semantic similarity.
        
        Args:
            query: Search query
            chunks: Available document chunks
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
        
        Returns:
            Tuple of (retrieved_chunks, scores)
        """
        try:
            if not chunks:
                return [], []
            
            # Check cache for query
            cache_key = f"semantic:{hash(query)}"
            if cache_key in self.cache:
                logger.debug("Using cached semantic results")
                return self.cache[cache_key]
            
            # Get embeddings for chunks that don't have them
            chunk_embeddings = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    chunk_embeddings.append(chunk.embedding)
                else:
                    # Generate embedding for chunk
                    embedding = self.embedding_model.encode([chunk.text])
                    chunk.embeddings = embedding[0]
                    chunk_embeddings.append(embedding[0])
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Get top-k chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            retrieved_chunks = []
            scores = []
            
            for idx in top_indices:
                if similarities[idx] >= min_score:
                    retrieved_chunks.append(chunks[idx])
                    scores.append(float(similarities[idx]))
            
            # Cache results
            self.cache[cache_key] = (retrieved_chunks, scores)
            
            logger.info(f"Semantic retrieval: {len(retrieved_chunks)} chunks found")
            
            return retrieved_chunks, scores
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            raise RetrievalError(f"Semantic retrieval failed: {e}")


class KeywordStrategy(RAGStrategy):
    """Keyword-based retrieval strategy using BM25."""
    
    def __init__(self, bm25_index: Optional[Any] = None):
        """
        Initialize keyword strategy.
        
        Args:
            bm25_index: Pre-built BM25 index (optional)
        """
        self.bm25_index = bm25_index
        self.cache = {}
    
    def _build_bm25_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from chunks."""
        try:
            if not BM25_AVAILABLE:
                logger.warning("BM25 not available, skipping keyword search")
                return
            
            # Tokenize chunks
            tokenized_chunks = []
            for chunk in chunks:
                # Simple tokenization (can be improved)
                tokens = re.findall(r'\b\w+\b', chunk.text.lower())
                tokenized_chunks.append(tokens)
            
            if tokenized_chunks:
                self.bm25_index = BM25Okapi(tokenized_chunks)
                logger.info("BM25 index built successfully")
            
        except Exception as e:
            logger.error(f"BM25 index building failed: {e}")
            raise RetrievalError(f"BM25 index building failed: {e}")
    
    @log_execution_time
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Retrieve chunks using keyword matching.
        
        Args:
            query: Search query
            chunks: Available document chunks
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
        
        Returns:
            Tuple of (retrieved_chunks, scores)
        """
        try:
            if not chunks:
                return [], []
            
            # Check cache for query
            cache_key = f"keyword:{hash(query)}"
            if cache_key in self.cache:
                logger.debug("Using cached keyword results")
                return self.cache[cache_key]
            
            # Build BM25 index if not available
            if self.bm25_index is None:
                self._build_bm25_index(chunks)
            
            if self.bm25_index is None:
                logger.warning("BM25 index not available, returning empty results")
                return [], []
            
            # Tokenize query
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Normalize scores to [0, 1] range
            if len(bm25_scores) > 0:
                max_score = np.max(bm25_scores)
                if max_score > 0:
                    bm25_scores = bm25_scores / max_score
            
            # Get top-k chunks
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            retrieved_chunks = []
            scores = []
            
            for idx in top_indices:
                if bm25_scores[idx] >= min_score:
                    retrieved_chunks.append(chunks[idx])
                    scores.append(float(bm25_scores[idx]))
            
            # Cache results
            self.cache[cache_key] = (retrieved_chunks, scores)
            
            logger.info(f"Keyword retrieval: {len(retrieved_chunks)} chunks found")
            
            return retrieved_chunks, scores
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            raise RetrievalError(f"Keyword retrieval failed: {e}")


class HybridStrategy(RAGStrategy):
    """Hybrid retrieval strategy combining semantic and keyword search."""
    
    def __init__(
        self,
        semantic_strategy: SemanticStrategy,
        keyword_strategy: KeywordStrategy,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid strategy.
        
        Args:
            semantic_strategy: Semantic retrieval strategy
            keyword_strategy: Keyword retrieval strategy
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
        """
        self.semantic_strategy = semantic_strategy
        self.keyword_strategy = keyword_strategy
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.cache = {}
    
    @log_execution_time
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Retrieve chunks using hybrid approach.
        
        Args:
            query: Search query
            chunks: Available document chunks
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
        
        Returns:
            Tuple of (retrieved_chunks, scores)
        """
        try:
            if not chunks:
                return [], []
            
            # Check cache for query
            cache_key = f"hybrid:{hash(query)}"
            if cache_key in self.cache:
                logger.debug("Using cached hybrid results")
                return self.cache[cache_key]
            
            # Get semantic results
            semantic_chunks, semantic_scores = self.semantic_strategy.retrieve(
                query, chunks, top_k * 2, min_score
            )
            
            # Get keyword results
            keyword_chunks, keyword_scores = self.keyword_strategy.retrieve(
                query, chunks, top_k * 2, min_score
            )
            
            # Combine results
            combined_results = self._combine_results(
                semantic_chunks, semantic_scores,
                keyword_chunks, keyword_scores,
                top_k, min_score
            )
            
            # Cache results
            self.cache[cache_key] = combined_results
            
            logger.info(f"Hybrid retrieval: {len(combined_results[0])} chunks found")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RetrievalError(f"Hybrid retrieval failed: {e}")
    
    def _combine_results(
        self,
        semantic_chunks: List[DocumentChunk],
        semantic_scores: List[float],
        keyword_chunks: List[DocumentChunk],
        keyword_scores: List[float],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """Combine semantic and keyword results."""
        try:
            # Create chunk ID to score mapping
            chunk_scores = {}
            
            # Add semantic scores
            for chunk, score in zip(semantic_chunks, semantic_scores):
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        'chunk': chunk,
                        'semantic_score': 0.0,
                        'keyword_score': 0.0
                    }
                chunk_scores[chunk_id]['semantic_score'] = score
            
            # Add keyword scores
            for chunk, score in zip(keyword_chunks, keyword_scores):
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        'chunk': chunk,
                        'semantic_score': 0.0,
                        'keyword_score': 0.0
                    }
                chunk_scores[chunk_id]['keyword_score'] = score
            
            # Calculate combined scores
            combined_chunks = []
            combined_scores = []
            
            for chunk_data in chunk_scores.values():
                combined_score = (
                    self.semantic_weight * chunk_data['semantic_score'] +
                    self.keyword_weight * chunk_data['keyword_score']
                )
                
                if combined_score >= min_score:
                    combined_chunks.append(chunk_data['chunk'])
                    combined_scores.append(combined_score)
            
            # Sort by combined score
            sorted_indices = np.argsort(combined_scores)[::-1]
            
            # Get top-k results
            top_chunks = [combined_chunks[i] for i in sorted_indices[:top_k]]
            top_scores = [combined_scores[i] for i in sorted_indices[:top_k]]
            
            return top_chunks, top_scores
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            raise RetrievalError(f"Result combination failed: {e}")


class RerankingStrategy:
    """Reranking strategy for improving retrieval results."""
    
    def __init__(self, reranker=None):
        """
        Initialize reranking strategy.
        
        Args:
            reranker: Reranking model (e.g., NVIDIA reranker)
        """
        self.reranker = reranker
        self.cache = {}
    
    @log_execution_time
    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        scores: List[float],
        top_k: int
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Rerank retrieved chunks.
        
        Args:
            query: Search query
            chunks: Retrieved chunks
            scores: Original scores
            top_k: Number of chunks to return
        
        Returns:
            Tuple of (reranked_chunks, reranked_scores)
        """
        try:
            if not chunks or not self.reranker:
                return chunks, scores
            
            # Check cache
            cache_key = f"rerank:{hash(query)}:{len(chunks)}"
            if cache_key in self.cache:
                logger.debug("Using cached reranking results")
                return self.cache[cache_key]
            
            # Prepare documents for reranking
            from langchain_core.documents import Document
            documents = [
                Document(
                    page_content=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "original_score": scores[i] if i < len(scores) else 0.0
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Rerank documents
            reranked_docs = self.reranker.compress_documents(documents, query)
            
            # Extract reranked chunks and scores
            reranked_chunks = []
            reranked_scores = []
            
            chunk_map = {chunk.chunk_id: (chunk, scores[i] if i < len(scores) else 0.0) 
                        for i, chunk in enumerate(chunks)}
            
            for doc in reranked_docs[:top_k]:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id in chunk_map:
                    chunk, score = chunk_map[chunk_id]
                    reranked_chunks.append(chunk)
                    reranked_scores.append(score)
            
            # Cache results
            self.cache[cache_key] = (reranked_chunks, reranked_scores)
            
            logger.info(f"Reranked {len(reranked_chunks)} chunks")
            
            return reranked_chunks, reranked_scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results if reranking fails
            return chunks[:top_k], scores[:top_k]

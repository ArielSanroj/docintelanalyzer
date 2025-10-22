"""
Abstract base class for RAG systems.
Defines the interface that all RAG implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str
    chunk_id: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    total_chunks_searched: int
    retrieval_type: str
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    relevant_chunks: List[DocumentChunk]
    confidence_score: float
    retrieval_info: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseRAGSystem(ABC):
    """Abstract base class for RAG systems."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 5,
        min_score: float = 0.3
    ):
        """
        Initialize base RAG system.
        
        Args:
            embedding_model: Name of the embedding model
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score
        
        # State
        self.document_chunks: List[DocumentChunk] = []
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the RAG system."""
        pass
    
    @abstractmethod
    def process_document(self, text: str, **kwargs) -> None:
        """
        Process a document for retrieval.
        
        Args:
            text: Document text to process
            **kwargs: Additional processing parameters
        """
        pass
    
    @abstractmethod
    def retrieve_documents(
        self,
        query: str,
        search_type: str = "hybrid",
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            search_type: Type of search (semantic, keyword, hybrid)
            **kwargs: Additional search parameters
        
        Returns:
            Retrieval result with relevant chunks
        """
        pass
    
    @abstractmethod
    def generate_response(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            context: Retrieved document chunks
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        pass
    
    @abstractmethod
    def clear_documents(self) -> None:
        """Clear all processed documents."""
        pass
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready for use."""
        return self.is_initialized and len(self.document_chunks) > 0
    
    def get_chunk_count(self) -> int:
        """Get the number of processed chunks."""
        return len(self.document_chunks)
    
    def get_total_characters(self) -> int:
        """Get total number of characters in processed documents."""
        return sum(len(chunk.text) for chunk in self.document_chunks)
    
    def get_total_words(self) -> int:
        """Get total number of words in processed documents."""
        return sum(len(chunk.text.split()) for chunk in self.document_chunks)


class RAGStrategy(ABC):
    """Abstract base class for RAG retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Retrieve relevant chunks using this strategy.
        
        Args:
            query: Search query
            chunks: Available document chunks
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
        
        Returns:
            Tuple of (retrieved_chunks, scores)
        """
        pass


class SemanticStrategy(RAGStrategy):
    """Semantic similarity-based retrieval strategy."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """Retrieve chunks using semantic similarity."""
        # Implementation would go here
        # This is a placeholder for the actual implementation
        return [], []


class KeywordStrategy(RAGStrategy):
    """Keyword-based retrieval strategy."""
    
    def __init__(self, bm25_index=None):
        self.bm25_index = bm25_index
    
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """Retrieve chunks using keyword matching."""
        # Implementation would go here
        # This is a placeholder for the actual implementation
        return [], []


class HybridStrategy(RAGStrategy):
    """Hybrid retrieval strategy combining semantic and keyword search."""
    
    def __init__(self, semantic_strategy: SemanticStrategy, keyword_strategy: KeywordStrategy):
        self.semantic_strategy = semantic_strategy
        self.keyword_strategy = keyword_strategy
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
    
    def retrieve(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """Retrieve chunks using hybrid approach."""
        # Get semantic results
        semantic_chunks, semantic_scores = self.semantic_strategy.retrieve(
            query, chunks, top_k * 2, min_score
        )
        
        # Get keyword results
        keyword_chunks, keyword_scores = self.keyword_strategy.retrieve(
            query, chunks, top_k * 2, min_score
        )
        
        # Combine results (simplified implementation)
        # In practice, this would be more sophisticated
        combined_chunks = semantic_chunks[:top_k]
        combined_scores = semantic_scores[:top_k]
        
        return combined_chunks, combined_scores

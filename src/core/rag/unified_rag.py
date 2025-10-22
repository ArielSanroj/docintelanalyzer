"""
Unified RAG system implementation.
Consolidates all RAG functionality into a single, efficient system.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import time
import json

# Core imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

# Local imports
from .base_rag import BaseRAGSystem, DocumentChunk, RetrievalResult, RAGResponse
from .strategies import SemanticStrategy, KeywordStrategy, HybridStrategy
from ..exceptions import RAGException, EmbeddingError, VectorStoreError, RetrievalError
from ..logging_config import get_logger, log_execution_time

# Optional imports
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

logger = get_logger(__name__)


class UnifiedRAGSystem(BaseRAGSystem):
    """
    Unified RAG system that consolidates all RAG functionality.
    Supports multiple retrieval strategies and vector stores.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 5,
        min_score: float = 0.3,
        search_type: str = "hybrid",
        vector_store_type: str = "faiss",
        vector_store_path: str = "./vector_store",
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        use_nvidia: bool = False
    ):
        """
        Initialize unified RAG system.
        
        Args:
            embedding_model: Name of the embedding model
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
            search_type: Type of search (semantic, keyword, hybrid)
            vector_store_type: Type of vector store (faiss, chroma)
            vector_store_path: Path to store vector indices
            enable_cache: Enable caching
            cache_ttl: Cache TTL in seconds
            use_nvidia: Use NVIDIA embeddings if available
        """
        super().__init__(embedding_model, chunk_size, chunk_overlap, top_k, min_score)
        
        self.search_type = search_type
        self.vector_store_type = vector_store_type
        self.vector_store_path = vector_store_path
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.use_nvidia = use_nvidia
        
        # Components
        self.embedding_model = None
        self.embedding_function = None
        self.nvidia_embeddings = None
        self.reranker = None
        self.vector_store = None
        self.bm25_index = None
        self.chunk_embeddings = None
        
        # Strategies
        self.semantic_strategy = None
        self.keyword_strategy = None
        self.hybrid_strategy = None
        
        # Cache
        self.embedding_cache = {}
        self.query_cache = {}
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "total_words": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retrieval_count": 0,
            "average_retrieval_time": 0.0
        }
    
    @log_execution_time
    def initialize(self) -> None:
        """Initialize the RAG system components."""
        logger.info("Initializing unified RAG system...")
        
        try:
            # Initialize embedding model
            self._initialize_embeddings()
            
            # Initialize vector store
            self._initialize_vector_store()
            
            # Initialize retrieval strategies
            self._initialize_strategies()
            
            # Initialize BM25 if available
            if BM25_AVAILABLE:
                self._initialize_bm25()
            
            # Initialize NVIDIA components if available
            if self.use_nvidia and NVIDIA_AVAILABLE:
                self._initialize_nvidia()
            
            self.is_initialized = True
            logger.info("Unified RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise RAGException(f"RAG system initialization failed: {e}")
    
    def _initialize_embeddings(self) -> None:
        """Initialize embedding model and function."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            
            # Load sentence transformer model
            cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            self.embedding_model = SentenceTransformer(
                self.embedding_model,
                cache_folder=cache_folder
            )
            
            # Initialize LangChain embedding function
            self.embedding_function = SentenceTransformerEmbeddings(
                model_name=self.embedding_model
            )
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Embedding model loading failed: {e}")
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store."""
        try:
            # Ensure vector store directory exists
            Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            if self.vector_store_type == "faiss":
                # FAISS will be created when documents are added
                logger.info("FAISS vector store ready")
            elif self.vector_store_type == "chroma":
                # Chroma will be created when documents are added
                logger.info("Chroma vector store ready")
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}")
    
    def _initialize_strategies(self) -> None:
        """Initialize retrieval strategies."""
        try:
            # Initialize semantic strategy
            self.semantic_strategy = SemanticStrategy(self.embedding_model)
            
            # Initialize keyword strategy
            self.keyword_strategy = KeywordStrategy()
            
            # Initialize hybrid strategy
            self.hybrid_strategy = HybridStrategy(
                self.semantic_strategy,
                self.keyword_strategy
            )
            
            logger.info("Retrieval strategies initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise RAGException(f"Strategy initialization failed: {e}")
    
    def _initialize_bm25(self) -> None:
        """Initialize BM25 index."""
        try:
            if BM25_AVAILABLE:
                logger.info("BM25 available for keyword search")
            else:
                logger.warning("BM25 not available, keyword search disabled")
                
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
    
    def _initialize_nvidia(self) -> None:
        """Initialize NVIDIA components if available."""
        try:
            if NVIDIA_AVAILABLE:
                # Initialize NVIDIA embeddings
                self.nvidia_embeddings = NVIDIAEmbeddings(
                    model=self.embedding_model
                )
                
                # Initialize NVIDIA reranker
                self.reranker = NVIDIARerank(
                    model="nvidia/llama-3.2-rerankqa-1b-v2"
                )
                
                logger.info("NVIDIA components initialized")
            else:
                logger.warning("NVIDIA components not available")
                
        except Exception as e:
            logger.warning(f"NVIDIA initialization failed: {e}")
    
    @log_execution_time
    def process_document(self, text: str, **kwargs) -> None:
        """
        Process a document for retrieval.
        
        Args:
            text: Document text to process
            **kwargs: Additional processing parameters
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Processing document: {len(text)} characters")
        
        try:
            # Chunk the document
            chunks = self._chunk_document(text)
            
            # Generate embeddings
            self._generate_embeddings(chunks)
            
            # Update document chunks
            self.document_chunks.extend(chunks)
            
            # Create vector store
            self._create_vector_store()
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] = len(self.document_chunks)
            self.stats["total_characters"] = sum(len(chunk.text) for chunk in self.document_chunks)
            self.stats["total_words"] = sum(len(chunk.text.split()) for chunk in self.document_chunks)
            
            logger.info(f"Document processed: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise RAGException(f"Document processing failed: {e}")
    
    def _chunk_document(self, text: str) -> List[DocumentChunk]:
        """Chunk document into smaller pieces."""
        try:
            # Use RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # Split text into chunks
            text_chunks = splitter.split_text(text)
            
            # Create DocumentChunk objects
            chunks = []
            start_pos = 0
            
            for i, chunk_text in enumerate(text_chunks):
                end_pos = start_pos + len(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=len(self.document_chunks) + i,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata={
                        "chunk_id": len(self.document_chunks) + i,
                        "length": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "has_numbers": any(c.isdigit() for c in chunk_text),
                        "has_dates": any(c in chunk_text for c in ["/", "-", "."])
                    }
                )
                chunks.append(chunk)
                start_pos = end_pos
            
            return chunks
            
        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            raise RAGException(f"Document chunking failed: {e}")
    
    def _generate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for document chunks."""
        try:
            # Check cache first
            if self.enable_cache:
                cached_embeddings = self._get_cached_embeddings(chunks)
                if cached_embeddings:
                    for chunk, embedding in zip(chunks, cached_embeddings):
                        chunk.embedding = embedding
                    self.stats["cache_hits"] += 1
                    return
            
            # Generate embeddings in batches
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True
            )
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Cache embeddings
            if self.enable_cache:
                self._cache_embeddings(chunks, embeddings)
            
            self.stats["cache_misses"] += 1
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def _get_cached_embeddings(self, chunks: List[DocumentChunk]) -> Optional[np.ndarray]:
        """Get cached embeddings for chunks."""
        cache_keys = [f"embedding:{hash(chunk.text)}" for chunk in chunks]
        cached_embeddings = []
        
        for key in cache_keys:
            if key in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[key])
            else:
                return None
        
        return np.array(cached_embeddings) if cached_embeddings else None
    
    def _cache_embeddings(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Cache embeddings for chunks."""
        for chunk, embedding in zip(chunks, embeddings):
            cache_key = f"embedding:{hash(chunk.text)}"
            self.embedding_cache[cache_key] = embedding
    
    def _create_vector_store(self) -> None:
        """Create vector store from document chunks."""
        try:
            if not self.document_chunks:
                return
            
            # Create LangChain documents
            documents = [
                Document(
                    page_content=chunk.text,
                    metadata=chunk.metadata
                )
                for chunk in self.document_chunks
            ]
            
            # Create vector store
            if self.vector_store_type == "faiss":
                if self.nvidia_embeddings:
                    self.vector_store = FAISS.from_documents(
                        documents, self.nvidia_embeddings
                    )
                else:
                    self.vector_store = FAISS.from_documents(
                        documents, self.embedding_function
                    )
            elif self.vector_store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_function,
                    persist_directory=self.vector_store_path
                )
            
            logger.info(f"Vector store created: {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise VectorStoreError(f"Vector store creation failed: {e}")
    
    @log_execution_time
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
        if not self.is_ready():
            raise RAGException("RAG system not ready")
        
        start_time = time.time()
        
        try:
            # Check query cache
            if self.enable_cache:
                cache_key = f"query:{hash(query)}:{search_type}"
                if cache_key in self.query_cache:
                    self.stats["cache_hits"] += 1
                    return self.query_cache[cache_key]
            
            # Retrieve documents based on search type
            if search_type == "semantic":
                chunks, scores = self.semantic_strategy.retrieve(
                    query, self.document_chunks, self.top_k, self.min_score
                )
            elif search_type == "keyword":
                chunks, scores = self.keyword_strategy.retrieve(
                    query, self.document_chunks, self.top_k, self.min_score
                )
            elif search_type == "hybrid":
                chunks, scores = self.hybrid_strategy.retrieve(
                    query, self.document_chunks, self.top_k, self.min_score
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Create retrieval result
            result = RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                total_chunks_searched=len(self.document_chunks),
                retrieval_type=search_type,
                metadata={
                    "retrieval_time": time.time() - start_time,
                    "chunks_found": len(chunks),
                    "avg_score": np.mean(scores) if scores else 0.0
                }
            )
            
            # Cache result
            if self.enable_cache:
                self.query_cache[cache_key] = result
            
            # Update statistics
            self.stats["retrieval_count"] += 1
            self.stats["cache_misses"] += 1
            retrieval_time = time.time() - start_time
            self.stats["average_retrieval_time"] = (
                (self.stats["average_retrieval_time"] * (self.stats["retrieval_count"] - 1) + retrieval_time) /
                self.stats["retrieval_count"]
            )
            
            logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise RetrievalError(f"Document retrieval failed: {e}")
    
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
        # This is a placeholder implementation
        # In practice, this would integrate with an LLM
        
        if not context:
            context = []
        
        # Simple response generation (placeholder)
        answer = f"Based on the retrieved context, here's what I found about '{query}':\n\n"
        
        for i, chunk in enumerate(context[:3]):  # Limit to top 3 chunks
            answer += f"Context {i+1}: {chunk.text[:200]}...\n\n"
        
        # Calculate confidence score
        confidence_score = min(1.0, len(context) / self.top_k)
        
        return RAGResponse(
            answer=answer,
            relevant_chunks=context,
            confidence_score=confidence_score,
            retrieval_info={
                "chunks_used": len(context),
                "avg_score": np.mean([getattr(chunk, 'score', 0.5) for chunk in context]) if context else 0.0
            },
            metadata={
                "generation_time": time.time(),
                "query_length": len(query)
            }
        )
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "is_ready": self.is_ready(),
            "chunk_count": len(self.document_chunks),
            "total_characters": self.get_total_characters(),
            "total_words": self.get_total_words(),
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "search_type": self.search_type,
            "cache_enabled": self.enable_cache,
            "cache_size": len(self.embedding_cache),
            "nvidia_available": NVIDIA_AVAILABLE,
            "bm25_available": BM25_AVAILABLE
        }
    
    def clear_documents(self) -> None:
        """Clear all processed documents."""
        self.document_chunks.clear()
        self.vector_store = None
        self.chunk_embeddings = None
        self.bm25_index = None
        
        # Clear cache
        self.embedding_cache.clear()
        self.query_cache.clear()
        
        # Reset statistics
        self.stats.update({
            "documents_processed": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "total_words": 0
        })
        
        logger.info("All documents cleared from RAG system")

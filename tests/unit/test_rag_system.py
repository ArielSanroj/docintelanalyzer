"""
Unit tests for RAG system components.
Tests core RAG functionality, chunking, embeddings, and retrieval.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.core.rag.unified_rag import UnifiedRAGSystem
from src.core.rag.base_rag import DocumentChunk, RetrievalResult, RAGResponse
from src.core.rag.strategies import SemanticStrategy, KeywordStrategy, HybridStrategy
from src.core.rag.embeddings_manager import EmbeddingsManager
from src.core.exceptions import RAGException, EmbeddingError, RetrievalError


class TestUnifiedRAGSystem:
    """Test cases for UnifiedRAGSystem."""
    
    def test_initialization(self, temp_dir):
        """Test RAG system initialization."""
        rag_system = UnifiedRAGSystem(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            min_score=0.3,
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        assert rag_system.embedding_model == "all-MiniLM-L6-v2"
        assert rag_system.chunk_size == 100
        assert rag_system.chunk_overlap == 20
        assert rag_system.top_k == 3
        assert rag_system.min_score == 0.3
        assert not rag_system.is_initialized
    
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_initialize(self, mock_sentence_transformer, temp_dir):
        """Test RAG system initialization."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        rag_system.initialize()
        
        assert rag_system.is_initialized
        assert rag_system.embedding_model is not None
    
    def test_chunk_document(self, sample_document, temp_dir):
        """Test document chunking."""
        rag_system = UnifiedRAGSystem(
            chunk_size=100,
            chunk_overlap=20,
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        chunks = rag_system._chunk_document(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.text for chunk in chunks)
        assert all(chunk.chunk_id >= 0 for chunk in chunks)
    
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer, temp_dir):
        """Test embedding generation."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_sentence_transformer.return_value = mock_model
        
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        rag_system.embedding_model = mock_model
        
        chunks = [
            DocumentChunk("Text 1", 0, 0, 10, {}),
            DocumentChunk("Text 2", 1, 10, 20, {}),
            DocumentChunk("Text 3", 2, 20, 30, {})
        ]
        
        rag_system._generate_embeddings(chunks)
        
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(isinstance(chunk.embedding, np.ndarray) for chunk in chunks)
    
    def test_retrieve_documents_not_ready(self, temp_dir):
        """Test retrieval when system is not ready."""
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        with pytest.raises(RAGException, match="RAG system not ready"):
            rag_system.retrieve_documents("test query")
    
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_retrieve_documents_semantic(self, mock_sentence_transformer, temp_dir):
        """Test semantic document retrieval."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer.return_value = mock_model
        
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        rag_system.embedding_model = mock_model
        rag_system.is_initialized = True
        
        # Add some test chunks
        chunk1 = DocumentChunk("AI and ML", 0, 0, 10, {})
        chunk1.embedding = np.random.rand(384)
        chunk2 = DocumentChunk("Deep learning", 1, 10, 20, {})
        chunk2.embedding = np.random.rand(384)
        rag_system.document_chunks = [chunk1, chunk2]
        
        result = rag_system.retrieve_documents("artificial intelligence", "semantic")
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "artificial intelligence"
        assert result.retrieval_type == "semantic"
        assert len(result.chunks) <= rag_system.top_k
    
    def test_generate_response(self, temp_dir):
        """Test response generation."""
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        chunks = [
            DocumentChunk("Context 1", 0, 0, 10, {}),
            DocumentChunk("Context 2", 1, 10, 20, {})
        ]
        
        response = rag_system.generate_response("test query", chunks)
        
        assert isinstance(response, RAGResponse)
        assert response.answer
        assert response.relevant_chunks == chunks
        assert 0 <= response.confidence_score <= 1
    
    def test_get_document_stats(self, temp_dir):
        """Test document statistics."""
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        # Add some test chunks
        chunk1 = DocumentChunk("Text 1", 0, 0, 10, {})
        chunk2 = DocumentChunk("Text 2", 1, 10, 20, {})
        rag_system.document_chunks = [chunk1, chunk2]
        
        stats = rag_system.get_document_stats()
        
        assert "chunk_count" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert stats["chunk_count"] == 2
    
    def test_clear_documents(self, temp_dir):
        """Test clearing documents."""
        rag_system = UnifiedRAGSystem(
            vector_store_path=str(temp_dir / "vector_store")
        )
        
        # Add some test chunks
        chunk1 = DocumentChunk("Text 1", 0, 0, 10, {})
        chunk2 = DocumentChunk("Text 2", 1, 10, 20, {})
        rag_system.document_chunks = [chunk1, chunk2]
        
        rag_system.clear_documents()
        
        assert len(rag_system.document_chunks) == 0
        assert rag_system.vector_store is None


class TestEmbeddingsManager:
    """Test cases for EmbeddingsManager."""
    
    @patch('src.core.rag.embeddings_manager.SentenceTransformer')
    def test_initialization(self, mock_sentence_transformer):
        """Test embeddings manager initialization."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingsManager(
            model_name="test-model",
            cache_dir="./test_cache",
            batch_size=16
        )
        
        manager.initialize()
        
        assert manager.model is not None
        assert manager.embedding_function is not None
    
    @patch('src.core.rag.embeddings_manager.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer):
        """Test embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingsManager()
        manager.model = mock_model
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = manager.generate_embeddings(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape[0] == 384 for emb in embeddings)
    
    @patch('src.core.rag.embeddings_manager.SentenceTransformer')
    def test_get_similarity(self, mock_sentence_transformer):
        """Test similarity calculation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1, 0, 0], [0, 1, 0]])
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingsManager()
        manager.model = mock_model
        
        similarity = manager.get_similarity("text 1", "text 2")
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_cache_operations(self):
        """Test cache operations."""
        manager = EmbeddingsManager(enable_cache=True)
        
        # Test cache stats
        stats = manager.get_cache_stats()
        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        
        # Test clearing cache
        manager.clear_cache()
        assert len(manager.cache) == 0


class TestRetrievalStrategies:
    """Test cases for retrieval strategies."""
    
    def test_semantic_strategy(self):
        """Test semantic retrieval strategy."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        
        strategy = SemanticStrategy(mock_model)
        
        chunks = [
            DocumentChunk("AI content", 0, 0, 10, {}),
            DocumentChunk("ML content", 1, 10, 20, {})
        ]
        for chunk in chunks:
            chunk.embedding = np.random.rand(384)
        
        retrieved_chunks, scores = strategy.retrieve("artificial intelligence", chunks, 2, 0.3)
        
        assert len(retrieved_chunks) <= 2
        assert len(scores) == len(retrieved_chunks)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_keyword_strategy(self):
        """Test keyword retrieval strategy."""
        strategy = KeywordStrategy()
        
        chunks = [
            DocumentChunk("AI and machine learning", 0, 0, 10, {}),
            DocumentChunk("Deep learning algorithms", 1, 10, 20, {})
        ]
        
        retrieved_chunks, scores = strategy.retrieve("machine learning", chunks, 2, 0.3)
        
        assert len(retrieved_chunks) <= 2
        assert len(scores) == len(retrieved_chunks)
    
    def test_hybrid_strategy(self):
        """Test hybrid retrieval strategy."""
        mock_semantic = Mock()
        mock_semantic.retrieve.return_value = ([], [])
        
        mock_keyword = Mock()
        mock_keyword.retrieve.return_value = ([], [])
        
        strategy = HybridStrategy(mock_semantic, mock_keyword)
        
        chunks = [
            DocumentChunk("AI content", 0, 0, 10, {}),
            DocumentChunk("ML content", 1, 10, 20, {})
        ]
        
        retrieved_chunks, scores = strategy.retrieve("artificial intelligence", chunks, 2, 0.3)
        
        assert len(retrieved_chunks) <= 2
        assert len(scores) == len(retrieved_chunks)


class TestDocumentChunk:
    """Test cases for DocumentChunk."""
    
    def test_chunk_creation(self):
        """Test document chunk creation."""
        chunk = DocumentChunk(
            text="Sample text",
            chunk_id=0,
            start_pos=0,
            end_pos=10,
            metadata={"length": 10}
        )
        
        assert chunk.text == "Sample text"
        assert chunk.chunk_id == 0
        assert chunk.start_pos == 0
        assert chunk.end_pos == 10
        assert chunk.metadata["length"] == 10
        assert chunk.embedding is None
    
    def test_chunk_with_embedding(self):
        """Test chunk with embedding."""
        embedding = np.random.rand(384)
        chunk = DocumentChunk(
            text="Sample text",
            chunk_id=0,
            start_pos=0,
            end_pos=10,
            metadata={"length": 10},
            embedding=embedding
        )
        
        assert chunk.embedding is not None
        assert np.array_equal(chunk.embedding, embedding)


class TestRetrievalResult:
    """Test cases for RetrievalResult."""
    
    def test_retrieval_result_creation(self):
        """Test retrieval result creation."""
        chunks = [
            DocumentChunk("Text 1", 0, 0, 10, {}),
            DocumentChunk("Text 2", 1, 10, 20, {})
        ]
        
        result = RetrievalResult(
            chunks=chunks,
            scores=[0.9, 0.8],
            query="test query",
            total_chunks_searched=10,
            retrieval_type="semantic",
            metadata={"time": 1.5}
        )
        
        assert result.chunks == chunks
        assert result.scores == [0.9, 0.8]
        assert result.query == "test query"
        assert result.total_chunks_searched == 10
        assert result.retrieval_type == "semantic"
        assert result.metadata["time"] == 1.5


class TestRAGResponse:
    """Test cases for RAGResponse."""
    
    def test_rag_response_creation(self):
        """Test RAG response creation."""
        chunks = [DocumentChunk("Context", 0, 0, 10, {})]
        
        response = RAGResponse(
            answer="Test answer",
            relevant_chunks=chunks,
            confidence_score=0.9,
            retrieval_info={"chunks_used": 1},
            metadata={"time": 2.0}
        )
        
        assert response.answer == "Test answer"
        assert response.relevant_chunks == chunks
        assert response.confidence_score == 0.9
        assert response.retrieval_info["chunks_used"] == 1
        assert response.metadata["time"] == 2.0

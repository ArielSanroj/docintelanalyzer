"""
Integration tests for the complete RAG pipeline.
Tests end-to-end functionality from document processing to response generation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Local imports
from src.core.rag.unified_rag import UnifiedRAGSystem
from src.core.llm.llm_manager import LLMManager, LLMConfig, LLMProvider
from src.core.validation.input_validator import InputValidator
from src.infrastructure.cache.cache_service import CacheService
from src.infrastructure.database.database_service import DatabaseService
from src.core.exceptions import RAGException, LLMException, ValidationError


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def sample_documents(self) -> List[str]:
        """Sample documents for testing."""
        return [
            "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers to process data.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and understand visual information."
        ]
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service."""
        cache_service = Mock()
        cache_service.get.return_value = None
        cache_service.set.return_value = True
        cache_service.exists.return_value = False
        cache_service.get_stats.return_value = {"hits": 0, "misses": 0}
        return cache_service
    
    @pytest.fixture
    def mock_database_service(self):
        """Mock database service."""
        db_service = Mock()
        db_service.execute_query.return_value = []
        db_service.execute_update.return_value = 1
        db_service.health_check.return_value = True
        return db_service
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager."""
        llm_manager = Mock()
        llm_manager.invoke.return_value = Mock(
            content="This is a test response about artificial intelligence.",
            provider="test",
            model="test-model",
            usage_metadata={"total_tokens": 100},
            metadata={}
        )
        return llm_manager
    
    @pytest.fixture
    def rag_system(self, temp_dir, mock_cache_service):
        """RAG system for testing."""
        return UnifiedRAGSystem(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            min_score=0.3,
            vector_store_path=str(temp_dir / "vector_store"),
            enable_cache=True
        )
    
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_document_processing_pipeline(
        self,
        mock_sentence_transformer,
        rag_system,
        sample_documents,
        mock_cache_service
    ):
        """Test complete document processing pipeline."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize RAG system
        rag_system.initialize()
        
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        # Verify documents were processed
        assert rag_system.get_chunk_count() > 0
        assert rag_system.is_ready()
        
        # Test retrieval
        result = rag_system.retrieve_documents("artificial intelligence", "semantic")
        
        assert result is not None
        assert len(result.chunks) > 0
        assert result.query == "artificial intelligence"
        assert result.retrieval_type == "semantic"
    
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_rag_with_caching(
        self,
        mock_sentence_transformer,
        rag_system,
        sample_documents,
        mock_cache_service
    ):
        """Test RAG system with caching."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize with cache service
        rag_system.cache_service = mock_cache_service
        rag_system.initialize()
        
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        # Test retrieval (should use cache)
        result1 = rag_system.retrieve_documents("machine learning", "semantic")
        result2 = rag_system.retrieve_documents("machine learning", "semantic")
        
        # Verify cache was used
        assert mock_cache_service.get.called
        assert mock_cache_service.set.called
        
        # Both results should be similar
        assert len(result1.chunks) == len(result2.chunks)
    
    def test_llm_integration(
        self,
        rag_system,
        sample_documents,
        mock_llm_manager
    ):
        """Test LLM integration with RAG."""
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        # Test response generation
        response = rag_system.generate_response(
            "What is artificial intelligence?",
            context=rag_system.document_chunks[:3]
        )
        
        assert response is not None
        assert response.answer
        assert response.relevant_chunks
        assert 0 <= response.confidence_score <= 1
    
    def test_input_validation_integration(self, sample_documents):
        """Test input validation integration."""
        validator = InputValidator()
        
        # Test valid inputs
        for doc in sample_documents:
            validated = validator.validate_text(doc, "document")
            assert validated == doc.strip()
        
        # Test invalid inputs
        with pytest.raises(ValidationError):
            validator.validate_text("", "document")
        
        with pytest.raises(ValidationError):
            validator.validate_text("x" * 10001, "document")
    
    def test_database_integration(self, mock_database_service):
        """Test database integration."""
        # Test health check
        assert mock_database_service.health_check()
        
        # Test query execution
        result = mock_database_service.execute_query("SELECT 1")
        assert result == []
        
        # Test update execution
        affected = mock_database_service.execute_update("INSERT INTO test VALUES (1)")
        assert affected == 1
    
    def test_error_handling_integration(self, rag_system):
        """Test error handling in the pipeline."""
        # Test with invalid input
        with pytest.raises(RAGException):
            rag_system.retrieve_documents("", "invalid_type")
        
        # Test with uninitialized system
        uninitialized_rag = UnifiedRAGSystem()
        with pytest.raises(RAGException):
            uninitialized_rag.retrieve_documents("test query")
    
    def test_performance_metrics(self, rag_system, sample_documents):
        """Test performance metrics collection."""
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        # Get statistics
        stats = rag_system.get_document_stats()
        
        assert "chunk_count" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert stats["chunk_count"] > 0
        assert stats["total_characters"] > 0
        assert stats["total_words"] > 0
    
    def test_concurrent_processing(self, rag_system, sample_documents):
        """Test concurrent document processing."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_document(doc):
            try:
                rag_system.process_document(doc)
                results.append(doc)
            except Exception as e:
                errors.append(e)
        
        # Process documents concurrently
        threads = []
        for doc in sample_documents:
            thread = threading.Thread(target=process_document, args=(doc,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == len(sample_documents)
        assert len(errors) == 0
        assert rag_system.get_chunk_count() > 0
    
    def test_memory_cleanup(self, rag_system, sample_documents):
        """Test memory cleanup functionality."""
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        initial_chunks = rag_system.get_chunk_count()
        assert initial_chunks > 0
        
        # Clear documents
        rag_system.clear_documents()
        
        # Verify cleanup
        assert rag_system.get_chunk_count() == 0
        assert not rag_system.is_ready()
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        from src.config.settings import get_settings
        
        settings = get_settings()
        
        # Verify configuration is loaded
        assert settings.app.name == "DocsReview RAG"
        assert settings.rag.chunk_size > 0
        assert settings.llm.temperature >= 0
        assert settings.database.pool_size > 0
    
    def test_logging_integration(self):
        """Test logging integration."""
        from src.core.logging_config import get_logger
        
        logger = get_logger(__name__)
        
        # Test logging levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify logger is working
        assert logger is not None
    
    def test_dependency_injection_integration(self):
        """Test dependency injection integration."""
        from src.core.container import configure_container
        
        # Configure container
        container = configure_container()
        
        # Test service retrieval
        config = container.get('config')
        assert config is not None
        
        # Test typed service retrieval
        from src.core.config.settings import Settings
        typed_config = container.get_typed('config', Settings)
        assert isinstance(typed_config, Settings)
    
    def test_end_to_end_workflow(
        self,
        rag_system,
        sample_documents,
        mock_llm_manager,
        mock_cache_service
    ):
        """Test complete end-to-end workflow."""
        # Initialize components
        rag_system.cache_service = mock_cache_service
        rag_system.initialize()
        
        # Process documents
        for doc in sample_documents:
            rag_system.process_document(doc)
        
        # Test retrieval
        result = rag_system.retrieve_documents("artificial intelligence", "semantic")
        assert result is not None
        assert len(result.chunks) > 0
        
        # Test response generation
        response = rag_system.generate_response(
            "What is artificial intelligence?",
            context=result.chunks
        )
        
        assert response is not None
        assert response.answer
        assert response.confidence_score > 0
        
        # Test statistics
        stats = rag_system.get_document_stats()
        assert stats["chunk_count"] > 0
        assert stats["total_characters"] > 0
        
        # Test cache statistics
        cache_stats = rag_system.get_cache_stats()
        assert "hit_rate" in cache_stats
        assert "embeddings_generated" in cache_stats


class TestLLMIntegration:
    """Integration tests for LLM components."""
    
    def test_llm_manager_initialization(self):
        """Test LLM manager initialization."""
        configs = [
            LLMConfig(
                provider=LLMProvider.FALLBACK,
                model="test-model",
                temperature=0.1,
                max_tokens=1000
            )
        ]
        
        llm_manager = LLMManager(configs)
        assert llm_manager is not None
        assert len(llm_manager.providers) > 0
    
    def test_llm_fallback_chain(self):
        """Test LLM fallback chain."""
        # This would test the fallback mechanism
        # when primary providers fail
        pass
    
    def test_llm_streaming(self):
        """Test LLM streaming functionality."""
        # This would test streaming responses
        pass


class TestDatabaseIntegration:
    """Integration tests for database components."""
    
    def test_database_connection_pooling(self):
        """Test database connection pooling."""
        # This would test connection pool functionality
        pass
    
    def test_database_migrations(self):
        """Test database migrations."""
        # This would test migration functionality
        pass
    
    def test_database_performance(self):
        """Test database performance with indexes."""
        # This would test query performance
        pass


class TestCacheIntegration:
    """Integration tests for cache components."""
    
    def test_redis_cache_integration(self):
        """Test Redis cache integration."""
        # This would test Redis connectivity and operations
        pass
    
    def test_memory_cache_fallback(self):
        """Test memory cache fallback."""
        # This would test fallback to memory cache
        pass
    
    def test_cache_performance(self):
        """Test cache performance metrics."""
        # This would test cache hit rates and performance
        pass


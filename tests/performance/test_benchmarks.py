"""
Performance benchmarks and memory profiling tests.
Tests system performance under various load conditions.
"""

import pytest
import time
import psutil
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import threading
import concurrent.futures

# Local imports
from src.core.rag.unified_rag import UnifiedRAGSystem
from src.core.rag.embeddings_manager import EmbeddingsManager
from src.core.llm.llm_manager import LLMManager, LLMConfig, LLMProvider
from src.infrastructure.cache.cache_service import CacheService
from src.infrastructure.database.database_service import DatabaseService


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def large_document_set(self) -> List[str]:
        """Large set of documents for performance testing."""
        base_text = "This is a sample document for performance testing. " * 100
        return [base_text + f" Document {i}." for i in range(100)]
    
    @pytest.fixture
    def performance_rag_system(self, temp_dir):
        """RAG system optimized for performance testing."""
        return UnifiedRAGSystem(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=50,
            top_k=5,
            min_score=0.3,
            vector_store_path=str(temp_dir / "vector_store"),
            enable_cache=True
        )
    
    @pytest.fixture
    def performance_cache_service(self):
        """Cache service for performance testing."""
        return CacheService(
            enable_cache=True,
            ttl=3600,
            max_memory_size=10000
        )
    
    @pytest.mark.benchmark
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_embedding_generation_performance(
        self,
        mock_sentence_transformer,
        performance_rag_system,
        large_document_set
    ):
        """Benchmark embedding generation performance."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(100, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize system
        performance_rag_system.initialize()
        
        # Measure embedding generation time
        start_time = time.time()
        
        for doc in large_document_set[:10]:  # Test with 10 documents
            performance_rag_system.process_document(doc)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 30.0  # Should complete within 30 seconds
        assert performance_rag_system.get_chunk_count() > 0
        
        # Calculate performance metrics
        chunks_per_second = performance_rag_system.get_chunk_count() / total_time
        assert chunks_per_second > 1.0  # Should process at least 1 chunk per second
        
        print(f"Embedding generation: {total_time:.2f}s for {performance_rag_system.get_chunk_count()} chunks")
        print(f"Performance: {chunks_per_second:.2f} chunks/second")
    
    @pytest.mark.benchmark
    @patch('src.core.rag.unified_rag.SentenceTransformer')
    def test_retrieval_performance(
        self,
        mock_sentence_transformer,
        performance_rag_system,
        large_document_set
    ):
        """Benchmark retrieval performance."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(100, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize and populate system
        performance_rag_system.initialize()
        
        for doc in large_document_set[:20]:  # Process 20 documents
            performance_rag_system.process_document(doc)
        
        # Measure retrieval performance
        queries = [
            "artificial intelligence",
            "machine learning algorithms",
            "neural networks",
            "deep learning",
            "computer vision"
        ]
        
        start_time = time.time()
        
        for query in queries:
            result = performance_rag_system.retrieve_documents(query, "semantic")
            assert result is not None
            assert len(result.chunks) > 0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 5.0  # Should complete within 5 seconds
        queries_per_second = len(queries) / total_time
        assert queries_per_second > 1.0  # Should handle at least 1 query per second
        
        print(f"Retrieval performance: {total_time:.2f}s for {len(queries)} queries")
        print(f"Performance: {queries_per_second:.2f} queries/second")
    
    @pytest.mark.benchmark
    def test_cache_performance(self, performance_cache_service):
        """Benchmark cache performance."""
        # Test cache set performance
        start_time = time.time()
        
        for i in range(1000):
            key = f"test_key_{i}"
            value = f"test_value_{i}" * 100  # Large value
            performance_cache_service.set(key, value, ttl=3600)
        
        set_time = time.time() - start_time
        
        # Test cache get performance
        start_time = time.time()
        
        for i in range(1000):
            key = f"test_key_{i}"
            value = performance_cache_service.get(key)
            assert value is not None
        
        get_time = time.time() - start_time
        
        # Performance assertions
        assert set_time < 10.0  # Should complete within 10 seconds
        assert get_time < 5.0   # Should complete within 5 seconds
        
        sets_per_second = 1000 / set_time
        gets_per_second = 1000 / get_time
        
        assert sets_per_second > 100  # Should handle at least 100 sets per second
        assert gets_per_second > 200  # Should handle at least 200 gets per second
        
        print(f"Cache set performance: {sets_per_second:.2f} operations/second")
        print(f"Cache get performance: {gets_per_second:.2f} operations/second")
    
    @pytest.mark.benchmark
    def test_memory_usage(self, performance_rag_system, large_document_set):
        """Test memory usage during processing."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process documents
        for doc in large_document_set[:50]:  # Process 50 documents
            performance_rag_system.process_document(doc)
        
        # Get memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 500  # Should not use more than 500MB
        assert final_memory < 1000   # Total memory should be less than 1GB
        
        print(f"Memory usage: {memory_increase:.2f}MB increase")
        print(f"Total memory: {final_memory:.2f}MB")
    
    @pytest.mark.benchmark
    def test_concurrent_processing_performance(
        self,
        performance_rag_system,
        large_document_set
    ):
        """Test concurrent processing performance."""
        def process_document(doc):
            try:
                performance_rag_system.process_document(doc)
                return True
            except Exception as e:
                print(f"Error processing document: {e}")
                return False
        
        # Test with different levels of concurrency
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(process_document, doc)
                    for doc in large_document_set[:20]
                ]
                
                results_list = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            success_rate = sum(results_list) / len(results_list)
            results[concurrency] = {
                'time': total_time,
                'success_rate': success_rate,
                'throughput': len(results_list) / total_time
            }
            
            print(f"Concurrency {concurrency}: {total_time:.2f}s, "
                  f"success rate: {success_rate:.2f}, "
                  f"throughput: {len(results_list)/total_time:.2f} docs/sec")
        
        # Verify that higher concurrency improves throughput
        assert results[4]['throughput'] > results[1]['throughput']
        assert results[8]['throughput'] > results[4]['throughput']
    
    @pytest.mark.benchmark
    def test_database_performance(self, temp_db):
        """Test database performance with connection pooling."""
        from database import get_db_connection
        
        # Test database operations
        start_time = time.time()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Insert test data
            for i in range(100):
                cursor.execute(
                    "INSERT INTO regulations (id, query, source_type, confirmed_source, language, final_summary) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"test_{i}", f"query_{i}", "test", "test_source", "en", f"summary_{i}")
                )
            
            conn.commit()
        
        insert_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regulations WHERE query LIKE ?", ("%query_%",))
            results = cursor.fetchall()
        
        query_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 5.0   # Should complete within 5 seconds
        assert query_time < 2.0    # Should complete within 2 seconds
        assert len(results) == 100
        
        print(f"Database insert performance: {100/insert_time:.2f} records/second")
        print(f"Database query performance: {len(results)/query_time:.2f} records/second")
    
    @pytest.mark.benchmark
    def test_llm_performance(self):
        """Test LLM performance with fallback chain."""
        configs = [
            LLMConfig(
                provider=LLMProvider.FALLBACK,
                model="test-model",
                temperature=0.1,
                max_tokens=1000,
                timeout=30
            )
        ]
        
        llm_manager = LLMManager(configs)
        
        # Test LLM response time
        messages = [{"role": "user", "content": "What is artificial intelligence?"}]
        
        start_time = time.time()
        response = llm_manager.invoke(messages)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Performance assertions
        assert response_time < 10.0  # Should respond within 10 seconds
        assert response is not None
        assert response.content is not None
        
        print(f"LLM response time: {response_time:.2f}s")
    
    @pytest.mark.benchmark
    def test_end_to_end_performance(
        self,
        performance_rag_system,
        large_document_set,
        performance_cache_service
    ):
        """Test end-to-end performance of the complete system."""
        # Initialize system with cache
        performance_rag_system.cache_service = performance_cache_service
        performance_rag_system.initialize()
        
        # Measure complete workflow
        start_time = time.time()
        
        # 1. Process documents
        for doc in large_document_set[:30]:
            performance_rag_system.process_document(doc)
        
        # 2. Perform retrievals
        queries = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "deep learning",
            "computer vision"
        ]
        
        for query in queries:
            result = performance_rag_system.retrieve_documents(query, "semantic")
            assert result is not None
        
        # 3. Generate responses
        for query in queries:
            result = performance_rag_system.retrieve_documents(query, "semantic")
            response = performance_rag_system.generate_response(
                query,
                context=result.chunks
            )
            assert response is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 60.0  # Should complete within 60 seconds
        
        # Calculate metrics
        documents_processed = 30
        queries_processed = len(queries) * 2  # Retrieval + generation
        
        doc_throughput = documents_processed / total_time
        query_throughput = queries_processed / total_time
        
        assert doc_throughput > 0.5  # Should process at least 0.5 docs per second
        assert query_throughput > 0.1  # Should process at least 0.1 queries per second
        
        print(f"End-to-end performance: {total_time:.2f}s")
        print(f"Document throughput: {doc_throughput:.2f} docs/second")
        print(f"Query throughput: {query_throughput:.2f} queries/second")
    
    @pytest.mark.benchmark
    def test_memory_leak_detection(self, performance_rag_system, large_document_set):
        """Test for memory leaks during processing."""
        process = psutil.Process()
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process documents multiple times
        for iteration in range(5):
            for doc in large_document_set[:10]:
                performance_rag_system.process_document(doc)
            
            # Clear documents
            performance_rag_system.clear_documents()
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory should not increase significantly
            assert memory_increase < 100  # Should not increase by more than 100MB
            
            print(f"Iteration {iteration + 1}: Memory increase: {memory_increase:.2f}MB")
    
    @pytest.mark.benchmark
    def test_scalability_benchmarks(self, performance_rag_system, large_document_set):
        """Test system scalability with increasing load."""
        document_sizes = [10, 25, 50, 100]
        results = {}
        
        for size in document_sizes:
            # Clear system
            performance_rag_system.clear_documents()
            
            # Process documents
            start_time = time.time()
            
            for doc in large_document_set[:size]:
                performance_rag_system.process_document(doc)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Test retrieval performance
            start_time = time.time()
            
            for _ in range(10):  # 10 queries
                result = performance_rag_system.retrieve_documents("test query", "semantic")
                assert result is not None
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            results[size] = {
                'processing_time': processing_time,
                'retrieval_time': retrieval_time,
                'chunks': performance_rag_system.get_chunk_count()
            }
            
            print(f"Size {size}: Processing: {processing_time:.2f}s, "
                  f"Retrieval: {retrieval_time:.2f}s, "
                  f"Chunks: {performance_rag_system.get_chunk_count()}")
        
        # Verify scalability
        assert results[100]['processing_time'] < results[10]['processing_time'] * 15  # Should scale reasonably
        assert results[100]['retrieval_time'] < results[10]['retrieval_time'] * 5   # Retrieval should scale well


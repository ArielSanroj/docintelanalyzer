"""
Pytest configuration and fixtures for DocsReview RAG tests.
Provides common fixtures and test utilities.
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import Mock, MagicMock
import sqlite3
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from src.core.rag.unified_rag import UnifiedRAGSystem
from src.core.llm.llm_manager import LLMManager, LLMConfig, LLMProvider
from src.core.validation.input_validator import InputValidator
from src.core.logging_config import setup_logging


@pytest.fixture(scope="session")
def test_logger():
    """Set up test logging."""
    return setup_logging({
        'level': 'DEBUG',
        'enable_console': False,
        'enable_structured': True,
        'enable_colors': False
    })


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_db(temp_dir: Path) -> Generator[str, None, None]:
    """Create temporary database for tests."""
    db_path = temp_dir / "test.db"
    
    # Create test database
    conn = sqlite3.connect(str(db_path))
    conn.execute('''CREATE TABLE IF NOT EXISTS regulations (
        id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        source_type TEXT NOT NULL,
        confirmed_source TEXT NOT NULL,
        language TEXT NOT NULL,
        final_summary TEXT NOT NULL,
        refs_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()
    
    yield str(db_path)
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_document() -> str:
    """Sample document text for testing."""
    return """
    This is a sample document for testing the RAG system.
    It contains multiple paragraphs with various information.
    
    The document discusses artificial intelligence and machine learning.
    It covers topics like natural language processing, computer vision,
    and deep learning algorithms.
    
    The document also mentions specific technologies like transformers,
    attention mechanisms, and neural networks.
    """


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Sample document chunks for testing."""
    return [
        {
            "text": "This is a sample document for testing the RAG system.",
            "chunk_id": 0,
            "start_pos": 0,
            "end_pos": 50,
            "metadata": {"length": 50, "word_count": 10}
        },
        {
            "text": "It contains multiple paragraphs with various information.",
            "chunk_id": 1,
            "start_pos": 51,
            "end_pos": 100,
            "metadata": {"length": 49, "word_count": 8}
        },
        {
            "text": "The document discusses artificial intelligence and machine learning.",
            "chunk_id": 2,
            "start_pos": 101,
            "end_pos": 150,
            "metadata": {"length": 49, "word_count": 8}
        }
    ]


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Mock LLM configuration for testing."""
    return LLMConfig(
        provider=LLMProvider.FALLBACK,
        model="test-model",
        temperature=0.1,
        max_tokens=1000,
        timeout=30
    )


@pytest.fixture
def mock_llm_manager(mock_llm_config: LLMConfig) -> LLMManager:
    """Mock LLM manager for testing."""
    return LLMManager([mock_llm_config])


@pytest.fixture
def mock_rag_system(temp_dir: Path) -> UnifiedRAGSystem:
    """Mock RAG system for testing."""
    return UnifiedRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=20,
        top_k=3,
        min_score=0.3,
        vector_store_path=str(temp_dir / "vector_store")
    )


@pytest.fixture
def input_validator() -> InputValidator:
    """Input validator for testing."""
    return InputValidator({
        "max_text_length": 1000,
        "min_text_length": 1,
        "max_file_size_mb": 10,
        "allowed_file_types": ["txt", "pdf"],
        "max_url_length": 1000,
        "allowed_languages": ["en", "es"]
    })


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]] * 3
    mock_model.get_sentence_embedding_dimension.return_value = 5
    return mock_model


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        Mock(page_content="Sample content 1", metadata={"score": 0.9}),
        Mock(page_content="Sample content 2", metadata={"score": 0.8}),
        Mock(page_content="Sample content 3", metadata={"score": 0.7})
    ]
    return mock_store


@pytest.fixture
def sample_queries() -> List[str]:
    """Sample queries for testing."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain deep learning algorithms",
        "What is natural language processing?"
    ]


@pytest.fixture
def sample_responses() -> List[Dict[str, Any]]:
    """Sample responses for testing."""
    return [
        {
            "content": "Artificial intelligence is a branch of computer science...",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
            "metadata": {"response_time": 1.5}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence...",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "usage": {"total_tokens": 120, "prompt_tokens": 60, "completion_tokens": 60},
            "metadata": {"response_time": 1.8}
        }
    ]


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.rowcount = 0
    return mock_conn


@pytest.fixture
def sample_file_data() -> Dict[str, Any]:
    """Sample file data for testing."""
    return {
        "filename": "test_document.txt",
        "content": "This is a test document content.",
        "size": 1000,
        "type": "text/plain",
        "path": "/tmp/test_document.txt"
    }


@pytest.fixture
def sample_url_data() -> Dict[str, Any]:
    """Sample URL data for testing."""
    return {
        "url": "https://example.com/document.pdf",
        "title": "Sample Document",
        "content": "This is the content from the URL.",
        "status_code": 200
    }


@pytest.fixture
def performance_metrics() -> Dict[str, Any]:
    """Performance metrics for testing."""
    return {
        "embedding_time": 1.5,
        "retrieval_time": 0.3,
        "generation_time": 2.1,
        "total_time": 3.9,
        "memory_usage": 500,
        "cpu_usage": 25.0
    }


@pytest.fixture
def error_scenarios() -> List[Dict[str, Any]]:
    """Error scenarios for testing."""
    return [
        {
            "error_type": "ValidationError",
            "message": "Invalid input provided",
            "field": "text",
            "value": ""
        },
        {
            "error_type": "FileSizeError",
            "message": "File size exceeds limit",
            "file_size": 1000000,
            "max_size": 500000
        },
        {
            "error_type": "LLMProviderError",
            "message": "LLM provider failed",
            "provider": "ollama",
            "status_code": 500
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after each test
    # Remove any test files or reset state
    pass


@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for testing."""
    session_state = {
        "reports": [],
        "current_doc_text": "",
        "rag_system": None,
        "chat_history": [],
        "user_id": "test_user"
    }
    return session_state


@pytest.fixture
def mock_streamlit_components():
    """Mock Streamlit components for testing."""
    components = {
        "file_uploader": Mock(return_value=None),
        "text_input": Mock(return_value="test input"),
        "selectbox": Mock(return_value="en"),
        "button": Mock(return_value=False),
        "success": Mock(),
        "error": Mock(),
        "warning": Mock(),
        "info": Mock()
    }
    return components


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add slow marker to tests that take more than 5 seconds
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

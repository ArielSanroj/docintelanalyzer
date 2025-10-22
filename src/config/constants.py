"""
Global constants for the DocsReview RAG application.
Centralizes magic numbers and configuration values.
"""

from typing import Dict, List, Any
from enum import Enum


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NVIDIA = "nvidia"
    FALLBACK = "fallback"


class RAGSearchType(str, Enum):
    """RAG search types."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    URL = "url"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


# Performance Constants
class Performance:
    """Performance-related constants."""
    
    # Embedding batch sizes
    EMBEDDING_BATCH_SIZE = 32
    MAX_EMBEDDING_BATCH_SIZE = 128
    
    # Chunking constants
    DEFAULT_CHUNK_SIZE = 800
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 120
    
    # Retrieval constants
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    MIN_SIMILARITY_SCORE = 0.3
    
    # Memory limits
    MAX_MEMORY_CACHE_SIZE = 1000
    MAX_SESSION_MEMORY_MB = 500
    
    # Timeout values (seconds)
    LLM_TIMEOUT = 120
    EMBEDDING_TIMEOUT = 60
    DATABASE_TIMEOUT = 30
    FILE_PROCESSING_TIMEOUT = 300


# API Constants
class API:
    """API-related constants."""
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 60  # requests per minute
    MAX_RATE_LIMIT = 1000
    
    # File upload limits
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_FILE_TYPES = ["pdf", "txt", "docx"]
    
    # URL limits
    MAX_URL_LENGTH = 2048
    URL_TIMEOUT = 30
    
    # Response limits
    MAX_RESPONSE_LENGTH = 8000
    MAX_CHUNKS_IN_RESPONSE = 10


# Database Constants
class Database:
    """Database-related constants."""
    
    # Connection pool
    DEFAULT_POOL_SIZE = 10
    MAX_POOL_SIZE = 50
    DEFAULT_MAX_OVERFLOW = 20
    
    # Query limits
    MAX_QUERY_LIMIT = 1000
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Index names
    INDEX_QUERY_SOURCE = "idx_query_source"
    INDEX_TIMESTAMP = "idx_timestamp"
    INDEX_SOURCE_TYPE = "idx_source_type"


# Logging Constants
class Logging:
    """Logging-related constants."""
    
    # Log levels
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    # Log formats
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    STRUCTURED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s"
    
    # File settings
    MAX_FILE_SIZE = 10485760  # 10MB
    BACKUP_COUNT = 5


# Security Constants
class Security:
    """Security-related constants."""
    
    # Input validation
    MAX_INPUT_LENGTH = 10000
    MIN_INPUT_LENGTH = 1
    
    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:8501",
        "https://*.streamlit.app",
        "https://*.herokuapp.com"
    ]
    
    # Headers
    CORRELATION_ID_HEADER = "X-Correlation-ID"
    USER_AGENT_HEADER = "User-Agent"
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS = 60


# Model Constants
class Models:
    """Model-related constants."""
    
    # Embedding models
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    FAST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # LLM models
    DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
    DEFAULT_OPENAI_MODEL = "gpt-4"
    DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
    
    # Model parameters
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_TOP_P = 0.9


# Cache Constants
class Cache:
    """Cache-related constants."""
    
    # TTL values (seconds)
    DEFAULT_TTL = 3600
    EMBEDDING_TTL = 7200
    QUERY_TTL = 1800
    MODEL_TTL = 86400
    
    # Cache keys
    EMBEDDING_PREFIX = "embedding:"
    QUERY_PREFIX = "query:"
    MODEL_PREFIX = "model:"
    SESSION_PREFIX = "session:"
    
    # Cache sizes
    MAX_MEMORY_CACHE_SIZE = 1000
    MAX_REDIS_MEMORY_MB = 100


# Error Messages
class ErrorMessages:
    """Standardized error messages."""
    
    # General errors
    INTERNAL_ERROR = "An internal error occurred. Please try again."
    VALIDATION_ERROR = "Invalid input provided."
    AUTHENTICATION_ERROR = "Authentication failed."
    AUTHORIZATION_ERROR = "Access denied."
    
    # File errors
    FILE_TOO_LARGE = f"File size exceeds {API.MAX_FILE_SIZE_MB}MB limit."
    INVALID_FILE_TYPE = f"File type not supported. Allowed types: {', '.join(API.ALLOWED_FILE_TYPES)}"
    FILE_PROCESSING_ERROR = "Error processing file."
    
    # LLM errors
    LLM_TIMEOUT = "LLM request timed out."
    LLM_RATE_LIMIT = "Rate limit exceeded for LLM requests."
    LLM_PROVIDER_ERROR = "LLM provider error."
    
    # RAG errors
    RAG_INITIALIZATION_ERROR = "RAG system initialization failed."
    RAG_RETRIEVAL_ERROR = "Document retrieval failed."
    RAG_EMBEDDING_ERROR = "Embedding generation failed."
    
    # Database errors
    DATABASE_CONNECTION_ERROR = "Database connection failed."
    DATABASE_QUERY_ERROR = "Database query failed."
    DATABASE_TIMEOUT = "Database operation timed out."


# Success Messages
class SuccessMessages:
    """Standardized success messages."""
    
    # General success
    OPERATION_SUCCESSFUL = "Operation completed successfully."
    DATA_SAVED = "Data saved successfully."
    DATA_DELETED = "Data deleted successfully."
    
    # File operations
    FILE_UPLOADED = "File uploaded successfully."
    FILE_PROCESSED = "File processed successfully."
    
    # RAG operations
    RAG_INITIALIZED = "RAG system initialized successfully."
    DOCUMENT_INDEXED = "Document indexed successfully."
    QUERY_PROCESSED = "Query processed successfully."


# UI Constants
class UI:
    """UI-related constants."""
    
    # Streamlit settings
    PAGE_TITLE = "DocsReview RAG"
    PAGE_ICON = "📄"
    LAYOUT = "wide"
    
    # Progress indicators
    LOADING_MESSAGE = "Processing..."
    SUCCESS_MESSAGE = "✅ Success"
    ERROR_MESSAGE = "❌ Error"
    WARNING_MESSAGE = "⚠️ Warning"
    
    # Display limits
    MAX_DISPLAY_CHUNKS = 5
    MAX_DISPLAY_CHARS = 1000
    MAX_CHAT_HISTORY = 20


# Feature Flags
class FeatureFlags:
    """Feature flag constants."""
    
    # RAG features
    ENABLE_ADVANCED_RAG = True
    ENABLE_MULTIMODAL_RAG = False
    ENABLE_RERANKING = True
    ENABLE_HYBRID_SEARCH = True
    
    # LLM features
    ENABLE_STREAMING = True
    ENABLE_FALLBACK_CHAIN = True
    ENABLE_RETRY_LOGIC = True
    
    # Caching features
    ENABLE_EMBEDDING_CACHE = True
    ENABLE_QUERY_CACHE = True
    ENABLE_MODEL_CACHE = True
    
    # Security features
    ENABLE_RATE_LIMITING = True
    ENABLE_INPUT_VALIDATION = True
    ENABLE_CORS = True


# Default Configurations
DEFAULT_CONFIG = {
    "app": {
        "name": "DocsReview RAG",
        "version": "2.0.0",
        "environment": "development",
        "debug": False
    },
    "database": {
        "url": "sqlite:///regulations.db",
        "pool_size": 10,
        "max_overflow": 20
    },
    "llm": {
        "primary_provider": "ollama",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "rag": {
        "chunk_size": 800,
        "chunk_overlap": 120,
        "top_k": 5,
        "min_score": 0.3
    },
    "cache": {
        "enable_cache": True,
        "ttl": 3600
    },
    "security": {
        "enable_rate_limiting": True,
        "max_file_size_mb": 50
    }
}


# Environment-specific overrides
ENVIRONMENT_CONFIGS = {
    Environment.DEVELOPMENT: {
        "app": {"debug": True},
        "logging": {"level": "DEBUG"},
        "cache": {"enable_cache": False}
    },
    Environment.STAGING: {
        "app": {"debug": False},
        "logging": {"level": "INFO"},
        "monitoring": {"sentry_environment": "staging"}
    },
    Environment.PRODUCTION: {
        "app": {"debug": False},
        "logging": {"level": "WARNING"},
        "monitoring": {"sentry_environment": "production"},
        "security": {"enable_rate_limiting": True}
    },
    Environment.TESTING: {
        "app": {"debug": True},
        "database": {"url": "sqlite:///:memory:"},
        "cache": {"enable_cache": False},
        "llm": {"primary_provider": "fallback"}
    }
}

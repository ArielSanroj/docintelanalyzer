"""
Configuration management using Pydantic BaseSettings.
Centralizes all application settings with validation and type safety.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="sqlite:///regulations.db", description="Database URL")
    pool_size: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    
    class Config:
        env_prefix = "DB_"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    # Primary provider
    primary_provider: str = Field(default="ollama", description="Primary LLM provider")
    primary_model: str = Field(default="llama3.1:8b", description="Primary model name")
    primary_base_url: str = Field(default="http://localhost:11434", description="Primary LLM base URL")
    
    # API Keys
    ollama_api_key: Optional[str] = Field(default=None, description="Ollama API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA API key")
    
    # Model parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2000, ge=1, le=8000, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    
    # Retry configuration
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay in seconds")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1, le=1000, description="Rate limit per minute")
    
    class Config:
        env_prefix = "LLM_"


class RAGSettings(BaseSettings):
    """RAG system configuration settings."""
    
    # Embedding settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    embedding_dimension: int = Field(default=384, description="Embedding dimension")
    embedding_batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for embeddings")
    
    # Chunking settings
    chunk_size: int = Field(default=800, ge=100, le=2000, description="Default chunk size")
    chunk_overlap: int = Field(default=120, ge=0, le=500, description="Chunk overlap")
    chunk_separators: List[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        description="Text separators for chunking"
    )
    
    # Retrieval settings
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    search_type: str = Field(default="hybrid", description="Search type: semantic, keyword, hybrid")
    
    # Vector store settings
    vector_store_type: str = Field(default="faiss", description="Vector store type")
    vector_store_path: str = Field(default="./vector_store", description="Vector store path")
    
    # Cache settings
    enable_cache: bool = Field(default=True, description="Enable embedding cache")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    
    class Config:
        env_prefix = "RAG_"


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    
    # Memory cache settings
    max_memory_cache_size: int = Field(default=1000, ge=10, le=10000, description="Max memory cache size")
    memory_cache_ttl: int = Field(default=1800, ge=60, le=86400, description="Memory cache TTL")
    
    class Config:
        env_prefix = "CACHE_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # API security
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    max_requests_per_minute: int = Field(default=60, ge=1, le=1000, description="Max requests per minute")
    max_file_size_mb: int = Field(default=50, ge=1, le=500, description="Max file size in MB")
    
    # Input validation
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "docx"],
        description="Allowed file types"
    )
    max_url_length: int = Field(default=2048, ge=100, le=8192, description="Max URL length")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:8501", "https://*.streamlit.app"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, ge=1024, le=104857600, description="Max log file size")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup files")
    
    # Structured logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    correlation_id_header: str = Field(default="X-Correlation-ID", description="Correlation ID header")
    
    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Sentry settings
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    sentry_environment: str = Field(default="development", description="Sentry environment")
    
    # Health check settings
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Health check interval")
    health_check_timeout: int = Field(default=10, ge=1, le=60, description="Health check timeout")
    
    # Metrics settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8000, ge=1000, le=65535, description="Metrics port")
    
    class Config:
        env_prefix = "MONITORING_"


class AppSettings(BaseSettings):
    """Main application settings."""
    
    # Application info
    name: str = Field(default="DocsReview RAG", description="Application name")
    version: str = Field(default="2.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8501, ge=1000, le=65535, description="Server port")
    
    # File upload settings
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    max_upload_size: int = Field(default=52428800, ge=1048576, le=524288000, description="Max upload size")
    
    # Session settings
    session_timeout: int = Field(default=3600, ge=300, le=86400, description="Session timeout")
    max_sessions: int = Field(default=100, ge=1, le=1000, description="Max concurrent sessions")
    
    class Config:
        env_prefix = "APP_"


class Settings(BaseSettings):
    """Main settings container."""
    
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @validator('app.upload_dir')
    def create_upload_dir(cls, v):
        """Ensure upload directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('rag.vector_store_path')
    def create_vector_store_dir(cls, v):
        """Ensure vector store directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings

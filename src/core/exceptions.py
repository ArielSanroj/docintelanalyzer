"""
Custom exceptions for the DocsReview RAG application.
Provides structured error handling with context and recovery information.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """Context information for errors."""
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocsReviewException(Exception):
    """Base exception for all DocsReview errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.retry_after = retry_after


class RAGException(DocsReviewException):
    """Base exception for RAG-related errors."""
    pass


class EmbeddingError(RAGException):
    """Errors related to embedding generation."""
    
    def __init__(
        self,
        message: str = "Embedding generation failed",
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size


class VectorStoreError(RAGException):
    """Errors related to vector store operations."""
    
    def __init__(
        self,
        message: str = "Vector store operation failed",
        operation: Optional[str] = None,
        store_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.store_type = store_type


class RetrievalError(RAGException):
    """Errors related to document retrieval."""
    
    def __init__(
        self,
        message: str = "Document retrieval failed",
        query: Optional[str] = None,
        search_type: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.query = query
        self.search_type = search_type
        self.top_k = top_k


class ChunkingError(RAGException):
    """Errors related to document chunking."""
    
    def __init__(
        self,
        message: str = "Document chunking failed",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        document_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.document_length = document_length


class LLMException(DocsReviewException):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMException):
    """Errors related to LLM provider communication."""
    
    def __init__(
        self,
        message: str = "LLM provider error",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.status_code = status_code


class LLMTimeoutError(LLMException):
    """LLM request timeout errors."""
    
    def __init__(
        self,
        message: str = "LLM request timed out",
        timeout_seconds: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMException):
    """LLM rate limit errors."""
    
    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        rate_limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.rate_limit = rate_limit
        self.reset_time = reset_time


class DocumentProcessingError(DocsReviewException):
    """Base exception for document processing errors."""
    pass


class FileProcessingError(DocumentProcessingError):
    """Errors related to file processing."""
    
    def __init__(
        self,
        message: str = "File processing failed",
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.file_type = file_type
        self.file_size = file_size


class OCRProcessingError(DocumentProcessingError):
    """Errors related to OCR processing."""
    
    def __init__(
        self,
        message: str = "OCR processing failed",
        language: Optional[str] = None,
        page_number: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.language = language
        self.page_number = page_number


class URLProcessingError(DocumentProcessingError):
    """Errors related to URL processing."""
    
    def __init__(
        self,
        message: str = "URL processing failed",
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code


class DatabaseError(DocsReviewException):
    """Base exception for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection errors."""
    
    def __init__(
        self,
        message: str = "Database connection failed",
        database_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.database_url = database_url


class DatabaseQueryError(DatabaseError):
    """Database query errors."""
    
    def __init__(
        self,
        message: str = "Database query failed",
        query: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.query = query
        self.parameters = parameters


class CacheError(DocsReviewException):
    """Base exception for cache-related errors."""
    pass


class CacheConnectionError(CacheError):
    """Cache connection errors."""
    
    def __init__(
        self,
        message: str = "Cache connection failed",
        cache_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.cache_url = cache_url


class CacheOperationError(CacheError):
    """Cache operation errors."""
    
    def __init__(
        self,
        message: str = "Cache operation failed",
        operation: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.key = key


class ValidationError(DocsReviewException):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        self.expected_type = expected_type


class SecurityError(DocsReviewException):
    """Security-related errors."""
    
    def __init__(
        self,
        message: str = "Security error",
        violation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.violation_type = violation_type


class RateLimitError(SecurityError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.limit = limit
        self.window = window


class FileSizeError(SecurityError):
    """File size limit errors."""
    
    def __init__(
        self,
        message: str = "File size exceeds limit",
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_size = file_size
        self.max_size = max_size


class FileTypeError(SecurityError):
    """File type validation errors."""
    
    def __init__(
        self,
        message: str = "File type not allowed",
        file_type: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_type = file_type
        self.allowed_types = allowed_types


class ConfigurationError(DocsReviewException):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        setting: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.setting = setting
        self.value = value


class ServiceUnavailableError(DocsReviewException):
    """Service unavailable errors."""
    
    def __init__(
        self,
        message: str = "Service unavailable",
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.service = service


# Error code mappings
ERROR_CODES = {
    "EMBEDDING_ERROR": EmbeddingError,
    "VECTOR_STORE_ERROR": VectorStoreError,
    "RETRIEVAL_ERROR": RetrievalError,
    "CHUNKING_ERROR": ChunkingError,
    "LLM_PROVIDER_ERROR": LLMProviderError,
    "LLM_TIMEOUT": LLMTimeoutError,
    "LLM_RATE_LIMIT": LLMRateLimitError,
    "FILE_PROCESSING_ERROR": FileProcessingError,
    "OCR_PROCESSING_ERROR": OCRProcessingError,
    "URL_PROCESSING_ERROR": URLProcessingError,
    "DATABASE_CONNECTION_ERROR": DatabaseConnectionError,
    "DATABASE_QUERY_ERROR": DatabaseQueryError,
    "CACHE_CONNECTION_ERROR": CacheConnectionError,
    "CACHE_OPERATION_ERROR": CacheOperationError,
    "VALIDATION_ERROR": ValidationError,
    "RATE_LIMIT_ERROR": RateLimitError,
    "FILE_SIZE_ERROR": FileSizeError,
    "FILE_TYPE_ERROR": FileTypeError,
    "CONFIGURATION_ERROR": ConfigurationError,
    "SERVICE_UNAVAILABLE": ServiceUnavailableError,
}


def get_exception_class(error_code: str) -> type:
    """Get exception class by error code."""
    return ERROR_CODES.get(error_code, DocsReviewException)


def create_exception(
    error_code: str,
    message: str,
    **kwargs
) -> DocsReviewException:
    """Create exception by error code."""
    exception_class = get_exception_class(error_code)
    return exception_class(message, error_code=error_code, **kwargs)

"""
Centralized logging configuration for the DocsReview RAG application.
Provides structured logging with correlation IDs and monitoring integration.
"""

import logging
import logging.handlers
import sys
import json
import time
import uuid
from typing import Optional, Dict, Any, Union
from pathlib import Path
from contextvars import ContextVar
from datetime import datetime

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "no-correlation-id"
        return True


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', 'no-correlation-id'),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'correlation_id'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    correlation_id = correlation_id_var.get()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation ID to all log messages."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message to add correlation ID."""
        correlation_id = get_correlation_id()
        return f"[{correlation_id}] {msg}", kwargs


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10485760,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_structured: bool = True,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        enable_console: Enable console logging
        enable_structured: Enable structured logging
        enable_colors: Enable colored console output
    
    Returns:
        Configured logger
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add correlation ID filter
    correlation_filter = CorrelationIDFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.addFilter(correlation_filter)
        
        if enable_colors:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.addFilter(correlation_filter)
        
        if enable_structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> LoggerAdapter:
    """
    Get logger with correlation ID support.
    
    Args:
        name: Logger name
    
    Returns:
        Logger adapter with correlation ID support
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, {})


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(f"Calling {func.__name__}", extra={
            'function': func.__name__,
            'module': func.__module__,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        })
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__}", extra={
                'function': func.__name__,
                'execution_time': execution_time,
                'success': True
            })
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__}: {str(e)}", extra={
                'function': func.__name__,
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
    
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Execution time for {func.__name__}: {execution_time:.3f}s", extra={
                'function': func.__name__,
                'execution_time': execution_time
            })
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Execution failed for {func.__name__} after {execution_time:.3f}s: {str(e)}", extra={
                'function': func.__name__,
                'execution_time': execution_time,
                'error': str(e)
            })
            raise
    
    return wrapper


class LoggingContext:
    """Context manager for logging with correlation ID."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.original_correlation_id = None
    
    def __enter__(self):
        self.original_correlation_id = correlation_id_var.get()
        set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_correlation_id:
            set_correlation_id(self.original_correlation_id)
        else:
            clear_correlation_id()


# Initialize logging
def initialize_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Initialize logging with configuration.
    
    Args:
        config: Logging configuration dictionary
    
    Returns:
        Configured logger
    """
    if config is None:
        config = {
            'level': 'INFO',
            'enable_console': True,
            'enable_structured': True,
            'enable_colors': True
        }
    
    return setup_logging(**config)


# Global logger instance
logger = get_logger(__name__)

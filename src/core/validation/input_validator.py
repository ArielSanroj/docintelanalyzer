"""
Comprehensive input validation and sanitization system.
Validates all user inputs to prevent security vulnerabilities.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse
from pathlib import Path
import magic
import mimetypes

# Local imports
from ..exceptions import ValidationError, SecurityError, FileSizeError, FileTypeError
from ..logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        
        # Default validation rules
        self.rules = {
            "max_text_length": 10000,
            "min_text_length": 1,
            "max_file_size_mb": 50,
            "allowed_file_types": ["pdf", "txt", "docx"],
            "max_url_length": 2048,
            "allowed_url_schemes": ["http", "https"],
            "max_query_length": 1000,
            "min_query_length": 1,
            "allowed_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            "max_chunk_size": 2000,
            "min_chunk_size": 100
        }
        
        # Update with custom config
        self.rules.update(self.config)
        
        # Compile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self.patterns = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            "phone": re.compile(r'^\+?[\d\s\-\(\)]{10,}$'),
            "sql_injection": re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)', re.IGNORECASE),
            "xss": re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            "path_traversal": re.compile(r'\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c', re.IGNORECASE),
            "command_injection": re.compile(r'[;&|`$()]', re.IGNORECASE),
            "html_tags": re.compile(r'<[^>]+>'),
            "special_chars": re.compile(r'[<>"\']'),
            "whitespace": re.compile(r'\s+'),
            "alphanumeric": re.compile(r'^[a-zA-Z0-9\s\-_.,!?]+$')
        }
    
    @log_execution_time
    def validate_text(self, text: str, field_name: str = "text") -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Text to validate
            field_name: Name of the field for error messages
        
        Returns:
            Sanitized text
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string")
        
        # Check length
        if len(text) < self.rules["min_text_length"]:
            raise ValidationError(f"{field_name} is too short (minimum {self.rules['min_text_length']} characters)")
        
        if len(text) > self.rules["max_text_length"]:
            raise ValidationError(f"{field_name} is too long (maximum {self.rules['max_text_length']} characters)")
        
        # Check for SQL injection attempts
        if self.patterns["sql_injection"].search(text):
            raise SecurityError(f"Potential SQL injection detected in {field_name}")
        
        # Check for XSS attempts
        if self.patterns["xss"].search(text):
            raise SecurityError(f"Potential XSS detected in {field_name}")
        
        # Check for command injection attempts
        if self.patterns["command_injection"].search(text):
            raise SecurityError(f"Potential command injection detected in {field_name}")
        
        # Check for path traversal attempts
        if self.patterns["path_traversal"].search(text):
            raise SecurityError(f"Potential path traversal detected in {field_name}")
        
        # Sanitize text
        sanitized = self._sanitize_text(text)
        
        logger.info(f"Text validation successful for {field_name}: {len(sanitized)} characters")
        return sanitized
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing dangerous content."""
        # Remove HTML tags
        sanitized = self.patterns["html_tags"].sub('', text)
        
        # Remove special characters that could be dangerous
        sanitized = self.patterns["special_chars"].sub('', sanitized)
        
        # Normalize whitespace
        sanitized = self.patterns["whitespace"].sub(' ', sanitized)
        
        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @log_execution_time
    def validate_file(self, file_path: str, file_size: int, file_type: str) -> Tuple[str, str]:
        """
        Validate file upload.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes
            file_type: MIME type of the file
        
        Returns:
            Tuple of (validated_path, validated_type)
        
        Raises:
            ValidationError: If validation fails
        """
        # Check file size
        max_size_bytes = self.rules["max_file_size_mb"] * 1024 * 1024
        if file_size > max_size_bytes:
            raise FileSizeError(
                f"File size {file_size} bytes exceeds limit of {max_size_bytes} bytes",
                file_size=file_size,
                max_size=max_size_bytes
            )
        
        # Check file type
        if file_type not in self.rules["allowed_file_types"]:
            raise FileTypeError(
                f"File type {file_type} not allowed",
                file_type=file_type,
                allowed_types=self.rules["allowed_file_types"]
            )
        
        # Validate file path
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        # Check for path traversal
        if self.patterns["path_traversal"].search(str(path)):
            raise SecurityError("Path traversal detected in file path")
        
        # Verify file type with magic
        try:
            detected_type = magic.from_file(str(path), mime=True)
            if detected_type != file_type:
                logger.warning(f"File type mismatch: declared {file_type}, detected {detected_type}")
        except Exception as e:
            logger.warning(f"Could not verify file type: {e}")
        
        logger.info(f"File validation successful: {file_path} ({file_size} bytes, {file_type})")
        return str(path), file_type
    
    @log_execution_time
    def validate_url(self, url: str) -> str:
        """
        Validate URL.
        
        Args:
            url: URL to validate
        
        Returns:
            Validated URL
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")
        
        # Check length
        if len(url) > self.rules["max_url_length"]:
            raise ValidationError(f"URL too long (maximum {self.rules['max_url_length']} characters)")
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")
        
        # Check scheme
        if parsed.scheme not in self.rules["allowed_url_schemes"]:
            raise ValidationError(f"URL scheme {parsed.scheme} not allowed")
        
        # Check for dangerous patterns
        if self.patterns["path_traversal"].search(url):
            raise SecurityError("Path traversal detected in URL")
        
        if self.patterns["command_injection"].search(url):
            raise SecurityError("Command injection detected in URL")
        
        # Validate URL format
        if not self.patterns["url"].match(url):
            raise ValidationError("Invalid URL format")
        
        logger.info(f"URL validation successful: {url}")
        return url
    
    @log_execution_time
    def validate_query(self, query: str) -> str:
        """
        Validate search query.
        
        Args:
            query: Query to validate
        
        Returns:
            Validated query
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(query, str):
            raise ValidationError("Query must be a string")
        
        # Check length
        if len(query) < self.rules["min_query_length"]:
            raise ValidationError(f"Query too short (minimum {self.rules['min_query_length']} characters)")
        
        if len(query) > self.rules["max_query_length"]:
            raise ValidationError(f"Query too long (maximum {self.rules['max_query_length']} characters)")
        
        # Check for dangerous patterns
        if self.patterns["sql_injection"].search(query):
            raise SecurityError("Potential SQL injection detected in query")
        
        if self.patterns["xss"].search(query):
            raise SecurityError("Potential XSS detected in query")
        
        if self.patterns["command_injection"].search(query):
            raise SecurityError("Potential command injection detected in query")
        
        # Sanitize query
        sanitized = self._sanitize_text(query)
        
        logger.info(f"Query validation successful: {len(sanitized)} characters")
        return sanitized
    
    @log_execution_time
    def validate_language(self, language: str) -> str:
        """
        Validate language code.
        
        Args:
            language: Language code to validate
        
        Returns:
            Validated language code
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(language, str):
            raise ValidationError("Language must be a string")
        
        # Check if language is allowed
        if language not in self.rules["allowed_languages"]:
            raise ValidationError(f"Language {language} not supported")
        
        logger.info(f"Language validation successful: {language}")
        return language
    
    @log_execution_time
    def validate_chunk_size(self, chunk_size: int) -> int:
        """
        Validate chunk size.
        
        Args:
            chunk_size: Chunk size to validate
        
        Returns:
            Validated chunk size
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(chunk_size, int):
            raise ValidationError("Chunk size must be an integer")
        
        if chunk_size < self.rules["min_chunk_size"]:
            raise ValidationError(f"Chunk size too small (minimum {self.rules['min_chunk_size']})")
        
        if chunk_size > self.rules["max_chunk_size"]:
            raise ValidationError(f"Chunk size too large (maximum {self.rules['max_chunk_size']})")
        
        logger.info(f"Chunk size validation successful: {chunk_size}")
        return chunk_size
    
    @log_execution_time
    def validate_email(self, email: str) -> str:
        """
        Validate email address.
        
        Args:
            email: Email to validate
        
        Returns:
            Validated email
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(email, str):
            raise ValidationError("Email must be a string")
        
        # Check email format
        if not self.patterns["email"].match(email):
            raise ValidationError("Invalid email format")
        
        # Check length
        if len(email) > 254:  # RFC 5321 limit
            raise ValidationError("Email too long")
        
        logger.info(f"Email validation successful: {email}")
        return email
    
    @log_execution_time
    def validate_phone(self, phone: str) -> str:
        """
        Validate phone number.
        
        Args:
            phone: Phone number to validate
        
        Returns:
            Validated phone number
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(phone, str):
            raise ValidationError("Phone must be a string")
        
        # Check phone format
        if not self.patterns["phone"].match(phone):
            raise ValidationError("Invalid phone number format")
        
        # Check length
        if len(phone) < 10 or len(phone) > 20:
            raise ValidationError("Phone number length invalid")
        
        logger.info(f"Phone validation successful: {phone}")
        return phone
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules."""
        return self.rules.copy()
    
    def update_validation_rules(self, new_rules: Dict[str, Any]) -> None:
        """Update validation rules."""
        self.rules.update(new_rules)
        logger.info(f"Updated validation rules: {new_rules}")
    
    def validate_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all fields in a data dictionary.
        
        Args:
            data: Dictionary of data to validate
        
        Returns:
            Dictionary of validated data
        
        Raises:
            ValidationError: If any validation fails
        """
        validated_data = {}
        
        for field, value in data.items():
            try:
                if field == "text":
                    validated_data[field] = self.validate_text(value, field)
                elif field == "url":
                    validated_data[field] = self.validate_url(value)
                elif field == "query":
                    validated_data[field] = self.validate_query(value)
                elif field == "language":
                    validated_data[field] = self.validate_language(value)
                elif field == "chunk_size":
                    validated_data[field] = self.validate_chunk_size(value)
                elif field == "email":
                    validated_data[field] = self.validate_email(value)
                elif field == "phone":
                    validated_data[field] = self.validate_phone(value)
                else:
                    # Generic text validation for unknown fields
                    validated_data[field] = self.validate_text(str(value), field)
                    
            except Exception as e:
                logger.error(f"Validation failed for field {field}: {e}")
                raise ValidationError(f"Validation failed for field {field}: {e}")
        
        logger.info(f"All fields validated successfully: {list(validated_data.keys())}")
        return validated_data

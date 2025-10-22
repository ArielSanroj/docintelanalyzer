"""
Centralized embedding management with caching and optimization.
Handles embedding generation, caching, and batch processing.
"""

import os
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pickle
import hashlib

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Local imports
from ..exceptions import EmbeddingError
from ..logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class EmbeddingsManager:
    """
    Centralized embedding management with caching and optimization.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./cache/embeddings",
        batch_size: int = 32,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        cache_service: Optional[Any] = None
    ):
        """
        Initialize embeddings manager.
        
        Args:
            model_name: Name of the embedding model
            cache_dir: Directory for embedding cache
            batch_size: Batch size for embedding generation
            enable_cache: Enable embedding caching
            cache_ttl: Cache TTL in seconds
            cache_service: External cache service (Redis/Memory)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.cache_service = cache_service
        
        # Initialize components
        self.model = None
        self.embedding_function = None
        self.local_cache = {}  # Local fallback cache
        self.cache_metadata = {}
        
        # Statistics
        self.stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "redis_hits": 0,
            "local_hits": 0
        }
    
    def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load sentence transformer model
            cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_folder
            )
            
            # Initialize LangChain embedding function
            self.embedding_function = SentenceTransformerEmbeddings(
                model_name=self.model_name
            )
            
            # Load existing cache
            if self.enable_cache:
                self._load_cache()
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Embedding model loading failed: {e}")
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        try:
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            metadata_file = self.cache_dir / "cache_metadata.pkl"
            
            if cache_file.exists() and metadata_file.exists():
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                
                with open(metadata_file, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
                
                # Clean expired entries
                self._clean_expired_cache()
                
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
            self.cache_metadata = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            if not self.enable_cache:
                return
            
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            metadata_file = self.cache_dir / "cache_metadata.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
            
            logger.debug("Cache saved to disk")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _clean_expired_cache(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.cache_metadata.items():
            if current_time - metadata['timestamp'] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_metadata:
                del self.cache_metadata[key]
        
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"embedding:{self.model_name}:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding from cache service or local cache."""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        
        # Try external cache service first (Redis)
        if self.cache_service:
            try:
                cached_value = self.cache_service.get(cache_key)
                if cached_value is not None:
                    self.stats["redis_hits"] += 1
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Redis cache hit: {cache_key}")
                    return np.array(cached_value)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to local cache
        if cache_key in self.local_cache:
            self.stats["local_hits"] += 1
            self.stats["cache_hits"] += 1
            logger.debug(f"Local cache hit: {cache_key}")
            return self.local_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        return None
    
    def _set_cached_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Set cached embedding in cache service and local cache."""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(text)
        
        # Store in external cache service (Redis)
        if self.cache_service:
            try:
                self.cache_service.set(cache_key, embedding.tolist(), ttl=self.cache_ttl)
                logger.debug(f"Stored in Redis cache: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache store error: {e}")
        
        # Store in local cache as fallback
        self.local_cache[cache_key] = embedding
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'model': self.model_name
        }
    
    @log_execution_time
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with optimized batch processing.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (optional)
        
        Returns:
            Array of embeddings
        """
        if not self.model:
            self.initialize()
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        start_time = time.time()
        
        try:
            # Check cache for all texts
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            if self.enable_cache:
                for i, text in enumerate(texts):
                    cached_embedding = self._get_cached_embedding(text)
                    if cached_embedding is not None:
                        cached_embeddings.append(cached_embedding)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
            
            # Generate embeddings for uncached texts with optimized batching
            if uncached_texts:
                logger.info(f"Generating embeddings for {len(uncached_texts)} texts in batches of {batch_size}")
                
                # Process in optimized batches
                all_embeddings = []
                total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
                
                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    
                    logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                    
                    # Optimized batch processing
                    batch_embeddings = self._process_batch(batch_texts)
                    all_embeddings.append(batch_embeddings)
                
                # Combine all embeddings efficiently
                if all_embeddings:
                    uncached_embeddings = np.vstack(all_embeddings)
                else:
                    uncached_embeddings = np.array([])
                
                # Cache new embeddings in batch
                if self.enable_cache:
                    self._cache_embeddings_batch(uncached_texts, uncached_embeddings)
            else:
                uncached_embeddings = np.array([])
            
            # Combine cached and new embeddings efficiently
            embeddings = self._combine_embeddings(
                texts, cached_embeddings, uncached_texts, uncached_embeddings
            )
            
            # Update statistics
            generation_time = time.time() - start_time
            self.stats["embeddings_generated"] += len(texts)
            self.stats["total_generation_time"] += generation_time
            self.stats["average_generation_time"] = (
                self.stats["total_generation_time"] / 
                max(1, self.stats["embeddings_generated"])
            )
            
            # Save cache periodically
            if self.enable_cache and len(self.cache) % 100 == 0:
                self._save_cache()
            
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.3f}s "
                       f"({len(embeddings)/generation_time:.1f} embeddings/sec)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def _process_batch(self, batch_texts: List[str]) -> np.ndarray:
        """
        Process a single batch of texts with optimizations.
        
        Args:
            batch_texts: List of texts in the batch
        
        Returns:
            Batch embeddings
        """
        try:
            # Pre-process texts for better performance
            processed_texts = [self._preprocess_text(text) for text in batch_texts]
            
            # Generate embeddings with optimized settings
            embeddings = self.model.encode(
                processed_texts,
                batch_size=len(processed_texts),  # Process entire batch at once
                show_progress_bar=False,  # Disable progress bar for batch processing
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for better similarity
                device='cpu'  # Use CPU for consistency
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise EmbeddingError(f"Batch processing failed: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (most models have limits)
        max_length = 512  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def _cache_embeddings_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> None:
        """
        Cache embeddings in batch for better performance.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
        """
        try:
            for text, embedding in zip(texts, embeddings):
                self._set_cached_embedding(text, embedding)
            
            logger.debug(f"Cached {len(texts)} embeddings in batch")
            
        except Exception as e:
            logger.error(f"Batch caching failed: {e}")
    
    def _combine_embeddings(
        self,
        all_texts: List[str],
        cached_embeddings: List[np.ndarray],
        uncached_texts: List[str],
        uncached_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Efficiently combine cached and uncached embeddings.
        
        Args:
            all_texts: All input texts
            cached_embeddings: Cached embedding results
            uncached_texts: Texts that weren't cached
            uncached_embeddings: Newly generated embeddings
        
        Returns:
            Combined embeddings array
        """
        if not cached_embeddings and len(uncached_embeddings) == 0:
            return np.array([])
        
        if not cached_embeddings:
            return uncached_embeddings
        
        if len(uncached_embeddings) == 0:
            return np.array(cached_embeddings)
        
        # Reconstruct full embedding array efficiently
        embedding_dim = self.model.get_sentence_embedding_dimension()
        full_embeddings = np.zeros((len(all_texts), embedding_dim))
        
        # Create mapping for efficient lookup
        uncached_text_set = set(uncached_texts)
        uncached_idx = 0
        cached_idx = 0
        
        for i, text in enumerate(all_texts):
            if text in uncached_text_set:
                full_embeddings[i] = uncached_embeddings[uncached_idx]
                uncached_idx += 1
            else:
                full_embeddings[i] = cached_embeddings[cached_idx]
                cached_idx += 1
        
        return full_embeddings
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.generate_embeddings([text1, text2])
            if len(embeddings) < 2:
                return 0.0
            
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        # Clear local cache
        self.local_cache.clear()
        self.cache_metadata.clear()
        
        # Clear external cache service
        if self.cache_service:
            try:
                # Clear all embedding cache keys
                self.cache_service.clear()
                logger.info("External cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear external cache: {e}")
        
        # Remove cache files
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        metadata_file = self.cache_dir / "cache_metadata.pkl"
        
        if cache_file.exists():
            cache_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / max(1, total_requests)
        
        stats = {
            "local_cache_size": len(self.local_cache),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "redis_hits": self.stats["redis_hits"],
            "local_hits": self.stats["local_hits"],
            "hit_rate": hit_rate,
            "embeddings_generated": self.stats["embeddings_generated"],
            "average_generation_time": self.stats["average_generation_time"],
            "cache_enabled": self.enable_cache,
            "cache_service_available": self.cache_service is not None
        }
        
        # Add external cache stats if available
        if self.cache_service:
            try:
                external_stats = self.cache_service.get_stats()
                stats["external_cache"] = external_stats
            except Exception as e:
                stats["external_cache_error"] = str(e)
        
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        if not self.model:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "cache_enabled": self.enable_cache,
            "cache_size": len(self.cache),
            "batch_size": self.batch_size
        }
    
    def __del__(self):
        """Save cache when object is destroyed."""
        try:
            if self.enable_cache and self.cache:
                self._save_cache()
        except Exception:
            pass  # Ignore errors during cleanup

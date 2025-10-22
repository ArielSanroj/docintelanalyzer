"""
Cache service with Redis and in-memory caching.
Provides unified caching interface for the application.
"""

import json
import time
import logging
from typing import Any, Optional, Dict, List, Union
from abc import ABC, abstractmethod

# Local imports
from ...core.exceptions import CacheError, CacheConnectionError, CacheOperationError
from ...core.logging_config import get_logger, log_execution_time

logger = get_logger(__name__)

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if entry.get('ttl') and time.time() > entry['ttl']:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Remove oldest entries if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Calculate TTL
            expire_time = None
            if ttl:
                expire_time = time.time() + ttl
            
            self.cache[key] = {
                'value': value,
                'ttl': expire_time,
                'created': time.time()
            }
            
            self.access_times[key] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        
        # Check TTL
        if entry.get('ttl') and time.time() > entry['ttl']:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return False
        
        return True
    
    def clear(self) -> bool:
        """Clear all memory cache entries."""
        self.cache.clear()
        self.access_times.clear()
        return True
    
    def _evict_oldest(self) -> None:
        """Evict oldest entries from cache."""
        if not self.access_times:
            return
        
        # Find oldest entry
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need to track hits/misses
            "backend": "memory"
        }


class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str, password: Optional[str] = None, db: int = 0):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            password: Redis password
            db: Redis database number
        """
        if not REDIS_AVAILABLE:
            raise CacheError("Redis not available. Install redis package.")
        
        self.redis_url = redis_url
        self.password = password
        self.db = db
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    def _connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self.client = redis.from_url(
                self.redis_url,
                password=self.password,
                db=self.db,
                decode_responses=True
            )
            
            # Test connection
            self.client.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheConnectionError(f"Redis connection failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._connected:
            self._connect()
        
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._connected:
            self._connect()
        
        try:
            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            result = self.client.set(key, serialized_value, ex=ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self._connected:
            self._connect()
        
        try:
            result = self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._connected:
            self._connect()
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all Redis cache entries."""
        if not self._connected:
            self._connect()
        
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self._connected:
            return {"backend": "redis", "connected": False}
        
        try:
            info = self.client.info()
            return {
                "backend": "redis",
                "connected": True,
                "used_memory": info.get("used_memory_human"),
                "keys": info.get("db0", {}).get("keys", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"backend": "redis", "connected": False, "error": str(e)}


class CacheService:
    """
    Unified cache service with multiple backends.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        enable_cache: bool = True,
        ttl: int = 3600,
        max_memory_size: int = 1000
    ):
        """
        Initialize cache service.
        
        Args:
            redis_url: Redis connection URL
            enable_cache: Enable caching
            ttl: Default TTL in seconds
            max_memory_size: Maximum memory cache size
        """
        self.enable_cache = enable_cache
        self.default_ttl = ttl
        self.redis_url = redis_url
        
        # Initialize backends
        self.memory_backend = MemoryCacheBackend(max_memory_size)
        self.redis_backend: Optional[RedisCacheBackend] = None
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        # Initialize Redis if available
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_backend = RedisCacheBackend(redis_url)
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_backend = None
    
    def _get_backend(self) -> CacheBackend:
        """Get the primary cache backend."""
        if self.redis_backend:
            return self.redis_backend
        return self.memory_backend
    
    @log_execution_time
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if not self.enable_cache:
            return None
        
        try:
            backend = self._get_backend()
            value = backend.get(key)
            
            if value is not None:
                self.stats["hits"] += 1
                logger.debug(f"Cache hit: {key}")
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
            
            return value
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    @log_execution_time
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        if not self.enable_cache:
            return False
        
        try:
            backend = self._get_backend()
            ttl = ttl or self.default_ttl
            success = backend.set(key, value, ttl)
            
            if success:
                self.stats["sets"] += 1
                logger.debug(f"Cache set: {key}")
            else:
                self.stats["errors"] += 1
                logger.warning(f"Cache set failed: {key}")
            
            return success
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    @log_execution_time
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful
        """
        if not self.enable_cache:
            return False
        
        try:
            backend = self._get_backend()
            success = backend.delete(key)
            
            if success:
                self.stats["deletes"] += 1
                logger.debug(f"Cache delete: {key}")
            
            return success
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enable_cache:
            return False
        
        try:
            backend = self._get_backend()
            return backend.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            # Clear both backends
            memory_success = self.memory_backend.clear()
            redis_success = True
            
            if self.redis_backend:
                redis_success = self.redis_backend.clear()
            
            success = memory_success and redis_success
            if success:
                logger.info("Cache cleared successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(1, total_requests)
        
        stats = {
            "enabled": self.enable_cache,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "errors": self.stats["errors"],
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
        
        # Add backend-specific stats
        if self.redis_backend:
            stats["redis"] = self.redis_backend.get_stats()
        else:
            stats["memory"] = self.memory_backend.get_stats()
        
        return stats
    
    def health_check(self) -> bool:
        """Check cache health."""
        try:
            # Test with a temporary key
            test_key = "health_check_test"
            test_value = "test"
            
            # Test set and get
            if not self.set(test_key, test_value, ttl=1):
                return False
            
            retrieved = self.get(test_key)
            if retrieved != test_value:
                return False
            
            # Clean up
            self.delete(test_key)
            return True
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False


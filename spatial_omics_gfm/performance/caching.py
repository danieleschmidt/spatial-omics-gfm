"""
Advanced caching system for spatial omics computations.

Provides memory caching, disk caching, and distributed caching
for expensive operations like graph construction and model inference.
"""

import os
import pickle
import hashlib
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union, List
from functools import wraps
import json
import gzip
import logging

logger = logging.getLogger(__name__)


class CacheKey:
    """Smart cache key generation with hash-based uniqueness."""
    
    @staticmethod
    def generate(
        func_name: str,
        args: tuple,
        kwargs: dict,
        include_types: bool = True
    ) -> str:
        """
        Generate a unique cache key for function call.
        
        Args:
            func_name: Name of the function
            args: Function arguments
            kwargs: Function keyword arguments
            include_types: Whether to include argument types in key
            
        Returns:
            Unique cache key string
        """
        # Convert arguments to hashable format
        key_data = {
            "function": func_name,
            "args": CacheKey._serialize_args(args, include_types),
            "kwargs": CacheKey._serialize_args(kwargs, include_types)
        }
        
        # Create deterministic hash
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
        
        return f"{func_name}_{key_hash}"
    
    @staticmethod
    def _serialize_args(args: Union[tuple, dict], include_types: bool) -> Union[list, dict]:
        """Serialize arguments for hashing."""
        if isinstance(args, dict):
            return {k: CacheKey._serialize_value(v, include_types) for k, v in args.items()}
        else:
            return [CacheKey._serialize_value(arg, include_types) for arg in args]
    
    @staticmethod
    def _serialize_value(value: Any, include_types: bool) -> Any:
        """Serialize a single value for hashing."""
        import numpy as np
        
        if value is None:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return (type(value).__name__, value) if include_types else value
        elif isinstance(value, (list, tuple)):
            serialized = [CacheKey._serialize_value(v, include_types) for v in value]
            return (type(value).__name__, serialized) if include_types else serialized
        elif isinstance(value, dict):
            serialized = {k: CacheKey._serialize_value(v, include_types) for k, v in value.items()}
            return ("dict", serialized) if include_types else serialized
        elif isinstance(value, np.ndarray):
            # For arrays, use shape, dtype, and hash of data
            array_info = {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "hash": hashlib.sha256(value.tobytes()).hexdigest()[:16]
            }
            return ("ndarray", array_info) if include_types else array_info
        else:
            # For other objects, try to use string representation
            str_repr = str(value)[:100]  # Limit length
            return (type(value).__name__, str_repr) if include_types else str_repr


class MemoryCache:
    """In-memory LRU cache with size and time limits."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 1024,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to cache
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._current_memory = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and time.time() - entry["timestamp"] > self.ttl_seconds:
                self._remove_entry(key)
                return None
            
            # Update access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            
            entry["access_count"] += 1
            entry["last_access"] = time.time()
            
            return entry["value"]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        import sys
        
        with self._lock:
            # Calculate size of new entry
            entry_size = sys.getsizeof(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check memory limit
            while (self._current_memory + entry_size > self.max_memory_bytes and 
                   self._access_order):
                oldest_key = self._access_order[0]
                self._remove_entry(oldest_key)
            
            # Check size limit
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order[0]
                self._remove_entry(oldest_key)
            
            # Add new entry
            now = time.time()
            self._cache[key] = {
                "value": value,
                "size": entry_size,
                "timestamp": now,
                "last_access": now,
                "access_count": 1
            }
            
            self._access_order.append(key)
            self._current_memory += entry_size
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry["size"]
            
            if key in self._access_order:
                self._access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(entry["access_count"] for entry in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "total_accesses": total_accesses,
                "hit_ratio": 0.0  # Would need to track misses to calculate
            }


class DiskCache:
    """Persistent disk cache with compression and metadata."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: int = 10,
        compression: bool = True,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in GB
            compression: Whether to compress cached data
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.compression = compression
        self.ttl_seconds = ttl_seconds
        
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        cache_file = self.cache_dir / f"{key}.cache"
        
        if not cache_file.exists():
            return None
        
        # Check metadata
        if key in self.metadata:
            entry_meta = self.metadata[key]
            
            # Check TTL
            if (self.ttl_seconds and 
                time.time() - entry_meta["timestamp"] > self.ttl_seconds):
                self._remove_entry(key)
                return None
        
        try:
            # Load data
            if self.compression:
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            # Update metadata
            if key in self.metadata:
                self.metadata[key]["access_count"] += 1
                self.metadata[key]["last_access"] = time.time()
                self._save_metadata()
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            self._remove_entry(key)
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in disk cache."""
        cache_file = self.cache_dir / f"{key}.cache"
        
        try:
            # Save data
            if self.compression:
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            now = time.time()
            
            self.metadata[key] = {
                "file_size": file_size,
                "timestamp": now,
                "last_access": now,
                "access_count": 1
            }
            
            # Check size limit and cleanup if needed
            self._cleanup_if_needed()
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to save cache entry {key}: {e}")
            if cache_file.exists():
                cache_file.unlink()
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        cache_file = self.cache_dir / f"{key}.cache"
        
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
    
    def _cleanup_if_needed(self) -> None:
        """Remove old entries if cache is too large."""
        total_size = sum(meta["file_size"] for meta in self.metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1]["last_access"]
        )
        
        # Remove entries until under size limit
        for key, meta in sorted_entries:
            if total_size <= self.max_size_bytes:
                break
            
            self._remove_entry(key)
            total_size -= meta["file_size"]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self.metadata.keys()):
            self._remove_entry(key)
        
        self.metadata.clear()
        self._save_metadata()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(meta["file_size"] for meta in self.metadata.values())
        total_accesses = sum(meta["access_count"] for meta in self.metadata.values())
        
        return {
            "size": len(self.metadata),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024),
            "total_accesses": total_accesses,
            "cache_dir": str(self.cache_dir)
        }


class CacheManager:
    """Unified cache manager with multiple cache backends."""
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        disk_cache: Optional[DiskCache] = None,
        use_memory: bool = True,
        use_disk: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            memory_cache: Optional memory cache instance
            disk_cache: Optional disk cache instance
            use_memory: Whether to use memory caching
            use_disk: Whether to use disk caching
        """
        self.use_memory = use_memory and memory_cache is not None
        self.use_disk = use_disk and disk_cache is not None
        
        self.memory_cache = memory_cache if self.use_memory else None
        self.disk_cache = disk_cache if self.use_disk else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        if self.use_memory:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try disk cache
        if self.use_disk:
            value = self.disk_cache.get(key)
            if value is not None:
                # Populate memory cache for future access
                if self.use_memory:
                    self.memory_cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache (both memory and disk)."""
        if self.use_memory:
            self.memory_cache.put(key, value)
        
        if self.use_disk:
            self.disk_cache.put(key, value)
    
    def clear(self) -> None:
        """Clear all caches."""
        if self.use_memory:
            self.memory_cache.clear()
        
        if self.use_disk:
            self.disk_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        stats = {}
        
        if self.use_memory:
            stats["memory"] = self.memory_cache.stats()
        
        if self.use_disk:
            stats["disk"] = self.disk_cache.stats()
        
        return stats


def cached_computation(
    cache_manager: CacheManager,
    ttl_seconds: Optional[int] = None,
    cache_key_func: Optional[Callable] = None
):
    """
    Decorator for caching expensive computations.
    
    Args:
        cache_manager: Cache manager instance
        ttl_seconds: Time-to-live for cached results
        cache_key_func: Custom function to generate cache keys
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = CacheKey.generate(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Compute result
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.put(cache_key, result)
            
            return result
        
        # Add cache control methods
        wrapper.cache_manager = cache_manager
        wrapper.clear_cache = lambda: cache_manager.clear()
        
        return wrapper
    
    return decorator


# Default cache instances
_default_memory_cache = MemoryCache(max_size=500, max_memory_mb=512)
_default_disk_cache = DiskCache(cache_dir="/tmp/spatial_omics_cache", max_size_gb=5)
_default_cache_manager = CacheManager(_default_memory_cache, _default_disk_cache)


def get_default_cache_manager() -> CacheManager:
    """Get the default cache manager instance."""
    return _default_cache_manager
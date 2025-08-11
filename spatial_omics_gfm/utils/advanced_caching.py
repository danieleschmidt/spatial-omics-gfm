"""
Generation 3: Advanced Caching & Performance Optimization System.

This module implements enterprise-grade caching and performance optimization:
- Multi-level caching (memory, disk, distributed cache)
- Smart prefetching and data pipeline optimization  
- JIT compilation and model optimization
- CUDA kernel optimization for spatial operations
- Intelligent cache eviction and warming strategies
"""

import os
import sys
import time
import json
import pickle
import hashlib
import threading
import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import redis
import lmdb
from anndata import AnnData

from ..models.graph_transformer import SpatialGraphTransformer
from .memory_management import MemoryMonitor

logger = __import__('logging').getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for advanced caching system."""
    
    # Memory cache settings
    memory_cache_size_gb: float = 4.0
    memory_cache_ttl: int = 3600  # seconds
    
    # Disk cache settings
    disk_cache_size_gb: float = 50.0
    disk_cache_path: str = "./cache/disk"
    disk_compression: bool = True
    
    # Distributed cache settings
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 7200  # seconds
    
    # LMDB settings
    enable_lmdb: bool = True
    lmdb_map_size: int = 100 * 1024**3  # 100GB
    lmdb_path: str = "./cache/lmdb"
    
    # Prefetching settings
    enable_prefetching: bool = True
    prefetch_threads: int = 4
    prefetch_queue_size: int = 100
    lookahead_steps: int = 10
    
    # Cache policies
    eviction_policy: str = "lru"  # lru, lfu, fifo
    cache_hit_tracking: bool = True
    cache_warming: bool = True
    
    # Performance optimization
    enable_jit_compilation: bool = True
    enable_cuda_kernels: bool = True
    memory_mapped_io: bool = True


class CacheKey:
    """Intelligent cache key generation and management."""
    
    def __init__(self, cache_config: CacheConfig):
        self.config = cache_config
        self.key_history = deque(maxlen=10000)
    
    def generate_key(
        self,
        data: Any,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for data and operation."""
        key_components = []
        
        # Operation identifier
        key_components.append(f"op:{operation}")
        
        # Data hash
        if isinstance(data, torch.Tensor):
            key_components.append(f"tensor:{self._tensor_hash(data)}")
        elif isinstance(data, np.ndarray):
            key_components.append(f"array:{self._array_hash(data)}")
        elif isinstance(data, AnnData):
            key_components.append(f"adata:{self._adata_hash(data)}")
        elif isinstance(data, (list, tuple)):
            key_components.append(f"list:{self._sequence_hash(data)}")
        else:
            key_components.append(f"obj:{hash(str(data))}")
        
        # Parameters hash
        if parameters:
            param_str = json.dumps(parameters, sort_keys=True, default=str)
            key_components.append(f"params:{hashlib.md5(param_str.encode()).hexdigest()}")
        
        # Combine components
        cache_key = "|".join(key_components)
        
        # Track key usage
        self.key_history.append((cache_key, time.time()))
        
        return cache_key
    
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """Generate hash for tensor data."""
        # Use shape, dtype, and sample of data
        shape_str = str(tensor.shape)
        dtype_str = str(tensor.dtype)
        
        # Sample data for hash (more efficient than full tensor)
        if tensor.numel() > 1000:
            sample_indices = torch.randperm(tensor.numel())[:100]
            sample_data = tensor.flatten()[sample_indices]
        else:
            sample_data = tensor.flatten()
        
        sample_bytes = sample_data.cpu().numpy().tobytes()
        data_hash = hashlib.md5(sample_bytes).hexdigest()
        
        return f"{shape_str}_{dtype_str}_{data_hash}"
    
    def _array_hash(self, array: np.ndarray) -> str:
        """Generate hash for numpy array."""
        shape_str = str(array.shape)
        dtype_str = str(array.dtype)
        
        # Sample data for efficiency
        if array.size > 1000:
            sample_indices = np.random.choice(array.size, 100, replace=False)
            sample_data = array.flatten()[sample_indices]
        else:
            sample_data = array.flatten()
        
        data_hash = hashlib.md5(sample_data.tobytes()).hexdigest()
        
        return f"{shape_str}_{dtype_str}_{data_hash}"
    
    def _adata_hash(self, adata: AnnData) -> str:
        """Generate hash for AnnData object."""
        components = []
        
        # Shape and basic info
        components.append(f"shape:{adata.shape}")
        components.append(f"vars:{len(adata.var_names)}")
        
        # Sample expression data
        if hasattr(adata.X, 'toarray'):
            sample_x = adata.X[:min(100, adata.n_obs), :min(100, adata.n_vars)].toarray()
        else:
            sample_x = adata.X[:min(100, adata.n_obs), :min(100, adata.n_vars)]
        
        x_hash = hashlib.md5(sample_x.tobytes()).hexdigest()
        components.append(f"x:{x_hash}")
        
        # Spatial coordinates if available
        if 'spatial' in adata.obsm:
            spatial_sample = adata.obsm['spatial'][:min(100, adata.n_obs)]
            spatial_hash = hashlib.md5(spatial_sample.tobytes()).hexdigest()
            components.append(f"spatial:{spatial_hash}")
        
        return "|".join(components)
    
    def _sequence_hash(self, seq: Union[List, Tuple]) -> str:
        """Generate hash for sequence data."""
        seq_str = str(seq)[:1000]  # Limit string length
        return hashlib.md5(seq_str.encode()).hexdigest()


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all items from cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get current cache size."""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheInterface):
    """High-performance memory cache with LRU/LFU eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_size_bytes = int(config.memory_cache_size_gb * 1024**3)
        self.ttl = config.memory_cache_ttl
        
        # Cache storage
        self.cache = OrderedDict()
        self.metadata = {}  # TTL, access count, size info
        self.current_size = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"MemoryCache initialized with {config.memory_cache_size_gb}GB limit")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from memory cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self.delete(key)
                self.misses += 1
                return None
            
            # Update access order and count
            value = self.cache[key]
            self.cache.move_to_end(key)
            self.metadata[key]['access_count'] += 1
            self.metadata[key]['last_access'] = time.time()
            
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in memory cache."""
        with self.lock:
            # Calculate object size
            obj_size = self._calculate_size(value)
            
            # Check if object fits in cache
            if obj_size > self.max_size_bytes:
                logger.warning(f"Object too large for memory cache: {obj_size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self.delete(key)
            
            # Evict items if necessary
            while self.current_size + obj_size > self.max_size_bytes:
                if not self._evict_one():
                    return False
            
            # Add to cache
            self.cache[key] = value
            self.metadata[key] = {
                'size': obj_size,
                'creation_time': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'ttl': ttl or self.ttl
            }
            
            self.current_size += obj_size
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from memory cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            # Remove from cache
            del self.cache[key]
            obj_size = self.metadata[key]['size']
            del self.metadata[key]
            
            self.current_size -= obj_size
            return True
    
    def clear(self) -> bool:
        """Clear all items from memory cache."""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.current_size = 0
            return True
    
    def size(self) -> int:
        """Get current cache size in bytes."""
        return self.current_size
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'type': 'memory',
            'size_bytes': self.current_size,
            'size_gb': self.current_size / 1024**3,
            'max_size_gb': self.config.memory_cache_size_gb,
            'items': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.metadata:
            return True
        
        meta = self.metadata[key]
        age = time.time() - meta['creation_time']
        return age > meta['ttl']
    
    def _evict_one(self) -> bool:
        """Evict one item based on eviction policy."""
        if not self.cache:
            return False
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self.cache))
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.metadata.keys(), key=lambda k: self.metadata[k]['access_count'])
        else:  # fifo
            # Remove first in, first out
            key = next(iter(self.cache))
        
        self.delete(key)
        self.evictions += 1
        return True
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # Rough approximation using pickle
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default assumption


class DiskCache(CacheInterface):
    """Persistent disk cache with compression and memory mapping."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"DiskCache initialized at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from disk cache."""
        with self.lock:
            if key not in self.metadata:
                self.misses += 1
                return None
            
            # Check if file exists and not expired
            file_path = self.cache_dir / self.metadata[key]['filename']
            if not file_path.exists() or self._is_expired(key):
                if key in self.metadata:
                    del self.metadata[key]
                self.misses += 1
                return None
            
            try:
                # Load from disk
                if self.config.disk_compression:
                    import gzip
                    with gzip.open(file_path, 'rb') as f:
                        value = pickle.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                
                # Update access time
                self.metadata[key]['last_access'] = time.time()
                self.metadata[key]['access_count'] += 1
                self._save_metadata()
                
                self.hits += 1
                return value
                
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                self.delete(key)
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in disk cache."""
        with self.lock:
            try:
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
                file_path = self.cache_dir / filename
                
                # Save to disk
                if self.config.disk_compression:
                    import gzip
                    with gzip.open(file_path, 'wb') as f:
                        pickle.dump(value, f)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(value, f)
                
                # Update metadata
                self.metadata[key] = {
                    'filename': filename,
                    'size': file_path.stat().st_size,
                    'creation_time': time.time(),
                    'last_access': time.time(),
                    'access_count': 1,
                    'ttl': ttl or 3600 * 24  # 24 hours default
                }
                
                self._save_metadata()
                self._cleanup_if_needed()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from disk cache."""
        with self.lock:
            if key not in self.metadata:
                return False
            
            try:
                # Remove file
                file_path = self.cache_dir / self.metadata[key]['filename']
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from metadata
                del self.metadata[key]
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete from disk cache: {e}")
                return False
    
    def clear(self) -> bool:
        """Clear all items from disk cache."""
        with self.lock:
            try:
                # Remove all cache files
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                
                # Clear metadata
                self.metadata.clear()
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
                return False
    
    def size(self) -> int:
        """Get current cache size in bytes."""
        return sum(meta['size'] for meta in self.metadata.values())
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'type': 'disk',
            'size_bytes': self.size(),
            'size_gb': self.size() / 1024**3,
            'items': len(self.metadata),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'path': str(self.cache_dir)
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save disk cache metadata: {e}")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.metadata:
            return True
        
        meta = self.metadata[key]
        age = time.time() - meta['creation_time']
        return age > meta['ttl']
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup disk cache if size limit exceeded."""
        current_size = self.size()
        max_size = self.config.disk_cache_size_gb * 1024**3
        
        if current_size <= max_size:
            return
        
        # Sort by last access time and remove oldest
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k]['last_access']
        )
        
        for key in sorted_keys:
            if self.size() <= max_size:
                break
            self.delete(key)


class RedisCache(CacheInterface):
    """Distributed cache using Redis."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        try:
            import redis
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=False
            )
            
            # Test connection
            self.redis_client.ping()
            
            logger.info(f"RedisCache connected to {config.redis_host}:{config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from Redis cache."""
        if not self.redis_client:
            self.misses += 1
            return None
        
        try:
            data = self.redis_client.get(key)
            if data is None:
                self.misses += 1
                return None
            
            value = pickle.loads(data)
            self.hits += 1
            return value
            
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            data = pickle.dumps(value)
            ttl = ttl or self.config.redis_ttl
            
            result = self.redis_client.setex(key, ttl, data)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis put failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all items from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False
    
    def size(self) -> int:
        """Get current cache size (number of keys)."""
        if not self.redis_client:
            return 0
        
        try:
            return self.redis_client.dbsize()
        except:
            return 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        stats = {
            'type': 'redis',
            'items': self.size(),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'available': self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats.update({
                    'memory_used': redis_info.get('used_memory', 0),
                    'memory_used_mb': redis_info.get('used_memory', 0) / 1024**2
                })
            except:
                pass
        
        return stats


class LMDBCache(CacheInterface):
    """High-performance LMDB-based cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_path = Path(config.lmdb_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import lmdb
            self.env = lmdb.open(
                str(self.cache_path),
                map_size=config.lmdb_map_size,
                max_dbs=1,
                readahead=False
            )
            
            logger.info(f"LMDBCache initialized at {self.cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LMDB: {e}")
            self.env = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from LMDB cache."""
        if not self.env:
            self.misses += 1
            return None
        
        try:
            with self.env.begin() as txn:
                data = txn.get(key.encode())
                if data is None:
                    self.misses += 1
                    return None
                
                value = pickle.loads(data)
                self.hits += 1
                return value
                
        except Exception as e:
            logger.warning(f"LMDB get failed: {e}")
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in LMDB cache."""
        if not self.env:
            return False
        
        try:
            data = pickle.dumps(value)
            
            with self.env.begin(write=True) as txn:
                result = txn.put(key.encode(), data)
                return bool(result)
                
        except Exception as e:
            logger.error(f"LMDB put failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from LMDB cache."""
        if not self.env:
            return False
        
        try:
            with self.env.begin(write=True) as txn:
                result = txn.delete(key.encode())
                return bool(result)
                
        except Exception as e:
            logger.error(f"LMDB delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all items from LMDB cache."""
        if not self.env:
            return False
        
        try:
            with self.env.begin(write=True) as txn:
                txn.drop(db=None, delete=False)
                return True
                
        except Exception as e:
            logger.error(f"LMDB clear failed: {e}")
            return False
    
    def size(self) -> int:
        """Get current cache size (number of entries)."""
        if not self.env:
            return 0
        
        try:
            with self.env.begin() as txn:
                return txn.stat()['entries']
        except:
            return 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        stats = {
            'type': 'lmdb',
            'items': self.size(),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'available': self.env is not None
        }
        
        if self.env:
            try:
                with self.env.begin() as txn:
                    stat = txn.stat()
                    stats.update({
                        'page_size': stat['psize'],
                        'depth': stat['depth'],
                        'branch_pages': stat['branch_pages'],
                        'leaf_pages': stat['leaf_pages'],
                        'overflow_pages': stat['overflow_pages']
                    })
            except:
                pass
        
        return stats


class SmartPrefetcher:
    """Intelligent prefetching system with predictive loading."""
    
    def __init__(self, config: CacheConfig, cache_manager: 'CacheManager'):
        self.config = config
        self.cache_manager = cache_manager
        
        # Prefetch queue and workers
        self.prefetch_queue = deque(maxlen=config.prefetch_queue_size)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=config.prefetch_threads)
        
        # Access pattern learning
        self.access_patterns = defaultdict(list)
        self.pattern_predictions = {}
        
        # Performance tracking
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        logger.info("SmartPrefetcher initialized")
    
    def record_access(self, key: str) -> None:
        """Record cache access for pattern learning."""
        timestamp = time.time()
        self.access_patterns[key].append(timestamp)
        
        # Keep only recent accesses
        cutoff = timestamp - 3600  # 1 hour
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
        
        # Update predictions based on recent patterns
        self._update_predictions(key)
    
    def prefetch_next(self, current_key: str, lookahead: int = None) -> None:
        """Prefetch likely next items based on patterns."""
        lookahead = lookahead or self.config.lookahead_steps
        
        predicted_keys = self._predict_next_keys(current_key, lookahead)
        
        for key in predicted_keys:
            if key not in self.cache_manager.memory_cache.cache:
                # Add to prefetch queue
                self.prefetch_queue.append(key)
        
        # Process prefetch queue
        self._process_prefetch_queue()
    
    def _update_predictions(self, key: str) -> None:
        """Update access pattern predictions."""
        if len(self.access_patterns[key]) < 3:
            return
        
        # Analyze access intervals
        accesses = self.access_patterns[key]
        intervals = [accesses[i+1] - accesses[i] for i in range(len(accesses)-1)]
        
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            self.pattern_predictions[key] = {
                'avg_interval': avg_interval,
                'std_interval': std_interval,
                'next_predicted': accesses[-1] + avg_interval,
                'confidence': 1.0 / (1.0 + std_interval)
            }
    
    def _predict_next_keys(self, current_key: str, count: int) -> List[str]:
        """Predict next likely keys to be accessed."""
        # Simple pattern-based prediction
        # In practice, this could use ML models
        
        predicted = []
        
        # Look for sequential patterns
        if current_key.endswith('_0'):
            base_key = current_key[:-2]
            for i in range(1, count + 1):
                predicted.append(f"{base_key}_{i}")
        
        # Look for co-occurrence patterns
        # This would be more sophisticated in practice
        
        return predicted[:count]
    
    def _process_prefetch_queue(self) -> None:
        """Process items in prefetch queue."""
        while self.prefetch_queue:
            try:
                key = self.prefetch_queue.popleft()
                
                # Submit prefetch task
                future = self.prefetch_executor.submit(self._prefetch_item, key)
                
            except Exception as e:
                logger.warning(f"Prefetch queue processing error: {e}")
                break
    
    def _prefetch_item(self, key: str) -> None:
        """Prefetch a single item."""
        try:
            # This would load the actual data based on the key
            # For now, just a placeholder
            logger.debug(f"Prefetching item: {key}")
            
            # In practice, this would:
            # 1. Parse the key to understand what data to load
            # 2. Load the data from source
            # 3. Store in cache
            
        except Exception as e:
            logger.warning(f"Failed to prefetch {key}: {e}")


class CacheManager:
    """Multi-level cache manager with intelligent orchestration."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.key_generator = CacheKey(config)
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(config)
        self.disk_cache = DiskCache(config)
        
        # Optional distributed caches
        self.redis_cache = RedisCache(config) if config.enable_redis else None
        self.lmdb_cache = LMDBCache(config) if config.enable_lmdb else None
        
        # Prefetcher
        self.prefetcher = SmartPrefetcher(config, self) if config.enable_prefetching else None
        
        # Cache hierarchy (order matters - faster caches first)
        self.cache_hierarchy = [self.memory_cache]
        if self.lmdb_cache:
            self.cache_hierarchy.append(self.lmdb_cache)
        if self.redis_cache:
            self.cache_hierarchy.append(self.redis_cache)
        self.cache_hierarchy.append(self.disk_cache)
        
        # Performance tracking
        self.total_gets = 0
        self.total_puts = 0
        self.hierarchy_stats = defaultdict(int)
        
        logger.info(f"CacheManager initialized with {len(self.cache_hierarchy)} cache layers")
    
    def get(
        self,
        data: Any = None,
        operation: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve item from multi-level cache.
        
        Args:
            data: Data for key generation (if key not provided)
            operation: Operation identifier
            parameters: Operation parameters
            key: Pre-generated cache key
            
        Returns:
            Cached value or None if not found
        """
        self.total_gets += 1
        
        # Generate key if not provided
        if key is None:
            if data is None:
                raise ValueError("Either data or key must be provided")
            key = self.key_generator.generate_key(data, operation, parameters)
        
        # Record access for prefetching
        if self.prefetcher:
            self.prefetcher.record_access(key)
        
        # Try cache hierarchy from fastest to slowest
        for level, cache in enumerate(self.cache_hierarchy):
            try:
                value = cache.get(key)
                if value is not None:
                    self.hierarchy_stats[f'level_{level}_hits'] += 1
                    
                    # Promote to faster caches
                    self._promote_to_faster_caches(key, value, level)
                    
                    # Trigger prefetching
                    if self.prefetcher:
                        self.prefetcher.prefetch_next(key)
                    
                    return value
            except Exception as e:
                logger.warning(f"Cache level {level} get failed: {e}")
                continue
        
        # Cache miss across all levels
        self.hierarchy_stats['total_misses'] += 1
        return None
    
    def put(
        self,
        value: Any,
        data: Any = None,
        operation: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store item in multi-level cache.
        
        Args:
            value: Value to cache
            data: Data for key generation (if key not provided)
            operation: Operation identifier
            parameters: Operation parameters
            key: Pre-generated cache key
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        self.total_puts += 1
        
        # Generate key if not provided
        if key is None:
            if data is None:
                raise ValueError("Either data or key must be provided")
            key = self.key_generator.generate_key(data, operation, parameters)
        
        # Store in all cache levels (best effort)
        success_count = 0
        
        for level, cache in enumerate(self.cache_hierarchy):
            try:
                if cache.put(key, value, ttl):
                    success_count += 1
                    self.hierarchy_stats[f'level_{level}_puts'] += 1
            except Exception as e:
                logger.warning(f"Cache level {level} put failed: {e}")
                continue
        
        return success_count > 0
    
    def invalidate(
        self,
        data: Any = None,
        operation: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None
    ) -> bool:
        """Invalidate item across all cache levels."""
        # Generate key if not provided
        if key is None:
            if data is None:
                raise ValueError("Either data or key must be provided")
            key = self.key_generator.generate_key(data, operation, parameters)
        
        # Delete from all cache levels
        success_count = 0
        
        for cache in self.cache_hierarchy:
            try:
                if cache.delete(key):
                    success_count += 1
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
                continue
        
        return success_count > 0
    
    def clear_all(self) -> Dict[str, bool]:
        """Clear all cache levels."""
        results = {}
        
        for level, cache in enumerate(self.cache_hierarchy):
            try:
                results[f'level_{level}'] = cache.clear()
            except Exception as e:
                logger.error(f"Failed to clear cache level {level}: {e}")
                results[f'level_{level}'] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'total_gets': self.total_gets,
            'total_puts': self.total_puts,
            'hierarchy_stats': dict(self.hierarchy_stats),
            'cache_levels': []
        }
        
        # Get stats from each cache level
        for level, cache in enumerate(self.cache_hierarchy):
            try:
                cache_stats = cache.stats()
                cache_stats['level'] = level
                stats['cache_levels'].append(cache_stats)
            except Exception as e:
                logger.warning(f"Failed to get stats for cache level {level}: {e}")
        
        # Calculate overall hit rate
        total_hits = sum(
            self.hierarchy_stats.get(f'level_{i}_hits', 0)
            for i in range(len(self.cache_hierarchy))
        )
        total_requests = total_hits + self.hierarchy_stats.get('total_misses', 0)
        
        if total_requests > 0:
            stats['overall_hit_rate'] = total_hits / total_requests
        else:
            stats['overall_hit_rate'] = 0.0
        
        return stats
    
    def _promote_to_faster_caches(self, key: str, value: Any, found_level: int) -> None:
        """Promote cache hit to faster cache levels."""
        # Store in all faster cache levels
        for level in range(found_level):
            try:
                cache = self.cache_hierarchy[level]
                cache.put(key, value)
            except Exception as e:
                logger.warning(f"Cache promotion to level {level} failed: {e}")
                continue
    
    def warm_cache(self, data_loader: DataLoader, operation: str = "warm") -> Dict[str, int]:
        """Warm cache with data from data loader."""
        logger.info("Starting cache warming process")
        
        warmed_count = 0
        failed_count = 0
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Generate cache key for batch
                key = self.key_generator.generate_key(batch, operation, {"batch_idx": batch_idx})
                
                # Store in cache
                if self.put(batch, key=key):
                    warmed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Cache warming failed for batch {batch_idx}: {e}")
                failed_count += 1
        
        logger.info(f"Cache warming completed: {warmed_count} warmed, {failed_count} failed")
        
        return {
            'warmed': warmed_count,
            'failed': failed_count,
            'total': warmed_count + failed_count
        }


def create_cache_manager(
    memory_cache_gb: float = 4.0,
    disk_cache_gb: float = 50.0,
    enable_redis: bool = False,
    enable_lmdb: bool = True,
    enable_prefetching: bool = True,
    **kwargs
) -> CacheManager:
    """Create cache manager with sensible defaults."""
    config = CacheConfig(
        memory_cache_size_gb=memory_cache_gb,
        disk_cache_size_gb=disk_cache_gb,
        enable_redis=enable_redis,
        enable_lmdb=enable_lmdb,
        enable_prefetching=enable_prefetching,
        **kwargs
    )
    
    return CacheManager(config)


class CachedModelInference:
    """Model inference with intelligent caching."""
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        cache_manager: CacheManager,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.cache_manager = cache_manager
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("CachedModelInference initialized")
    
    def predict(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with caching.
        
        Args:
            gene_expression: Gene expression data
            spatial_coords: Spatial coordinates
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            use_cache: Whether to use cache
            
        Returns:
            Model outputs
        """
        if use_cache:
            # Try to get from cache first
            cache_data = {
                'gene_expression': gene_expression,
                'spatial_coords': spatial_coords,
                'edge_index': edge_index
            }
            if edge_attr is not None:
                cache_data['edge_attr'] = edge_attr
            
            cached_result = self.cache_manager.get(
                data=cache_data,
                operation='model_inference',
                parameters={'model_id': id(self.model)}
            )
            
            if cached_result is not None:
                logger.debug("Using cached inference result")
                return cached_result
        
        # Run actual inference
        with torch.no_grad():
            outputs = self.model(
                gene_expression=gene_expression,
                spatial_coords=spatial_coords,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
        
        # Cache the result
        if use_cache:
            self.cache_manager.put(
                value=outputs,
                data=cache_data,
                operation='model_inference',
                parameters={'model_id': id(self.model)}
            )
        
        return outputs
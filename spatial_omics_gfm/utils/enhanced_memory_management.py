"""
Enhanced memory management utilities with advanced resource control and optimization.
Implements intelligent memory allocation, dynamic batch sizing, and resource pooling.
"""

import logging
import gc
import psutil
import threading
import time
import warnings
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List, Callable, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import weakref
import mmap
from anndata import AnnData
import h5py

from .memory_management import MemoryConfig, MemoryMonitor, DataChunker, SwapManager

logger = logging.getLogger(__name__)


@dataclass
class ResourcePool:
    """Configuration for resource pooling."""
    max_size: int = 10
    timeout_seconds: float = 30.0
    cleanup_interval: float = 60.0
    enable_preallocation: bool = True
    

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive memory management."""
    target_memory_usage: float = 0.7  # Target 70% memory usage
    adaptation_window: int = 10  # Number of operations to consider for adaptation
    min_batch_size: int = 1
    max_batch_size: int = 1024
    memory_check_frequency: int = 5  # Check memory every N operations
    enable_predictive_scaling: bool = True


class IntelligentBatchSizer:
    """Dynamically adjusts batch sizes based on available memory and performance."""
    
    def __init__(self, config: AdaptiveConfig, initial_batch_size: int = 32):
        self.config = config
        self.current_batch_size = initial_batch_size
        self.performance_history = deque(maxlen=config.adaptation_window)
        self.memory_history = deque(maxlen=config.adaptation_window)
        self.operation_count = 0
        self.last_adjustment_time = time.time()
        
        # Statistics
        self.total_adaptations = 0
        self.successful_increases = 0
        self.forced_decreases = 0
        
        logger.info(f"Initialized IntelligentBatchSizer with batch size {initial_batch_size}")
    
    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        self.operation_count += 1
        
        # Check memory and adjust if needed
        if self.operation_count % self.config.memory_check_frequency == 0:
            self._check_and_adjust()
        
        return max(self.config.min_batch_size, min(self.current_batch_size, self.config.max_batch_size))
    
    def record_performance(self, batch_size: int, processing_time: float, memory_used: float, success: bool):
        """Record performance metrics for a batch operation."""
        if not success:
            # Failed operation, reduce batch size
            self._reduce_batch_size("operation_failed")
            return
        
        # Calculate throughput (items per second per GB memory)
        throughput = batch_size / (processing_time * max(memory_used, 0.1))
        
        self.performance_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'memory_used': memory_used,
            'throughput': throughput,
            'timestamp': time.time()
        })
        
        self.memory_history.append(memory_used)
    
    def _check_and_adjust(self):
        """Check memory usage and adjust batch size if needed."""
        current_memory = self._get_memory_usage()
        
        if current_memory > 0.95:  # Critical memory usage
            self._reduce_batch_size("critical_memory")
        elif current_memory > self.config.target_memory_usage + 0.1:  # High memory usage
            self._reduce_batch_size("high_memory")
        elif current_memory < self.config.target_memory_usage - 0.1:  # Low memory usage
            self._maybe_increase_batch_size()
    
    def _reduce_batch_size(self, reason: str):
        """Reduce batch size due to memory pressure or failures."""
        old_size = self.current_batch_size
        reduction_factor = 0.8 if reason == "high_memory" else 0.5
        
        self.current_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * reduction_factor)
        )
        
        if self.current_batch_size != old_size:
            self.total_adaptations += 1
            self.forced_decreases += 1
            logger.info(f"Reduced batch size: {old_size} -> {self.current_batch_size} (reason: {reason})")
    
    def _maybe_increase_batch_size(self):
        """Consider increasing batch size if conditions are favorable."""
        if len(self.performance_history) < 3:
            return
        
        # Check if recent performance is stable
        recent_performance = list(self.performance_history)[-3:]
        throughputs = [p['throughput'] for p in recent_performance]
        
        # Only increase if throughput is stable or improving
        if len(throughputs) >= 2:
            trend = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
            if trend >= 0:  # Stable or improving
                old_size = self.current_batch_size
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * 1.2)
                )
                
                if self.current_batch_size != old_size:
                    self.total_adaptations += 1
                    self.successful_increases += 1
                    logger.info(f"Increased batch size: {old_size} -> {self.current_batch_size}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction of total."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.5  # Conservative estimate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch sizing statistics."""
        recent_throughputs = [p['throughput'] for p in self.performance_history]
        
        return {
            'current_batch_size': self.current_batch_size,
            'total_adaptations': self.total_adaptations,
            'successful_increases': self.successful_increases,
            'forced_decreases': self.forced_decreases,
            'average_throughput': np.mean(recent_throughputs) if recent_throughputs else 0,
            'performance_samples': len(self.performance_history),
            'memory_usage': self._get_memory_usage()
        }


class ResourcePoolManager:
    """Manages pools of reusable resources (tensors, arrays, etc.)."""
    
    def __init__(self, config: ResourcePool):
        self.config = config
        self.pools = {}
        self.pool_stats = {}
        self.cleanup_thread = None
        self.running = False
        self._lock = threading.Lock()
        
        if config.cleanup_interval > 0:
            self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.cleanup_thread is not None:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Resource pool cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
            self.cleanup_thread = None
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                self._cleanup_expired_resources()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in resource pool cleanup: {e}")
    
    def get_tensor_pool(self, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cpu") -> 'TensorPool':
        """Get or create a tensor pool for specific shape/dtype/device."""
        pool_key = f"tensor_{shape}_{dtype}_{device}"
        
        with self._lock:
            if pool_key not in self.pools:
                self.pools[pool_key] = TensorPool(shape, dtype, device, self.config)
                self.pool_stats[pool_key] = {'created': time.time(), 'hits': 0, 'misses': 0}
        
        return self.pools[pool_key]
    
    def get_array_pool(self, shape: Tuple[int, ...], dtype: np.dtype) -> 'ArrayPool':
        """Get or create a numpy array pool for specific shape/dtype."""
        pool_key = f"array_{shape}_{dtype}"
        
        with self._lock:
            if pool_key not in self.pools:
                self.pools[pool_key] = ArrayPool(shape, dtype, self.config)
                self.pool_stats[pool_key] = {'created': time.time(), 'hits': 0, 'misses': 0}
        
        return self.pools[pool_key]
    
    def _cleanup_expired_resources(self):
        """Clean up expired resources from all pools."""
        with self._lock:
            for pool in self.pools.values():
                if hasattr(pool, 'cleanup_expired'):
                    pool.cleanup_expired()
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics for all resource pools."""
        with self._lock:
            stats = {}
            for pool_key, pool in self.pools.items():
                pool_stat = self.pool_stats.get(pool_key, {})
                stats[pool_key] = {
                    'size': len(pool.available),
                    'max_size': pool.config.max_size,
                    'hits': pool_stat.get('hits', 0),
                    'misses': pool_stat.get('misses', 0),
                    'created': pool_stat.get('created')
                }
        return stats


class TensorPool:
    """Pool of reusable PyTorch tensors."""
    
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: str, config: ResourcePool):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.config = config
        self.available = deque()
        self.in_use = set()
        self.creation_times = {}
        self._lock = threading.Lock()
        
        # Pre-allocate tensors if enabled
        if config.enable_preallocation:
            self._preallocate()
    
    def _preallocate(self):
        """Pre-allocate tensors for better performance."""
        initial_count = min(3, self.config.max_size)
        for _ in range(initial_count):
            tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            self.available.append(tensor)
            self.creation_times[id(tensor)] = time.time()
    
    @contextmanager
    def get_tensor(self):
        """Get a tensor from the pool."""
        tensor = None
        
        with self._lock:
            if self.available:
                tensor = self.available.popleft()
            else:
                # Create new tensor if pool is empty and under limit
                if len(self.in_use) < self.config.max_size:
                    tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
                    self.creation_times[id(tensor)] = time.time()
        
        if tensor is None:
            # Pool exhausted, create temporary tensor
            tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            logger.warning(f"Tensor pool exhausted, creating temporary tensor {self.shape}")
        else:
            with self._lock:
                self.in_use.add(id(tensor))
        
        try:
            # Zero out tensor for clean slate
            tensor.zero_()
            yield tensor
        finally:
            # Return tensor to pool
            if id(tensor) in self.in_use:
                with self._lock:
                    self.in_use.remove(id(tensor))
                    if len(self.available) < self.config.max_size:
                        self.available.append(tensor)
                    else:
                        # Pool is full, discard tensor
                        if id(tensor) in self.creation_times:
                            del self.creation_times[id(tensor)]
    
    def cleanup_expired(self):
        """Remove old tensors from the pool."""
        if not hasattr(self.config, 'max_age_seconds'):
            return
        
        current_time = time.time()
        max_age = getattr(self.config, 'max_age_seconds', 3600)  # 1 hour default
        
        with self._lock:
            expired_tensors = []
            for tensor in list(self.available):
                tensor_id = id(tensor)
                if tensor_id in self.creation_times:
                    age = current_time - self.creation_times[tensor_id]
                    if age > max_age:
                        expired_tensors.append(tensor)
            
            for tensor in expired_tensors:
                self.available.remove(tensor)
                tensor_id = id(tensor)
                if tensor_id in self.creation_times:
                    del self.creation_times[tensor_id]
            
            if expired_tensors:
                logger.debug(f"Cleaned up {len(expired_tensors)} expired tensors from pool")


class ArrayPool:
    """Pool of reusable NumPy arrays."""
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, config: ResourcePool):
        self.shape = shape
        self.dtype = dtype
        self.config = config
        self.available = deque()
        self.in_use = set()
        self.creation_times = {}
        self._lock = threading.Lock()
        
        if config.enable_preallocation:
            self._preallocate()
    
    def _preallocate(self):
        """Pre-allocate arrays for better performance."""
        initial_count = min(3, self.config.max_size)
        for _ in range(initial_count):
            array = np.empty(self.shape, dtype=self.dtype)
            self.available.append(array)
            self.creation_times[id(array)] = time.time()
    
    @contextmanager
    def get_array(self):
        """Get an array from the pool."""
        array = None
        
        with self._lock:
            if self.available:
                array = self.available.popleft()
            else:
                if len(self.in_use) < self.config.max_size:
                    array = np.empty(self.shape, dtype=self.dtype)
                    self.creation_times[id(array)] = time.time()
        
        if array is None:
            array = np.empty(self.shape, dtype=self.dtype)
            logger.warning(f"Array pool exhausted, creating temporary array {self.shape}")
        else:
            with self._lock:
                self.in_use.add(id(array))
        
        try:
            # Zero out array
            array.fill(0)
            yield array
        finally:
            if id(array) in self.in_use:
                with self._lock:
                    self.in_use.remove(id(array))
                    if len(self.available) < self.config.max_size:
                        self.available.append(array)
                    else:
                        if id(array) in self.creation_times:
                            del self.creation_times[id(array)]


class MemoryMappedDataLoader:
    """Memory-mapped data loader for extremely large datasets."""
    
    def __init__(self, file_path: Union[str, Path], chunk_size: int = 1000):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.file_handle = None
        self.memory_map = None
        self.data_info = None
    
    def __enter__(self):
        """Enter context manager."""
        self._open_file()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._close_file()
    
    def _open_file(self):
        """Open file and create memory map."""
        if self.file_path.suffix == '.h5ad':
            self._open_h5ad()
        elif self.file_path.suffix == '.h5':
            self._open_h5()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
    
    def _open_h5ad(self):
        """Open H5AD file with memory mapping."""
        try:
            # Use backed mode for memory efficiency
            import scanpy as sc
            self.data_info = {
                'backed': True,
                'file_path': self.file_path
            }
            logger.info(f"Opened H5AD file in backed mode: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to open H5AD file: {e}")
            raise
    
    def _open_h5(self):
        """Open HDF5 file with memory mapping."""
        try:
            self.file_handle = h5py.File(self.file_path, 'r', rdcc_nbytes=1024**2, rdcc_nslots=10007)
            self.data_info = {
                'shape': self.file_handle['X'].shape if 'X' in self.file_handle else None,
                'chunks': self.file_handle['X'].chunks if 'X' in self.file_handle else None
            }
            logger.info(f"Opened HDF5 file: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to open HDF5 file: {e}")
            raise
    
    def _close_file(self):
        """Close file and cleanup."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        if self.memory_map:
            self.memory_map.close()
            self.memory_map = None
    
    def iter_chunks(self) -> Iterator[np.ndarray]:
        """Iterate over data chunks."""
        if self.file_handle and 'X' in self.file_handle:
            dataset = self.file_handle['X']
            n_obs = dataset.shape[0]
            
            for start_idx in range(0, n_obs, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_obs)
                chunk = dataset[start_idx:end_idx]
                
                # Convert to dense array if sparse
                if hasattr(chunk, 'toarray'):
                    chunk = chunk.toarray()
                
                yield chunk
        else:
            raise RuntimeError("No suitable dataset found or file not opened")


class AdaptiveMemoryManager:
    """Adaptive memory manager that adjusts strategies based on system state."""
    
    def __init__(
        self, 
        base_config: Optional[MemoryConfig] = None,
        adaptive_config: Optional[AdaptiveConfig] = None
    ):
        self.base_config = base_config or MemoryConfig()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        
        # Components
        self.batch_sizer = IntelligentBatchSizer(self.adaptive_config)
        self.resource_pool = ResourcePoolManager(ResourcePool())
        self.base_monitor = MemoryMonitor(self.base_config)
        
        # Adaptive state
        self.system_pressure_history = deque(maxlen=20)
        self.gc_frequency_adaptive = self.base_config.gc_frequency
        self.last_adaptation_time = time.time()
        
        # Statistics
        self.adaptations_made = 0
        self.memory_pressure_events = 0
        self.gc_triggered_count = 0
        
        logger.info("Initialized AdaptiveMemoryManager")
    
    def start_monitoring(self):
        """Start adaptive memory monitoring."""
        self.base_monitor.start_monitoring()
        logger.info("Adaptive memory monitoring started")
    
    def stop_monitoring(self):
        """Stop adaptive memory monitoring."""
        self.base_monitor.stop_monitoring()
        self.resource_pool.stop_cleanup_thread()
        logger.info("Adaptive memory monitoring stopped")
    
    @contextmanager
    def adaptive_batch_processing(
        self, 
        data_iterator: Iterator,
        processing_func: Callable,
        initial_batch_size: Optional[int] = None
    ):
        """Context manager for adaptive batch processing."""
        if initial_batch_size:
            self.batch_sizer.current_batch_size = initial_batch_size
        
        batch_data = []
        processed_count = 0
        
        try:
            for item in data_iterator:
                batch_data.append(item)
                
                if len(batch_data) >= self.batch_sizer.get_batch_size():
                    # Process batch
                    start_time = time.time()
                    start_memory = self._get_memory_usage_gb()
                    
                    try:
                        processing_func(batch_data)
                        
                        # Record successful batch processing
                        end_time = time.time()
                        end_memory = self._get_memory_usage_gb()
                        
                        self.batch_sizer.record_performance(
                            len(batch_data),
                            end_time - start_time,
                            end_memory,
                            True
                        )
                        
                        processed_count += len(batch_data)
                        batch_data = []
                        
                        # Adaptive garbage collection
                        self._adaptive_gc_check()
                        
                    except Exception as e:
                        # Record failed batch processing
                        self.batch_sizer.record_performance(
                            len(batch_data),
                            time.time() - start_time,
                            self._get_memory_usage_gb(),
                            False
                        )
                        
                        # Try with smaller batches
                        if len(batch_data) > 1:
                            # Split batch and retry
                            mid = len(batch_data) // 2
                            for sub_batch in [batch_data[:mid], batch_data[mid:]]:
                                try:
                                    processing_func(sub_batch)
                                    processed_count += len(sub_batch)
                                except Exception:
                                    logger.error(f"Failed to process sub-batch of size {len(sub_batch)}")
                        else:
                            logger.error(f"Failed to process single item: {e}")
                        
                        batch_data = []
            
            # Process remaining items
            if batch_data:
                try:
                    processing_func(batch_data)
                    processed_count += len(batch_data)
                except Exception as e:
                    logger.error(f"Failed to process final batch: {e}")
            
            yield processed_count
            
        finally:
            # Cleanup
            self._force_cleanup()
    
    def _adaptive_gc_check(self):
        """Perform garbage collection based on adaptive frequency."""
        memory_usage = self._get_memory_usage_gb()
        total_memory = psutil.virtual_memory().total / (1024**3)
        memory_fraction = memory_usage / total_memory
        
        self.system_pressure_history.append(memory_fraction)
        
        # Adapt GC frequency based on memory pressure
        if memory_fraction > 0.8:  # High pressure
            self.gc_frequency_adaptive = max(1, self.base_config.gc_frequency // 4)
            self.memory_pressure_events += 1
        elif memory_fraction > 0.6:  # Medium pressure
            self.gc_frequency_adaptive = max(5, self.base_config.gc_frequency // 2)
        else:  # Low pressure
            self.gc_frequency_adaptive = self.base_config.gc_frequency * 2
        
        # Trigger GC if needed
        if self.gc_triggered_count % self.gc_frequency_adaptive == 0:
            collected = gc.collect()
            logger.debug(f"Adaptive GC collected {collected} objects")
            
            # Clear PyTorch cache if memory pressure is high
            if memory_fraction > 0.7 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.gc_triggered_count += 1
    
    def _force_cleanup(self):
        """Force cleanup of resources."""
        # Aggressive garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        logger.debug("Forced memory cleanup completed")
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        try:
            return psutil.virtual_memory().used / (1024**3)
        except Exception:
            return 0.0
    
    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get adaptive memory management statistics."""
        batch_stats = self.batch_sizer.get_statistics()
        pool_stats = self.resource_pool.get_pool_statistics()
        
        avg_pressure = np.mean(self.system_pressure_history) if self.system_pressure_history else 0
        
        return {
            'batch_sizing': batch_stats,
            'resource_pools': pool_stats,
            'memory_pressure': {
                'average': avg_pressure,
                'current': self._get_memory_usage_gb() / (psutil.virtual_memory().total / (1024**3)),
                'pressure_events': self.memory_pressure_events
            },
            'garbage_collection': {
                'adaptive_frequency': self.gc_frequency_adaptive,
                'base_frequency': self.base_config.gc_frequency,
                'total_triggered': self.gc_triggered_count
            },
            'adaptations': {
                'total_made': self.adaptations_made,
                'last_adaptation_time': self.last_adaptation_time
            }
        }
    
    @contextmanager
    def optimized_tensor_operations(self, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cpu"):
        """Context manager for optimized tensor operations using resource pooling."""
        tensor_pool = self.resource_pool.get_tensor_pool(shape, dtype, device)
        
        with tensor_pool.get_tensor() as tensor:
            yield tensor
    
    @contextmanager
    def optimized_array_operations(self, shape: Tuple[int, ...], dtype: np.dtype):
        """Context manager for optimized array operations using resource pooling."""
        array_pool = self.resource_pool.get_array_pool(shape, dtype)
        
        with array_pool.get_array() as array:
            yield array


def create_adaptive_memory_context(
    target_memory_usage: float = 0.7,
    initial_batch_size: int = 32,
    enable_resource_pooling: bool = True
) -> AdaptiveMemoryManager:
    """Create an adaptive memory management context."""
    adaptive_config = AdaptiveConfig(
        target_memory_usage=target_memory_usage,
        min_batch_size=1,
        max_batch_size=1024
    )
    
    base_config = MemoryConfig(
        max_memory_gb=psutil.virtual_memory().total / (1024**3) * target_memory_usage,
        enable_monitoring=True
    )
    
    manager = AdaptiveMemoryManager(base_config, adaptive_config)
    
    if initial_batch_size != 32:
        manager.batch_sizer.current_batch_size = initial_batch_size
    
    return manager


@contextmanager
def memory_optimized_operation(operation_name: str = "operation"):
    """Context manager for memory-optimized operations."""
    manager = create_adaptive_memory_context()
    
    try:
        manager.start_monitoring()
        
        start_time = time.time()
        start_memory = manager._get_memory_usage_gb()
        
        logger.info(f"Starting memory-optimized operation: {operation_name}")
        
        yield manager
        
        # Log performance statistics
        end_time = time.time()
        end_memory = manager._get_memory_usage_gb()
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        stats = manager.get_adaptive_statistics()
        
        logger.info(
            f"Completed {operation_name}: {duration:.2f}s, "
            f"memory delta: {memory_delta:+.2f}GB, "
            f"batch adaptations: {stats['batch_sizing']['total_adaptations']}"
        )
        
    finally:
        manager.stop_monitoring()
"""
Performance optimization utilities for spatial omics computations.

Provides profiling, batch processing, and automatic optimization
for computationally intensive operations.
"""

import time
import cProfile
import pstats
import io
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple
import numpy as np
from functools import wraps
import logging
import gc
import psutil

logger = logging.getLogger(__name__)


class ProfilerContext:
    """Context manager for profiling code execution."""
    
    def __init__(self, name: str = "profiling", enable: bool = True):
        """
        Initialize profiler context.
        
        Args:
            name: Name for the profiling session
            enable: Whether to actually perform profiling
        """
        self.name = name
        self.enable = enable
        self.profiler = None
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
    
    def __enter__(self):
        if not self.enable:
            return self
        
        # Record start time and memory
        self.start_time = time.perf_counter()
        self.memory_before = psutil.Process().memory_info().rss
        
        # Start profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        
        # Stop profiling
        self.profiler.disable()
        
        # Record end time and memory
        self.end_time = time.perf_counter()
        self.memory_after = psutil.Process().memory_info().rss
    
    def get_stats(self, sort_by: str = 'cumulative', top_n: int = 20) -> Dict[str, Any]:
        """
        Get profiling statistics.
        
        Args:
            sort_by: How to sort the stats ('cumulative', 'time', 'calls')
            top_n: Number of top functions to include
            
        Returns:
            Dictionary with profiling results
        """
        if not self.enable or not self.profiler:
            return {}
        
        # Get string representation of stats
        string_buffer = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=string_buffer)
        ps.sort_stats(sort_by)
        ps.print_stats(top_n)
        profile_text = string_buffer.getvalue()
        
        return {
            "name": self.name,
            "execution_time_seconds": self.end_time - self.start_time if self.end_time else None,
            "memory_delta_mb": ((self.memory_after - self.memory_before) / (1024*1024)) 
                               if self.memory_after else None,
            "profile_text": profile_text,
            "sort_by": sort_by,
            "top_n": top_n
        }
    
    def print_stats(self, sort_by: str = 'cumulative', top_n: int = 20):
        """Print profiling statistics to console."""
        stats = self.get_stats(sort_by, top_n)
        
        print(f"\n=== Profiling Results: {stats['name']} ===")
        if stats.get("execution_time_seconds"):
            print(f"Execution time: {stats['execution_time_seconds']:.4f} seconds")
        if stats.get("memory_delta_mb"):
            print(f"Memory delta: {stats['memory_delta_mb']:.2f} MB")
        print(stats.get("profile_text", "No profiling data available"))


class BatchProcessor:
    """Efficient batch processing for large datasets."""
    
    def __init__(
        self,
        batch_size: int = 1000,
        overlap: int = 0,
        memory_limit_mb: int = 1024,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            overlap: Number of overlapping items between batches
            memory_limit_mb: Memory limit for batch processing
            progress_callback: Optional callback for progress updates
        """
        self.batch_size = batch_size
        self.overlap = overlap
        self.memory_limit_mb = memory_limit_mb
        self.progress_callback = progress_callback
    
    def process_array(
        self,
        array: np.ndarray,
        process_func: Callable,
        axis: int = 0,
        **kwargs
    ) -> np.ndarray:
        """
        Process large array in batches.
        
        Args:
            array: Input array to process
            process_func: Function to apply to each batch
            axis: Axis along which to batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed array
        """
        logger.info(f"Processing array of shape {array.shape} in batches of {self.batch_size}")
        
        array_size = array.shape[axis]
        results = []
        
        for i in range(0, array_size, self.batch_size - self.overlap):
            # Calculate batch boundaries
            start_idx = max(0, i - self.overlap) if i > 0 else 0
            end_idx = min(array_size, i + self.batch_size)
            
            # Extract batch
            if axis == 0:
                batch = array[start_idx:end_idx]
            elif axis == 1:
                batch = array[:, start_idx:end_idx]
            else:
                # For higher dimensions, use np.take
                batch = np.take(array, range(start_idx, end_idx), axis=axis)
            
            # Process batch
            try:
                batch_result = process_func(batch, **kwargs)
                results.append(batch_result)
                
                # Progress callback
                if self.progress_callback:
                    progress = (end_idx / array_size) * 100
                    self.progress_callback(progress, f"Processed {end_idx}/{array_size} items")
                
                # Memory management
                if i % 10 == 0:  # Check memory every 10 batches
                    self._check_memory_usage()
                
            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx}: {e}")
                raise
        
        # Combine results
        if results:
            combined = np.concatenate(results, axis=axis)
            logger.info(f"Batch processing completed. Result shape: {combined.shape}")
            return combined
        else:
            logger.warning("No results from batch processing")
            return np.array([])
    
    def process_list(
        self,
        items: List[Any],
        process_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Process list of items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        logger.info(f"Processing {len(items)} items in batches of {self.batch_size}")
        
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            try:
                batch_result = process_func(batch, **kwargs)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
                
                # Progress callback
                if self.progress_callback:
                    progress = ((i + len(batch)) / len(items)) * 100
                    self.progress_callback(progress, f"Processed {i + len(batch)}/{len(items)} items")
                
                # Memory management
                if i % (10 * self.batch_size) == 0:
                    self._check_memory_usage()
                
            except Exception as e:
                logger.error(f"Error processing batch {i}-{i + len(batch)}: {e}")
                raise
        
        logger.info(f"Batch processing completed. {len(results)} results.")
        return results
    
    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed."""
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        if memory_mb > self.memory_limit_mb:
            logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds limit ({self.memory_limit_mb} MB)")
            gc.collect()  # Force garbage collection
            
            # Check again after cleanup
            memory_mb_after = psutil.Process().memory_info().rss / (1024 * 1024)
            logger.info(f"Memory usage after cleanup: {memory_mb_after:.1f} MB")


class PerformanceOptimizer:
    """Automatic performance optimization for common operations."""
    
    def __init__(self):
        self.optimization_history = {}
        self._lock = threading.Lock()
    
    def optimize_matrix_operations(
        self,
        operation_name: str,
        matrix_shape: Tuple[int, ...],
        operation_func: Callable,
        test_sizes: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Automatically optimize matrix operations by testing different parameters.
        
        Args:
            operation_name: Name of the operation for caching
            matrix_shape: Shape of the matrix to optimize for
            operation_func: Function to optimize
            test_sizes: List of batch sizes to test
            **kwargs: Additional arguments for operation_func
            
        Returns:
            Optimization results with best parameters
        """
        if test_sizes is None:
            # Default test sizes based on matrix size
            matrix_size = np.prod(matrix_shape)
            if matrix_size < 10000:
                test_sizes = [100, 500, 1000]
            elif matrix_size < 100000:
                test_sizes = [500, 1000, 2000, 5000]
            else:
                test_sizes = [1000, 2000, 5000, 10000]
        
        logger.info(f"Optimizing {operation_name} for matrix shape {matrix_shape}")
        
        # Create test matrix
        test_matrix = np.random.rand(*matrix_shape).astype(np.float32)
        
        results = []
        
        for batch_size in test_sizes:
            try:
                # Time the operation
                with ProfilerContext(f"{operation_name}_batch_{batch_size}") as profiler:
                    processor = BatchProcessor(batch_size=batch_size)
                    result = processor.process_array(test_matrix, operation_func, **kwargs)
                
                stats = profiler.get_stats()
                
                results.append({
                    "batch_size": batch_size,
                    "execution_time": stats.get("execution_time_seconds", float('inf')),
                    "memory_delta": stats.get("memory_delta_mb", 0),
                    "result_shape": result.shape if hasattr(result, 'shape') else None
                })
                
                logger.debug(f"Batch size {batch_size}: {stats.get('execution_time_seconds', 0):.4f}s")
                
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                results.append({
                    "batch_size": batch_size,
                    "execution_time": float('inf'),
                    "memory_delta": float('inf'),
                    "error": str(e)
                })
        
        # Find best batch size (optimize for time)
        valid_results = [r for r in results if r["execution_time"] != float('inf')]
        
        if valid_results:
            best_result = min(valid_results, key=lambda x: x["execution_time"])
            
            optimization_result = {
                "operation_name": operation_name,
                "matrix_shape": matrix_shape,
                "best_batch_size": best_result["batch_size"],
                "best_time": best_result["execution_time"],
                "all_results": results,
                "speedup_factor": max(r["execution_time"] for r in valid_results) / best_result["execution_time"]
            }
            
            # Cache result
            with self._lock:
                cache_key = f"{operation_name}_{matrix_shape}"
                self.optimization_history[cache_key] = optimization_result
            
            logger.info(f"Best batch size for {operation_name}: {best_result['batch_size']} "
                       f"(speedup: {optimization_result['speedup_factor']:.2f}x)")
            
            return optimization_result
        
        else:
            logger.error(f"All batch sizes failed for {operation_name}")
            return {
                "operation_name": operation_name,
                "matrix_shape": matrix_shape,
                "best_batch_size": test_sizes[0],  # Default fallback
                "error": "All optimizations failed"
            }
    
    def get_cached_optimization(
        self,
        operation_name: str,
        matrix_shape: Tuple[int, ...]
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization result."""
        cache_key = f"{operation_name}_{matrix_shape}"
        
        with self._lock:
            return self.optimization_history.get(cache_key)


def optimize_computation(
    operation_name: str,
    matrix_shape: Optional[Tuple[int, ...]] = None,
    auto_optimize: bool = True
):
    """
    Decorator for automatic computation optimization.
    
    Args:
        operation_name: Name of the operation
        matrix_shape: Expected matrix shape (for optimization)
        auto_optimize: Whether to automatically optimize
    """
    def decorator(func: Callable):
        optimizer = PerformanceOptimizer()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not auto_optimize or not matrix_shape:
                return func(*args, **kwargs)
            
            # Check for cached optimization
            cached_opt = optimizer.get_cached_optimization(operation_name, matrix_shape)
            
            if cached_opt:
                # Use cached optimal batch size
                best_batch_size = cached_opt["best_batch_size"]
                logger.debug(f"Using cached optimal batch size: {best_batch_size}")
                
                processor = BatchProcessor(batch_size=best_batch_size)
                
                # Apply optimization if first argument is an array
                if args and hasattr(args[0], 'shape'):
                    return processor.process_array(args[0], func, *args[1:], **kwargs)
            
            # Fallback to original function
            return func(*args, **kwargs)
        
        # Add optimization method
        wrapper.optimize = lambda shape=matrix_shape: optimizer.optimize_matrix_operations(
            operation_name, shape, func
        )
        
        return wrapper
    
    return decorator


@contextmanager
def performance_monitor(name: str = "operation") -> Iterator[Dict[str, Any]]:
    """
    Context manager for monitoring performance metrics.
    
    Args:
        name: Name of the operation being monitored
        
    Yields:
        Dictionary to store custom metrics
    """
    metrics = {"name": name}
    
    # Record start metrics
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield metrics
        
    finally:
        # Record end metrics
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        metrics.update({
            "execution_time_seconds": end_time - start_time,
            "memory_delta_mb": (end_memory - start_memory) / (1024 * 1024),
            "peak_memory_mb": end_memory / (1024 * 1024)
        })
        
        logger.info(f"Performance metrics for {name}: "
                   f"{metrics['execution_time_seconds']:.4f}s, "
                   f"{metrics['memory_delta_mb']:.2f}MB delta")
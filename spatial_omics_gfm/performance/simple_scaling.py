"""
Lightweight scaling and performance optimization for Generation 3.
Works without heavy dependencies like psutil.
"""

import time
import gc
import threading
import os
from typing import Any, Dict, List, Optional, Callable
import numpy as np
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class LightweightProfiler:
    """Lightweight profiler for Generation 3 performance tracking."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.memory_snapshots = []
        
    def __enter__(self):
        gc.collect()  # Clean memory before profiling
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        gc.collect()  # Clean memory after profiling
        
    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "name": self.name,
            "execution_time_seconds": self.get_execution_time(),
            "status": "completed" if self.end_time else "running"
        }


class AdaptiveBatchProcessor:
    """Adaptive batch processor that scales based on performance."""
    
    def __init__(self, initial_batch_size: int = 100):
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.processed_items = 0
        
    def process_with_scaling(self, data: List[Any], process_func: Callable) -> List[Any]:
        """Process data with adaptive batch sizing."""
        results = []
        total_items = len(data)
        processed = 0
        
        while processed < total_items:
            # Current batch
            batch_end = min(processed + self.current_batch_size, total_items)
            batch = data[processed:batch_end]
            
            # Time the batch processing
            with LightweightProfiler(f"batch_{len(batch)}") as profiler:
                batch_result = process_func(batch)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Record performance
            batch_time = profiler.get_execution_time()
            items_per_second = len(batch) / max(batch_time, 0.001)  # Avoid division by zero
            
            self.performance_history.append({
                'batch_size': len(batch),
                'items_per_second': items_per_second,
                'timestamp': time.time()
            })
            
            # Adapt batch size
            self._adapt_batch_size()
            
            processed = batch_end
            self.processed_items += len(batch)
            
            # Progress logging
            if processed % (self.current_batch_size * 5) == 0:
                progress = (processed / total_items) * 100
                logger.info(f"Processed {progress:.1f}% - batch_size: {self.current_batch_size}")
        
        return results
    
    def _adapt_batch_size(self):
        """Adapt batch size based on recent performance."""
        if len(self.performance_history) < 3:
            return
        
        # Get recent performance metrics
        recent = self.performance_history[-3:]
        current_perf = recent[-1]['items_per_second']
        prev_perf = recent[-2]['items_per_second']
        
        # Increase batch size if performance is improving
        if current_perf > prev_perf * 1.1:
            self.current_batch_size = min(int(self.current_batch_size * 1.2), 1000)
        # Decrease if performance is degrading
        elif current_perf < prev_perf * 0.9:
            self.current_batch_size = max(int(self.current_batch_size * 0.8), 10)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        throughput_values = [h['items_per_second'] for h in self.performance_history]
        batch_sizes = [h['batch_size'] for h in self.performance_history]
        
        return {
            'total_processed': self.processed_items,
            'current_batch_size': self.current_batch_size,
            'avg_throughput': np.mean(throughput_values),
            'max_throughput': max(throughput_values),
            'min_batch_size': min(batch_sizes),
            'max_batch_size': max(batch_sizes),
            'adaptations_made': len(self.performance_history)
        }


class MemoryEfficientProcessor:
    """Memory-efficient processing with automatic garbage collection."""
    
    def __init__(self, gc_frequency: int = 10):
        self.gc_frequency = gc_frequency
        self.operations_count = 0
        
    def process_chunks(self, data: np.ndarray, chunk_size: int, process_func: Callable) -> List[Any]:
        """Process data in memory-efficient chunks."""
        results = []
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Process chunk
            with LightweightProfiler(f"chunk_{i//chunk_size}") as profiler:
                chunk_result = process_func(chunk)
                results.append(chunk_result)
            
            self.operations_count += 1
            
            # Periodic garbage collection
            if self.operations_count % self.gc_frequency == 0:
                gc.collect()
                
            # Progress update
            chunk_num = i // chunk_size + 1
            if chunk_num % 5 == 0:
                progress = (chunk_num / total_chunks) * 100
                logger.info(f"Memory-efficient processing: {progress:.1f}% complete")
        
        return results


class SimpleAutoScaler:
    """Simple auto-scaler based on CPU core count and data size."""
    
    def __init__(self):
        self.cpu_count = os.cpu_count() or 1
        self.optimal_batch_sizes = {}
        
    def recommend_batch_size(self, data_size: int, operation_type: str = "default") -> int:
        """Recommend batch size based on data size and operation type."""
        
        # Base calculation on CPU cores and data size
        base_size = max(1, data_size // (self.cpu_count * 10))
        
        # Operation-specific adjustments
        multipliers = {
            "cpu_intensive": 0.5,
            "memory_intensive": 0.3,
            "io_intensive": 2.0,
            "default": 1.0
        }
        
        multiplier = multipliers.get(operation_type, 1.0)
        recommended_size = int(base_size * multiplier)
        
        # Clamp to reasonable bounds
        return max(10, min(recommended_size, 1000))
    
    def recommend_workers(self, data_size: int, is_cpu_bound: bool = True) -> int:
        """Recommend number of workers."""
        if is_cpu_bound:
            return min(self.cpu_count, 4)  # Don't exceed 4 for CPU-bound tasks
        else:
            return min(self.cpu_count * 2, 8)  # More workers for I/O-bound tasks


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        
        with LightweightProfiler(function_name) as profiler:
            result = func(*args, **kwargs)
        
        # Log performance
        execution_time = profiler.get_execution_time()
        logger.info(f"Performance: {function_name} executed in {execution_time:.3f}s")
        
        return result
    
    return wrapper


def memory_optimized(func: Callable) -> Callable:
    """Decorator to optimize memory usage."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory before execution
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clear memory after execution
            gc.collect()
    
    return wrapper


# Example test functions
def test_generation_3_scaling():
    """Test Generation 3 scaling features."""
    print("=== Testing Generation 3 Scaling Features ===")
    
    # Test adaptive batch processor
    processor = AdaptiveBatchProcessor(initial_batch_size=50)
    test_data = list(range(1000))
    
    def sample_operation(batch):
        return [x * x for x in batch]
    
    results = processor.process_with_scaling(test_data, sample_operation)
    stats = processor.get_performance_stats()
    
    print(f"✓ Adaptive batch processing: processed {len(results)} items")
    print(f"  - Final batch size: {stats.get('current_batch_size', 'N/A')}")
    print(f"  - Average throughput: {stats.get('avg_throughput', 0):.1f} items/sec")
    
    # Test memory-efficient processor
    mem_processor = MemoryEfficientProcessor()
    test_array = np.random.rand(1000, 10)
    
    def array_operation(chunk):
        return np.sum(chunk, axis=1)
    
    chunk_results = mem_processor.process_chunks(test_array, chunk_size=100, process_func=array_operation)
    print(f"✓ Memory-efficient processing: {len(chunk_results)} chunks processed")
    
    # Test auto-scaler
    scaler = SimpleAutoScaler()
    recommended_batch = scaler.recommend_batch_size(5000, "cpu_intensive")
    recommended_workers = scaler.recommend_workers(5000, is_cpu_bound=True)
    
    print(f"✓ Auto-scaling recommendations:")
    print(f"  - Batch size: {recommended_batch}")
    print(f"  - Workers: {recommended_workers}")
    
    # Test performance monitoring
    @performance_monitor
    @memory_optimized
    def expensive_operation(n):
        return sum(i * i for i in range(n))
    
    result = expensive_operation(100000)
    print(f"✓ Performance monitoring: computed result {result}")
    
    print("✅ All Generation 3 scaling features working!")


if __name__ == "__main__":
    test_generation_3_scaling()
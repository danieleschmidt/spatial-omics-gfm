"""
Simple memory management utilities without heavy dependencies.
Provides basic memory monitoring and managed operations.
"""

import gc
import psutil
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import warnings


class MemoryMonitor:
    """Simple memory monitoring without heavy dependencies."""
    
    def __init__(self):
        self.monitoring_enabled = True
        self.warning_threshold = 0.8  # 80% memory usage
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent / 100.0,
                'swap_percent': psutil.swap_memory().percent / 100.0
            }
        except Exception:
            # Fallback if psutil not available
            return {
                'total_memory': 0,
                'available_memory': 0,
                'used_memory': 0,
                'memory_percent': 0.0,
                'swap_percent': 0.0
            }
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        stats = self.get_memory_stats()
        memory_percent = stats.get('memory_percent', 0.0)
        
        if memory_percent > self.warning_threshold:
            warnings.warn(f"High memory usage: {memory_percent:.1%}")
            return False
        
        return True
    
    def memory_managed_operation(self, func: Callable) -> Callable:
        """Decorator for memory-managed operations."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before operation
            initial_stats = self.get_memory_stats()
            
            # Force garbage collection
            gc.collect()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Clean up after operation
                gc.collect()
                
                # Check memory after operation
                final_stats = self.get_memory_stats()
                memory_increase = final_stats['memory_percent'] - initial_stats['memory_percent']
                
                if memory_increase > 0.1:  # 10% increase
                    warnings.warn(f"Operation increased memory usage by {memory_increase:.1%}")
        
        return wrapper


def memory_managed_operation(func: Callable) -> Callable:
    """Standalone decorator for memory-managed operations."""
    monitor = MemoryMonitor()
    return monitor.memory_managed_operation(func)


def check_available_memory() -> Dict[str, Any]:
    """Check available memory without creating a monitor instance."""
    monitor = MemoryMonitor()
    return monitor.get_memory_stats()


def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()


class SimpleMemoryManager:
    """Simple memory manager for batch processing."""
    
    def __init__(self, max_memory_percent: float = 0.8):
        self.max_memory_percent = max_memory_percent
        self.monitor = MemoryMonitor()
    
    def get_optimal_batch_size(self, total_items: int, base_batch_size: int = 1000) -> int:
        """Calculate optimal batch size based on available memory."""
        stats = self.monitor.get_memory_stats()
        available_percent = 1.0 - stats['memory_percent']
        
        if available_percent < 0.2:  # Less than 20% memory available
            return max(base_batch_size // 4, 100)
        elif available_percent < 0.5:  # Less than 50% memory available
            return max(base_batch_size // 2, 250)
        else:
            return base_batch_size
    
    def process_in_batches(self, items: list, process_func: Callable, batch_size: Optional[int] = None):
        """Process items in memory-managed batches."""
        if batch_size is None:
            batch_size = self.get_optimal_batch_size(len(items))
        
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Check memory before processing batch
            if not self.monitor.check_memory_usage():
                # Force garbage collection if memory is high
                gc.collect()
                
                # Reduce batch size if still high memory
                if not self.monitor.check_memory_usage():
                    batch_size = max(batch_size // 2, 10)
                    batch = items[i:i + batch_size]
            
            # Process batch
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            # Clean up after batch
            gc.collect()
        
        return results


if __name__ == "__main__":
    # Test memory monitoring
    monitor = MemoryMonitor()
    stats = monitor.get_memory_stats()
    
    print("=== Memory Statistics ===")
    print(f"Total Memory: {stats['total_memory'] / (1024**3):.1f} GB")
    print(f"Available Memory: {stats['available_memory'] / (1024**3):.1f} GB") 
    print(f"Memory Usage: {stats['memory_percent']:.1%}")
    print(f"Swap Usage: {stats['swap_percent']:.1%}")
    
    # Test memory-managed operation
    @memory_managed_operation
    def test_operation():
        # Simulate some work
        import numpy as np
        data = np.random.rand(1000, 1000)
        return np.sum(data)
    
    print(f"\nTest operation result: {test_operation()}")
    
    print("\n✅ Simple memory management working")
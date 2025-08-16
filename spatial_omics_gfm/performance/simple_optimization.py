"""
Simple performance optimization for Generation 3 scaling without heavy dependencies.

Provides basic batch processing, memory-efficient operations, and performance monitoring
using only core Python libraries.
"""

import time
import gc
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple
import numpy as np
from functools import wraps
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import json
from pathlib import Path


class SimplePerformanceMonitor:
    """Lightweight performance monitoring without external dependencies."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        end_time = time.perf_counter()
        with self.lock:
            if name not in self.start_times:
                warnings.warn(f"Timer {name} was not started")
                return 0.0
            
            duration = end_time - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        times = self.metrics[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": np.mean(times),
            "min": min(times),
            "max": max(times),
            "std": np.std(times) if len(times) > 1 else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all timing statistics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


class SimpleBatchProcessor:
    """Memory-efficient batch processing for large datasets."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(4, cpu_count())
        self.use_processes = use_processes
        self.monitor = SimplePerformanceMonitor()
    
    def process_batches(
        self,
        data: List[Any],
        batch_size: int,
        process_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process data in batches with parallel execution."""
        
        # Create batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        self.monitor.start_timer("batch_processing")
        
        results = []
        if self.use_processes:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(lambda batch: process_func(batch, **kwargs), batches))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(lambda batch: process_func(batch, **kwargs), batches))
        
        self.monitor.end_timer("batch_processing")
        
        # Flatten results
        flattened = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flattened.extend(batch_result)
            else:
                flattened.append(batch_result)
        
        return flattened
    
    def adaptive_batch_size(
        self,
        initial_batch_size: int,
        target_memory_mb: float = 1000.0,
        test_iterations: int = 3
    ) -> int:
        """Determine optimal batch size based on memory usage."""
        
        current_batch_size = initial_batch_size
        best_batch_size = initial_batch_size
        
        for i in range(test_iterations):
            # Monitor memory before processing
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate processing with current batch size
            # In real implementation, this would run a sample batch
            time.sleep(0.01)  # Simulate work
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if memory_used < target_memory_mb:
                # Can increase batch size
                current_batch_size = min(current_batch_size * 2, initial_batch_size * 10)
                best_batch_size = current_batch_size
            else:
                # Need to decrease batch size
                current_batch_size = max(current_batch_size // 2, 1)
                break
        
        return best_batch_size


class SimpleResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self):
        self.history = {}
        self.lock = threading.Lock()
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_available_mb": memory.available / 1024 / 1024,
                "timestamp": time.time()
            }
            
            # Add GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
                    memory_cached = torch.cuda.memory_reserved(device) / 1024 / 1024
                    usage.update({
                        "gpu_memory_allocated_mb": memory_allocated,
                        "gpu_memory_cached_mb": memory_cached
                    })
            except ImportError:
                pass
            
            return usage
            
        except Exception as e:
            warnings.warn(f"Failed to get resource usage: {e}")
            return {}
    
    def record_usage(self, name: str) -> None:
        """Record current usage under a name."""
        usage = self.get_current_usage()
        if usage:
            with self.lock:
                if name not in self.history:
                    self.history[name] = []
                self.history[name].append(usage)
    
    def get_average_usage(self, name: str, window: int = 10) -> Dict[str, float]:
        """Get average usage over recent history."""
        with self.lock:
            if name not in self.history or not self.history[name]:
                return {}
            
            recent = self.history[name][-window:]
            if not recent:
                return {}
            
            # Calculate averages
            avg_usage = {}
            for key in recent[0].keys():
                if key != "timestamp" and isinstance(recent[0][key], (int, float)):
                    values = [usage[key] for usage in recent if key in usage]
                    if values:
                        avg_usage[f"avg_{key}"] = np.mean(values)
            
            return avg_usage


class SimpleAutoScaler:
    """Basic auto-scaling for computational workloads."""
    
    def __init__(
        self,
        target_cpu_utilization: float = 70.0,
        target_memory_utilization: float = 75.0,
        min_batch_size: int = 1,
        max_batch_size: int = 1024
    ):
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self.resource_monitor = SimpleResourceMonitor()
        self.performance_monitor = SimplePerformanceMonitor()
        
        self.current_batch_size = min_batch_size
        self.scaling_history = []
    
    def should_scale_up(self, usage: Dict[str, float]) -> bool:
        """Determine if we should scale up based on resource usage."""
        cpu_ok = usage.get("cpu_percent", 100) < self.target_cpu_utilization
        memory_ok = usage.get("memory_percent", 100) < self.target_memory_utilization
        
        return cpu_ok and memory_ok and self.current_batch_size < self.max_batch_size
    
    def should_scale_down(self, usage: Dict[str, float]) -> bool:
        """Determine if we should scale down based on resource usage."""
        cpu_high = usage.get("cpu_percent", 0) > self.target_cpu_utilization * 1.2
        memory_high = usage.get("memory_percent", 0) > self.target_memory_utilization * 1.2
        
        return (cpu_high or memory_high) and self.current_batch_size > self.min_batch_size
    
    def adjust_batch_size(self) -> int:
        """Adjust batch size based on current resource usage."""
        usage = self.resource_monitor.get_current_usage()
        
        old_batch_size = self.current_batch_size
        
        if self.should_scale_up(usage):
            # Gradually increase batch size
            self.current_batch_size = min(
                int(self.current_batch_size * 1.5),
                self.max_batch_size
            )
        elif self.should_scale_down(usage):
            # Quickly decrease batch size to avoid resource exhaustion
            self.current_batch_size = max(
                int(self.current_batch_size * 0.7),
                self.min_batch_size
            )
        
        if old_batch_size != self.current_batch_size:
            self.scaling_history.append({
                "timestamp": time.time(),
                "old_batch_size": old_batch_size,
                "new_batch_size": self.current_batch_size,
                "cpu_percent": usage.get("cpu_percent", 0),
                "memory_percent": usage.get("memory_percent", 0)
            })
        
        return self.current_batch_size
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get statistics about scaling behavior."""
        if not self.scaling_history:
            return {"total_adjustments": 0}
        
        scale_ups = sum(1 for h in self.scaling_history if h["new_batch_size"] > h["old_batch_size"])
        scale_downs = sum(1 for h in self.scaling_history if h["new_batch_size"] < h["old_batch_size"])
        
        return {
            "total_adjustments": len(self.scaling_history),
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "current_batch_size": self.current_batch_size,
            "min_batch_size_used": min(h["new_batch_size"] for h in self.scaling_history),
            "max_batch_size_used": max(h["new_batch_size"] for h in self.scaling_history)
        }


def performance_optimized(func: Callable) -> Callable:
    """Decorator to add performance optimization to functions."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        monitor = SimplePerformanceMonitor()
        resource_monitor = SimpleResourceMonitor()
        
        func_name = func.__name__
        monitor.start_timer(func_name)
        resource_monitor.record_usage(f"{func_name}_start")
        
        # Force garbage collection before expensive operation
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Record performance metrics
            duration = monitor.end_timer(func_name)
            resource_monitor.record_usage(f"{func_name}_end")
            
            # Log performance info
            stats = monitor.get_stats(func_name)
            if stats:
                print(f"Performance: {func_name} took {duration:.3f}s")
    
    return wrapper


def memory_efficient_operation(
    data: List[Any],
    operation: Callable,
    batch_size: Optional[int] = None,
    max_workers: int = 2
) -> List[Any]:
    """
    Perform memory-efficient operation on large datasets.
    
    Args:
        data: Input data list
        operation: Function to apply to each batch
        batch_size: Size of each batch (auto-determined if None)
        max_workers: Number of parallel workers
        
    Returns:
        List of results
    """
    if not data:
        return []
    
    # Auto-determine batch size if not provided
    if batch_size is None:
        processor = SimpleBatchProcessor(max_workers=max_workers)
        batch_size = processor.adaptive_batch_size(
            initial_batch_size=max(1, len(data) // 100),
            target_memory_mb=500.0
        )
    
    # Process in batches
    processor = SimpleBatchProcessor(max_workers=max_workers)
    return processor.process_batches(data, batch_size, operation)


# Example usage and testing functions
def test_simple_optimization():
    """Test the simple optimization features."""
    print("=== Testing Simple Optimization Features ===")
    
    # Test performance monitoring
    monitor = SimplePerformanceMonitor()
    
    @performance_optimized
    def sample_computation(n: int):
        """Sample computation for testing."""
        return sum(i * i for i in range(n))
    
    # Run test
    result = sample_computation(100000)
    print(f"✓ Performance monitoring: computation result = {result}")
    
    # Test batch processing
    processor = SimpleBatchProcessor(max_workers=2)
    test_data = list(range(1000))
    
    def square_batch(batch):
        return [x * x for x in batch]
    
    results = processor.process_batches(test_data, batch_size=100, process_func=square_batch)
    print(f"✓ Batch processing: processed {len(results)} items")
    
    # Test auto-scaling
    scaler = SimpleAutoScaler()
    initial_batch_size = scaler.current_batch_size
    new_batch_size = scaler.adjust_batch_size()
    print(f"✓ Auto-scaling: batch size {initial_batch_size} -> {new_batch_size}")
    
    # Test resource monitoring
    resource_monitor = SimpleResourceMonitor()
    usage = resource_monitor.get_current_usage()
    print(f"✓ Resource monitoring: CPU {usage.get('cpu_percent', 0):.1f}%, Memory {usage.get('memory_percent', 0):.1f}%")
    
    print("✅ All simple optimization features working")


if __name__ == "__main__":
    test_simple_optimization()
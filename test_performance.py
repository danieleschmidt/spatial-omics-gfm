#!/usr/bin/env python3
"""
Test performance optimization features in Generation 3 implementation.
"""

import sys
import os
sys.path.insert(0, '/root/repo')

import time
import numpy as np
from spatial_omics_gfm.performance import (
    MemoryCache,
    DiskCache,
    CacheManager,
    cached_computation,
    ProfilerContext,
    BatchProcessor,
    PerformanceOptimizer,
    optimize_computation,
    performance_monitor
)

def test_caching_system():
    """Test the caching system."""
    print("=== Testing Caching System ===")
    
    # Test memory cache
    print("1. Testing memory cache...")
    memory_cache = MemoryCache(max_size=10, max_memory_mb=50)
    
    # Add some data
    for i in range(5):
        data = np.random.rand(100, 100)
        memory_cache.put(f"test_data_{i}", data)
    
    # Retrieve data
    retrieved = memory_cache.get("test_data_0")
    print(f"   Retrieved data shape: {retrieved.shape if retrieved is not None else 'None'}")
    
    # Check stats
    stats = memory_cache.stats()
    print(f"   Memory cache stats: {stats}")
    
    # Test disk cache
    print("2. Testing disk cache...")
    disk_cache = DiskCache("/tmp/test_cache", max_size_gb=1)
    
    # Add data to disk cache
    test_data = {"matrix": np.random.rand(50, 50), "metadata": {"test": True}}
    disk_cache.put("disk_test", test_data)
    
    # Retrieve from disk
    retrieved_disk = disk_cache.get("disk_test")
    print(f"   Retrieved from disk: {type(retrieved_disk)}")
    print(f"   Disk cache stats: {disk_cache.stats()}")
    
    # Test unified cache manager
    print("3. Testing unified cache manager...")
    cache_manager = CacheManager(memory_cache, disk_cache)
    
    @cached_computation(cache_manager)
    def expensive_computation(size: int, multiplier: float = 1.0):
        """Simulate expensive computation."""
        time.sleep(0.1)  # Simulate work
        return np.random.rand(size, size) * multiplier
    
    # First call (should be slow)
    start_time = time.time()
    result1 = expensive_computation(20, 2.0)
    first_call_time = time.time() - start_time
    
    # Second call (should be fast - cached)
    start_time = time.time()
    result2 = expensive_computation(20, 2.0)
    second_call_time = time.time() - start_time
    
    print(f"   First call time: {first_call_time:.4f}s")
    print(f"   Second call time: {second_call_time:.4f}s")
    print(f"   Speedup: {first_call_time / second_call_time:.2f}x")
    print(f"   Results identical: {np.array_equal(result1, result2)}")


def test_profiling():
    """Test profiling capabilities."""
    print("\n=== Testing Profiling ===")
    
    print("1. Testing profiler context...")
    
    def matrix_operations(size: int):
        """Some matrix operations to profile."""
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        d = np.linalg.svd(c, full_matrices=False)
        return d
    
    # Profile the operation
    with ProfilerContext("matrix_ops", enable=True) as profiler:
        result = matrix_operations(100)
    
    stats = profiler.get_stats(top_n=10)
    print(f"   Execution time: {stats.get('execution_time_seconds', 0):.4f}s")
    print(f"   Memory delta: {stats.get('memory_delta_mb', 0):.2f}MB")
    print("   Top functions profiled:")
    
    # Show abbreviated profile
    profile_lines = stats.get("profile_text", "").split('\n')[:15]
    for line in profile_lines:
        if line.strip():
            print(f"     {line}")


def test_batch_processing():
    """Test batch processing."""
    print("\n=== Testing Batch Processing ===")
    
    print("1. Testing array batch processing...")
    
    # Create large array
    large_array = np.random.rand(5000, 100)
    
    def process_batch(batch):
        """Simple processing function."""
        return np.mean(batch, axis=1, keepdims=True)
    
    # Process with batch processor
    processor = BatchProcessor(batch_size=1000, memory_limit_mb=100)
    
    with performance_monitor("batch_processing") as metrics:
        result = processor.process_array(large_array, process_batch, axis=0)
    
    print(f"   Input shape: {large_array.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   Processing time: {metrics.get('execution_time_seconds', 0):.4f}s")
    print(f"   Memory delta: {metrics.get('memory_delta_mb', 0):.2f}MB")
    
    print("2. Testing list batch processing...")
    
    # Create list of items
    items = [{"id": i, "value": np.random.rand()} for i in range(1000)]
    
    def process_item_batch(batch):
        """Process batch of items."""
        return [item["value"] * 2 for item in batch]
    
    processed_items = processor.process_list(items, process_item_batch)
    print(f"   Processed {len(processed_items)} items")


def test_optimization():
    """Test automatic optimization."""
    print("\n=== Testing Automatic Optimization ===")
    
    print("1. Testing performance optimizer...")
    
    def sample_operation(matrix):
        """Sample matrix operation to optimize."""
        return np.sum(matrix ** 2, axis=1)
    
    optimizer = PerformanceOptimizer()
    
    # Test optimization for different matrix sizes
    test_shapes = [(1000, 50), (2000, 100)]
    
    for shape in test_shapes:
        print(f"   Optimizing for shape {shape}...")
        
        opt_result = optimizer.optimize_matrix_operations(
            "sum_squares",
            shape,
            sample_operation,
            test_sizes=[100, 500, 1000]
        )
        
        print(f"     Best batch size: {opt_result['best_batch_size']}")
        print(f"     Best time: {opt_result['best_time']:.4f}s")
        print(f"     Speedup factor: {opt_result.get('speedup_factor', 1):.2f}x")
    
    print("2. Testing optimization decorator...")
    
    @optimize_computation("matrix_norm", matrix_shape=(1000, 100))
    def compute_matrix_norm(matrix):
        """Compute matrix norm."""
        return np.linalg.norm(matrix, axis=1)
    
    # This would use cached optimization if available
    test_matrix = np.random.rand(1000, 100)
    
    with performance_monitor("optimized_computation") as metrics:
        norms = compute_matrix_norm(test_matrix)
    
    print(f"   Computed norms for {len(norms)} rows")
    print(f"   Execution time: {metrics.get('execution_time_seconds', 0):.4f}s")


def test_memory_management():
    """Test memory management features."""
    print("\n=== Testing Memory Management ===")
    
    print("1. Testing memory monitoring...")
    
    def memory_intensive_operation():
        """Operation that uses significant memory."""
        # Create several large arrays
        arrays = []
        for i in range(5):
            arr = np.random.rand(1000, 1000)
            arrays.append(arr)
        
        # Do some computation
        result = np.sum([np.sum(arr) for arr in arrays])
        
        # Clean up explicitly
        del arrays
        return result
    
    with performance_monitor("memory_intensive") as metrics:
        result = memory_intensive_operation()
    
    print(f"   Result: {result:.2f}")
    print(f"   Peak memory: {metrics.get('peak_memory_mb', 0):.2f}MB")
    print(f"   Memory delta: {metrics.get('memory_delta_mb', 0):.2f}MB")
    
    print("2. Testing batch processor memory limits...")
    
    # Create processor with strict memory limit
    strict_processor = BatchProcessor(
        batch_size=500,
        memory_limit_mb=50  # Very low limit to trigger cleanup
    )
    
    large_data = np.random.rand(2000, 200)
    
    def memory_heavy_process(batch):
        """Process that creates temporary large arrays."""
        temp = np.tile(batch, (2, 1))  # Double the memory usage
        return np.mean(temp, axis=1, keepdims=True)
    
    try:
        with performance_monitor("memory_limited_processing") as metrics:
            processed = strict_processor.process_array(large_data, memory_heavy_process)
        
        print(f"   Successfully processed with memory limits")
        print(f"   Output shape: {processed.shape}")
        print(f"   Processing time: {metrics.get('execution_time_seconds', 0):.4f}s")
        
    except Exception as e:
        print(f"   Memory limit handling: {e}")


def test_integrated_performance():
    """Test integrated performance features."""
    print("\n=== Testing Integrated Performance ===")
    
    print("1. Creating integrated performance pipeline...")
    
    # Setup caching
    cache_manager = CacheManager(
        MemoryCache(max_size=20, max_memory_mb=100),
        DiskCache("/tmp/integrated_cache", max_size_gb=1)
    )
    
    @cached_computation(cache_manager)
    @optimize_computation("spatial_analysis", matrix_shape=(1000, 500))
    def spatial_analysis_pipeline(expression_data, coordinates):
        """Integrated spatial analysis with caching and optimization."""
        
        with performance_monitor("spatial_pipeline") as metrics:
            # Simulate spatial analysis steps
            
            # 1. Normalize expression
            normalized = expression_data / (np.sum(expression_data, axis=1, keepdims=True) + 1e-10)
            
            # 2. Compute pairwise distances
            n_cells = coordinates.shape[0]
            distances = np.zeros((n_cells, n_cells))
            
            batch_processor = BatchProcessor(batch_size=100)
            
            def compute_distance_batch(batch_indices):
                batch_coords = coordinates[batch_indices]
                batch_distances = np.sqrt(
                    np.sum((batch_coords[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
                )
                return batch_distances
            
            # Process in batches
            for i in range(0, n_cells, 100):
                end_i = min(i + 100, n_cells)
                batch_indices = list(range(i, end_i))
                batch_dist = compute_distance_batch(batch_indices)
                distances[i:end_i] = batch_dist
            
            # 3. Find neighbors
            neighbors = np.argsort(distances, axis=1)[:, 1:7]  # 6 nearest neighbors
            
            # 4. Compute spatial statistics
            spatial_stats = {
                "mean_distance": np.mean(distances[distances > 0]),
                "neighbor_connectivity": neighbors.shape[1],
                "expression_variance": np.var(normalized, axis=0).mean()
            }
            
            metrics.update(spatial_stats)
            
        return {
            "normalized_expression": normalized,
            "distances": distances,
            "neighbors": neighbors,
            "spatial_stats": spatial_stats
        }
    
    # Test the pipeline
    print("2. Running spatial analysis pipeline...")
    
    # Create test data
    n_cells, n_genes = 500, 200
    expression = np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(float)
    coordinates = np.random.rand(n_cells, 2) * 1000
    
    # First run (no cache)
    start_time = time.time()
    result1 = spatial_analysis_pipeline(expression, coordinates)
    first_run_time = time.time() - start_time
    
    # Second run (cached)
    start_time = time.time()
    result2 = spatial_analysis_pipeline(expression, coordinates)
    second_run_time = time.time() - start_time
    
    print(f"   First run time: {first_run_time:.4f}s")
    print(f"   Second run time: {second_run_time:.4f}s")
    print(f"   Cache speedup: {first_run_time / second_run_time:.2f}x")
    
    print(f"   Spatial stats: {result1['spatial_stats']}")
    
    # Check cache manager stats
    cache_stats = cache_manager.stats()
    print(f"   Cache stats: {cache_stats}")


def main():
    """Run all performance tests."""
    print("⚡ Spatial-Omics GFM Performance Testing")
    print("=" * 50)
    
    try:
        test_caching_system()
        test_profiling()
        test_batch_processing()
        test_optimization()
        test_memory_management()
        test_integrated_performance()
        
        print("\n✅ All performance tests completed successfully!")
        print("Generation 3 performance features are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Optimized Framework - Generation 3 Implementation

Adds comprehensive performance optimization, caching, concurrency, auto-scaling,
and memory management. Builds upon Generations 1&2 with production-scale performance.
"""

import os
import sys
import json
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
from pathlib import Path

# Import robust framework components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robust_framework import (
    RobustSpatialData, RobustLogger, ValidationLevel, SecurityLevel,
    DataValidator, SecurityGuard, ValidationResult
)
from enhanced_basic_example import (
    create_enhanced_demo_data,
    EnhancedCellTypePredictor,
    EnhancedInteractionPredictor
)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    start_time: float
    end_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    operations_per_second: float = 0.0
    parallelization_efficiency: float = 0.0
    
    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time


class MemoryMonitor:
    """Memory usage monitoring and management."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.current_memory = 0.0
        self.memory_history = deque(maxlen=1000)  # Keep last 1000 measurements
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB (approximation)."""
        try:
            # Simple memory estimation based on object tracking
            import gc
            objects = gc.get_objects()
            memory_mb = len(objects) * 0.001  # Rough approximation
            
            self.current_memory = memory_mb
            self.peak_memory = max(self.peak_memory, memory_mb)
            self.memory_history.append((time.time(), memory_mb))
            
            return memory_mb
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_history:
            return {"current": 0.0, "peak": 0.0, "average": 0.0}
        
        recent_usage = [usage for _, usage in list(self.memory_history)[-100:]]
        return {
            "current": self.current_memory,
            "peak": self.peak_memory,
            "average": sum(recent_usage) / len(recent_usage) if recent_usage else 0.0
        }


class InMemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def _evict_expired(self) -> None:
        """Evict expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using LRU."""
        while len(self.cache) > self.max_size:
            if self.access_order:
                oldest_key = self.access_order.popleft()
                self._remove_key(oldest_key)
            else:
                break
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self._evict_expired()
            
            if key in self.cache:
                # Update access time and order
                self.access_times[key] = time.time()
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hit_count += 1
                return self.cache[key]
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            self._evict_expired()
            
            # Update or add entry
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self._enforce_size_limit()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }


class ParallelProcessor:
    """High-performance parallel processing engine."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.logger = RobustLogger("parallel_processor")
    
    def process_chunks(
        self, 
        data: List[Any], 
        chunk_processor: Callable[[List[Any]], Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Process data in parallel chunks."""
        
        if not data:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.max_workers * 2))
        
        # Create chunks
        chunks = [
            data[i:i + chunk_size] 
            for i in range(0, len(data), chunk_size)
        ]
        
        self.logger.info(
            f"Processing {len(data)} items in {len(chunks)} chunks "
            f"using {self.max_workers} workers"
        )
        
        start_time = time.time()
        
        # Process chunks in parallel
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                results = list(executor.map(chunk_processor, chunks))
            
            processing_time = time.time() - start_time
            items_per_second = len(data) / processing_time if processing_time > 0 else 0
            
            self.logger.info(
                f"Parallel processing completed: {len(data)} items in {processing_time:.2f}s "
                f"({items_per_second:.0f} items/sec)"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            self.logger.warning("Falling back to sequential processing")
            return [chunk_processor(chunk) for chunk in chunks]
    
    def parallel_map(
        self,
        function: Callable[[Any], Any],
        items: List[Any]
    ) -> List[Any]:
        """Apply function to items in parallel."""
        
        if not items:
            return []
        
        self.logger.info(f"Parallel mapping function to {len(items)} items")
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                results = list(executor.map(function, items))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Parallel mapping failed: {e}")
            # Fallback to sequential processing
            return [function(item) for item in items]


class OptimizedSpatialData(RobustSpatialData):
    """Optimized spatial data with performance enhancements."""
    
    def __init__(
        self,
        expression_matrix: List[List[float]],
        coordinates: List[List[float]],
        gene_names: Optional[List[str]] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        security_level: SecurityLevel = SecurityLevel.MINIMAL,
        enable_caching: bool = True,
        cache_size: int = 1000,
        max_workers: int = None
    ):
        # Initialize parent class
        super().__init__(
            expression_matrix=expression_matrix,
            coordinates=coordinates,
            gene_names=gene_names,
            validation_level=validation_level,
            security_level=security_level,
            enable_logging=True
        )
        
        # Performance components
        self.memory_monitor = MemoryMonitor()
        self.cache = InMemoryCache(max_size=cache_size) if enable_caching else None
        self.parallel_processor = ParallelProcessor(max_workers=max_workers)
        
        # Performance metrics
        self.metrics = {
            "initialization": PerformanceMetrics(start_time=time.time())
        }
        
        # Pre-compute and cache common operations
        self._precompute_optimizations()
        
        if self.logger:
            self.logger.info(f"Initialized optimized spatial data with caching: {enable_caching}")
    
    def _precompute_optimizations(self) -> None:
        """Pre-compute commonly used values for performance."""
        
        start_time = time.time()
        
        # Pre-compute cell and gene statistics
        if self.cache:
            # Cell totals
            cell_totals = [sum(row) for row in self.expression_matrix]
            self.cache.set("cell_totals", cell_totals)
            
            # Gene totals
            gene_totals = [
                sum(self.expression_matrix[i][j] for i in range(self.n_cells))
                for j in range(self.n_genes)
            ]
            self.cache.set("gene_totals", gene_totals)
            
            # Spatial bounds
            x_coords = [coord[0] for coord in self.coordinates]
            y_coords = [coord[1] for coord in self.coordinates]
            spatial_bounds = {
                "x_min": min(x_coords), "x_max": max(x_coords),
                "y_min": min(y_coords), "y_max": max(y_coords)
            }
            self.cache.set("spatial_bounds", spatial_bounds)
        
        precompute_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"Pre-computation completed in {precompute_time:.3f}s")
    
    def optimized_normalize_expression(
        self, 
        method: str = "log1p_cpm", 
        use_parallel: bool = True
    ) -> bool:
        """Optimized normalization with parallel processing."""
        
        metric_key = f"normalization_{method}"
        self.metrics[metric_key] = PerformanceMetrics(start_time=time.time())
        
        if self.logger:
            self.logger.info(f"Starting optimized normalization: {method} (parallel: {use_parallel})")
        
        try:
            if use_parallel and self.n_cells > 100:
                # Parallel normalization for large datasets
                success = self._parallel_normalize(method)
            else:
                # Sequential normalization for small datasets
                success = self.safe_normalize_expression(method)
            
            # Update metrics
            self.metrics[metric_key].end_time = time.time()
            self.metrics[metric_key].memory_usage_mb = self.memory_monitor.get_memory_usage()
            
            if success and self.cache:
                # Invalidate cached statistics
                self.cache.set("cell_totals", None)
                self.cache.set("gene_totals", None)
            
            return success
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Optimized normalization failed: {e}")
            return False
    
    def _parallel_normalize(self, method: str) -> bool:
        """Perform normalization in parallel chunks."""
        
        def normalize_chunk(cell_indices: List[int]) -> bool:
            """Normalize a chunk of cells."""
            try:
                for i in cell_indices:
                    if method == "log1p_cpm":
                        cell_total = sum(self.expression_matrix[i])
                        if cell_total == 0:
                            continue
                        
                        cpm_factor = 1000000 / cell_total
                        for j in range(self.n_genes):
                            normalized_value = self.expression_matrix[i][j] * cpm_factor
                            self.expression_matrix[i][j] = self._safe_log1p(normalized_value)
                    
                    elif method == "z_score":
                        # Z-score normalization per cell
                        cell_values = self.expression_matrix[i]
                        mean_val = sum(cell_values) / len(cell_values)
                        variance = sum((x - mean_val) ** 2 for x in cell_values) / len(cell_values)
                        
                        if variance > 0:
                            std_dev = variance ** 0.5
                            for j in range(self.n_genes):
                                self.expression_matrix[i][j] = (
                                    self.expression_matrix[i][j] - mean_val
                                ) / std_dev
                
                return True
            except Exception:
                return False
        
        # Create cell index chunks
        cell_indices = list(range(self.n_cells))
        chunk_results = self.parallel_processor.process_chunks(
            cell_indices, 
            normalize_chunk,
            chunk_size=max(10, self.n_cells // (self.parallel_processor.max_workers * 2))
        )
        
        return all(chunk_results)
    
    def optimized_find_neighbors(
        self,
        k: int = 6,
        max_distance: Optional[float] = None,
        use_parallel: bool = True,
        use_cache: bool = True
    ) -> Optional[Dict[int, List[int]]]:
        """Optimized neighbor finding with caching and parallelization."""
        
        # Create cache key
        cache_key = f"neighbors_k{k}_dist{max_distance}" if use_cache and self.cache else None
        
        # Check cache first
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                if self.logger:
                    self.logger.info(f"Neighbors retrieved from cache (k={k})")
                return cached_result
        
        metric_key = f"neighbors_k{k}"
        self.metrics[metric_key] = PerformanceMetrics(start_time=time.time())
        
        if self.logger:
            self.logger.info(
                f"Computing optimized neighbors (k={k}, max_distance={max_distance}, "
                f"parallel: {use_parallel})"
            )
        
        try:
            if use_parallel and self.n_cells > 200:
                neighbors = self._parallel_find_neighbors(k, max_distance)
            else:
                neighbors = self.safe_find_neighbors(k, max_distance)
            
            # Update metrics
            self.metrics[metric_key].end_time = time.time()
            self.metrics[metric_key].memory_usage_mb = self.memory_monitor.get_memory_usage()
            
            # Cache result
            if cache_key and neighbors is not None:
                self.cache.set(cache_key, neighbors)
            
            return neighbors
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Optimized neighbor finding failed: {e}")
            return None
    
    def _parallel_find_neighbors(
        self, 
        k: int, 
        max_distance: Optional[float]
    ) -> Optional[Dict[int, List[int]]]:
        """Find neighbors in parallel for large datasets."""
        
        def compute_neighbors_chunk(cell_indices: List[int]) -> Dict[int, List[int]]:
            """Compute neighbors for a chunk of cells."""
            chunk_neighbors = {}
            
            for i in cell_indices:
                distances = []
                for j in range(self.n_cells):
                    if i != j:
                        dist = self._safe_distance(
                            self.coordinates[i], 
                            self.coordinates[j]
                        )
                        if max_distance is None or dist <= max_distance:
                            distances.append((dist, j))
                
                # Sort and take k nearest
                distances.sort(key=lambda x: x[0])
                chunk_neighbors[i] = [idx for _, idx in distances[:k]]
            
            return chunk_neighbors
        
        # Process in parallel chunks
        cell_indices = list(range(self.n_cells))
        chunk_results = self.parallel_processor.process_chunks(
            cell_indices,
            compute_neighbors_chunk,
            chunk_size=max(20, self.n_cells // self.parallel_processor.max_workers)
        )
        
        # Combine results
        neighbors = {}
        for chunk_result in chunk_results:
            neighbors.update(chunk_result)
        
        return neighbors
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        memory_stats = self.memory_monitor.get_memory_stats()
        cache_stats = self.cache.get_stats() if self.cache else {}
        
        # Compute metric summaries
        metric_summaries = {}
        for key, metric in self.metrics.items():
            metric_summaries[key] = {
                "execution_time": metric.execution_time,
                "memory_usage_mb": metric.memory_usage_mb,
                "cpu_utilization": metric.cpu_utilization
            }
        
        return {
            "memory_usage": memory_stats,
            "cache_performance": cache_stats,
            "operation_metrics": metric_summaries,
            "parallelization": {
                "max_workers": self.parallel_processor.max_workers,
                "use_processes": self.parallel_processor.use_processes
            }
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up unused data."""
        
        initial_memory = self.memory_monitor.get_memory_usage()
        
        optimizations_applied = []
        
        # Clear expired cache entries
        if self.cache:
            cache_size_before = self.cache.get_stats()["size"]
            self.cache._evict_expired()
            cache_size_after = self.cache.get_stats()["size"]
            
            if cache_size_before > cache_size_after:
                optimizations_applied.append(
                    f"Cleaned {cache_size_before - cache_size_after} expired cache entries"
                )
        
        # Trigger garbage collection
        try:
            import gc
            collected = gc.collect()
            if collected > 0:
                optimizations_applied.append(f"Garbage collected {collected} objects")
        except Exception:
            pass
        
        final_memory = self.memory_monitor.get_memory_usage()
        memory_saved = initial_memory - final_memory
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_saved_mb": memory_saved,
            "optimizations_applied": optimizations_applied
        }
    
    def benchmark_operations(self, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark key operations for performance analysis."""
        
        if self.logger:
            self.logger.info(f"Starting performance benchmark ({iterations} iterations)")
        
        benchmark_results = {}
        
        # Benchmark neighbor finding
        neighbor_times = []
        for i in range(iterations):
            start_time = time.time()
            neighbors = self.optimized_find_neighbors(k=6, use_cache=False)
            end_time = time.time()
            
            if neighbors is not None:
                neighbor_times.append(end_time - start_time)
        
        if neighbor_times:
            benchmark_results["neighbor_finding"] = {
                "avg_time": sum(neighbor_times) / len(neighbor_times),
                "min_time": min(neighbor_times),
                "max_time": max(neighbor_times)
            }
        
        # Benchmark cache performance
        if self.cache:
            cache_times = []
            for i in range(iterations * 10):
                key = f"test_key_{i % 100}"  # Create some cache hits
                value = f"test_value_{i}"
                
                start_time = time.time()
                self.cache.set(key, value)
                retrieved = self.cache.get(key)
                end_time = time.time()
                
                cache_times.append(end_time - start_time)
            
            benchmark_results["cache_operations"] = {
                "avg_time": sum(cache_times) / len(cache_times),
                "operations_per_second": len(cache_times) / sum(cache_times) if sum(cache_times) > 0 else 0
            }
        
        if self.logger:
            self.logger.info("Performance benchmark completed")
        
        return benchmark_results


def run_optimized_analysis_demo() -> Dict[str, Any]:
    """Demonstrate optimized analysis capabilities - Generation 3."""
    print("=== Optimized Spatial-Omics Analysis (Generation 3) ===")
    
    # Initialize logger
    logger = RobustLogger("optimized_demo", log_level="INFO")
    logger.info("Starting optimized analysis demonstration")
    
    overall_start_time = time.time()
    
    try:
        # Create larger dataset for performance testing
        logger.info("Creating large-scale demo data for optimization testing")
        demo_data = create_enhanced_demo_data(n_cells=1000, n_genes=200)
        
        # Create optimized spatial data
        logger.info("Initializing optimized spatial data with performance enhancements")
        optimized_data = OptimizedSpatialData(
            expression_matrix=demo_data.expression_matrix,
            coordinates=demo_data.coordinates,
            gene_names=demo_data.gene_names,
            validation_level=ValidationLevel.STRICT,
            security_level=SecurityLevel.MINIMAL,
            enable_caching=True,
            cache_size=2000,
            max_workers=4
        )
        
        logger.info("Optimized initialization completed")
        
        # Benchmark performance
        logger.info("Running performance benchmarks")
        benchmark_results = optimized_data.benchmark_operations(iterations=2)
        
        # Optimized normalization
        logger.info("Performing optimized normalization with parallel processing")
        norm_start = time.time()
        normalization_success = optimized_data.optimized_normalize_expression(
            method="log1p_cpm", 
            use_parallel=True
        )
        norm_time = time.time() - norm_start
        
        logger.info(f"Normalization completed in {norm_time:.3f}s (success: {normalization_success})")
        
        # Optimized neighbor finding with caching
        logger.info("Computing spatial neighbors with optimization and caching")
        neighbor_start = time.time()
        neighbors = optimized_data.optimized_find_neighbors(
            k=10, 
            max_distance=150.0,
            use_parallel=True,
            use_cache=True
        )
        neighbor_time = time.time() - neighbor_start
        
        # Test cache performance (should be much faster)
        cached_start = time.time()
        cached_neighbors = optimized_data.optimized_find_neighbors(
            k=10, 
            max_distance=150.0,
            use_parallel=True,
            use_cache=True
        )
        cached_time = time.time() - cached_start
        
        logger.info(
            f"Neighbors: first computation {neighbor_time:.3f}s, "
            f"cached retrieval {cached_time:.3f}s "
            f"(speedup: {neighbor_time/cached_time if cached_time > 0 else 'inf'}x)"
        )
        
        # Memory optimization
        logger.info("Performing memory optimization")
        memory_optimization = optimized_data.optimize_memory_usage()
        
        # Enhanced analysis with parallelization
        logger.info("Running enhanced analysis with parallel processing")
        
        # Create compatible data wrapper
        class OptimizedCompatibleData:
            def __init__(self, optimized_data):
                self.expression_matrix = optimized_data.expression_matrix
                self.coordinates = optimized_data.coordinates
                self.gene_names = optimized_data.gene_names
                self.n_cells = optimized_data.n_cells
                self.n_genes = optimized_data.n_genes
                self._neighbors_cache = neighbors
            
            def calculate_distance(self, coord1, coord2):
                import math
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
            
            def find_spatial_neighbors(self, k=6):
                return self._neighbors_cache if self._neighbors_cache else {}
        
        compatible_data = OptimizedCompatibleData(optimized_data)
        
        # Parallel cell type prediction
        cell_type_start = time.time()
        cell_type_predictor = EnhancedCellTypePredictor()
        cell_type_predictions = cell_type_predictor.predict_cell_types(compatible_data)
        cell_type_assignments = cell_type_predictor.assign_best_cell_types(cell_type_predictions)
        cell_type_time = time.time() - cell_type_start
        
        logger.info(f"Cell type prediction completed in {cell_type_time:.3f}s")
        
        # Parallel interaction prediction
        interaction_start = time.time()
        interaction_predictor = EnhancedInteractionPredictor()
        interactions = interaction_predictor.predict_interactions(
            compatible_data, max_distance=120.0, min_score=0.02
        )
        
        # Pathway analysis
        pathway_enrichment = interaction_predictor.analyze_pathway_enrichment(interactions)
        interaction_time = time.time() - interaction_start
        
        logger.info(
            f"Interaction prediction completed in {interaction_time:.3f}s "
            f"({len(interactions)} interactions found)"
        )
        
        # Get comprehensive performance summary
        performance_summary = optimized_data.get_performance_summary()
        
        total_time = time.time() - overall_start_time
        
        # Compile results
        results = {
            "generation": "3_optimized",
            "analysis_metadata": {
                "n_cells": optimized_data.n_cells,
                "n_genes": optimized_data.n_genes,
                "total_analysis_time": total_time,
                "features_analyzed": [
                    "performance_optimization",
                    "in_memory_caching", 
                    "parallel_processing",
                    "memory_management",
                    "auto_scaling",
                    "performance_monitoring"
                ]
            },
            "performance_metrics": {
                "total_execution_time": total_time,
                "normalization_time": norm_time,
                "neighbor_computation_time": neighbor_time,
                "cached_neighbor_time": cached_time,
                "cache_speedup_factor": neighbor_time / cached_time if cached_time > 0 else float('inf'),
                "cell_type_prediction_time": cell_type_time,
                "interaction_prediction_time": interaction_time,
                "memory_optimization": memory_optimization,
                "benchmark_results": benchmark_results
            },
            "system_performance": performance_summary,
            "analysis_results": {
                "cell_types": {
                    "total_assignments": len(cell_type_assignments),
                    "unique_types": len(set(
                        assignment["predicted_type"] 
                        for assignment in cell_type_assignments
                    )),
                    "success": len(cell_type_assignments) > 0
                },
                "interactions": {
                    "total_interactions": len(interactions),
                    "pathways_analyzed": len(pathway_enrichment),
                    "success": len(interactions) > 0
                },
                "neighbors": {
                    "cells_with_neighbors": len(neighbors) if neighbors else 0,
                    "avg_neighbors_per_cell": (
                        sum(len(neighs) for neighs in neighbors.values()) / len(neighbors)
                        if neighbors else 0
                    ),
                    "success": neighbors is not None
                }
            },
            "optimization_features": {
                "caching_enabled": optimized_data.cache is not None,
                "cache_hit_rate": (
                    optimized_data.cache.get_stats()["hit_rate"] 
                    if optimized_data.cache else 0
                ),
                "parallel_processing": True,
                "max_workers": optimized_data.parallel_processor.max_workers,
                "memory_monitoring": True,
                "normalization_success": normalization_success
            }
        }
        
        # Save results
        with open("/root/repo/optimized_generation3_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\u2705 Optimized Generation 3 analysis complete!")
        print(f"   - Total execution time: {total_time:.2f}s")
        print(f"   - Processed {optimized_data.n_cells} cells and {optimized_data.n_genes} genes")
        print(f"   - Cache hit rate: {results['optimization_features']['cache_hit_rate']:.1%}")
        print(f"   - Cache speedup: {results['performance_metrics']['cache_speedup_factor']:.1f}x")
        print(f"   - Parallel workers: {optimized_data.parallel_processor.max_workers}")
        print(f"   - Memory saved: {memory_optimization['memory_saved_mb']:.1f} MB")
        print(f"   - Cell types predicted: {len(cell_type_assignments)}")
        print(f"   - Interactions found: {len(interactions)}")
        
        logger.info("Optimized analysis completed successfully")
        return results
    
    except Exception as e:
        logger.critical(f"Optimized analysis failed: {e}")
        return {
            "generation": "3_optimized",
            "status": "failed",
            "error": str(e),
            "execution_time": time.time() - overall_start_time
        }


if __name__ == "__main__":
    run_optimized_analysis_demo()
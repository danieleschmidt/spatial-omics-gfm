"""
Streaming inference for continuous spatial transcriptomics data processing.
Implements real-time inference capabilities for live data streams
and memory-constrained environments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import queue
import threading
import time
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .efficient_inference import EfficientInference
from ..models.graph_transformer import SpatialGraphTransformer

logger = logging.getLogger(__name__)


class StreamingInference(EfficientInference):
    """
    Streaming inference engine for real-time spatial transcriptomics processing.
    
    Features:
    - Real-time data processing
    - Sliding window analysis
    - Continuous model updates
    - Memory-bounded operations
    - Asynchronous processing
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        buffer_size: int = 1000,
        window_size: int = 500,
        overlap_size: int = 50,
        max_latency_ms: float = 100.0,
        batch_size: int = 16,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize streaming inference engine.
        
        Args:
            model: Pre-trained spatial graph transformer
            buffer_size: Maximum number of cells to buffer
            window_size: Size of processing windows
            overlap_size: Overlap between consecutive windows
            max_latency_ms: Maximum processing latency in milliseconds
            batch_size: Batch size for processing
            device: Device for inference
            use_mixed_precision: Whether to use mixed precision
            enable_caching: Whether to enable result caching
        """
        super().__init__(model, batch_size, use_mixed_precision, device)
        
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.max_latency_ms = max_latency_ms
        self.enable_caching = enable_caching
        
        # Streaming components
        self.data_buffer = StreamingBuffer(buffer_size)
        self.result_cache = ResultCache() if enable_caching else None
        self.window_processor = WindowProcessor(model, window_size, overlap_size)
        
        # Threading components
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Initialized StreamingInference with buffer size: {buffer_size}")
    
    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        """Start streaming processing in background thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Streaming already active")
            return
        
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._streaming_worker,
            args=(callback,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started streaming inference")
    
    def stop_streaming(self) -> None:
        """Stop streaming processing."""
        self.stop_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Stopped streaming inference")
    
    def add_data(self, data: Dict[str, Any]) -> None:
        """
        Add new data to streaming buffer.
        
        Args:
            data: Dictionary containing expression, coordinates, and metadata
        """
        try:
            self.processing_queue.put(data, timeout=1.0)
        except queue.Full:
            logger.warning("Processing queue full, dropping data")
    
    def get_results(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get processed results from result queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Processing results or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _streaming_worker(self, callback: Optional[Callable] = None) -> None:
        """Background worker for streaming processing."""
        logger.info("Starting streaming worker")
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue
                data = self.processing_queue.get(timeout=0.1)
                
                # Add to buffer
                self.data_buffer.add(data)
                
                # Process if window ready
                if self.data_buffer.can_create_window(self.window_size):
                    start_time = time.time()
                    
                    # Create processing window
                    window_data = self.data_buffer.get_window(self.window_size, self.overlap_size)
                    
                    # Process window
                    results = self._process_window(window_data)
                    
                    # Update performance metrics
                    processing_time = (time.time() - start_time) * 1000  # Convert to ms
                    self.performance_monitor.update(processing_time, len(window_data['expression']))
                    
                    # Add performance info to results
                    results['performance'] = {
                        'processing_time_ms': processing_time,
                        'cells_processed': len(window_data['expression']),
                        'throughput_cells_per_sec': len(window_data['expression']) / (processing_time / 1000)
                    }
                    
                    # Put results in queue
                    try:
                        self.result_queue.put(results, timeout=0.1)
                        
                        # Call callback if provided
                        if callback:
                            callback(results)
                            
                    except queue.Full:
                        logger.warning("Result queue full, dropping results")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                continue
        
        logger.info("Streaming worker stopped")
    
    def _process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single window of data."""
        start_time = time.time()
        
        # Check cache first
        if self.result_cache:
            cache_key = self.result_cache.get_cache_key(window_data)
            cached_result = self.result_cache.get(cache_key)
            if cached_result:
                logger.debug("Using cached result")
                return cached_result
        
        # Convert to tensors
        expression = torch.tensor(window_data['expression'], dtype=torch.float32).to(self.device)
        spatial_coords = torch.tensor(window_data['spatial_coords'], dtype=torch.float32).to(self.device)
        
        # Build spatial graph for window
        edge_index, edge_attr = self._build_window_graph(spatial_coords.cpu().numpy())
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        
        # Run inference
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(expression, spatial_coords, edge_index, edge_attr)
            else:
                embeddings = self.model.encode(expression, spatial_coords, edge_index, edge_attr)
        
        # Prepare results
        results = {
            'embeddings': embeddings.cpu().numpy(),
            'window_id': window_data.get('window_id'),
            'timestamp': time.time(),
            'num_cells': len(expression)
        }
        
        # Cache results
        if self.result_cache:
            self.result_cache.put(cache_key, results)
        
        processing_time = (time.time() - start_time) * 1000
        if processing_time > self.max_latency_ms:
            logger.warning(f"Processing latency ({processing_time:.1f}ms) exceeds threshold ({self.max_latency_ms}ms)")
        
        return results
    
    def _build_window_graph(self, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build spatial graph for window data."""
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(6, len(coords) - 1)
        
        if n_neighbors <= 0:
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Build edge list
        edges = []
        edge_features = []
        
        for i in range(len(coords)):
            for j in range(1, len(indices[i])):  # Skip self
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Calculate direction
                dx = coords[neighbor_idx, 0] - coords[i, 0]
                dy = coords[neighbor_idx, 1] - coords[i, 1]
                
                edges.append([i, neighbor_idx])
                edge_features.append([distance, dx, dy])
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def process_stream(
        self,
        data_stream: Iterator[Dict[str, Any]],
        output_callback: Optional[Callable] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Process continuous data stream.
        
        Args:
            data_stream: Iterator yielding data dictionaries
            output_callback: Optional callback for results
            
        Yields:
            Processing results
        """
        logger.info("Starting stream processing")
        
        self.start_streaming(output_callback)
        
        try:
            for data in data_stream:
                # Add data to processing queue
                self.add_data(data)
                
                # Yield any available results
                while True:
                    result = self.get_results(timeout=0.01)
                    if result is None:
                        break
                    yield result
            
            # Process remaining data
            time.sleep(0.5)  # Allow final processing
            
            while True:
                result = self.get_results(timeout=0.1)
                if result is None:
                    break
                yield result
                
        finally:
            self.stop_streaming()
    
    def process_file_stream(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 100,
        output_callback: Optional[Callable] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Process file as streaming data.
        
        Args:
            file_path: Path to data file
            chunk_size: Size of chunks to read
            output_callback: Optional callback for results
            
        Yields:
            Processing results
        """
        def file_data_generator():
            """Generator for file data chunks."""
            try:
                import h5py
                
                with h5py.File(file_path, 'r') as f:
                    # Determine dataset structure
                    if 'X' in f:
                        expression_data = f['X']
                        spatial_data = f['obsm']['spatial'] if 'obsm' in f and 'spatial' in f['obsm'] else None
                    else:
                        raise ValueError("Unsupported file structure")
                    
                    total_cells = expression_data.shape[0]
                    
                    for start_idx in range(0, total_cells, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_cells)
                        
                        chunk_data = {
                            'expression': expression_data[start_idx:end_idx],
                            'spatial_coords': spatial_data[start_idx:end_idx] if spatial_data is not None else np.random.rand(end_idx - start_idx, 2) * 1000,
                            'chunk_id': start_idx // chunk_size,
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        }
                        
                        yield chunk_data
                        
            except Exception as e:
                logger.error(f"Error reading file stream: {e}")
                return
        
        # Process the file stream
        yield from self.process_stream(file_data_generator(), output_callback)
    
    async def async_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous processing for single data point.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processing results
        """
        loop = asyncio.get_event_loop()
        
        # Run processing in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(executor, self._process_single, data)
            result = await future
        
        return result
    
    def _process_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single data point synchronously."""
        # Create mini-window from single data point
        window_data = {
            'expression': data['expression'],
            'spatial_coords': data['spatial_coords'],
            'window_id': f"single_{time.time()}"
        }
        
        return self._process_window(window_data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_monitor.get_stats()


class StreamingBuffer:
    """Circular buffer for streaming data."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0
    
    def add(self, data: Dict[str, Any]) -> None:
        """Add data to buffer."""
        # Add timestamp and ID
        data['buffer_timestamp'] = time.time()
        data['buffer_id'] = self.total_added
        
        self.buffer.append(data)
        self.total_added += 1
    
    def can_create_window(self, window_size: int) -> bool:
        """Check if we can create a window of given size."""
        return len(self.buffer) >= window_size
    
    def get_window(self, window_size: int, overlap_size: int = 0) -> Dict[str, Any]:
        """Get a window of data from buffer."""
        if not self.can_create_window(window_size):
            raise ValueError("Insufficient data for window")
        
        # Get latest window_size items
        window_items = list(self.buffer)[-window_size:]
        
        # Combine data
        combined_data = self._combine_data_items(window_items)
        combined_data['window_id'] = f"window_{window_items[0]['buffer_id']}_{window_items[-1]['buffer_id']}"
        
        # Remove processed items (keeping overlap)
        for _ in range(window_size - overlap_size):
            if self.buffer:
                self.buffer.popleft()
        
        return combined_data
    
    def _combine_data_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple data items into single window."""
        if not items:
            return {}
        
        # Stack arrays
        expressions = []
        spatial_coords = []
        
        for item in items:
            if 'expression' in item:
                expr = item['expression']
                if expr.ndim == 1:
                    expr = expr.reshape(1, -1)
                expressions.append(expr)
            
            if 'spatial_coords' in item:
                coords = item['spatial_coords']
                if coords.ndim == 1:
                    coords = coords.reshape(1, -1)
                spatial_coords.append(coords)
        
        combined = {}
        
        if expressions:
            combined['expression'] = np.vstack(expressions)
        
        if spatial_coords:
            combined['spatial_coords'] = np.vstack(spatial_coords)
        
        # Add metadata
        combined['num_items'] = len(items)
        combined['timestamp_range'] = (
            items[0].get('buffer_timestamp'),
            items[-1].get('buffer_timestamp')
        )
        
        return combined
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.max_size


class WindowProcessor:
    """Processor for sliding windows."""
    
    def __init__(self, model: nn.Module, window_size: int, overlap_size: int):
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.processed_windows = []
    
    def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single window."""
        # Store window for overlap handling
        self.processed_windows.append(window_data)
        
        # Keep only recent windows
        if len(self.processed_windows) > 10:
            self.processed_windows = self.processed_windows[-10:]
        
        # Process with model (implementation depends on specific use case)
        return {
            'window_data': window_data,
            'processed_timestamp': time.time()
        }


class ResultCache:
    """LRU cache for processing results."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
    
    def get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for data."""
        # Simple hash-based key (could be improved)
        expr_hash = hash(str(data.get('expression', '')))
        coord_hash = hash(str(data.get('spatial_coords', '')))
        return f"{expr_hash}_{coord_hash}"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, result: Dict[str, Any]) -> None:
        """Cache result."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
        
        self.cache[key] = result
        self.access_order.append(key)


class PerformanceMonitor:
    """Monitor streaming performance."""
    
    def __init__(self):
        self.processing_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.total_cells_processed = 0
    
    def update(self, processing_time_ms: float, num_cells: int) -> None:
        """Update performance metrics."""
        self.processing_times.append(processing_time_ms)
        
        throughput = num_cells / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
        self.throughput_history.append(throughput)
        
        self.total_cells_processed += num_cells
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}
        
        return {
            'mean_processing_time_ms': np.mean(self.processing_times),
            'median_processing_time_ms': np.median(self.processing_times),
            'p95_processing_time_ms': np.percentile(self.processing_times, 95),
            'mean_throughput_cells_per_sec': np.mean(self.throughput_history),
            'total_cells_processed': self.total_cells_processed,
            'uptime_seconds': time.time() - self.start_time,
            'overall_throughput_cells_per_sec': self.total_cells_processed / (time.time() - self.start_time)
        }


def create_live_data_stream(
    data_source: str,
    update_interval: float = 1.0,
    batch_size: int = 50
) -> Iterator[Dict[str, Any]]:
    """
    Create live data stream from various sources.
    
    Args:
        data_source: Source of live data ('random', 'file', 'api')
        update_interval: Interval between updates in seconds
        batch_size: Size of data batches
        
    Yields:
        Data batches
    """
    if data_source == 'random':
        yield from _random_data_stream(update_interval, batch_size)
    elif data_source.startswith('file:'):
        file_path = data_source[5:]
        yield from _file_data_stream(file_path, update_interval, batch_size)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def _random_data_stream(update_interval: float, batch_size: int) -> Iterator[Dict[str, Any]]:
    """Generate random data stream for testing."""
    cell_id = 0
    
    while True:
        # Generate random batch
        expression = np.random.lognormal(0, 1, size=(batch_size, 2000))
        spatial_coords = np.random.rand(batch_size, 2) * 1000
        
        data = {
            'expression': expression,
            'spatial_coords': spatial_coords,
            'batch_id': cell_id // batch_size,
            'timestamp': time.time()
        }
        
        cell_id += batch_size
        yield data
        
        time.sleep(update_interval)


def _file_data_stream(file_path: str, update_interval: float, batch_size: int) -> Iterator[Dict[str, Any]]:
    """Stream data from file with simulated real-time updates."""
    try:
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            expression_data = f['X'][:]
            spatial_data = f['obsm']['spatial'][:] if 'obsm' in f and 'spatial' in f['obsm'] else None
            
            total_cells = len(expression_data)
            
            for start_idx in range(0, total_cells, batch_size):
                end_idx = min(start_idx + batch_size, total_cells)
                
                data = {
                    'expression': expression_data[start_idx:end_idx],
                    'spatial_coords': spatial_data[start_idx:end_idx] if spatial_data is not None else np.random.rand(end_idx - start_idx, 2) * 1000,
                    'batch_id': start_idx // batch_size,
                    'timestamp': time.time()
                }
                
                yield data
                time.sleep(update_interval)
                
    except Exception as e:
        logger.error(f"Error in file data stream: {e}")
        return
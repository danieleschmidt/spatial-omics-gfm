"""
Generation 3: Enhanced Streaming & Real-time Processing.

This module extends the existing streaming capabilities with:
- Incremental learning and model updates
- Event-driven processing architecture
- Large-scale data ingestion pipelines
- Advanced stream processing patterns
- Real-time analytics and monitoring
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Iterator, Union, AsyncIterator
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import numpy as np
import pandas as pd
from queue import Queue, Empty, Full
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from anndata import AnnData

from .streaming_inference import StreamingInference, StreamingBuffer, PerformanceMonitor
from ..models.graph_transformer import SpatialGraphTransformer
from ..training.fine_tuning import FineTuningConfig
from ..utils.advanced_caching import CacheManager, create_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for enhanced streaming processing."""
    
    # Core streaming settings
    buffer_size: int = 10000
    window_size: int = 1000
    overlap_size: int = 100
    max_latency_ms: float = 50.0
    batch_size: int = 32
    
    # Incremental learning
    enable_incremental_learning: bool = True
    learning_rate: float = 1e-5
    adaptation_frequency: int = 100  # Update model every N samples
    forget_factor: float = 0.99  # Exponential forgetting for old data
    
    # Event-driven processing
    enable_event_driven: bool = True
    event_queue_size: int = 1000
    event_processing_threads: int = 4
    
    # Data ingestion
    max_ingestion_rate: float = 1000.0  # samples/second
    compression_enabled: bool = True
    checkpointing_interval: int = 1000  # samples
    
    # Performance optimization
    prefetch_batches: int = 5
    parallel_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Monitoring and analytics
    enable_real_time_analytics: bool = True
    analytics_window_size: int = 5000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'latency_ms': 100.0,
        'error_rate': 0.05,
        'throughput_drop': 0.3
    })


class StreamEvent:
    """Base class for stream events."""
    
    def __init__(self, event_type: str, data: Any, timestamp: float = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.event_id = f"{event_type}_{int(self.timestamp * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp
        }


class DataEvent(StreamEvent):
    """Event containing new data for processing."""
    
    def __init__(self, data: Dict[str, Any], metadata: Optional[Dict] = None):
        super().__init__('data_arrival', data)
        self.metadata = metadata or {}


class ModelUpdateEvent(StreamEvent):
    """Event triggering model update."""
    
    def __init__(self, update_data: Dict[str, Any]):
        super().__init__('model_update', update_data)


class AlertEvent(StreamEvent):
    """Event for system alerts."""
    
    def __init__(self, alert_type: str, message: str, severity: str = 'warning'):
        super().__init__('alert', {
            'alert_type': alert_type,
            'message': message,
            'severity': severity
        })


class EventProcessor(ABC):
    """Abstract base class for event processors."""
    
    @abstractmethod
    async def process_event(self, event: StreamEvent) -> Optional[Any]:
        """Process a stream event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if processor can handle event type."""
        pass


class DataProcessor(EventProcessor):
    """Processor for data events."""
    
    def __init__(self, model: SpatialGraphTransformer, cache_manager: CacheManager):
        self.model = model
        self.cache_manager = cache_manager
        
    async def process_event(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process data event with model inference."""
        if not isinstance(event, DataEvent):
            return None
        
        try:
            data = event.data
            
            # Run inference (potentially cached)
            result = await self._run_inference_async(data)
            
            return {
                'event_id': event.event_id,
                'result': result,
                'processing_time': time.time() - event.timestamp
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return None
    
    def can_handle(self, event_type: str) -> bool:
        """Check if can handle event type."""
        return event_type == 'data_arrival'
    
    async def _run_inference_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference asynchronously."""
        # Convert to tensors
        expression = torch.tensor(data['expression'], dtype=torch.float32)
        spatial_coords = torch.tensor(data['spatial_coords'], dtype=torch.float32)
        
        # Build simple graph
        edge_index, edge_attr = self._build_graph(spatial_coords)
        
        # Run inference
        loop = asyncio.get_event_loop()
        
        def _inference():
            with torch.no_grad():
                return self.model(
                    gene_expression=expression,
                    spatial_coords=spatial_coords,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
        
        result = await loop.run_in_executor(None, _inference)
        
        return {
            'embeddings': result['embeddings'].numpy() if 'embeddings' in result else result.numpy(),
            'timestamp': time.time()
        }
    
    def _build_graph(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build spatial graph for coordinates."""
        from sklearn.neighbors import NearestNeighbors
        
        n_points = len(coords)
        k = min(6, n_points - 1)
        
        if k <= 0:
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[1.0]], dtype=torch.float32)
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords.numpy())
        distances, indices = nbrs.kneighbors(coords.numpy())
        
        edges = []
        edge_features = []
        
        for i in range(n_points):
            for j in range(1, len(indices[i])):
                neighbor = indices[i][j]
                dist = distances[i][j]
                edges.append([i, neighbor])
                edge_features.append([dist])
        
        if not edges:
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[1.0]], dtype=torch.float32)
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return edge_index, edge_attr


class ModelUpdateProcessor(EventProcessor):
    """Processor for model update events."""
    
    def __init__(self, model: SpatialGraphTransformer, config: StreamingConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
    async def process_event(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process model update event."""
        if not isinstance(event, ModelUpdateEvent):
            return None
        
        try:
            update_data = event.data
            
            # Perform incremental update
            result = await self._update_model_async(update_data)
            
            return {
                'event_id': event.event_id,
                'update_result': result,
                'model_version': self._get_model_version()
            }
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return None
    
    def can_handle(self, event_type: str) -> bool:
        """Check if can handle event type."""
        return event_type == 'model_update'
    
    async def _update_model_async(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update model with new data asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _update():
            # Convert update data to tensors
            if 'loss' in update_data:
                # Direct loss update
                loss = torch.tensor(update_data['loss'], dtype=torch.float32)
            else:
                # Compute loss from data
                expression = torch.tensor(update_data['expression'], dtype=torch.float32)
                target = torch.tensor(update_data['target'], dtype=torch.float32)
                
                # Forward pass
                self.model.train()
                output = self.model(
                    gene_expression=expression,
                    spatial_coords=torch.tensor(update_data['spatial_coords'], dtype=torch.float32),
                    edge_index=torch.tensor(update_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(update_data.get('edge_attr', [[1.0]]), dtype=torch.float32)
                )
                
                embeddings = output['embeddings'] if 'embeddings' in output else output
                loss = F.mse_loss(embeddings, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.model.eval()
            
            return {
                'loss': loss.item(),
                'updated': True
            }
        
        result = await loop.run_in_executor(None, _update)
        return result
    
    def _get_model_version(self) -> str:
        """Get current model version identifier."""
        # Simple version based on parameter hash
        param_hash = hash(str([p.data.sum().item() for p in self.model.parameters()]))
        return f"v_{abs(param_hash) % 1000000}"


class AlertProcessor(EventProcessor):
    """Processor for alert events."""
    
    def __init__(self, alert_handlers: Optional[List[Callable]] = None):
        self.alert_handlers = alert_handlers or []
        self.alert_history = deque(maxlen=1000)
    
    async def process_event(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process alert event."""
        if not isinstance(event, AlertEvent):
            return None
        
        alert_data = event.data
        self.alert_history.append(event)
        
        # Log alert
        severity = alert_data.get('severity', 'warning')
        message = alert_data.get('message', 'Unknown alert')
        
        if severity == 'critical':
            logger.critical(f"ALERT: {message}")
        elif severity == 'error':
            logger.error(f"ALERT: {message}")
        else:
            logger.warning(f"ALERT: {message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await self._call_handler_async(handler, event)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        return {
            'event_id': event.event_id,
            'handled': True,
            'severity': severity
        }
    
    def can_handle(self, event_type: str) -> bool:
        """Check if can handle event type."""
        return event_type == 'alert'
    
    async def _call_handler_async(self, handler: Callable, event: StreamEvent) -> None:
        """Call alert handler asynchronously."""
        if asyncio.iscoroutinefunction(handler):
            await handler(event)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, handler, event)


class EventDispatcher:
    """Dispatches events to appropriate processors."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.processors: List[EventProcessor] = []
        self.event_queue = asyncio.Queue(maxsize=config.event_queue_size)
        self.processing_tasks: List[asyncio.Task] = []
        self.running = False
        
    def add_processor(self, processor: EventProcessor) -> None:
        """Add event processor."""
        self.processors.append(processor)
        logger.info(f"Added event processor: {type(processor).__name__}")
    
    async def dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch event to queue."""
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
    
    async def start_processing(self) -> None:
        """Start event processing tasks."""
        if self.running:
            return
        
        self.running = True
        
        # Start processing tasks
        for i in range(self.config.event_processing_threads):
            task = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            self.processing_tasks.append(task)
        
        logger.info(f"Started {len(self.processing_tasks)} event processing workers")
    
    async def stop_processing(self) -> None:
        """Stop event processing."""
        self.running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        logger.info("Stopped event processing")
    
    async def _processing_worker(self, worker_id: str) -> None:
        """Event processing worker."""
        logger.info(f"Event processing worker {worker_id} started")
        
        while self.running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Find processors that can handle this event
                handled = False
                for processor in self.processors:
                    if processor.can_handle(event.event_type):
                        try:
                            result = await processor.process_event(event)
                            if result:
                                handled = True
                                logger.debug(f"Worker {worker_id} processed {event.event_type}")
                        except Exception as e:
                            logger.error(f"Processor failed: {e}")
                
                if not handled:
                    logger.warning(f"No processor for event type: {event.event_type}")
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Event processing worker {worker_id} stopped")


class IncrementalLearner:
    """Manages incremental learning for streaming data."""
    
    def __init__(self, model: SpatialGraphTransformer, config: StreamingConfig):
        self.model = model
        self.config = config
        
        # Learning state
        self.samples_processed = 0
        self.last_update = time.time()
        self.learning_history = deque(maxlen=1000)
        
        # Optimizer for incremental updates
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Exponential moving averages for stability
        self.loss_ema = None
        self.gradient_ema = None
        
        logger.info("IncrementalLearner initialized")
    
    def should_update_model(self) -> bool:
        """Determine if model should be updated."""
        return (
            self.config.enable_incremental_learning and
            self.samples_processed % self.config.adaptation_frequency == 0 and
            self.samples_processed > 0
        )
    
    async def update_model(
        self,
        data_batch: Dict[str, Any],
        target_batch: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update model with new batch of data."""
        if not self.should_update_model():
            return {'updated': False, 'reason': 'not_scheduled'}
        
        try:
            # Prepare training data
            expression = torch.tensor(data_batch['expression'], dtype=torch.float32)
            spatial_coords = torch.tensor(data_batch['spatial_coords'], dtype=torch.float32)
            
            # Use self-supervised loss if no targets provided
            if target_batch is None:
                target_batch = self._generate_self_supervised_targets(data_batch)
            
            targets = torch.tensor(target_batch['targets'], dtype=torch.float32)
            
            # Forward pass
            self.model.train()
            
            # Build graph
            edge_index, edge_attr = self._build_graph_batch(spatial_coords)
            
            outputs = self.model(
                gene_expression=expression,
                spatial_coords=spatial_coords,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            
            embeddings = outputs['embeddings'] if 'embeddings' in outputs else outputs
            
            # Compute loss
            loss = self._compute_adaptive_loss(embeddings, targets)
            
            # Apply exponential forgetting
            if self.loss_ema is not None:
                loss = self.loss_ema * self.config.forget_factor + loss * (1 - self.config.forget_factor)
            
            self.loss_ema = loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.model.eval()
            
            # Record learning event
            self.learning_history.append({
                'timestamp': time.time(),
                'loss': loss.item(),
                'samples_processed': self.samples_processed,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            self.last_update = time.time()
            
            return {
                'updated': True,
                'loss': loss.item(),
                'samples_processed': self.samples_processed
            }
            
        except Exception as e:
            logger.error(f"Incremental learning failed: {e}")
            return {'updated': False, 'error': str(e)}
    
    def process_sample(self, sample: Dict[str, Any]) -> None:
        """Process a single sample (increment counter)."""
        self.samples_processed += 1
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get incremental learning statistics."""
        if not self.learning_history:
            return {
                'samples_processed': self.samples_processed,
                'updates_performed': 0,
                'last_loss': self.loss_ema,
                'last_update': self.last_update
            }
        
        return {
            'samples_processed': self.samples_processed,
            'updates_performed': len(self.learning_history),
            'last_loss': self.learning_history[-1]['loss'],
            'avg_loss': np.mean([h['loss'] for h in self.learning_history]),
            'last_update': self.last_update,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def _generate_self_supervised_targets(self, data_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Generate self-supervised targets for unsupervised learning."""
        # Simple example: predict masked gene expression
        expression = np.array(data_batch['expression'])
        
        # Random masking
        mask_prob = 0.15
        mask = np.random.random(expression.shape) < mask_prob
        
        targets = expression.copy()
        expression_masked = expression.copy()
        expression_masked[mask] = 0
        
        return {
            'targets': targets,
            'masked_expression': expression_masked,
            'mask': mask
        }
    
    def _compute_adaptive_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive loss with regularization."""
        # Base reconstruction loss
        reconstruction_loss = F.mse_loss(embeddings, targets)
        
        # Add regularization for stability
        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
        regularization_loss = 1e-5 * l2_reg
        
        total_loss = reconstruction_loss + regularization_loss
        
        return total_loss
    
    def _build_graph_batch(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph for batch of coordinates."""
        # Simplified batch graph construction
        batch_size, n_points, spatial_dim = coords.shape
        
        all_edges = []
        all_edge_attrs = []
        
        for b in range(batch_size):
            batch_coords = coords[b]
            
            # Build graph for this batch
            from sklearn.neighbors import NearestNeighbors
            
            k = min(6, n_points - 1)
            if k <= 0:
                continue
            
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(batch_coords.numpy())
            distances, indices = nbrs.kneighbors(batch_coords.numpy())
            
            for i in range(n_points):
                for j in range(1, len(indices[i])):
                    neighbor = indices[i][j]
                    dist = distances[i][j]
                    
                    # Add batch offset
                    src = b * n_points + i
                    dst = b * n_points + neighbor
                    
                    all_edges.append([src, dst])
                    all_edge_attrs.append([dist])
        
        if not all_edges:
            # Return dummy graph
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[1.0]], dtype=torch.float32)
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).T
        edge_attr = torch.tensor(all_edge_attrs, dtype=torch.float32)
        
        return edge_index, edge_attr


class StreamAnalytics:
    """Real-time analytics for streaming data."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.analytics_window_size)
        self.alerts = deque(maxlen=1000)
        
        # Performance tracking
        self.throughput_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        
        # Analytics state
        self.start_time = time.time()
        self.total_samples = 0
        self.total_errors = 0
        
    def record_sample(
        self,
        processing_time: float,
        success: bool = True,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record processing of a sample."""
        timestamp = time.time()
        
        self.total_samples += 1
        if not success:
            self.total_errors += 1
        
        # Record metrics
        sample_metric = {
            'timestamp': timestamp,
            'processing_time': processing_time,
            'success': success,
            'metadata': metadata or {}
        }
        
        self.metrics_buffer.append(sample_metric)
        self.latency_history.append(processing_time * 1000)  # Convert to ms
        
        # Update throughput
        current_throughput = self._calculate_throughput()
        self.throughput_history.append(current_throughput)
        
        # Check for alerts
        self._check_alerts(processing_time * 1000, current_throughput, success)
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time analytics statistics."""
        if not self.metrics_buffer:
            return {
                'total_samples': self.total_samples,
                'total_errors': self.total_errors,
                'uptime_seconds': time.time() - self.start_time
            }
        
        recent_metrics = list(self.metrics_buffer)[-100:]  # Last 100 samples
        
        # Calculate statistics
        processing_times = [m['processing_time'] * 1000 for m in recent_metrics]  # ms
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        
        stats = {
            'total_samples': self.total_samples,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(self.total_samples, 1),
            'uptime_seconds': time.time() - self.start_time,
            
            # Recent performance
            'recent_avg_latency_ms': np.mean(processing_times) if processing_times else 0,
            'recent_p95_latency_ms': np.percentile(processing_times, 95) if processing_times else 0,
            'recent_success_rate': success_rate,
            'current_throughput': self._calculate_throughput(),
            
            # Trends
            'latency_trend': self._calculate_trend(self.latency_history),
            'throughput_trend': self._calculate_trend(self.throughput_history),
            
            # Alerts
            'active_alerts': len(self.alerts),
            'recent_alerts': list(self.alerts)[-10:]  # Last 10 alerts
        }
        
        return stats
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (samples/second)."""
        if len(self.metrics_buffer) < 2:
            return 0.0
        
        # Use last 60 seconds of data
        cutoff_time = time.time() - 60
        recent_samples = [m for m in self.metrics_buffer if m['timestamp'] >= cutoff_time]
        
        if len(recent_samples) < 2:
            return 0.0
        
        time_span = recent_samples[-1]['timestamp'] - recent_samples[0]['timestamp']
        if time_span <= 0:
            return 0.0
        
        return len(recent_samples) / time_span
    
    def _calculate_trend(self, values: deque, window_size: int = 50) -> str:
        """Calculate trend direction for values."""
        if len(values) < window_size:
            return 'insufficient_data'
        
        recent_values = list(values)[-window_size:]
        
        # Simple linear regression
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _check_alerts(self, latency_ms: float, throughput: float, success: bool) -> None:
        """Check for alert conditions."""
        current_time = time.time()
        
        # Latency alert
        if latency_ms > self.config.alert_thresholds['latency_ms']:
            alert = AlertEvent(
                'high_latency',
                f'Processing latency {latency_ms:.2f}ms exceeds threshold {self.config.alert_thresholds["latency_ms"]:.2f}ms'
            )
            self.alerts.append(alert)
        
        # Error rate alert
        recent_error_rate = self.total_errors / max(self.total_samples, 1)
        if recent_error_rate > self.config.alert_thresholds['error_rate']:
            alert = AlertEvent(
                'high_error_rate',
                f'Error rate {recent_error_rate:.3f} exceeds threshold {self.config.alert_thresholds["error_rate"]:.3f}',
                severity='error'
            )
            self.alerts.append(alert)
        
        # Throughput drop alert
        if len(self.throughput_history) > 10:
            recent_avg_throughput = np.mean(list(self.throughput_history)[-10:])
            baseline_throughput = np.mean(list(self.throughput_history)[-100:-10]) if len(self.throughput_history) > 100 else recent_avg_throughput
            
            if baseline_throughput > 0:
                throughput_drop = 1 - (recent_avg_throughput / baseline_throughput)
                if throughput_drop > self.config.alert_thresholds['throughput_drop']:
                    alert = AlertEvent(
                        'throughput_drop',
                        f'Throughput dropped by {throughput_drop:.2%}',
                        severity='warning'
                    )
                    self.alerts.append(alert)


class EnhancedStreamingInference(StreamingInference):
    """Enhanced streaming inference with event-driven architecture and incremental learning."""
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        config: StreamingConfig,
        cache_manager: Optional[CacheManager] = None
    ):
        # Initialize parent with compatible parameters
        super().__init__(
            model=model,
            buffer_size=config.buffer_size,
            window_size=config.window_size,
            overlap_size=config.overlap_size,
            max_latency_ms=config.max_latency_ms,
            batch_size=config.batch_size,
            enable_caching=cache_manager is not None
        )
        
        self.config = config
        self.cache_manager = cache_manager or create_cache_manager()
        
        # Enhanced components
        self.event_dispatcher = EventDispatcher(config)
        self.incremental_learner = IncrementalLearner(model, config)
        self.analytics = StreamAnalytics(config)
        
        # Setup event processors
        self.data_processor = DataProcessor(model, self.cache_manager)
        self.model_update_processor = ModelUpdateProcessor(model, config)
        self.alert_processor = AlertProcessor()
        
        self.event_dispatcher.add_processor(self.data_processor)
        self.event_dispatcher.add_processor(self.model_update_processor)
        self.event_dispatcher.add_processor(self.alert_processor)
        
        logger.info("EnhancedStreamingInference initialized")
    
    async def start_enhanced_streaming(self) -> None:
        """Start enhanced streaming with event-driven processing."""
        # Start event processing
        await self.event_dispatcher.start_processing()
        
        # Start parent streaming
        self.start_streaming()
        
        logger.info("Enhanced streaming started")
    
    async def stop_enhanced_streaming(self) -> None:
        """Stop enhanced streaming."""
        # Stop event processing
        await self.event_dispatcher.stop_processing()
        
        # Stop parent streaming
        self.stop_streaming()
        
        logger.info("Enhanced streaming stopped")
    
    async def process_data_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data asynchronously with event system."""
        start_time = time.time()
        
        try:
            # Create data event
            data_event = DataEvent(data)
            
            # Dispatch event
            await self.event_dispatcher.dispatch_event(data_event)
            
            # Process sample for incremental learning
            self.incremental_learner.process_sample(data)
            
            # Check if model update is needed
            if self.incremental_learner.should_update_model():
                # Create model update event
                update_event = ModelUpdateEvent({
                    'expression': data['expression'],
                    'spatial_coords': data['spatial_coords'],
                    'timestamp': time.time()
                })
                await self.event_dispatcher.dispatch_event(update_event)
            
            # Run inference (use cached result if available)
            result = await self._run_inference_with_cache(data)
            
            # Record analytics
            processing_time = time.time() - start_time
            self.analytics.record_sample(processing_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.analytics.record_sample(processing_time, success=False)
            
            # Create alert event
            alert_event = AlertEvent('processing_error', str(e), severity='error')
            await self.event_dispatcher.dispatch_event(alert_event)
            
            raise
    
    async def _run_inference_with_cache(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with caching support."""
        # Try cache first
        cached_result = self.cache_manager.get(
            data=data,
            operation='streaming_inference'
        )
        
        if cached_result is not None:
            return cached_result
        
        # Run inference
        result = await self.data_processor._run_inference_async(data)
        
        # Cache result
        self.cache_manager.put(
            value=result,
            data=data,
            operation='streaming_inference'
        )
        
        return result
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        base_stats = self.get_performance_stats()
        
        enhanced_stats = {
            'base_streaming': base_stats,
            'incremental_learning': self.incremental_learner.get_learning_stats(),
            'real_time_analytics': self.analytics.get_real_time_stats(),
            'cache_stats': self.cache_manager.get_stats(),
            'event_queue_size': self.event_dispatcher.event_queue.qsize()
        }
        
        return enhanced_stats
    
    async def process_stream_enhanced(
        self,
        data_stream: AsyncIterator[Dict[str, Any]],
        output_callback: Optional[Callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process enhanced data stream with full capabilities."""
        await self.start_enhanced_streaming()
        
        try:
            async for data in data_stream:
                try:
                    result = await self.process_data_async(data)
                    
                    if output_callback:
                        await output_callback(result)
                    
                    yield result
                    
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    continue
        
        finally:
            await self.stop_enhanced_streaming()


# Convenience functions and utilities
async def create_enhanced_streaming_processor(
    model: SpatialGraphTransformer,
    config: Optional[StreamingConfig] = None,
    enable_caching: bool = True,
    enable_incremental_learning: bool = True
) -> EnhancedStreamingInference:
    """Create enhanced streaming processor with defaults."""
    
    if config is None:
        config = StreamingConfig(
            enable_incremental_learning=enable_incremental_learning
        )
    
    cache_manager = create_cache_manager() if enable_caching else None
    
    processor = EnhancedStreamingInference(
        model=model,
        config=config,
        cache_manager=cache_manager
    )
    
    return processor


async def process_large_scale_stream(
    model: SpatialGraphTransformer,
    data_source: Union[str, AsyncIterator],
    output_handler: Optional[Callable] = None,
    config: Optional[StreamingConfig] = None
) -> Dict[str, Any]:
    """Process large-scale data stream with enhanced capabilities."""
    
    processor = await create_enhanced_streaming_processor(model, config)
    
    # Create data stream
    if isinstance(data_source, str):
        # File-based stream
        data_stream = _create_file_stream(data_source)
    else:
        data_stream = data_source
    
    # Process stream
    results_count = 0
    total_processing_time = 0
    
    async for result in processor.process_stream_enhanced(data_stream, output_handler):
        results_count += 1
        total_processing_time += result.get('processing_time', 0)
    
    # Get final statistics
    final_stats = processor.get_enhanced_stats()
    final_stats.update({
        'total_results': results_count,
        'total_processing_time': total_processing_time,
        'average_processing_time': total_processing_time / max(results_count, 1)
    })
    
    return final_stats


async def _create_file_stream(file_path: str) -> AsyncIterator[Dict[str, Any]]:
    """Create async iterator from file."""
    # Simplified file streaming - would be more sophisticated in practice
    import h5py
    
    try:
        with h5py.File(file_path, 'r') as f:
            expression_data = f['X']
            spatial_data = f['obsm']['spatial'] if 'obsm' in f and 'spatial' in f['obsm'] else None
            
            n_samples = expression_data.shape[0]
            
            for i in range(0, n_samples, 100):  # Process in chunks
                end_idx = min(i + 100, n_samples)
                
                chunk_data = {
                    'expression': expression_data[i:end_idx],
                    'spatial_coords': spatial_data[i:end_idx] if spatial_data is not None else np.random.rand(end_idx - i, 2) * 1000,
                    'chunk_id': i // 100,
                    'sample_indices': list(range(i, end_idx))
                }
                
                yield chunk_data
                
                # Allow other async operations
                await asyncio.sleep(0.01)
                
    except Exception as e:
        logger.error(f"Error reading file stream: {e}")
        return
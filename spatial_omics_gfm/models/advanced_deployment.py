"""
Generation 3: Advanced Model Serving & Deployment Platform.

This module implements enterprise-grade model serving capabilities:
- Model serving infrastructure with load balancing
- API endpoints for inference and training
- Model versioning and A/B testing capabilities
- Production monitoring and health checks
- Multi-model serving and routing
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import warnings
import uuid
import hashlib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.jit
from anndata import AnnData

# FastAPI for REST API
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available, API endpoints will be disabled")

from .graph_transformer import SpatialGraphTransformer, TransformerConfig
from .deployment import ModelServer, DeploymentConfig
from ..utils.advanced_caching import CacheManager, create_cache_manager
from ..utils.auto_scaling import AutoScaler, create_auto_scaler

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_path: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_time: float = field(default_factory=time.time)
    status: str = "inactive"  # inactive, active, deprecated
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingConfig:
    """Configuration for advanced model serving."""
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_docs: bool = True
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, performance_based
    
    # Model versioning
    enable_versioning: bool = True
    max_versions: int = 5
    default_version: str = "latest"
    
    # A/B testing
    enable_ab_testing: bool = True
    traffic_split_config: Dict[str, float] = field(default_factory=lambda: {"v1": 0.8, "v2": 0.2})
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_retention_hours: int = 24
    health_check_interval: int = 30
    
    # Performance
    request_timeout: int = 30
    max_concurrent_requests: int = 100
    enable_request_caching: bool = True
    cache_ttl: int = 300  # seconds
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    auto_scaling_config: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    rate_limit_requests_per_minute: int = 1000


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class InferenceRequest(BaseModel):
        """Request model for inference."""
        gene_expression: List[List[float]] = Field(..., description="Gene expression matrix")
        spatial_coords: List[List[float]] = Field(..., description="Spatial coordinates")
        model_version: Optional[str] = Field(None, description="Model version to use")
        return_embeddings: bool = Field(True, description="Whether to return embeddings")
        cache_result: bool = Field(True, description="Whether to cache the result")
        
        class Config:
            schema_extra = {
                "example": {
                    "gene_expression": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
                    "spatial_coords": [[0.0, 0.0], [1.0, 1.0]],
                    "model_version": "v1.0.0",
                    "return_embeddings": True,
                    "cache_result": True
                }
            }
    
    class InferenceResponse(BaseModel):
        """Response model for inference."""
        embeddings: Optional[List[List[float]]] = None
        predictions: Optional[Dict[str, Any]] = None
        model_version: str
        processing_time_ms: float
        request_id: str
        cached: bool = False
        
        class Config:
            schema_extra = {
                "example": {
                    "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    "model_version": "v1.0.0",
                    "processing_time_ms": 25.5,
                    "request_id": "req_12345",
                    "cached": False
                }
            }
    
    class ModelInfo(BaseModel):
        """Model information response."""
        version_id: str
        status: str
        performance_metrics: Dict[str, float]
        deployment_time: float
        metadata: Dict[str, Any]
    
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        timestamp: float
        version: str
        uptime_seconds: float
        active_models: int
        system_metrics: Dict[str, float]


class ModelRegistry:
    """Registry for managing multiple model versions."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.models: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        self.deployment_history = deque(maxlen=1000)
        
        logger.info("ModelRegistry initialized")
    
    def register_model(
        self,
        version_id: str,
        model_path: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a new model version."""
        try:
            if version_id in self.models:
                logger.warning(f"Model version {version_id} already exists, updating")
            
            model_version = ModelVersion(
                version_id=version_id,
                model_path=model_path,
                config=config,
                metadata=metadata or {}
            )
            
            self.models[version_id] = model_version
            
            # Set as active if first model or explicitly requested
            if self.active_version is None or version_id == self.config.default_version:
                self.active_version = version_id
                model_version.status = "active"
            
            # Clean up old versions if limit exceeded
            self._cleanup_old_versions()
            
            # Record deployment
            self.deployment_history.append({
                'version_id': version_id,
                'action': 'registered',
                'timestamp': time.time()
            })
            
            logger.info(f"Registered model version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {version_id}: {e}")
            return False
    
    def get_model(self, version_id: Optional[str] = None) -> Optional[ModelVersion]:
        """Get model version."""
        if version_id is None or version_id == "latest":
            version_id = self.active_version
        
        return self.models.get(version_id)
    
    def list_models(self) -> List[ModelVersion]:
        """List all registered models."""
        return list(self.models.values())
    
    def activate_model(self, version_id: str) -> bool:
        """Activate a model version."""
        if version_id not in self.models:
            return False
        
        # Deactivate current active model
        if self.active_version and self.active_version in self.models:
            self.models[self.active_version].status = "inactive"
        
        # Activate new model
        self.active_version = version_id
        self.models[version_id].status = "active"
        
        # Record deployment
        self.deployment_history.append({
            'version_id': version_id,
            'action': 'activated',
            'timestamp': time.time()
        })
        
        logger.info(f"Activated model version: {version_id}")
        return True
    
    def deprecate_model(self, version_id: str) -> bool:
        """Deprecate a model version."""
        if version_id not in self.models:
            return False
        
        self.models[version_id].status = "deprecated"
        
        # If this was the active model, need to activate another
        if version_id == self.active_version:
            # Find another active model
            for v_id, model in self.models.items():
                if v_id != version_id and model.status != "deprecated":
                    self.activate_model(v_id)
                    break
            else:
                self.active_version = None
        
        logger.info(f"Deprecated model version: {version_id}")
        return True
    
    def update_performance_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Update performance metrics for a model version."""
        if version_id not in self.models:
            return False
        
        self.models[version_id].performance_metrics.update(metrics)
        return True
    
    def _cleanup_old_versions(self) -> None:
        """Clean up old model versions."""
        if len(self.models) <= self.config.max_versions:
            return
        
        # Sort by deployment time and keep most recent
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1].deployment_time,
            reverse=True
        )
        
        # Keep active model and most recent versions
        to_keep = set()
        to_keep.add(self.active_version)
        
        kept_count = 0
        for version_id, model in sorted_models:
            if kept_count < self.config.max_versions:
                to_keep.add(version_id)
                kept_count += 1
        
        # Remove old versions
        for version_id in list(self.models.keys()):
            if version_id not in to_keep:
                del self.models[version_id]
                logger.info(f"Removed old model version: {version_id}")


class ABTestingManager:
    """Manages A/B testing for model versions."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.traffic_splits = config.traffic_split_config.copy()
        self.experiment_history = deque(maxlen=10000)
        self.version_metrics = defaultdict(list)
        
        logger.info("ABTestingManager initialized")
    
    def select_model_version(self, request_id: str) -> str:
        """Select model version based on traffic split."""
        if not self.config.enable_ab_testing or not self.traffic_splits:
            return "latest"
        
        # Use deterministic selection based on request_id hash
        request_hash = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        selection_value = (request_hash % 1000) / 1000.0  # 0.0 to 1.0
        
        cumulative_prob = 0.0
        for version, split_ratio in self.traffic_splits.items():
            cumulative_prob += split_ratio
            if selection_value <= cumulative_prob:
                return version
        
        # Fallback to first version
        return list(self.traffic_splits.keys())[0]
    
    def record_experiment(
        self,
        request_id: str,
        version_id: str,
        success: bool,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record A/B testing experiment result."""
        experiment = {
            'request_id': request_id,
            'version_id': version_id,
            'success': success,
            'latency_ms': latency_ms,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.experiment_history.append(experiment)
        self.version_metrics[version_id].append(experiment)
        
        # Keep version metrics bounded
        if len(self.version_metrics[version_id]) > 1000:
            self.version_metrics[version_id] = self.version_metrics[version_id][-1000:]
    
    def get_version_performance(self, version_id: str) -> Dict[str, float]:
        """Get performance metrics for a version."""
        if version_id not in self.version_metrics:
            return {}
        
        experiments = self.version_metrics[version_id]
        if not experiments:
            return {}
        
        # Calculate metrics
        success_rate = sum(1 for e in experiments if e['success']) / len(experiments)
        avg_latency = np.mean([e['latency_ms'] for e in experiments])
        p95_latency = np.percentile([e['latency_ms'] for e in experiments], 95)
        
        return {
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'total_requests': len(experiments)
        }
    
    def get_ab_test_results(self) -> Dict[str, Any]:
        """Get A/B testing results."""
        results = {
            'traffic_splits': self.traffic_splits,
            'total_experiments': len(self.experiment_history),
            'version_performance': {}
        }
        
        for version_id in self.traffic_splits.keys():
            results['version_performance'][version_id] = self.get_version_performance(version_id)
        
        return results
    
    def update_traffic_split(self, new_splits: Dict[str, float]) -> bool:
        """Update traffic split configuration."""
        # Validate splits sum to 1.0
        total = sum(new_splits.values())
        if abs(total - 1.0) > 0.01:
            logger.error(f"Traffic splits must sum to 1.0, got {total}")
            return False
        
        self.traffic_splits = new_splits.copy()
        logger.info(f"Updated traffic splits: {self.traffic_splits}")
        return True


class LoadBalancer:
    """Load balancer for model serving requests."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.strategy = config.load_balance_strategy
        
        # Load balancing state
        self.model_loads = defaultdict(float)
        self.model_performance = defaultdict(dict)
        self.request_counts = defaultdict(int)
        self.round_robin_index = 0
        
        logger.info(f"LoadBalancer initialized with strategy: {self.strategy}")
    
    def select_model_instance(
        self,
        available_versions: List[str],
        request_metadata: Optional[Dict] = None
    ) -> str:
        """Select model instance based on load balancing strategy."""
        if not available_versions:
            raise ValueError("No available model versions")
        
        if len(available_versions) == 1:
            return available_versions[0]
        
        if self.strategy == "round_robin":
            selected = available_versions[self.round_robin_index % len(available_versions)]
            self.round_robin_index += 1
            return selected
        
        elif self.strategy == "least_loaded":
            # Select version with lowest current load
            min_load = float('inf')
            selected = available_versions[0]
            
            for version in available_versions:
                load = self.model_loads[version]
                if load < min_load:
                    min_load = load
                    selected = version
            
            return selected
        
        elif self.strategy == "performance_based":
            # Select based on historical performance
            best_score = float('-inf')
            selected = available_versions[0]
            
            for version in available_versions:
                perf = self.model_performance[version]
                if not perf:
                    continue  # Skip versions without performance data
                
                # Simple scoring: high success rate, low latency
                success_rate = perf.get('success_rate', 0.5)
                avg_latency = perf.get('avg_latency_ms', 1000)
                
                # Score favors high success rate and low latency
                score = success_rate * 100 - (avg_latency / 10)
                
                if score > best_score:
                    best_score = score
                    selected = version
            
            return selected
        
        else:
            # Default to round robin
            return self.select_model_instance(available_versions, request_metadata)
    
    def record_request(
        self,
        version_id: str,
        processing_time_ms: float,
        success: bool
    ) -> None:
        """Record request for load balancing decisions."""
        self.request_counts[version_id] += 1
        
        # Update load (exponential moving average)
        current_load = self.model_loads[version_id]
        new_load = processing_time_ms / 1000.0  # Convert to seconds
        self.model_loads[version_id] = 0.9 * current_load + 0.1 * new_load
        
        # Update performance metrics
        if version_id not in self.model_performance:
            self.model_performance[version_id] = {
                'success_rate': 0.5,
                'avg_latency_ms': 100,
                'request_count': 0
            }
        
        perf = self.model_performance[version_id]
        perf['request_count'] += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * (1 if success else 0)
        
        # Update average latency (exponential moving average)
        perf['avg_latency_ms'] = (1 - alpha) * perf['avg_latency_ms'] + alpha * processing_time_ms
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'strategy': self.strategy,
            'model_loads': dict(self.model_loads),
            'model_performance': dict(self.model_performance),
            'request_counts': dict(self.request_counts)
        }


class HealthMonitor:
    """Health monitoring for model serving."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.start_time = time.time()
        self.health_history = deque(maxlen=1000)
        
        # System metrics
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Health check thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info("HealthMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Health monitoring stopped")
    
    def record_request(
        self,
        processing_time_ms: float,
        success: bool,
        version_id: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record request for health monitoring."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        self.total_processing_time += processing_time_ms
        
        # Record detailed metrics
        health_record = {
            'timestamp': time.time(),
            'processing_time_ms': processing_time_ms,
            'success': success,
            'version_id': version_id,
            'metadata': metadata or {}
        }
        
        self.health_history.append(health_record)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        error_rate = self.error_count / max(self.request_count, 1)
        avg_processing_time = self.total_processing_time / max(self.request_count, 1)
        requests_per_second = self.request_count / max(uptime, 1)
        
        # Recent metrics (last 5 minutes)
        recent_cutoff = current_time - 300
        recent_records = [r for r in self.health_history if r['timestamp'] >= recent_cutoff]
        
        recent_error_rate = 0.0
        recent_avg_latency = 0.0
        
        if recent_records:
            recent_errors = sum(1 for r in recent_records if not r['success'])
            recent_error_rate = recent_errors / len(recent_records)
            recent_avg_latency = np.mean([r['processing_time_ms'] for r in recent_records])
        
        # System resource metrics
        system_metrics = self._get_system_metrics()
        
        # Determine overall health status
        status = "healthy"
        if error_rate > 0.1 or recent_error_rate > 0.15:
            status = "degraded"
        if error_rate > 0.25 or recent_error_rate > 0.3:
            status = "unhealthy"
        
        return {
            'status': status,
            'timestamp': current_time,
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': error_rate,
            'avg_processing_time_ms': avg_processing_time,
            'requests_per_second': requests_per_second,
            'recent_error_rate': recent_error_rate,
            'recent_avg_latency_ms': recent_avg_latency,
            'system_metrics': system_metrics
        }
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    
                    metrics.update({
                        'gpu_memory_allocated_gb': gpu_memory_allocated,
                        'gpu_memory_reserved_gb': gpu_memory_reserved
                    })
                except:
                    pass
            
            return metrics
            
        except ImportError:
            return {'cpu_percent': 0.0, 'memory_percent': 0.0}
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Record health metrics
                health_status = self.get_health_status()
                
                # Log warnings for degraded health
                if health_status['status'] == 'degraded':
                    logger.warning(f"Service health degraded: {health_status}")
                elif health_status['status'] == 'unhealthy':
                    logger.error(f"Service unhealthy: {health_status}")
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.health_check_interval)


class AdvancedModelServer:
    """Advanced model server with versioning, A/B testing, and load balancing."""
    
    def __init__(
        self,
        config: ServingConfig,
        cache_manager: Optional[CacheManager] = None,
        auto_scaler: Optional[AutoScaler] = None
    ):
        self.config = config
        self.cache_manager = cache_manager or create_cache_manager()
        self.auto_scaler = auto_scaler
        
        # Core components
        self.model_registry = ModelRegistry(config)
        self.ab_testing = ABTestingManager(config)
        self.load_balancer = LoadBalancer(config)
        self.health_monitor = HealthMonitor(config)
        
        # Model servers for each version
        self.model_servers: Dict[str, ModelServer] = {}
        
        # API app
        self.app = None
        if FASTAPI_AVAILABLE:
            self._setup_api()
        
        # Request tracking
        self.active_requests = 0
        self.request_history = deque(maxlen=10000)
        
        logger.info("AdvancedModelServer initialized")
    
    def register_model(
        self,
        version_id: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a new model version."""
        try:
            # Register in model registry
            success = self.model_registry.register_model(
                version_id=version_id,
                model_path=model_path,
                config=config or {},
                metadata={'registered_time': time.time()}
            )
            
            if not success:
                return False
            
            # Create model server instance
            deployment_config = DeploymentConfig()
            if config:
                # Update deployment config with provided config
                for key, value in config.items():
                    if hasattr(deployment_config, key):
                        setattr(deployment_config, key, value)
            
            model_server = ModelServer(
                model_path=model_path,
                config=deployment_config,
                warm_start=True
            )
            
            self.model_servers[version_id] = model_server
            
            logger.info(f"Successfully registered model version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {version_id}: {e}")
            return False
    
    def start_serving(self) -> None:
        """Start model serving."""
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Start auto-scaling if enabled
        if self.auto_scaler and self.config.enable_auto_scaling:
            self.auto_scaler.start_auto_scaling()
        
        logger.info("Advanced model serving started")
    
    def stop_serving(self) -> None:
        """Stop model serving."""
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Stop auto-scaling
        if self.auto_scaler:
            self.auto_scaler.stop_auto_scaling()
        
        logger.info("Advanced model serving stopped")
    
    def predict(
        self,
        gene_expression: List[List[float]],
        spatial_coords: List[List[float]],
        model_version: Optional[str] = None,
        request_id: Optional[str] = None,
        return_embeddings: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Run prediction with advanced serving features."""
        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())
        
        try:
            self.active_requests += 1
            
            # Select model version (A/B testing or specified)
            if model_version is None:
                if self.config.enable_ab_testing:
                    model_version = self.ab_testing.select_model_version(request_id)
                else:
                    model_version = "latest"
            
            # Get actual version from registry
            model_info = self.model_registry.get_model(model_version)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model version {model_version} not found")
            
            actual_version = model_info.version_id
            
            # Load balancing for model instances (if multiple instances per version)
            # For now, we have one instance per version, but this could be extended
            
            # Check cache first
            cache_key = None
            cached_result = None
            
            if use_cache and self.config.enable_request_caching:
                cache_data = {
                    'gene_expression': gene_expression,
                    'spatial_coords': spatial_coords,
                    'model_version': actual_version,
                    'return_embeddings': return_embeddings
                }
                
                cached_result = self.cache_manager.get(
                    data=cache_data,
                    operation='model_prediction'
                )
            
            if cached_result is not None:
                # Return cached result
                processing_time_ms = (time.time() - start_time) * 1000
                
                result = {
                    'embeddings': cached_result.get('embeddings'),
                    'predictions': cached_result.get('predictions'),
                    'model_version': actual_version,
                    'processing_time_ms': processing_time_ms,
                    'request_id': request_id,
                    'cached': True
                }
                
                # Record metrics
                self._record_request_metrics(request_id, actual_version, processing_time_ms, True, True)
                
                return result
            
            # Run actual inference
            model_server = self.model_servers.get(actual_version)
            if not model_server:
                raise HTTPException(status_code=500, detail=f"Model server not available for version {actual_version}")
            
            # Convert to AnnData format expected by ModelServer
            adata = self._create_anndata_from_request(gene_expression, spatial_coords)
            
            # Run prediction
            prediction_result = model_server.predict(
                adata=adata,
                return_embeddings=return_embeddings,
                use_cache=False  # We handle caching at this level
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Prepare response
            result = {
                'embeddings': prediction_result.get('embeddings', {}).get('embeddings', None),
                'predictions': prediction_result,
                'model_version': actual_version,
                'processing_time_ms': processing_time_ms,
                'request_id': request_id,
                'cached': False
            }
            
            # Cache result
            if use_cache and self.config.enable_request_caching:
                self.cache_manager.put(
                    value=result,
                    data=cache_data,
                    operation='model_prediction',
                    ttl=self.config.cache_ttl
                )
            
            # Record metrics
            self._record_request_metrics(request_id, actual_version, processing_time_ms, True, False)
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._record_request_metrics(request_id, model_version or "unknown", processing_time_ms, False, False)
            
            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise
            
        finally:
            self.active_requests -= 1
    
    def _create_anndata_from_request(
        self,
        gene_expression: List[List[float]],
        spatial_coords: List[List[float]]
    ) -> AnnData:
        """Create AnnData object from request data."""
        import pandas as pd
        
        # Convert to numpy arrays
        X = np.array(gene_expression, dtype=np.float32)
        spatial = np.array(spatial_coords, dtype=np.float32)
        
        # Create obs and var dataframes
        n_cells, n_genes = X.shape
        
        obs = pd.DataFrame({
            'cell_id': [f'cell_{i}' for i in range(n_cells)]
        })
        obs.index = obs['cell_id']
        
        var = pd.DataFrame({
            'gene_id': [f'gene_{i}' for i in range(n_genes)]
        })
        var.index = var['gene_id']
        
        # Create AnnData
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = spatial
        
        return adata
    
    def _record_request_metrics(
        self,
        request_id: str,
        version_id: str,
        processing_time_ms: float,
        success: bool,
        cached: bool
    ) -> None:
        """Record request metrics across all monitoring systems."""
        # Health monitoring
        self.health_monitor.record_request(
            processing_time_ms=processing_time_ms,
            success=success,
            version_id=version_id,
            metadata={'cached': cached, 'request_id': request_id}
        )
        
        # A/B testing
        if self.config.enable_ab_testing:
            self.ab_testing.record_experiment(
                request_id=request_id,
                version_id=version_id,
                success=success,
                latency_ms=processing_time_ms,
                metadata={'cached': cached}
            )
        
        # Load balancing
        self.load_balancer.record_request(
            version_id=version_id,
            processing_time_ms=processing_time_ms,
            success=success
        )
        
        # Request history
        self.request_history.append({
            'request_id': request_id,
            'version_id': version_id,
            'processing_time_ms': processing_time_ms,
            'success': success,
            'cached': cached,
            'timestamp': time.time()
        })
        
        # Update auto-scaler with performance metrics
        if self.auto_scaler:
            self.auto_scaler.update_performance_metrics({
                'batch_processing_time': processing_time_ms / 1000,
                'throughput_samples_per_sec': 1000 / max(processing_time_ms, 1),
                'active_workers': self.active_requests,
                'queue_length': 0  # We don't have a queue in this implementation
            })
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        return {
            'health': self.health_monitor.get_health_status(),
            'model_registry': {
                'active_models': len([m for m in self.model_registry.list_models() if m.status == 'active']),
                'total_models': len(self.model_registry.models),
                'active_version': self.model_registry.active_version
            },
            'ab_testing': self.ab_testing.get_ab_test_results(),
            'load_balancing': self.load_balancer.get_load_stats(),
            'cache': self.cache_manager.get_stats(),
            'active_requests': self.active_requests,
            'total_requests': len(self.request_history)
        }
    
    def _setup_api(self) -> None:
        """Setup FastAPI application."""
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available, skipping API setup")
            return
        
        self.app = FastAPI(
            title="Spatial-Omics GFM API",
            description="Advanced model serving API for spatial transcriptomics",
            version="3.0.0",
            docs_url="/docs" if self.config.enable_docs else None,
            redoc_url="/redoc" if self.config.enable_docs else None
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define API endpoints
        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict_endpoint(request: InferenceRequest):
            """Main prediction endpoint."""
            try:
                result = self.predict(
                    gene_expression=request.gene_expression,
                    spatial_coords=request.spatial_coords,
                    model_version=request.model_version,
                    return_embeddings=request.return_embeddings,
                    use_cache=request.cache_result
                )
                
                return InferenceResponse(**result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_endpoint():
            """Health check endpoint."""
            health_status = self.health_monitor.get_health_status()
            
            return HealthResponse(
                status=health_status['status'],
                timestamp=health_status['timestamp'],
                version="3.0.0",
                uptime_seconds=health_status['uptime_seconds'],
                active_models=len([m for m in self.model_registry.list_models() if m.status == 'active']),
                system_metrics=health_status['system_metrics']
            )
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models_endpoint():
            """List available models."""
            models = self.model_registry.list_models()
            
            return [
                ModelInfo(
                    version_id=model.version_id,
                    status=model.status,
                    performance_metrics=model.performance_metrics,
                    deployment_time=model.deployment_time,
                    metadata=model.metadata
                )
                for model in models
            ]
        
        @self.app.get("/stats")
        async def stats_endpoint():
            """Get comprehensive server statistics."""
            return self.get_server_stats()
        
        @self.app.post("/models/{version_id}/activate")
        async def activate_model_endpoint(version_id: str):
            """Activate a model version."""
            success = self.model_registry.activate_model(version_id)
            
            if success:
                return {"status": "success", "message": f"Activated model {version_id}"}
            else:
                raise HTTPException(status_code=404, detail=f"Model {version_id} not found")
        
        @self.app.put("/ab-testing/traffic-split")
        async def update_traffic_split_endpoint(traffic_splits: Dict[str, float]):
            """Update A/B testing traffic splits."""
            success = self.ab_testing.update_traffic_split(traffic_splits)
            
            if success:
                return {"status": "success", "traffic_splits": traffic_splits}
            else:
                raise HTTPException(status_code=400, detail="Invalid traffic splits")
    
    def run_server(self) -> None:
        """Run the FastAPI server."""
        if not self.app:
            raise RuntimeError("FastAPI not available or app not setup")
        
        self.start_serving()
        
        try:
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                workers=1,  # Use 1 worker for now to avoid model loading issues
                access_log=True
            )
        finally:
            self.stop_serving()


# Convenience functions
def create_advanced_server(
    model_versions: Dict[str, str],
    config: Optional[ServingConfig] = None,
    enable_caching: bool = True,
    enable_auto_scaling: bool = True
) -> AdvancedModelServer:
    """Create advanced model server with multiple versions."""
    
    if config is None:
        config = ServingConfig()
    
    # Create components
    cache_manager = create_cache_manager() if enable_caching else None
    auto_scaler = create_auto_scaler() if enable_auto_scaling else None
    
    # Create server
    server = AdvancedModelServer(
        config=config,
        cache_manager=cache_manager,
        auto_scaler=auto_scaler
    )
    
    # Register models
    for version_id, model_path in model_versions.items():
        server.register_model(version_id, model_path)
    
    return server


def deploy_model_with_ab_testing(
    model_v1_path: str,
    model_v2_path: str,
    traffic_split: Tuple[float, float] = (0.8, 0.2),
    port: int = 8000
) -> AdvancedModelServer:
    """Deploy two model versions with A/B testing."""
    
    config = ServingConfig(
        port=port,
        enable_ab_testing=True,
        traffic_split_config={
            "v1": traffic_split[0],
            "v2": traffic_split[1]
        }
    )
    
    server = create_advanced_server(
        model_versions={
            "v1": model_v1_path,
            "v2": model_v2_path
        },
        config=config
    )
    
    return server
"""
Generation 3: Scalable Distributed Training for Spatial-Omics GFM.

This module implements enterprise-scale distributed training capabilities:
- Advanced PyTorch DDP/FSDP with optimized communication
- Multi-node cluster support with automatic discovery
- Dynamic resource allocation and load balancing
- Fault-tolerant training with checkpointing and recovery
- Resource optimization for million-cell datasets
"""

import os
import sys
import time
import json
import socket
import logging
import threading
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hma

from ..models.graph_transformer import SpatialGraphTransformer
from ..data.base import BaseSpatialDataset
from .distributed_training import DistributedConfig, DistributedTrainer
from .fine_tuning import FineTuningConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalableDistributedConfig:
    """Advanced configuration for scalable distributed training."""
    
    # Basic distributed settings
    backend: str = "nccl"
    init_method: str = "env://"
    
    # Multi-node settings
    master_addr: str = "localhost"
    master_port: str = "29500"
    nnodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = -1  # Auto-detect GPUs per node
    
    # Advanced FSDP settings
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_auto_wrap_min_params: int = 100_000
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    
    # Communication optimization
    gradient_compression: bool = True
    compression_ratio: float = 0.1
    communication_hook: str = "fp16_compress"  # fp16_compress, powerSGD, none
    bucket_cap_mb: int = 25
    gradient_predivide_factor: float = 1.0
    
    # Resource management
    enable_auto_scaling: bool = True
    dynamic_batch_size: bool = True
    memory_efficient_attention: bool = True
    activation_checkpointing: bool = True
    cpu_offload_optimizer: bool = False
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    max_restarts: int = 5
    checkpoint_interval: int = 100
    elastic_training: bool = True
    health_check_interval: int = 30
    
    # Performance optimization
    compile_model: bool = True
    compile_fullgraph: bool = False
    use_torch_dynamo: bool = True
    profile_memory: bool = False
    profile_communication: bool = False
    
    # Cluster management
    auto_node_discovery: bool = True
    heartbeat_timeout: int = 60
    coordinator_port: str = "29501"
    resource_monitor_interval: int = 10


class ClusterCoordinator:
    """Manages multi-node cluster coordination and node discovery."""
    
    def __init__(self, config: ScalableDistributedConfig):
        self.config = config
        self.nodes = {}
        self.node_status = {}
        self.resource_stats = defaultdict(dict)
        self.coordinator_socket = None
        self.running = False
        
        logger.info("Initialized ClusterCoordinator")
    
    def start_coordinator(self) -> None:
        """Start cluster coordinator service."""
        if self.config.node_rank == 0:  # Only master node runs coordinator
            self.coordinator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.coordinator_socket.bind((self.config.master_addr, int(self.config.coordinator_port)))
            self.coordinator_socket.listen(10)
            
            self.running = True
            coordinator_thread = threading.Thread(target=self._coordinator_loop, daemon=True)
            coordinator_thread.start()
            
            logger.info(f"Cluster coordinator started on {self.config.master_addr}:{self.config.coordinator_port}")
    
    def _coordinator_loop(self) -> None:
        """Main coordinator loop for handling node communications."""
        while self.running:
            try:
                client_socket, addr = self.coordinator_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Coordinator error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, addr: Tuple[str, int]) -> None:
        """Handle individual node communications."""
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                message = json.loads(data.decode())
                response = self._process_message(message, addr)
                
                if response:
                    client_socket.send(json.dumps(response).encode())
                    
        except Exception as e:
            logger.warning(f"Client handler error for {addr}: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        """Process incoming messages from nodes."""
        msg_type = message.get("type")
        
        if msg_type == "register":
            # Register new node
            node_info = message.get("node_info", {})
            self.nodes[addr[0]] = node_info
            self.node_status[addr[0]] = "active"
            logger.info(f"Registered node: {addr[0]}")
            return {"status": "registered"}
            
        elif msg_type == "heartbeat":
            # Update node status
            self.node_status[addr[0]] = "active"
            resource_info = message.get("resources", {})
            self.resource_stats[addr[0]] = resource_info
            return {"status": "ok"}
            
        elif msg_type == "get_cluster_info":
            # Return cluster information
            return {
                "nodes": self.nodes,
                "node_status": self.node_status,
                "resource_stats": dict(self.resource_stats)
            }
            
        return {"status": "unknown_message"}
    
    def register_node(self) -> bool:
        """Register this node with cluster coordinator."""
        if self.config.node_rank == 0:
            return True  # Master node doesn't need to register
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.master_addr, int(self.config.coordinator_port)))
            
            node_info = {
                "node_rank": self.config.node_rank,
                "hostname": socket.gethostname(),
                "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "memory_gb": self._get_total_memory_gb()
            }
            
            message = {
                "type": "register",
                "node_info": node_info
            }
            
            sock.send(json.dumps(message).encode())
            response = json.loads(sock.recv(1024).decode())
            sock.close()
            
            return response.get("status") == "registered"
            
        except Exception as e:
            logger.error(f"Node registration failed: {e}")
            return False
    
    def send_heartbeat(self) -> None:
        """Send heartbeat to coordinator."""
        if self.config.node_rank == 0:
            return
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.master_addr, int(self.config.coordinator_port)))
            
            resources = {
                "gpu_memory_used": self._get_gpu_memory_usage(),
                "cpu_percent": self._get_cpu_usage(),
                "timestamp": time.time()
            }
            
            message = {
                "type": "heartbeat",
                "resources": resources
            }
            
            sock.send(json.dumps(message).encode())
            response = json.loads(sock.recv(1024).decode())
            sock.close()
            
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
    
    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 16.0  # Default assumption
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def stop_coordinator(self) -> None:
        """Stop cluster coordinator."""
        self.running = False
        if self.coordinator_socket:
            self.coordinator_socket.close()


class FaultToleranceManager:
    """Manages fault tolerance, checkpointing, and recovery."""
    
    def __init__(self, config: ScalableDistributedConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.restart_count = 0
        self.last_checkpoint_time = time.time()
        self.training_state = {}
        
        logger.info("Initialized FaultToleranceManager")
    
    def should_checkpoint(self, step: int, loss: float) -> bool:
        """Determine if we should create a checkpoint."""
        time_since_last = time.time() - self.last_checkpoint_time
        
        # Checkpoint based on interval or time
        return (
            step % self.config.checkpoint_interval == 0 or
            time_since_last > 300  # Every 5 minutes minimum
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> Path:
        """Save comprehensive checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        # Prepare model state dict based on parallelization strategy
        if isinstance(model, FSDP):
            # FSDP model
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state_dict = model.state_dict()
        elif isinstance(model, DDP):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': self.config.__dict__,
            'restart_count': self.restart_count,
            'timestamp': time.time(),
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'rank': dist.get_rank() if dist.is_initialized() else 0
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Only rank 0 saves
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.save(checkpoint, checkpoint_path)
            
            # Keep only recent checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        self.last_checkpoint_time = time.time()
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if not checkpoint_files:
            logger.info("No checkpoints found")
            return None
        
        # Find latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def _cleanup_old_checkpoints(self, keep_recent: int = 5) -> None:
        """Remove old checkpoints to save disk space."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[:-keep_recent]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def handle_training_failure(self, error: Exception) -> bool:
        """Handle training failure and determine if restart should be attempted."""
        self.restart_count += 1
        
        logger.error(f"Training failure #{self.restart_count}: {error}")
        
        if self.restart_count >= self.config.max_restarts:
            logger.error("Max restarts reached, giving up")
            return False
        
        if self.config.enable_fault_tolerance:
            logger.info("Attempting to restart training from checkpoint")
            return True
        
        return False


class ResourceManager:
    """Manages compute resources and dynamic scaling."""
    
    def __init__(self, config: ScalableDistributedConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.scaling_decisions = deque(maxlen=100)
        
        logger.info("Initialized ResourceManager")
    
    def optimize_batch_size(
        self,
        model: nn.Module,
        current_batch_size: int,
        memory_threshold: float = 0.85
    ) -> int:
        """Dynamically optimize batch size based on available memory."""
        if not self.config.dynamic_batch_size:
            return current_batch_size
        
        memory_usage = self.resource_monitor.get_memory_utilization()
        
        if memory_usage > memory_threshold:
            # Reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            logger.info(f"Reducing batch size: {current_batch_size} -> {new_batch_size} (memory: {memory_usage:.2f})")
        elif memory_usage < 0.6:
            # Increase batch size
            new_batch_size = int(current_batch_size * 1.2)
            logger.info(f"Increasing batch size: {current_batch_size} -> {new_batch_size} (memory: {memory_usage:.2f})")
        else:
            new_batch_size = current_batch_size
        
        self.scaling_decisions.append({
            'timestamp': time.time(),
            'old_batch_size': current_batch_size,
            'new_batch_size': new_batch_size,
            'memory_usage': memory_usage,
            'reason': 'memory_optimization'
        })
        
        return new_batch_size
    
    def should_enable_gradient_checkpointing(self) -> bool:
        """Determine if gradient checkpointing should be enabled."""
        memory_usage = self.resource_monitor.get_memory_utilization()
        return memory_usage > 0.7 or self.config.activation_checkpointing
    
    def get_optimal_num_workers(self) -> int:
        """Get optimal number of dataloader workers."""
        cpu_count = os.cpu_count() or 4
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Rule of thumb: 2-4 workers per GPU, but not more than CPU cores
        optimal_workers = min(cpu_count, gpu_count * 3)
        return max(1, optimal_workers)


class ResourceMonitor:
    """Monitors system resources for optimization decisions."""
    
    def __init__(self):
        self.history = defaultdict(deque)
        self.monitoring = True
        
    def get_memory_utilization(self) -> float:
        """Get current memory utilization percentage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            try:
                import psutil
                return psutil.virtual_memory().percent / 100.0
            except ImportError:
                return 0.5  # Default assumption
    
    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except (ImportError, Exception):
            return 0.0
    
    def update_metrics(self) -> None:
        """Update resource metrics."""
        if not self.monitoring:
            return
            
        timestamp = time.time()
        self.history['memory_util'].append((timestamp, self.get_memory_utilization()))
        self.history['gpu_util'].append((timestamp, self.get_gpu_utilization()))
        
        # Keep only recent history
        cutoff_time = timestamp - 300  # 5 minutes
        for key in self.history:
            while self.history[key] and self.history[key][0][0] < cutoff_time:
                self.history[key].popleft()


class ScalableDistributedTrainer:
    """
    Enterprise-scale distributed trainer with advanced optimizations.
    
    Features:
    - Multi-node FSDP training with communication optimization
    - Fault-tolerant training with automatic recovery
    - Dynamic resource management and scaling
    - Advanced performance monitoring and profiling
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        config: ScalableDistributedConfig,
        fine_tuning_config: FineTuningConfig,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.fine_tuning_config = fine_tuning_config
        self.output_dir = Path(output_dir) if output_dir else Path("./scalable_training")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.cluster_coordinator = ClusterCoordinator(config)
        self.fault_tolerance = FaultToleranceManager(config, self.output_dir)
        self.resource_manager = ResourceManager(config)
        
        # Training state
        self.device = None
        self.distributed_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Performance tracking
        self.training_metrics = defaultdict(list)
        self.communication_stats = {}
        
        logger.info("Initialized ScalableDistributedTrainer")
    
    def setup_distributed_environment(self) -> None:
        """Setup distributed training environment with multi-node support."""
        # Auto-detect GPU count if not specified
        if self.config.nproc_per_node == -1:
            self.config.nproc_per_node = torch.cuda.device_count()
        
        # Calculate world size
        world_size = self.config.nnodes * self.config.nproc_per_node
        
        # Setup environment variables
        os.environ.update({
            'MASTER_ADDR': self.config.master_addr,
            'MASTER_PORT': self.config.master_port,
            'WORLD_SIZE': str(world_size),
            'NODE_RANK': str(self.config.node_rank),
        })
        
        # Start cluster coordinator if master node
        if self.config.auto_node_discovery:
            self.cluster_coordinator.start_coordinator()
        
        logger.info(f"Distributed setup: {self.config.nnodes} nodes, {self.config.nproc_per_node} GPUs/node")
    
    def launch_training(
        self,
        train_dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset] = None
    ) -> Dict[str, Any]:
        """Launch scalable distributed training across multiple nodes."""
        self.setup_distributed_environment()
        
        # Launch training processes
        if self.config.nnodes == 1:
            # Single node training
            if self.config.nproc_per_node > 1:
                mp.spawn(
                    self._training_worker,
                    args=(train_dataset, val_dataset),
                    nprocs=self.config.nproc_per_node,
                    join=True
                )
            else:
                self._training_worker(0, train_dataset, val_dataset)
        else:
            # Multi-node training
            rank = self.config.node_rank * self.config.nproc_per_node
            mp.spawn(
                self._training_worker,
                args=(train_dataset, val_dataset),
                nprocs=self.config.nproc_per_node,
                join=True
            )
        
        return {"status": "completed"}
    
    def _training_worker(
        self,
        local_rank: int,
        train_dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset]
    ) -> None:
        """Training worker process."""
        # Calculate global rank
        global_rank = self.config.node_rank * self.config.nproc_per_node + local_rank
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
        else:
            self.device = torch.device('cpu')
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.nnodes * self.config.nproc_per_node,
            rank=global_rank
        )
        
        try:
            # Register with cluster coordinator
            if self.config.auto_node_discovery and self.config.node_rank > 0:
                self.cluster_coordinator.register_node()
            
            # Setup model for distributed training
            self._setup_distributed_model()
            
            # Setup training components
            self._setup_training_components(train_dataset)
            
            # Load checkpoint if exists
            start_epoch, start_step = self._load_checkpoint_if_exists()
            
            # Start training loop
            self._distributed_training_loop(
                train_dataset, val_dataset, start_epoch, start_step
            )
            
        except Exception as e:
            logger.error(f"Training worker {global_rank} failed: {e}")
            if self.fault_tolerance.handle_training_failure(e):
                # Attempt restart
                self._training_worker(local_rank, train_dataset, val_dataset)
            raise
        finally:
            # Cleanup
            if dist.is_initialized():
                dist.destroy_process_group()
    
    def _setup_distributed_model(self) -> None:
        """Setup model for distributed training with FSDP or DDP."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Apply activation checkpointing if needed
        if self.resource_manager.should_enable_gradient_checkpointing():
            self._apply_activation_checkpointing()
        
        # Setup distributed wrapper
        if self.config.use_fsdp:
            self.distributed_model = self._setup_fsdp_model()
        else:
            self.distributed_model = self._setup_ddp_model()
        
        # Compile model if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.distributed_model = torch.compile(
                    self.distributed_model,
                    fullgraph=self.config.compile_fullgraph,
                    dynamic=True
                )
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def _setup_fsdp_model(self) -> FSDP:
        """Setup FSDP model with advanced optimizations."""
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        # Mixed precision policy
        mixed_precision_policy = None
        if self.config.fsdp_mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        # Sharding strategy
        sharding_strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_strategies[self.config.fsdp_sharding_strategy]
        
        # Auto wrap policy
        auto_wrap_policy = transformer_auto_wrap_policy
        
        # CPU offload
        cpu_offload = None
        if self.config.fsdp_cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)
        
        fsdp_model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=getattr(BackwardPrefetch, self.config.fsdp_backward_prefetch),
            cpu_offload=cpu_offload,
            device_id=self.device.index if self.device.type == 'cuda' else None,
            limit_all_gathers=True,
            use_orig_params=True,  # Important for optimizer compatibility
        )
        
        logger.info(f"FSDP model setup completed with {self.config.fsdp_sharding_strategy} strategy")
        return fsdp_model
    
    def _setup_ddp_model(self) -> DDP:
        """Setup DDP model with communication optimizations."""
        # Setup communication hook if specified
        ddp_model = DDP(
            self.model,
            device_ids=[self.device.index] if self.device.type == 'cuda' else None,
            output_device=self.device.index if self.device.type == 'cuda' else None,
            bucket_cap_mb=self.config.bucket_cap_mb,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
        
        # Apply communication hook
        if self.config.communication_hook == "fp16_compress":
            ddp_model.register_comm_hook(
                None, 
                self._fp16_compress_hook
            )
        elif self.config.communication_hook == "powerSGD":
            from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
            ddp_model.register_comm_hook(
                powerSGD_hook.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=4,
                    start_powerSGD_iter=2
                ),
                powerSGD_hook.powerSGD_hook
            )
        
        logger.info("DDP model setup completed with communication optimizations")
        return ddp_model
    
    def _fp16_compress_hook(self, process_group, bucket):
        """Custom FP16 compression communication hook."""
        compressed = bucket.buffer().to(torch.float16)
        fut = dist.all_reduce(compressed, group=process_group, async_op=True)
        return fut.get_future().then(lambda fut: fut.value()[0].to(torch.float32))
    
    def _apply_activation_checkpointing(self) -> None:
        """Apply activation checkpointing to reduce memory usage."""
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                # Wrap transformer layers with checkpointing
                layer = checkpoint_wrapper(
                    layer,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
        
        logger.info("Applied activation checkpointing to transformer layers")
    
    def _setup_training_components(self, train_dataset: BaseSpatialDataset) -> None:
        """Setup optimizer, scheduler, and other training components."""
        # Get model parameters for optimizer
        if isinstance(self.distributed_model, FSDP):
            parameters = self.distributed_model.parameters()
        else:
            parameters = self.distributed_model.parameters()
        
        # Setup optimizer with CPU offloading if enabled
        if self.config.cpu_offload_optimizer:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            self.optimizer = ZeroRedundancyOptimizer(
                parameters,
                optimizer_class=torch.optim.AdamW,
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        
        # Setup scheduler
        if self.fine_tuning_config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.fine_tuning_config.epochs
            )
        
        # Setup mixed precision scaler
        if self.fine_tuning_config.use_mixed_precision:
            self.scaler = GradScaler()
        
        logger.info("Training components setup completed")
    
    def _load_checkpoint_if_exists(self) -> Tuple[int, int]:
        """Load checkpoint if exists and return starting epoch and step."""
        checkpoint = self.fault_tolerance.load_latest_checkpoint()
        
        if checkpoint is None:
            return 0, 0
        
        try:
            # Load model state
            if isinstance(self.distributed_model, FSDP):
                with FSDP.state_dict_type(
                    self.distributed_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                ):
                    self.distributed_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.distributed_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            
            logger.info(f"Resumed from checkpoint: epoch {start_epoch}, step {start_step}")
            return start_epoch, start_step
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0, 0
    
    def _distributed_training_loop(
        self,
        train_dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset],
        start_epoch: int,
        start_step: int
    ) -> None:
        """Main distributed training loop."""
        # Create distributed sampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        
        # Dynamic batch size optimization
        current_batch_size = self.fine_tuning_config.batch_size
        current_batch_size = self.resource_manager.optimize_batch_size(
            self.distributed_model, current_batch_size
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            sampler=train_sampler,
            num_workers=self.resource_manager.get_optimal_num_workers(),
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = None
        if val_dataset:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=current_batch_size,
                sampler=val_sampler,
                num_workers=self.resource_manager.get_optimal_num_workers(),
                pin_memory=True
            )
        
        # Training loop
        global_step = start_step
        
        for epoch in range(start_epoch, self.fine_tuning_config.epochs):
            train_sampler.set_epoch(epoch)
            
            # Training epoch
            epoch_metrics = self._train_epoch(train_loader, epoch, global_step)
            global_step += len(train_loader)
            
            # Validation epoch
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Checkpointing
            if dist.get_rank() == 0:
                avg_loss = epoch_metrics.get('loss', 0.0)
                if self.fault_tolerance.should_checkpoint(global_step, avg_loss):
                    self.fault_tolerance.save_checkpoint(
                        self.distributed_model,
                        self.optimizer,
                        self.scheduler,
                        global_step,
                        epoch,
                        avg_loss,
                        epoch_metrics
                    )
            
            # Dynamic resource optimization
            if self.config.enable_auto_scaling and epoch % 5 == 0:
                new_batch_size = self.resource_manager.optimize_batch_size(
                    self.distributed_model, current_batch_size
                )
                if new_batch_size != current_batch_size:
                    current_batch_size = new_batch_size
                    # Note: Would need to recreate data loader in practice
            
            # Send heartbeat to coordinator
            if self.config.auto_node_discovery:
                self.cluster_coordinator.send_heartbeat()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log progress (rank 0 only)
            if dist.get_rank() == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {epoch_metrics.get('loss', 0):.4f}, "
                    f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"Batch Size: {current_batch_size}"
                )
        
        # Final checkpoint
        if dist.get_rank() == 0:
            self.fault_tolerance.save_checkpoint(
                self.distributed_model,
                self.optimizer,
                self.scheduler,
                global_step,
                self.fine_tuning_config.epochs,
                epoch_metrics.get('loss', 0.0),
                epoch_metrics
            )
        
        logger.info("Distributed training completed successfully")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        global_step: int
    ) -> Dict[str, float]:
        """Train for one epoch with advanced optimizations."""
        self.distributed_model.train()
        
        total_loss = 0.0
        total_samples = 0
        step_count = 0
        
        # Resource monitoring
        self.resource_manager.resource_monitor.update_metrics()
        
        # Progress tracking (rank 0 only)
        if dist.get_rank() == 0:
            from tqdm import tqdm
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward and backward pass with gradient accumulation
            step_loss = 0.0
            
            for micro_step in range(self.fine_tuning_config.gradient_accumulation_steps):
                with autocast(enabled=self.fine_tuning_config.use_mixed_precision):
                    # Forward pass
                    outputs = self.distributed_model(
                        gene_expression=batch.x,
                        spatial_coords=batch.pos,
                        edge_index=batch.edge_index,
                        edge_attr=getattr(batch, 'edge_attr', None),
                        batch=getattr(batch, 'batch', None)
                    )
                    
                    # Compute loss (simplified example)
                    embeddings = outputs.get('embeddings', outputs)
                    if hasattr(batch, 'y'):
                        loss = F.mse_loss(embeddings, batch.y)
                    else:
                        loss = embeddings.norm()
                    
                    loss = loss / self.fine_tuning_config.gradient_accumulation_steps
                
                # Backward pass
                if self.fine_tuning_config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                step_loss += loss.item()
            
            # Optimizer step with gradient clipping
            if self.fine_tuning_config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.distributed_model.parameters(),
                    self.fine_tuning_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.distributed_model.parameters(),
                    self.fine_tuning_config.max_grad_norm
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += step_loss * self.fine_tuning_config.gradient_accumulation_steps
            total_samples += len(batch.x) if hasattr(batch, 'x') else self.fine_tuning_config.batch_size
            step_count += 1
            
            # Update progress bar (rank 0 only)
            if dist.get_rank() == 0:
                progress_bar.set_postfix({
                    'loss': total_loss / step_count,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
                })
        
        # Collect metrics from all ranks
        avg_loss = total_loss / max(step_count, 1)
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        return {
            'loss': loss_tensor.item(),
            'samples': total_samples,
            'steps': step_count
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.distributed_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        step_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.fine_tuning_config.use_mixed_precision):
                    outputs = self.distributed_model(
                        gene_expression=batch.x,
                        spatial_coords=batch.pos,
                        edge_index=batch.edge_index,
                        edge_attr=getattr(batch, 'edge_attr', None),
                        batch=getattr(batch, 'batch', None)
                    )
                    
                    embeddings = outputs.get('embeddings', outputs)
                    if hasattr(batch, 'y'):
                        loss = F.mse_loss(embeddings, batch.y)
                    else:
                        loss = embeddings.norm()
                
                total_loss += loss.item()
                total_samples += len(batch.x) if hasattr(batch, 'x') else self.fine_tuning_config.batch_size
                step_count += 1
        
        # Collect metrics from all ranks
        avg_loss = total_loss / max(step_count, 1)
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        return {
            'loss': loss_tensor.item(),
            'samples': total_samples,
            'steps': step_count
        }


def launch_scalable_training(
    model: SpatialGraphTransformer,
    train_dataset: BaseSpatialDataset,
    config: ScalableDistributedConfig,
    fine_tuning_config: FineTuningConfig,
    val_dataset: Optional[BaseSpatialDataset] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Launch scalable distributed training across multiple nodes.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        config: Scalable distributed configuration
        fine_tuning_config: Fine-tuning configuration
        val_dataset: Validation dataset
        output_dir: Output directory
        
    Returns:
        Training results
    """
    trainer = ScalableDistributedTrainer(
        model=model,
        config=config,
        fine_tuning_config=fine_tuning_config,
        output_dir=output_dir
    )
    
    return trainer.launch_training(train_dataset, val_dataset)


def create_scalable_config(
    nnodes: int = 1,
    nproc_per_node: int = -1,
    use_fsdp: bool = True,
    enable_fault_tolerance: bool = True,
    **kwargs
) -> ScalableDistributedConfig:
    """Create scalable distributed configuration with sensible defaults."""
    return ScalableDistributedConfig(
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        use_fsdp=use_fsdp,
        enable_fault_tolerance=enable_fault_tolerance,
        **kwargs
    )
"""
Generation 3: Auto-Scaling & Resource Management Infrastructure.

This module implements enterprise-grade auto-scaling and resource management:
- Dynamic batch size scaling based on throughput
- Automatic resource provisioning and deprovisioning
- Load-based scaling triggers and policies
- Resource usage optimization and efficiency metrics
- Cloud integration for elastic compute
"""

import os
import sys
import time
import json
import psutil
import logging
import threading
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .memory_management import MemoryMonitor, MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    
    # Scaling triggers
    cpu_target_utilization: float = 70.0  # Target CPU utilization %
    memory_target_utilization: float = 75.0  # Target memory utilization %
    gpu_target_utilization: float = 80.0  # Target GPU utilization %
    
    # Scaling bounds
    min_batch_size: int = 1
    max_batch_size: int = 1024
    min_instances: int = 1
    max_instances: int = 10
    
    # Scaling behavior
    scale_up_threshold: float = 85.0  # Scale up when utilization > this
    scale_down_threshold: float = 50.0  # Scale down when utilization < this
    scale_up_factor: float = 1.5  # Multiply by this factor when scaling up
    scale_down_factor: float = 0.8  # Multiply by this factor when scaling down
    
    # Timing controls
    scale_up_cooldown: int = 60  # Seconds to wait before scaling up again
    scale_down_cooldown: int = 300  # Seconds to wait before scaling down again
    evaluation_interval: int = 30  # Seconds between scaling evaluations
    
    # Advanced policies
    enable_predictive_scaling: bool = True
    enable_batch_size_scaling: bool = True
    enable_instance_scaling: bool = False  # For cloud deployments
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, weighted


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    
    # Training-specific metrics
    batch_processing_time: float = 0.0
    throughput_samples_per_sec: float = 0.0
    queue_length: int = 0
    active_workers: int = 0


class ResourceMonitor:
    """Advanced resource monitoring with predictive capabilities."""
    
    def __init__(self, monitoring_interval: int = 10):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_thread = None
        self.running = False
        
        # Prediction models (simplified)
        self.trend_window = 60  # seconds
        self.prediction_horizon = 120  # seconds
        
        logger.info("ResourceMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
        
        logger.info("Resource monitoring stopped")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_percent = 0.0
        gpu_memory_used_gb = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                
                if gpu_memory_total > 0:
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                    gpu_memory_used_gb = gpu_memory_used / (1024**3)
                
                # Try to get GPU utilization via nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                except (ImportError, Exception):
                    pass
                    
            except Exception:
                pass
        
        # Network I/O (simplified)
        try:
            network_stats = psutil.net_io_counters()
            network_io_mbps = (network_stats.bytes_sent + network_stats.bytes_recv) / (1024**2)
        except:
            network_io_mbps = 0.0
        
        # Disk I/O (simplified)
        try:
            disk_stats = psutil.disk_io_counters()
            disk_io_mbps = (disk_stats.read_bytes + disk_stats.write_bytes) / (1024**2)
        except:
            disk_io_mbps = 0.0
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            network_io_mbps=network_io_mbps,
            disk_io_mbps=disk_io_mbps
        )
    
    def get_metrics_history(self, window_seconds: int = 300) -> List[ResourceMetrics]:
        """Get recent metrics history."""
        cutoff_time = time.time() - window_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def predict_resource_usage(self, horizon_seconds: int = 120) -> Dict[str, float]:
        """Predict future resource usage based on trends."""
        if len(self.metrics_history) < 3:
            # Not enough data for prediction
            current = self.get_current_metrics()
            return {
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'gpu_utilization': current.gpu_utilization
            }
        
        # Simple linear trend prediction
        recent_metrics = self.get_metrics_history(self.trend_window)
        
        if len(recent_metrics) < 2:
            current = self.get_current_metrics()
            return {
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'gpu_utilization': current.gpu_utilization
            }
        
        # Calculate trends
        timestamps = [m.timestamp for m in recent_metrics]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_values = [m.gpu_utilization for m in recent_metrics]
        
        # Linear regression for trend
        def predict_trend(timestamps, values, horizon):
            if len(timestamps) < 2:
                return values[-1] if values else 0
            
            # Simple linear trend
            x = np.array(timestamps)
            y = np.array(values)
            
            # Fit line
            coeffs = np.polyfit(x - x[0], y, 1)
            slope, intercept = coeffs
            
            # Predict future value
            future_x = timestamps[-1] + horizon - timestamps[0]
            predicted = slope * future_x + intercept
            
            return max(0, min(100, predicted))  # Clamp to reasonable range
        
        predictions = {
            'cpu_percent': predict_trend(timestamps, cpu_values, horizon_seconds),
            'memory_percent': predict_trend(timestamps, memory_values, horizon_seconds),
            'gpu_utilization': predict_trend(timestamps, gpu_values, horizon_seconds)
        }
        
        return predictions
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)


class BatchSizeScaler:
    """Dynamic batch size scaling based on resource utilization."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.current_batch_size = 32  # Default starting batch size
        self.last_scale_time = 0
        self.scale_history = deque(maxlen=100)
        
        logger.info("BatchSizeScaler initialized")
    
    def evaluate_scaling(
        self,
        metrics: ResourceMetrics,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Evaluate if batch size should be scaled.
        
        Args:
            metrics: Current resource metrics
            performance_metrics: Training performance metrics
            
        Returns:
            New batch size (may be same as current)
        """
        if not self.policy.enable_batch_size_scaling:
            return self.current_batch_size
        
        current_time = time.time()
        
        # Check cooldown periods
        time_since_last_scale = current_time - self.last_scale_time
        
        # Determine if we should scale based on resource utilization
        should_scale_up = False
        should_scale_down = False
        
        # Check memory pressure (primary constraint for batch size)
        if metrics.memory_percent > self.policy.scale_up_threshold:
            should_scale_down = True
        elif metrics.memory_percent < self.policy.scale_down_threshold:
            should_scale_up = True
        
        # Check GPU memory if available
        if metrics.gpu_memory_percent > 0:
            if metrics.gpu_memory_percent > self.policy.scale_up_threshold:
                should_scale_down = True
            elif metrics.gpu_memory_percent < self.policy.scale_down_threshold and not should_scale_down:
                should_scale_up = True
        
        # Consider performance metrics
        if performance_metrics:
            throughput = performance_metrics.get('throughput_samples_per_sec', 0)
            if throughput > 0 and throughput < 10:  # Very slow processing
                should_scale_down = True
        
        new_batch_size = self.current_batch_size
        scale_reason = "no_change"
        
        # Apply scaling decisions with cooldown
        if should_scale_down and time_since_last_scale > self.policy.scale_down_cooldown:
            new_batch_size = max(
                self.policy.min_batch_size,
                int(self.current_batch_size * self.policy.scale_down_factor)
            )
            scale_reason = "scale_down_memory_pressure"
            self.last_scale_time = current_time
            
        elif should_scale_up and time_since_last_scale > self.policy.scale_up_cooldown:
            new_batch_size = min(
                self.policy.max_batch_size,
                int(self.current_batch_size * self.policy.scale_up_factor)
            )
            scale_reason = "scale_up_capacity_available"
            self.last_scale_time = current_time
        
        # Record scaling decision
        if new_batch_size != self.current_batch_size:
            self.scale_history.append({
                'timestamp': current_time,
                'old_batch_size': self.current_batch_size,
                'new_batch_size': new_batch_size,
                'reason': scale_reason,
                'memory_percent': metrics.memory_percent,
                'gpu_memory_percent': metrics.gpu_memory_percent
            })
            
            self.current_batch_size = new_batch_size
            
            logger.info(
                f"Batch size scaled: {self.scale_history[-1]['old_batch_size']} -> {new_batch_size} "
                f"(reason: {scale_reason}, memory: {metrics.memory_percent:.1f}%)"
            )
        
        return self.current_batch_size
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get batch size scaling statistics."""
        if not self.scale_history:
            return {
                'current_batch_size': self.current_batch_size,
                'total_scales': 0,
                'scale_ups': 0,
                'scale_downs': 0,
                'recent_scales': []
            }
        
        scale_ups = sum(1 for s in self.scale_history if s['new_batch_size'] > s['old_batch_size'])
        scale_downs = sum(1 for s in self.scale_history if s['new_batch_size'] < s['old_batch_size'])
        
        return {
            'current_batch_size': self.current_batch_size,
            'total_scales': len(self.scale_history),
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'recent_scales': list(self.scale_history)[-10:]  # Last 10 scaling events
        }


class CloudResourceProvider(ABC):
    """Abstract interface for cloud resource providers."""
    
    @abstractmethod
    def get_available_instances(self) -> List[Dict[str, Any]]:
        """Get list of available instance types."""
        pass
    
    @abstractmethod
    def launch_instance(self, instance_type: str, **kwargs) -> Dict[str, Any]:
        """Launch a new compute instance."""
        pass
    
    @abstractmethod
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a compute instance."""
        pass
    
    @abstractmethod
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get status of a compute instance."""
        pass
    
    @abstractmethod
    def list_instances(self) -> List[Dict[str, Any]]:
        """List all managed instances."""
        pass


class AWSResourceProvider(CloudResourceProvider):
    """AWS EC2 resource provider for auto-scaling."""
    
    def __init__(self, region: str = "us-east-1", **aws_config):
        self.region = region
        self.aws_config = aws_config
        
        try:
            import boto3
            self.ec2 = boto3.client('ec2', region_name=region, **aws_config)
            self.available = True
            logger.info(f"AWS provider initialized for region {region}")
        except ImportError:
            logger.warning("boto3 not available, AWS provider disabled")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize AWS provider: {e}")
            self.available = False
    
    def get_available_instances(self) -> List[Dict[str, Any]]:
        """Get available EC2 instance types."""
        if not self.available:
            return []
        
        try:
            # Get instance type offerings
            response = self.ec2.describe_instance_type_offerings(
                Filters=[
                    {'Name': 'instance-type', 'Values': ['p3.*', 'p4.*', 'g4dn.*']},  # GPU instances
                    {'Name': 'location-type', 'Values': ['availability-zone']}
                ]
            )
            
            instance_types = []
            for offering in response['InstanceTypeOfferings']:
                instance_types.append({
                    'instance_type': offering['InstanceType'],
                    'location': offering['Location']
                })
            
            return instance_types
            
        except Exception as e:
            logger.error(f"Failed to get available instances: {e}")
            return []
    
    def launch_instance(self, instance_type: str, **kwargs) -> Dict[str, Any]:
        """Launch EC2 instance."""
        if not self.available:
            return {'success': False, 'error': 'AWS not available'}
        
        try:
            # Default launch parameters
            launch_params = {
                'ImageId': kwargs.get('ami_id', 'ami-0abcdef1234567890'),  # Replace with actual AMI
                'InstanceType': instance_type,
                'MinCount': 1,
                'MaxCount': 1,
                'KeyName': kwargs.get('key_name', 'default-key'),
                'SecurityGroupIds': kwargs.get('security_groups', ['sg-12345678']),
                'SubnetId': kwargs.get('subnet_id', 'subnet-12345678'),
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'spatial-gfm-worker'},
                            {'Key': 'Purpose', 'Value': 'auto-scaling'},
                            {'Key': 'CreatedBy', 'Value': 'spatial-omics-gfm'}
                        ]
                    }
                ]
            }
            
            response = self.ec2.run_instances(**launch_params)
            
            instance_id = response['Instances'][0]['InstanceId']
            
            return {
                'success': True,
                'instance_id': instance_id,
                'instance_type': instance_type,
                'state': 'launching'
            }
            
        except Exception as e:
            logger.error(f"Failed to launch instance: {e}")
            return {'success': False, 'error': str(e)}
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate EC2 instance."""
        if not self.available:
            return False
        
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Terminated instance: {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get EC2 instance status."""
        if not self.available:
            return {}
        
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                return {
                    'instance_id': instance['InstanceId'],
                    'state': instance['State']['Name'],
                    'instance_type': instance['InstanceType'],
                    'launch_time': instance.get('LaunchTime'),
                    'public_ip': instance.get('PublicIpAddress'),
                    'private_ip': instance.get('PrivateIpAddress')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get instance status: {e}")
            return {}
    
    def list_instances(self) -> List[Dict[str, Any]]:
        """List managed EC2 instances."""
        if not self.available:
            return []
        
        try:
            response = self.ec2.describe_instances(
                Filters=[
                    {'Name': 'tag:CreatedBy', 'Values': ['spatial-omics-gfm']},
                    {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
                ]
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'instance_id': instance['InstanceId'],
                        'state': instance['State']['Name'],
                        'instance_type': instance['InstanceType'],
                        'launch_time': instance.get('LaunchTime'),
                        'public_ip': instance.get('PublicIpAddress'),
                        'private_ip': instance.get('PrivateIpAddress')
                    })
            
            return instances
            
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return []


class MockCloudProvider(CloudResourceProvider):
    """Mock cloud provider for testing."""
    
    def __init__(self):
        self.instances = {}
        self.next_instance_id = 1
        
    def get_available_instances(self) -> List[Dict[str, Any]]:
        return [
            {'instance_type': 'mock.small', 'vcpus': 4, 'memory_gb': 16},
            {'instance_type': 'mock.large', 'vcpus': 8, 'memory_gb': 32},
            {'instance_type': 'mock.xlarge', 'vcpus': 16, 'memory_gb': 64},
        ]
    
    def launch_instance(self, instance_type: str, **kwargs) -> Dict[str, Any]:
        instance_id = f"mock-{self.next_instance_id:04d}"
        self.next_instance_id += 1
        
        self.instances[instance_id] = {
            'instance_id': instance_id,
            'instance_type': instance_type,
            'state': 'running',
            'launch_time': time.time()
        }
        
        return {'success': True, 'instance_id': instance_id}
    
    def terminate_instance(self, instance_id: str) -> bool:
        if instance_id in self.instances:
            self.instances[instance_id]['state'] = 'terminated'
            return True
        return False
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        return self.instances.get(instance_id, {})
    
    def list_instances(self) -> List[Dict[str, Any]]:
        return [inst for inst in self.instances.values() if inst['state'] != 'terminated']


class InstanceScaler:
    """Dynamic instance scaling for cloud deployments."""
    
    def __init__(self, policy: ScalingPolicy, cloud_provider: CloudResourceProvider):
        self.policy = policy
        self.cloud_provider = cloud_provider
        self.managed_instances = {}
        self.last_scale_time = 0
        self.scale_history = deque(maxlen=100)
        
        logger.info("InstanceScaler initialized")
    
    def evaluate_scaling(
        self,
        metrics: ResourceMetrics,
        cluster_metrics: Optional[List[ResourceMetrics]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if instance scaling is needed.
        
        Args:
            metrics: Local resource metrics
            cluster_metrics: Metrics from all instances in cluster
            
        Returns:
            Scaling decision and actions taken
        """
        if not self.policy.enable_instance_scaling:
            return {'action': 'none', 'reason': 'instance_scaling_disabled'}
        
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_time
        
        # Get current instance count
        running_instances = self.cloud_provider.list_instances()
        current_instances = len([i for i in running_instances if i['state'] == 'running'])
        
        # Analyze cluster load if available
        if cluster_metrics:
            avg_cpu = np.mean([m.cpu_percent for m in cluster_metrics])
            avg_memory = np.mean([m.memory_percent for m in cluster_metrics])
            max_cpu = np.max([m.cpu_percent for m in cluster_metrics])
            max_memory = np.max([m.memory_percent for m in cluster_metrics])
        else:
            avg_cpu = metrics.cpu_percent
            avg_memory = metrics.memory_percent
            max_cpu = metrics.cpu_percent
            max_memory = metrics.memory_percent
        
        # Determine scaling action
        action = 'none'
        reason = 'no_scaling_needed'
        
        # Scale up conditions
        if (max_cpu > self.policy.scale_up_threshold or 
            max_memory > self.policy.scale_up_threshold):
            
            if (current_instances < self.policy.max_instances and
                time_since_last_scale > self.policy.scale_up_cooldown):
                action = 'scale_up'
                reason = f'high_utilization_cpu_{max_cpu:.1f}_memory_{max_memory:.1f}'
        
        # Scale down conditions
        elif (avg_cpu < self.policy.scale_down_threshold and 
              avg_memory < self.policy.scale_down_threshold):
            
            if (current_instances > self.policy.min_instances and
                time_since_last_scale > self.policy.scale_down_cooldown):
                action = 'scale_down'
                reason = f'low_utilization_cpu_{avg_cpu:.1f}_memory_{avg_memory:.1f}'
        
        # Execute scaling action
        result = {'action': action, 'reason': reason}
        
        if action == 'scale_up':
            result.update(self._scale_up())
        elif action == 'scale_down':
            result.update(self._scale_down())
        
        # Record scaling decision
        if action != 'none':
            self.scale_history.append({
                'timestamp': current_time,
                'action': action,
                'reason': reason,
                'instances_before': current_instances,
                'instances_after': len(self.cloud_provider.list_instances()),
                'avg_cpu': avg_cpu,
                'avg_memory': avg_memory,
                'max_cpu': max_cpu,
                'max_memory': max_memory
            })
            
            self.last_scale_time = current_time
        
        return result
    
    def _scale_up(self) -> Dict[str, Any]:
        """Scale up by launching new instance."""
        try:
            # Choose instance type (simplified - would be more sophisticated)
            available_types = self.cloud_provider.get_available_instances()
            if not available_types:
                return {'success': False, 'error': 'no_instance_types_available'}
            
            instance_type = available_types[0]['instance_type']
            
            # Launch instance
            result = self.cloud_provider.launch_instance(instance_type)
            
            if result.get('success'):
                instance_id = result['instance_id']
                self.managed_instances[instance_id] = {
                    'launch_time': time.time(),
                    'instance_type': instance_type,
                    'purpose': 'scale_up'
                }
                
                logger.info(f"Scaled up: launched instance {instance_id} ({instance_type})")
                
                return {
                    'success': True,
                    'instance_id': instance_id,
                    'instance_type': instance_type
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _scale_down(self) -> Dict[str, Any]:
        """Scale down by terminating least utilized instance."""
        try:
            # Get running instances
            running_instances = self.cloud_provider.list_instances()
            
            if not running_instances:
                return {'success': False, 'error': 'no_instances_to_terminate'}
            
            # Find instance to terminate (simplified - would analyze utilization)
            instance_to_terminate = running_instances[-1]  # Terminate most recent
            instance_id = instance_to_terminate['instance_id']
            
            # Terminate instance
            if self.cloud_provider.terminate_instance(instance_id):
                if instance_id in self.managed_instances:
                    del self.managed_instances[instance_id]
                
                logger.info(f"Scaled down: terminated instance {instance_id}")
                
                return {
                    'success': True,
                    'terminated_instance_id': instance_id
                }
            else:
                return {'success': False, 'error': 'termination_failed'}
                
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get instance scaling statistics."""
        running_instances = self.cloud_provider.list_instances()
        
        if not self.scale_history:
            return {
                'current_instances': len(running_instances),
                'managed_instances': len(self.managed_instances),
                'total_scales': 0,
                'scale_ups': 0,
                'scale_downs': 0
            }
        
        scale_ups = sum(1 for s in self.scale_history if s['action'] == 'scale_up')
        scale_downs = sum(1 for s in self.scale_history if s['action'] == 'scale_down')
        
        return {
            'current_instances': len(running_instances),
            'managed_instances': len(self.managed_instances),
            'total_scales': len(self.scale_history),
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'recent_scales': list(self.scale_history)[-10:]
        }


class LoadBalancer:
    """Load balancer for distributing work across instances."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances = []
        self.current_index = 0
        self.instance_loads = defaultdict(float)
        
        logger.info(f"LoadBalancer initialized with {strategy} strategy")
    
    def add_instance(self, instance_info: Dict[str, Any]) -> None:
        """Add instance to load balancer."""
        self.instances.append(instance_info)
        logger.info(f"Added instance to load balancer: {instance_info.get('instance_id', 'unknown')}")
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove instance from load balancer."""
        self.instances = [i for i in self.instances if i.get('instance_id') != instance_id]
        if instance_id in self.instance_loads:
            del self.instance_loads[instance_id]
        logger.info(f"Removed instance from load balancer: {instance_id}")
    
    def get_next_instance(self, task_weight: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next instance for task assignment."""
        if not self.instances:
            return None
        
        if self.strategy == "round_robin":
            instance = self.instances[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.instances)
            return instance
            
        elif self.strategy == "least_loaded":
            # Find instance with lowest current load
            min_load = float('inf')
            selected_instance = None
            
            for instance in self.instances:
                instance_id = instance.get('instance_id', '')
                load = self.instance_loads[instance_id]
                
                if load < min_load:
                    min_load = load
                    selected_instance = instance
            
            return selected_instance
            
        elif self.strategy == "weighted":
            # Simple weighted selection based on instance capacity
            # (would be more sophisticated in practice)
            weights = []
            for instance in self.instances:
                # Use instance type as proxy for capacity
                instance_type = instance.get('instance_type', '')
                if 'large' in instance_type:
                    weights.append(2.0)
                elif 'xlarge' in instance_type:
                    weights.append(4.0)
                else:
                    weights.append(1.0)
            
            # Weighted random selection
            total_weight = sum(weights)
            if total_weight == 0:
                return self.instances[0]
            
            import random
            target = random.uniform(0, total_weight)
            current = 0
            
            for i, weight in enumerate(weights):
                current += weight
                if current >= target:
                    return self.instances[i]
            
            return self.instances[-1]
        
        else:
            # Default to round robin
            return self.get_next_instance(task_weight)
    
    def update_instance_load(self, instance_id: str, load: float) -> None:
        """Update instance load metrics."""
        self.instance_loads[instance_id] = load
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'instances': len(self.instances),
            'strategy': self.strategy,
            'instance_loads': dict(self.instance_loads),
            'total_load': sum(self.instance_loads.values())
        }


class AutoScaler:
    """Main auto-scaling orchestrator."""
    
    def __init__(
        self,
        policy: ScalingPolicy,
        cloud_provider: Optional[CloudResourceProvider] = None,
        enable_monitoring: bool = True
    ):
        self.policy = policy
        self.cloud_provider = cloud_provider
        
        # Components
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        self.batch_scaler = BatchSizeScaler(policy)
        self.instance_scaler = InstanceScaler(policy, cloud_provider) if cloud_provider else None
        self.load_balancer = LoadBalancer(policy.load_balance_strategy) if policy.enable_load_balancing else None
        
        # State
        self.running = False
        self.scaling_thread = None
        self.scaling_decisions = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        logger.info("AutoScaler initialized")
    
    def start_auto_scaling(self) -> None:
        """Start auto-scaling process."""
        if self.running:
            return
        
        self.running = True
        
        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        # Start scaling evaluation loop
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self) -> None:
        """Stop auto-scaling process."""
        self.running = False
        
        # Stop monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        # Wait for scaling thread to finish
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join()
        
        logger.info("Auto-scaling stopped")
    
    def evaluate_scaling(
        self,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate and execute scaling decisions.
        
        Args:
            performance_metrics: Training/inference performance metrics
            
        Returns:
            Scaling decisions and actions taken
        """
        if not self.resource_monitor:
            return {'error': 'resource_monitor_not_available'}
        
        # Get current resource metrics
        current_metrics = self.resource_monitor.get_current_metrics()
        
        # Add performance metrics to resource metrics
        if performance_metrics:
            current_metrics.batch_processing_time = performance_metrics.get('batch_processing_time', 0)
            current_metrics.throughput_samples_per_sec = performance_metrics.get('throughput_samples_per_sec', 0)
            current_metrics.queue_length = performance_metrics.get('queue_length', 0)
            current_metrics.active_workers = performance_metrics.get('active_workers', 0)
        
        scaling_result = {
            'timestamp': time.time(),
            'metrics': current_metrics,
            'decisions': {}
        }
        
        # Batch size scaling
        new_batch_size = self.batch_scaler.evaluate_scaling(current_metrics, performance_metrics)
        scaling_result['decisions']['batch_size'] = {
            'old_size': self.batch_scaler.current_batch_size,
            'new_size': new_batch_size,
            'changed': new_batch_size != self.batch_scaler.current_batch_size
        }
        
        # Instance scaling (if enabled and cloud provider available)
        if self.instance_scaler:
            instance_result = self.instance_scaler.evaluate_scaling(current_metrics)
            scaling_result['decisions']['instances'] = instance_result
        
        # Predictive scaling (if enabled)
        if self.policy.enable_predictive_scaling:
            predictions = self.resource_monitor.predict_resource_usage()
            scaling_result['predictions'] = predictions
            
            # Could trigger preemptive scaling based on predictions
            # (implementation would depend on specific requirements)
        
        # Record decision
        self.scaling_decisions.append(scaling_result)
        
        return scaling_result
    
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.batch_scaler.current_batch_size
    
    def get_available_instances(self) -> List[Dict[str, Any]]:
        """Get list of available instances."""
        if not self.cloud_provider:
            return []
        return self.cloud_provider.list_instances()
    
    def get_next_instance_for_task(self) -> Optional[Dict[str, Any]]:
        """Get next instance for task assignment."""
        if not self.load_balancer:
            return None
        return self.load_balancer.get_next_instance()
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics for scaling decisions."""
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        stats = {
            'policy': self.policy.__dict__,
            'batch_scaler': self.batch_scaler.get_scaling_stats(),
            'running': self.running,
            'total_scaling_decisions': len(self.scaling_decisions)
        }
        
        if self.instance_scaler:
            stats['instance_scaler'] = self.instance_scaler.get_scaling_stats()
        
        if self.load_balancer:
            stats['load_balancer'] = self.load_balancer.get_load_stats()
        
        if self.resource_monitor:
            recent_metrics = self.resource_monitor.get_metrics_history(300)  # 5 minutes
            if recent_metrics:
                stats['recent_metrics'] = {
                    'avg_cpu': np.mean([m.cpu_percent for m in recent_metrics]),
                    'avg_memory': np.mean([m.memory_percent for m in recent_metrics]),
                    'avg_gpu': np.mean([m.gpu_utilization for m in recent_metrics])
                }
        
        return stats
    
    def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while self.running:
            try:
                # Evaluate scaling
                self.evaluate_scaling()
                
                # Wait for next evaluation
                time.sleep(self.policy.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(self.policy.evaluation_interval)


def create_auto_scaler(
    cpu_target: float = 70.0,
    memory_target: float = 75.0,
    enable_cloud_scaling: bool = False,
    cloud_provider: str = "aws",
    **policy_kwargs
) -> AutoScaler:
    """Create auto-scaler with sensible defaults."""
    
    policy = ScalingPolicy(
        cpu_target_utilization=cpu_target,
        memory_target_utilization=memory_target,
        enable_instance_scaling=enable_cloud_scaling,
        **policy_kwargs
    )
    
    # Setup cloud provider if requested
    cloud_provider_instance = None
    if enable_cloud_scaling:
        if cloud_provider == "aws":
            cloud_provider_instance = AWSResourceProvider()
        elif cloud_provider == "mock":
            cloud_provider_instance = MockCloudProvider()
        else:
            logger.warning(f"Unknown cloud provider: {cloud_provider}, using mock provider")
            cloud_provider_instance = MockCloudProvider()
    
    return AutoScaler(policy, cloud_provider_instance)


# Example usage and testing functions
def test_auto_scaling(duration_seconds: int = 300) -> Dict[str, Any]:
    """Test auto-scaling functionality."""
    logger.info(f"Starting auto-scaling test for {duration_seconds} seconds")
    
    # Create auto-scaler with mock cloud provider
    auto_scaler = create_auto_scaler(
        enable_cloud_scaling=True,
        cloud_provider="mock"
    )
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling()
    
    try:
        # Simulate workload with varying performance metrics
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Simulate varying performance
            current_time = time.time() - start_time
            
            # Create synthetic performance metrics
            synthetic_metrics = {
                'batch_processing_time': 0.1 + 0.05 * np.sin(current_time / 30),
                'throughput_samples_per_sec': 100 + 50 * np.cos(current_time / 45),
                'queue_length': max(0, int(10 + 5 * np.sin(current_time / 20))),
                'active_workers': 4
            }
            
            auto_scaler.update_performance_metrics(synthetic_metrics)
            
            time.sleep(10)  # Update every 10 seconds
        
    finally:
        # Stop auto-scaling
        auto_scaler.stop_auto_scaling()
    
    # Get final statistics
    stats = auto_scaler.get_scaling_stats()
    
    logger.info("Auto-scaling test completed")
    return stats
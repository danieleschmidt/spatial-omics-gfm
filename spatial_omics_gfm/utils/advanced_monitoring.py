"""
Advanced monitoring and profiling utilities for Spatial-Omics GFM.
Implements comprehensive metrics collection, real-time monitoring, and performance analysis.
"""

import logging
import time
import threading
import psutil
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np
import torch
from pathlib import Path
import warnings
import traceback
from concurrent.futures import ThreadPoolExecutor
import queue
import atexit

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """Individual metric sample with timestamp."""
    timestamp: float
    value: Union[float, int, str]
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'tags': self.tags
        }


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: Union[float, int]
    duration_seconds: float = 60.0  # Alert after condition is true for this long
    callback: Optional[Callable] = None
    enabled: bool = True
    cooldown_seconds: float = 300.0  # Wait before alerting again
    
    def __post_init__(self):
        if self.condition not in ['gt', 'lt', 'eq', 'ne']:
            raise ValueError(f"Invalid condition: {self.condition}")


class MetricsCollector:
    """Thread-safe metrics collector with various aggregation functions."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.Lock()
        self.start_time = time.time()
    
    def record(self, metric_name: str, value: Union[float, int, str], tags: Optional[Dict[str, str]] = None):
        """Record a metric sample."""
        sample = MetricSample(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[metric_name].append(sample)
    
    def get_samples(self, metric_name: str, since: Optional[float] = None) -> List[MetricSample]:
        """Get metric samples, optionally filtered by time."""
        with self._lock:
            samples = list(self.metrics[metric_name])
        
        if since is not None:
            samples = [s for s in samples if s.timestamp >= since]
        
        return samples
    
    def get_latest(self, metric_name: str) -> Optional[MetricSample]:
        """Get the latest sample for a metric."""
        with self._lock:
            samples = self.metrics[metric_name]
            return samples[-1] if samples else None
    
    def get_aggregated(
        self, 
        metric_name: str, 
        aggregation: str = 'mean',
        window_seconds: Optional[float] = None
    ) -> Optional[float]:
        """Get aggregated metric value."""
        if window_seconds:
            since = time.time() - window_seconds
            samples = self.get_samples(metric_name, since)
        else:
            samples = self.get_samples(metric_name)
        
        if not samples:
            return None
        
        numeric_values = [s.value for s in samples if isinstance(s.value, (int, float))]
        if not numeric_values:
            return None
        
        if aggregation == 'mean':
            return np.mean(numeric_values)
        elif aggregation == 'median':
            return np.median(numeric_values)
        elif aggregation == 'min':
            return np.min(numeric_values)
        elif aggregation == 'max':
            return np.max(numeric_values)
        elif aggregation == 'sum':
            return np.sum(numeric_values)
        elif aggregation == 'count':
            return len(numeric_values)
        elif aggregation == 'std':
            return np.std(numeric_values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(self.metrics.keys())
    
    def clear_metric(self, metric_name: str):
        """Clear all samples for a metric."""
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in various formats."""
        with self._lock:
            data = {}
            for metric_name, samples in self.metrics.items():
                data[metric_name] = [sample.to_dict() for sample in samples]
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unknown export format: {format}")


class SystemMonitor:
    """Comprehensive system resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, interval_seconds: float = 1.0):
        self.metrics_collector = metrics_collector
        self.interval_seconds = interval_seconds
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start system monitoring."""
        if self.monitoring:
            logger.warning("System monitoring already running")
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics_collector.record('system.cpu.percent', cpu_percent)
        
        cpu_count = psutil.cpu_count()
        self.metrics_collector.record('system.cpu.count', cpu_count)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record('system.memory.percent', memory.percent)
        self.metrics_collector.record('system.memory.used_gb', memory.used / (1024**3))
        self.metrics_collector.record('system.memory.available_gb', memory.available / (1024**3))
        self.metrics_collector.record('system.memory.total_gb', memory.total / (1024**3))
        
        # Disk metrics
        try:
            disk = psutil.disk_usage('/')
            self.metrics_collector.record('system.disk.percent', disk.percent)
            self.metrics_collector.record('system.disk.used_gb', disk.used / (1024**3))
            self.metrics_collector.record('system.disk.free_gb', disk.free / (1024**3))
        except Exception:
            pass  # Disk monitoring might fail on some systems
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    self.metrics_collector.record(
                        f'system.gpu.{i}.memory_allocated_gb', 
                        gpu_memory_allocated,
                        tags={'device': str(i)}
                    )
                    self.metrics_collector.record(
                        f'system.gpu.{i}.memory_reserved_gb', 
                        gpu_memory_reserved,
                        tags={'device': str(i)}
                    )
                    
                    # GPU utilization (if nvidia-ml-py is available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        self.metrics_collector.record(
                            f'system.gpu.{i}.utilization_percent',
                            utilization.gpu,
                            tags={'device': str(i)}
                        )
                        self.metrics_collector.record(
                            f'system.gpu.{i}.temperature_celsius',
                            temperature,
                            tags={'device': str(i)}
                        )
                    except ImportError:
                        pass  # nvidia-ml-py not available
                    except Exception:
                        pass  # GPU monitoring failed
            except Exception:
                pass  # CUDA not properly initialized


class ProcessMonitor:
    """Monitor specific process metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.process = psutil.Process()
    
    def collect_metrics(self):
        """Collect current process metrics."""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            self.metrics_collector.record('process.cpu.percent', cpu_percent)
            
            # Memory usage
            memory_info = self.process.memory_info()
            self.metrics_collector.record('process.memory.rss_mb', memory_info.rss / (1024**2))
            self.metrics_collector.record('process.memory.vms_mb', memory_info.vms / (1024**2))
            
            # Thread count
            num_threads = self.process.num_threads()
            self.metrics_collector.record('process.threads.count', num_threads)
            
            # File descriptors (Unix only)
            try:
                num_fds = self.process.num_fds()
                self.metrics_collector.record('process.fds.count', num_fds)
            except AttributeError:
                pass  # Not available on Windows
            
            # IO counters
            try:
                io_counters = self.process.io_counters()
                self.metrics_collector.record('process.io.read_bytes', io_counters.read_bytes)
                self.metrics_collector.record('process.io.write_bytes', io_counters.write_bytes)
            except AttributeError:
                pass  # Not available on all systems
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning("Cannot access process information")


class AlertManager:
    """Manages monitoring alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.alert_states = {}
        self.last_alert_times = {}
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.alert_states[rule.name] = {'triggered': False, 'first_trigger_time': None}
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            del self.alert_states[rule_name]
            if rule_name in self.last_alert_times:
                del self.last_alert_times[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def start_monitoring(self, check_interval: float = 10.0):
        """Start alert monitoring."""
        if self.monitoring:
            logger.warning("Alert monitoring already running")
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._alert_loop, 
            args=(check_interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Alert monitoring stopped")
    
    def _alert_loop(self, check_interval: float):
        """Main alert checking loop."""
        while not self._stop_event.wait(check_interval):
            try:
                self._check_alerts()
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
    
    def _check_alerts(self):
        """Check all alert rules."""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get latest metric value
                latest_sample = self.metrics_collector.get_latest(rule.metric_name)
                if latest_sample is None:
                    continue
                
                # Check if condition is met
                condition_met = self._evaluate_condition(latest_sample.value, rule)
                
                state = self.alert_states[rule_name]
                
                if condition_met:
                    if not state['triggered']:
                        # First time condition is met
                        state['triggered'] = True
                        state['first_trigger_time'] = current_time
                    elif current_time - state['first_trigger_time'] >= rule.duration_seconds:
                        # Condition has been met for required duration
                        self._trigger_alert(rule, latest_sample, current_time)
                else:
                    # Condition not met, reset state
                    state['triggered'] = False
                    state['first_trigger_time'] = None
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, value: Union[float, int, str], rule: AlertRule) -> bool:
        """Evaluate alert condition."""
        if not isinstance(value, (int, float)):
            return False
        
        if rule.condition == 'gt':
            return value > rule.threshold
        elif rule.condition == 'lt':
            return value < rule.threshold
        elif rule.condition == 'eq':
            return value == rule.threshold
        elif rule.condition == 'ne':
            return value != rule.threshold
        
        return False
    
    def _trigger_alert(self, rule: AlertRule, sample: MetricSample, current_time: float):
        """Trigger an alert."""
        # Check cooldown
        if rule.name in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[rule.name]
            if time_since_last < rule.cooldown_seconds:
                return
        
        # Record alert time
        self.last_alert_times[rule.name] = current_time
        
        # Create alert message
        alert_message = f"ALERT: {rule.name} - {rule.metric_name} = {sample.value} {rule.condition} {rule.threshold}"
        
        logger.warning(alert_message)
        
        # Call callback if provided
        if rule.callback:
            try:
                rule.callback(rule, sample, alert_message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Record alert as metric
        self.metrics_collector.record(
            'alerts.triggered',
            1,
            tags={'rule_name': rule.name, 'metric_name': rule.metric_name}
        )


class PerformanceProfiler:
    """Advanced performance profiling with call stack analysis."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for profiling operations."""
        profile_id = f"{operation_name}_{threading.get_ident()}_{time.time()}"
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Record start
        with self._lock:
            self.active_profiles[profile_id] = {
                'operation': operation_name,
                'start_time': start_time,
                'start_memory': start_memory,
                'tags': tags or {}
            }
        
        try:
            yield profile_id
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record metrics
            self.metrics_collector.record(
                f'performance.{operation_name}.duration_seconds',
                duration,
                tags=tags
            )
            
            self.metrics_collector.record(
                f'performance.{operation_name}.memory_delta_mb',
                memory_delta,
                tags=tags
            )
            
            # Clean up
            with self._lock:
                if profile_id in self.active_profiles:
                    del self.active_profiles[profile_id]
            
            logger.debug(f"Profiled {operation_name}: {duration:.3f}s, {memory_delta:.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except Exception:
            return 0.0
    
    def profile_function(self, func_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Decorator for profiling functions."""
        def decorator(func):
            operation_name = func_name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                with self.profile(operation_name, tags):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_active_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active profiles."""
        with self._lock:
            return dict(self.active_profiles)


class TrainingProgressMonitor:
    """Specialized monitor for training progress."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.epoch_start_times = {}
        self.best_metrics = {}
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_times[epoch] = time.time()
        self.metrics_collector.record('training.epoch.started', epoch)
    
    def end_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Mark the end of an epoch with metrics."""
        if epoch in self.epoch_start_times:
            duration = time.time() - self.epoch_start_times[epoch]
            self.metrics_collector.record('training.epoch.duration_seconds', duration, tags={'epoch': str(epoch)})
            del self.epoch_start_times[epoch]
        
        # Record training metrics
        for metric_name, value in metrics.items():
            self.metrics_collector.record(f'training.{metric_name}', value, tags={'epoch': str(epoch)})
            
            # Track best metrics
            if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]['value']:
                self.best_metrics[metric_name] = {'value': value, 'epoch': epoch, 'timestamp': time.time()}
        
        self.metrics_collector.record('training.epoch.completed', epoch)
    
    def record_batch_metrics(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Record metrics for a training batch."""
        tags = {'epoch': str(epoch), 'batch': str(batch)}
        
        for metric_name, value in metrics.items():
            self.metrics_collector.record(f'training.batch.{metric_name}', value, tags=tags)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training progress summary."""
        summary = {
            'best_metrics': self.best_metrics,
            'current_epoch': self.metrics_collector.get_latest('training.epoch.completed'),
            'total_epochs': self.metrics_collector.get_aggregated('training.epoch.completed', 'count'),
            'average_epoch_duration': self.metrics_collector.get_aggregated('training.epoch.duration_seconds', 'mean')
        }
        
        return summary


class MonitoringDashboard:
    """Real-time monitoring dashboard data provider."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def get_dashboard_data(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get dashboard data for the last N minutes."""
        window_seconds = window_minutes * 60
        current_time = time.time()
        since = current_time - window_seconds
        
        dashboard_data = {
            'timestamp': current_time,
            'window_minutes': window_minutes,
            'metrics': {}
        }
        
        # System metrics
        system_metrics = [
            'system.cpu.percent',
            'system.memory.percent',
            'system.memory.used_gb',
            'system.disk.percent'
        ]
        
        for metric_name in system_metrics:
            samples = self.metrics_collector.get_samples(metric_name, since)
            if samples:
                values = [s.value for s in samples if isinstance(s.value, (int, float))]
                if values:
                    dashboard_data['metrics'][metric_name] = {
                        'current': values[-1],
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'samples': len(values)
                    }
        
        # GPU metrics (if available)
        gpu_metrics = [name for name in self.metrics_collector.get_metric_names() if 'gpu' in name]
        for metric_name in gpu_metrics:
            samples = self.metrics_collector.get_samples(metric_name, since)
            if samples:
                values = [s.value for s in samples if isinstance(s.value, (int, float))]
                if values:
                    dashboard_data['metrics'][metric_name] = {
                        'current': values[-1],
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values)
                    }
        
        # Training metrics (if available)
        training_metrics = [name for name in self.metrics_collector.get_metric_names() if 'training' in name]
        for metric_name in training_metrics[:10]:  # Limit to avoid too much data
            latest = self.metrics_collector.get_latest(metric_name)
            if latest:
                dashboard_data['metrics'][metric_name] = {
                    'current': latest.value,
                    'timestamp': latest.timestamp
                }
        
        return dashboard_data
    
    def export_dashboard_json(self, window_minutes: int = 10) -> str:
        """Export dashboard data as JSON."""
        data = self.get_dashboard_data(window_minutes)
        return json.dumps(data, indent=2, default=str)


class MonitoringManager:
    """Central manager for all monitoring components."""
    
    def __init__(
        self, 
        metrics_retention_samples: int = 100000,
        system_monitor_interval: float = 1.0,
        alert_check_interval: float = 10.0
    ):
        self.metrics_collector = MetricsCollector(max_samples=metrics_retention_samples)
        self.system_monitor = SystemMonitor(self.metrics_collector, system_monitor_interval)
        self.process_monitor = ProcessMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.profiler = PerformanceProfiler(self.metrics_collector)
        self.training_monitor = TrainingProgressMonitor(self.metrics_collector)
        self.dashboard = MonitoringDashboard(self.metrics_collector)
        
        self.alert_check_interval = alert_check_interval
        self.running = False
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
    
    def start(self):
        """Start all monitoring components."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        
        # Start system monitoring
        self.system_monitor.start()
        
        # Start alert monitoring
        self.alert_manager.start_monitoring(self.alert_check_interval)
        
        logger.info("Monitoring manager started")
    
    def stop(self):
        """Stop all monitoring components."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.system_monitor.stop()
        self.alert_manager.stop_monitoring()
        
        logger.info("Monitoring manager stopped")
    
    def shutdown(self):
        """Shutdown monitoring (called on exit)."""
        self.stop()
    
    def add_standard_alerts(self):
        """Add standard system alerts."""
        # High CPU usage alert
        self.alert_manager.add_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system.cpu.percent",
            condition="gt",
            threshold=80.0,
            duration_seconds=60.0,
            callback=self._standard_alert_callback
        ))
        
        # High memory usage alert
        self.alert_manager.add_rule(AlertRule(
            name="high_memory_usage",
            metric_name="system.memory.percent",
            condition="gt",
            threshold=90.0,
            duration_seconds=30.0,
            callback=self._standard_alert_callback
        ))
        
        # High GPU memory usage (if available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.alert_manager.add_rule(AlertRule(
                    name=f"high_gpu_{i}_memory",
                    metric_name=f"system.gpu.{i}.memory_allocated_gb",
                    condition="gt",
                    threshold=0.9 * (torch.cuda.get_device_properties(i).total_memory / (1024**3)),
                    duration_seconds=60.0,
                    callback=self._standard_alert_callback
                ))
    
    def _standard_alert_callback(self, rule: AlertRule, sample: MetricSample, message: str):
        """Standard callback for system alerts."""
        logger.warning(f"SYSTEM ALERT: {message}")
        
        # Could add additional actions here like:
        # - Send notification
        # - Trigger memory cleanup
        # - Save current state
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'monitoring_active': self.running,
            'components': {}
        }
        
        # Check system metrics
        cpu_usage = self.metrics_collector.get_latest('system.cpu.percent')
        memory_usage = self.metrics_collector.get_latest('system.memory.percent')
        
        if cpu_usage and cpu_usage.value > 90:
            health['status'] = 'warning'
            health['components']['cpu'] = f'High CPU usage: {cpu_usage.value:.1f}%'
        
        if memory_usage and memory_usage.value > 95:
            health['status'] = 'critical'
            health['components']['memory'] = f'Critical memory usage: {memory_usage.value:.1f}%'
        
        # Check for recent alerts
        recent_alerts = self.metrics_collector.get_samples('alerts.triggered', time.time() - 300)  # Last 5 minutes
        if recent_alerts:
            health['status'] = 'warning' if health['status'] == 'healthy' else health['status']
            health['components']['alerts'] = f'{len(recent_alerts)} alerts in last 5 minutes'
        
        return health


# Global monitoring instance
global_monitoring_manager = None

def get_monitoring_manager() -> MonitoringManager:
    """Get or create global monitoring manager."""
    global global_monitoring_manager
    if global_monitoring_manager is None:
        global_monitoring_manager = MonitoringManager()
    return global_monitoring_manager

def start_monitoring():
    """Start global monitoring."""
    manager = get_monitoring_manager()
    manager.start()
    manager.add_standard_alerts()

def stop_monitoring():
    """Stop global monitoring."""
    manager = get_monitoring_manager()
    manager.stop()

def record_metric(metric_name: str, value: Union[float, int, str], tags: Optional[Dict[str, str]] = None):
    """Convenient function to record a metric."""
    manager = get_monitoring_manager()
    manager.metrics_collector.record(metric_name, value, tags)

@contextmanager
def monitor_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for monitoring operations."""
    manager = get_monitoring_manager()
    with manager.profiler.profile(operation_name, tags) as profile_id:
        yield profile_id
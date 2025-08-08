"""
Comprehensive logging configuration for Spatial-Omics GFM.
Implements structured logging, error tracking, and performance monitoring.
"""

import logging
import logging.handlers
import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings


class SpatialOmicsFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Create structured log entry
        log_entry = {}
        
        if self.include_timestamp:
            log_entry['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        if self.include_level:
            log_entry['level'] = record.levelname
        
        log_entry['logger'] = record.name
        log_entry['message'] = record.getMessage()
        
        # Add module and function info
        if hasattr(record, 'funcName'):
            log_entry['function'] = record.funcName
        if hasattr(record, 'lineno'):
            log_entry['line'] = record.lineno
        if hasattr(record, 'pathname'):
            log_entry['file'] = os.path.basename(record.pathname)
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Logger for performance monitoring and profiling."""
    
    def __init__(self, logger_name: str = 'spatial_omics_gfm.performance'):
        self.logger = logging.getLogger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, **kwargs) -> float:
        """End timing an operation and log duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(
            f"Completed operation: {operation}",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'performance_data': kwargs
            }
        )
        
        return duration
    
    def log_memory_usage(self, operation: str, **kwargs) -> None:
        """Log memory usage information."""
        try:
            import psutil
            import torch
            
            memory_info = {
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            if torch.cuda.is_available():
                memory_info.update({
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'gpu_memory_cached_gb': torch.cuda.memory_reserved() / (1024**3),
                    'gpu_memory_percent': (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
                })
            
            memory_info.update(kwargs)
            
            self.logger.info(
                f"Memory usage for {operation}",
                extra={
                    'operation': operation,
                    'memory_info': memory_info
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")


class ErrorTracker:
    """Centralized error tracking and reporting."""
    
    def __init__(self, logger_name: str = 'spatial_omics_gfm.errors'):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
    
    def log_error(
        self,
        error: Exception,
        context: str,
        operation: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log error with context and tracking."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error frequency
        error_key = f"{error_type}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error with structured information
        self.logger.error(
            f"Error in {context}: {error_message}",
            extra={
                'error_type': error_type,
                'error_message': error_message,
                'context': context,
                'operation': operation,
                'error_count': self.error_counts[error_key],
                'additional_info': kwargs
            },
            exc_info=True
        )
    
    def log_warning(
        self,
        message: str,
        context: str,
        **kwargs
    ) -> None:
        """Log warning with context."""
        self.logger.warning(
            f"Warning in {context}: {message}",
            extra={
                'context': context,
                'warning_info': kwargs
            }
        )
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return self.error_counts.copy()


class SpatialOmicsLogger:
    """Main logger configuration for Spatial-Omics GFM."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5
    ):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level
            log_dir: Directory for log files
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._configure_logging()
        
        # Initialize specialized loggers
        self.performance_logger = PerformanceLogger()
        self.error_tracker = ErrorTracker()
        
        # Log initialization
        main_logger = logging.getLogger('spatial_omics_gfm')
        main_logger.info("Logging system initialized", extra={
            'log_level': log_level,
            'log_dir': str(self.log_dir),
            'file_logging': enable_file_logging,
            'console_logging': enable_console_logging
        })
    
    def _configure_logging(self) -> None:
        """Configure logging handlers and formatters."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure main logger
        main_logger = logging.getLogger('spatial_omics_gfm')
        main_logger.setLevel(self.log_level)
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            # Use simpler format for console
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            main_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            # Main log file
            main_log_file = self.log_dir / "spatial_omics_gfm.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            main_handler.setLevel(self.log_level)
            main_handler.setFormatter(SpatialOmicsFormatter())
            main_logger.addHandler(main_handler)
            
            # Error log file
            error_log_file = self.log_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(SpatialOmicsFormatter())
            main_logger.addHandler(error_handler)
            
            # Performance log file
            perf_log_file = self.log_dir / "performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(SpatialOmicsFormatter())
            
            # Configure performance logger
            perf_logger = logging.getLogger('spatial_omics_gfm.performance')
            perf_logger.setLevel(logging.INFO)
            perf_logger.addHandler(perf_handler)
            
            # Configure error logger
            error_logger = logging.getLogger('spatial_omics_gfm.errors')
            error_logger.setLevel(logging.WARNING)
            error_logger.addHandler(error_handler)
        
        # Suppress some noisy loggers
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        # Configure warnings to be logged
        logging.captureWarnings(True)
        warnings.filterwarnings('default')
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(f'spatial_omics_gfm.{name}')
    
    def log_system_info(self) -> None:
        """Log system information for debugging."""
        logger = self.get_logger('system')
        
        try:
            import platform
            import torch
            import numpy as np
            import pandas as pd
            
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'cuda_available': torch.cuda.is_available(),
            }
            
            if torch.cuda.is_available():
                system_info.update({
                    'cuda_version': torch.version.cuda,
                    'cudnn_version': torch.backends.cudnn.version(),
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                })
            
            logger.info("System information", extra={'system_info': system_info})
            
        except Exception as e:
            logger.warning(f"Failed to log system information: {e}")
    
    def close(self) -> None:
        """Close all logging handlers."""
        for logger_name in ['spatial_omics_gfm', 'spatial_omics_gfm.performance', 'spatial_omics_gfm.errors']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    **kwargs
) -> SpatialOmicsLogger:
    """
    Setup logging configuration for Spatial-Omics GFM.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        **kwargs: Additional arguments for SpatialOmicsLogger
        
    Returns:
        Configured logger instance
    """
    return SpatialOmicsLogger(log_level=log_level, log_dir=log_dir, **kwargs)


# Context managers for logging operations
class LoggedOperation:
    """Context manager for logging operations with automatic timing and error handling."""
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        log_start: bool = True,
        log_end: bool = True,
        log_errors: bool = True,
        **context_data
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger('spatial_omics_gfm')
        self.log_start = log_start
        self.log_end = log_end
        self.log_errors = log_errors
        self.context_data = context_data
        self.start_time = None
        self.performance_logger = PerformanceLogger()
        self.error_tracker = ErrorTracker()
    
    def __enter__(self):
        if self.log_start:
            self.logger.info(f"Starting operation: {self.operation_name}", extra=self.context_data)
        
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is not None:
            if self.log_errors:
                self.error_tracker.log_error(
                    exc_val, 
                    self.operation_name,
                    operation=self.operation_name,
                    duration_seconds=duration,
                    **self.context_data
                )
            return False  # Re-raise exception
        
        if self.log_end:
            self.logger.info(
                f"Completed operation: {self.operation_name}",
                extra={
                    'operation': self.operation_name,
                    'duration_seconds': duration,
                    **self.context_data
                }
            )
        
        return False


# Decorators for automatic logging
def log_function_call(
    logger_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_performance: bool = True
):
    """
    Decorator to automatically log function calls.
    
    Args:
        logger_name: Name of logger to use
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_performance: Whether to log performance metrics
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or f'spatial_omics_gfm.{func.__module__}')
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Prepare log context
            log_context = {'function': func.__name__, 'module': func.__module__}
            if log_args:
                log_context['args'] = str(args)[:200]  # Truncate long args
                log_context['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            with LoggedOperation(
                operation_name,
                logger,
                log_start=True,
                log_end=log_performance,
                **log_context
            ):
                result = func(*args, **kwargs)
                
                if log_result:
                    logger.debug(f"Function {func.__name__} returned", extra={
                        'result_type': type(result).__name__,
                        'result_summary': str(result)[:100]
                    })
                
                return result
        
        return wrapper
    return decorator


# Global logger instance (can be initialized once and used throughout)
_global_logger_instance = None

def get_global_logger() -> Optional[SpatialOmicsLogger]:
    """Get the global logger instance."""
    return _global_logger_instance

def init_global_logger(**kwargs) -> SpatialOmicsLogger:
    """Initialize the global logger instance."""
    global _global_logger_instance
    _global_logger_instance = setup_logging(**kwargs)
    return _global_logger_instance
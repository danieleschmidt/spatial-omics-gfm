"""
Utility functions and helpers for Spatial-Omics GFM.

This module contains common utility functions used across
the package for data processing, validation, and analysis.
"""

from .validators import SpatialDataValidator, ModelInputValidator, validate_file_format, validate_model_config
from .enhanced_validators import (
    RobustValidator, ValidationConfig, ValidationResult, ValidationException,
    DataIntegrityValidator, AdversarialInputDetector, FilePathSanitizer,
    validate_model_config_robust
)
from .security import (
    SecurityConfig, InputSanitizer, SecureFileHandler, ModelSecurity,
    sanitize_user_input, secure_file_operation, create_security_context
)
from .config_manager import (
    ConfigManager, ExperimentConfig, ModelConfig, TrainingConfig,
    DataConfig, SystemConfig, SecurityConfig as SecurityConfigClass,
    create_default_config, load_config_from_file, validate_config_file
)
from .helpers import compute_spatial_distance, normalize_coordinates
from .metrics import SpatialMetrics, evaluate_model_performance
from .logging_config import setup_logging, SpatialOmicsLogger, LoggedOperation
from .optimization import ModelOptimizer, BatchProcessor, PerformanceProfiler, optimize_for_production
from .memory_management import (
    MemoryMonitor, DataChunker, SwapManager, memory_managed_operation,
    MemoryEfficientDataLoader, get_memory_recommendations
)

__all__ = [
    # Original validators
    "SpatialDataValidator",
    "ModelInputValidator",
    "validate_file_format", 
    "validate_model_config",
    
    # Enhanced validation and robustness
    "RobustValidator",
    "ValidationConfig",
    "ValidationResult",
    "ValidationException",
    "DataIntegrityValidator",
    "AdversarialInputDetector",
    "FilePathSanitizer",
    "validate_model_config_robust",
    
    # Security utilities
    "SecurityConfig",
    "InputSanitizer",
    "SecureFileHandler",
    "ModelSecurity",
    "sanitize_user_input",
    "secure_file_operation",
    "create_security_context",
    
    # Configuration management
    "ConfigManager",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "SystemConfig",
    "SecurityConfigClass",
    "create_default_config",
    "load_config_from_file",
    "validate_config_file",
    
    # Original utilities
    "compute_spatial_distance",
    "normalize_coordinates", 
    "SpatialMetrics",
    "evaluate_model_performance",
    "setup_logging",
    "SpatialOmicsLogger", 
    "LoggedOperation",
    "ModelOptimizer",
    "BatchProcessor",
    "PerformanceProfiler",
    "optimize_for_production",
    "MemoryMonitor",
    "DataChunker",
    "SwapManager",
    "memory_managed_operation",
    "MemoryEfficientDataLoader",
    "get_memory_recommendations",
]
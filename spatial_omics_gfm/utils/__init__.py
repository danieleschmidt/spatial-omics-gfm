"""
Utility functions and helpers for Spatial-Omics GFM.

This module contains common utility functions used across
the package for data processing, validation, and analysis.
"""

import warnings

# Core utilities (always available)
from .helpers import compute_spatial_distance, normalize_coordinates

# Conditional imports with graceful fallback for robustness
try:
    from .validators import SpatialDataValidator, ModelInputValidator, validate_file_format, validate_model_config
    TORCH_VALIDATORS_AVAILABLE = True
except ImportError:
    TORCH_VALIDATORS_AVAILABLE = False
    SpatialDataValidator = ModelInputValidator = None
    validate_file_format = validate_model_config = None
    warnings.warn("PyTorch-based validators not available - install torch for full functionality", ImportWarning)

try:
    from .enhanced_validators import (
        RobustValidator, ValidationConfig, ValidationResult, ValidationException,
        DataIntegrityValidator, AdversarialInputDetector, FilePathSanitizer,
        validate_model_config_robust
    )
    ENHANCED_VALIDATORS_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATORS_AVAILABLE = False
    RobustValidator = ValidationConfig = ValidationResult = ValidationException = None
    DataIntegrityValidator = AdversarialInputDetector = FilePathSanitizer = None
    validate_model_config_robust = None
    warnings.warn("Enhanced validators not available - some features limited", ImportWarning)

try:
    from .security import (
        SecurityConfig, InputSanitizer, SecureFileHandler, ModelSecurity,
        sanitize_user_input, secure_file_operation, create_security_context
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    SecurityConfig = InputSanitizer = SecureFileHandler = ModelSecurity = None
    sanitize_user_input = secure_file_operation = create_security_context = None
    warnings.warn("Security features not available - some dependencies missing", ImportWarning)

try:
    from .config_manager import (
        ConfigManager, ExperimentConfig, ModelConfig, TrainingConfig,
        DataConfig, SystemConfig, SecurityConfig as SecurityConfigClass,
        create_default_config, load_config_from_file, validate_config_file
    )
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    ConfigManager = ExperimentConfig = ModelConfig = TrainingConfig = None
    DataConfig = SystemConfig = SecurityConfigClass = None
    create_default_config = load_config_from_file = validate_config_file = None
    warnings.warn("Configuration manager not available", ImportWarning)

try:
    from .metrics import SpatialMetrics, evaluate_model_performance
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    SpatialMetrics = evaluate_model_performance = None
    warnings.warn("Advanced metrics not available", ImportWarning)

try:
    from .logging_config import setup_logging, SpatialOmicsLogger, LoggedOperation
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    setup_logging = SpatialOmicsLogger = LoggedOperation = None
    warnings.warn("Advanced logging not available", ImportWarning)

try:
    from .optimization import ModelOptimizer, BatchProcessor, PerformanceProfiler, optimize_for_production
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    ModelOptimizer = BatchProcessor = PerformanceProfiler = optimize_for_production = None
    warnings.warn("Optimization utilities not available", ImportWarning)

try:
    from .memory_management import (
        MemoryMonitor, DataChunker, SwapManager, memory_managed_operation,
        MemoryEfficientDataLoader, get_memory_recommendations
    )
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False
    MemoryMonitor = DataChunker = SwapManager = memory_managed_operation = None
    MemoryEfficientDataLoader = get_memory_recommendations = None
    warnings.warn("Memory management utilities not available", ImportWarning)

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
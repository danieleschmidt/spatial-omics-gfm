"""
Enhanced validation utilities for spatial transcriptomics data with comprehensive robustness features.
Implements advanced error handling, recovery strategies, and security validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
import warnings
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pathlib import Path
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation processes."""
    strict_mode: bool = False
    auto_fix: bool = True
    fail_fast: bool = False
    max_warnings: int = 100
    validation_timeout: int = 300  # seconds
    custom_validators: List[Callable] = field(default_factory=list)
    excluded_checks: Set[str] = field(default_factory=set)
    parallel_validation: bool = True
    max_workers: int = 4


@dataclass
class ValidationResult:
    """Comprehensive validation result with enhanced metadata."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str, error_code: str, context: Optional[Dict] = None):
        """Add structured error with metadata."""
        self.errors.append({
            'message': message,
            'error_code': error_code,
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'severity': 'error'
        })
        self.is_valid = False
    
    def add_warning(self, message: str, warning_code: str, context: Optional[Dict] = None):
        """Add structured warning with metadata."""
        self.warnings.append({
            'message': message,
            'warning_code': warning_code,
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'severity': 'warning'
        })
    
    def add_recommendation(self, message: str, action: str, priority: str = 'medium'):
        """Add structured recommendation."""
        self.recommendations.append({
            'message': message,
            'action': action,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_fix(self, description: str, fix_type: str, affected_data: Optional[str] = None):
        """Add applied fix with metadata."""
        self.fixes_applied.append({
            'description': description,
            'fix_type': fix_type,
            'affected_data': affected_data,
            'timestamp': datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'quality_metrics': self.quality_metrics,
            'fixes_applied': self.fixes_applied,
            'validation_metadata': self.validation_metadata,
            'performance_stats': self.performance_stats
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ValidationException(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, error_code: str, validation_result: Optional[ValidationResult] = None):
        self.error_code = error_code
        self.validation_result = validation_result
        super().__init__(message)


class RecoveryStrategy:
    """Base class for recovery strategies when validation fails."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_recover(self, error: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError
    
    def apply_recovery(self, data: Any, error: Dict[str, Any]) -> Tuple[Any, bool]:
        """Apply recovery strategy and return (recovered_data, success)."""
        raise NotImplementedError


class DataIntegrityValidator:
    """Advanced validator for data integrity checks."""
    
    def __init__(self):
        self.checksums = {}
        self.data_signatures = {}
    
    def compute_data_checksum(self, data: Any) -> str:
        """Compute checksum for data integrity verification."""
        if isinstance(data, AnnData):
            # Create signature from key properties
            signature_data = {
                'shape': data.shape,
                'obs_columns': list(data.obs.columns),
                'var_columns': list(data.var.columns),
                'obsm_keys': list(data.obsm.keys()),
                'varm_keys': list(data.varm.keys())
            }
            if hasattr(data.X, 'toarray'):
                x_hash = hashlib.md5(data.X.data.tobytes()).hexdigest()[:16]
            else:
                x_hash = hashlib.md5(data.X.tobytes()).hexdigest()[:16]
            signature_data['x_hash'] = x_hash
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            signature_data = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'hash': hashlib.md5(data.tobytes()).hexdigest()[:16]
            }
        else:
            signature_data = {
                'type': type(data).__name__, 
                'str_hash': hashlib.md5(str(data).encode()).hexdigest()[:16]
            }
        
        return hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    
    def verify_data_integrity(self, data: Any, expected_checksum: str) -> bool:
        """Verify data integrity against expected checksum."""
        current_checksum = self.compute_data_checksum(data)
        return current_checksum == expected_checksum
    
    def detect_data_corruption(self, data: AnnData, result: ValidationResult) -> None:
        """Detect potential data corruption patterns."""
        if hasattr(data.X, 'toarray'):
            X = data.X.toarray()
        else:
            X = data.X
        
        # Check for suspicious patterns
        # 1. Uniform values (potential overwrite corruption)
        unique_ratio = len(np.unique(X)) / X.size
        if unique_ratio < 0.01 and X.size > 1000:  # Very low diversity
            result.add_warning(
                f"Suspiciously low data diversity: {unique_ratio:.4f} unique ratio",
                "DATA_CORRUPTION_UNIFORM",
                {'unique_ratio': unique_ratio, 'total_elements': X.size}
            )
        
        # 2. Repeated patterns (potential memory corruption)
        if X.shape[0] > 10:
            first_row = X[0]
            identical_rows = np.sum([np.array_equal(X[i], first_row) for i in range(min(10, X.shape[0]))])
            if identical_rows > 5:
                result.add_warning(
                    f"Multiple identical rows detected: {identical_rows}/10",
                    "DATA_CORRUPTION_REPEATED",
                    {'identical_rows': identical_rows}
                )
        
        # 3. Extreme outliers (potential bit corruption)
        if X.size > 0:
            percentiles = np.percentile(X[X > 0], [95, 99, 99.9]) if np.any(X > 0) else [0, 0, 0]
            if len(percentiles) >= 3 and percentiles[2] > percentiles[1] * 100:
                result.add_warning(
                    f"Extreme outliers detected: 99.9th percentile = {percentiles[2]:.2f}, 99th = {percentiles[1]:.2f}",
                    "DATA_CORRUPTION_OUTLIERS",
                    {'percentiles': {'95th': percentiles[0], '99th': percentiles[1], '99.9th': percentiles[2]}}
                )


class AdversarialInputDetector:
    """Detector for adversarial inputs and potential attacks."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'[<>"\'\\/]',     # Potential injection characters
            r'\$\{.*\}',       # Variable expansion patterns
            r'\[.*\]',         # Bracket patterns
            r'\.\.[\\/]',      # Directory traversal
        ]
        self.max_string_length = 1000
        self.max_array_size = 10**8  # 100M elements
    
    def detect_adversarial_patterns(self, data: Any, result: ValidationResult) -> None:
        """Detect patterns that might indicate adversarial inputs."""
        
        # Check string-based metadata for injection patterns
        self._check_string_fields(data, result)
        
        # Check for suspiciously large arrays
        self._check_array_sizes(data, result)
        
        # Check for unusual data distributions
        self._check_data_distributions(data, result)
    
    def _check_string_fields(self, data: AnnData, result: ValidationResult) -> None:
        """Check string fields for suspicious patterns."""
        string_fields = []
        
        # Check observation names
        if data.obs.index.dtype == 'object':
            string_fields.extend(data.obs.index.astype(str))
        
        # Check variable names
        if data.var.index.dtype == 'object':
            string_fields.extend(data.var.index.astype(str))
        
        # Check metadata string columns
        for col in data.obs.select_dtypes(include=['object']).columns:
            string_fields.extend(data.obs[col].astype(str))
        
        for col in data.var.select_dtypes(include=['object']).columns:
            string_fields.extend(data.var[col].astype(str))
        
        # Check each string field
        for field in string_fields[:1000]:  # Limit checks to avoid performance issues
            if len(field) > self.max_string_length:
                result.add_warning(
                    f"Suspiciously long string field: {len(field)} characters",
                    "ADVERSARIAL_LONG_STRING",
                    {'field_length': len(field), 'field_preview': field[:100]}
                )
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, field):
                    result.add_warning(
                        f"Suspicious pattern detected in string field: {pattern}",
                        "ADVERSARIAL_PATTERN",
                        {'pattern': pattern, 'field_preview': field[:100]}
                    )
                    break
    
    def _check_array_sizes(self, data: AnnData, result: ValidationResult) -> None:
        """Check for suspiciously large arrays."""
        total_elements = data.X.size
        
        if total_elements > self.max_array_size:
            result.add_warning(
                f"Suspiciously large data array: {total_elements} elements",
                "ADVERSARIAL_LARGE_ARRAY",
                {'array_size': total_elements, 'shape': data.shape}
            )
    
    def _check_data_distributions(self, data: AnnData, result: ValidationResult) -> None:
        """Check for unusual data distributions that might indicate adversarial inputs."""
        if hasattr(data.X, 'toarray'):
            X = data.X.toarray()
        else:
            X = data.X
        
        # Check for all-zero or all-constant data
        if np.all(X == 0):
            result.add_warning(
                "All-zero data matrix detected",
                "ADVERSARIAL_ALL_ZERO",
                {'matrix_shape': X.shape}
            )
        
        # Check for suspiciously regular patterns
        if X.shape[0] > 1 and X.shape[1] > 1:
            # Check if all rows are identical
            if np.all([np.array_equal(X[0], X[i]) for i in range(min(10, X.shape[0]))]):
                result.add_warning(
                    "Identical rows detected in expression matrix",
                    "ADVERSARIAL_IDENTICAL_ROWS",
                    {'checked_rows': min(10, X.shape[0])}
                )
            
            # Check if all columns are identical
            if np.all([np.array_equal(X[:, 0], X[:, i]) for i in range(min(10, X.shape[1]))]):
                result.add_warning(
                    "Identical columns detected in expression matrix",
                    "ADVERSARIAL_IDENTICAL_COLS",
                    {'checked_cols': min(10, X.shape[1])}
                )


class FilePathSanitizer:
    """Sanitizer for file paths and user inputs to prevent security vulnerabilities."""
    
    def __init__(self, allowed_extensions: Optional[Set[str]] = None, base_directories: Optional[Set[str]] = None):
        self.allowed_extensions = allowed_extensions or {'.h5ad', '.h5', '.csv', '.tsv', '.zarr', '.json', '.yaml'}
        self.base_directories = set(str(Path(d).resolve()) for d in base_directories or [Path.cwd()])
        self.dangerous_patterns = [
            r'\.\.[\\/]',      # Directory traversal
            r'^[\\/]',         # Absolute paths (sometimes dangerous)
            r'[<>:"|?*]',      # Windows illegal characters
            r'[\x00-\x1f]',    # Control characters
        ]
    
    def sanitize_file_path(self, file_path: Union[str, Path]) -> Path:
        """Sanitize and validate file path."""
        path = Path(file_path)
        path_str = str(path)
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, path_str):
                raise ValidationException(
                    f"Dangerous pattern detected in file path: {pattern}",
                    "SECURITY_DANGEROUS_PATH"
                )
        
        # Resolve path and check if it's within allowed directories
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError):
            raise ValidationException(
                f"Cannot resolve file path: {path}",
                "SECURITY_UNRESOLVABLE_PATH"
            )
        
        # Check if path is within allowed base directories
        if self.base_directories:
            path_is_allowed = any(
                str(resolved_path).startswith(base_dir) 
                for base_dir in self.base_directories
            )
            if not path_is_allowed:
                raise ValidationException(
                    f"File path outside allowed directories: {resolved_path}",
                    "SECURITY_PATH_OUTSIDE_BASE"
                )
        
        # Check file extension
        if resolved_path.suffix.lower() not in self.allowed_extensions:
            raise ValidationException(
                f"File extension not allowed: {resolved_path.suffix}",
                "SECURITY_INVALID_EXTENSION"
            )
        
        return resolved_path
    
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent injection attacks."""
        # Remove or escape dangerous characters
        sanitized = re.sub(r'[<>"\'\\/\x00-\x1f]', '', input_str)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"String input truncated to {max_length} characters")
        
        return sanitized


class RobustValidator:
    """Main robust validator with comprehensive error handling and recovery."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.data_integrity_validator = DataIntegrityValidator()
        self.adversarial_detector = AdversarialInputDetector()
        self.path_sanitizer = FilePathSanitizer()
        self.recovery_strategies = []
        self._lock = threading.Lock()
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a recovery strategy."""
        with self._lock:
            self.recovery_strategies.append(strategy)
            logger.info(f"Registered recovery strategy: {strategy.name}")
    
    @contextmanager
    def validation_timeout(self, timeout: int):
        """Context manager for validation timeout."""
        def timeout_handler():
            raise TimeoutError(f"Validation timeout after {timeout} seconds")
        
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    
    def validate_with_timeout(self, validation_func: Callable, *args, **kwargs) -> Any:
        """Execute validation function with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(validation_func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.validation_timeout)
            except TimeoutError:
                raise ValidationException(
                    f"Validation timeout after {self.config.validation_timeout} seconds",
                    "VALIDATION_TIMEOUT"
                )
    
    def validate_spatial_data_robust(
        self, 
        adata: AnnData, 
        file_path: Optional[Union[str, Path]] = None,
        expected_checksum: Optional[str] = None
    ) -> ValidationResult:
        """Comprehensive robust validation of spatial data."""
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Initialize validation metadata
            result.validation_metadata = {
                'validator_version': '2.0.0',
                'validation_start_time': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'data_shape': adata.shape
            }
            
            # 1. File path validation (if provided)
            if file_path is not None:
                try:
                    sanitized_path = self.path_sanitizer.sanitize_file_path(file_path)
                    result.validation_metadata['sanitized_file_path'] = str(sanitized_path)
                except ValidationException as e:
                    result.add_error(str(e), e.error_code)
                    if self.config.fail_fast:
                        return result
            
            # 2. Data integrity validation
            if expected_checksum:
                try:
                    integrity_valid = self.data_integrity_validator.verify_data_integrity(
                        adata, expected_checksum
                    )
                    if not integrity_valid:
                        result.add_error(
                            "Data integrity check failed",
                            "DATA_INTEGRITY_MISMATCH"
                        )
                        if self.config.fail_fast:
                            return result
                except Exception as e:
                    result.add_warning(
                        f"Data integrity check failed: {str(e)}",
                        "DATA_INTEGRITY_ERROR"
                    )
            
            # 3. Adversarial input detection
            try:
                self.adversarial_detector.detect_adversarial_patterns(adata, result)
            except Exception as e:
                result.add_warning(
                    f"Adversarial detection failed: {str(e)}",
                    "ADVERSARIAL_DETECTION_ERROR"
                )
            
            # 4. Data corruption detection
            try:
                self.data_integrity_validator.detect_data_corruption(adata, result)
            except Exception as e:
                result.add_warning(
                    f"Corruption detection failed: {str(e)}",
                    "CORRUPTION_DETECTION_ERROR"
                )
            
            # 5. Standard validation checks (using existing validators)
            from .validators import SpatialDataValidator
            try:
                standard_validator = SpatialDataValidator()
                standard_result = standard_validator.validate_adata(adata, fix_issues=self.config.auto_fix)
                
                # Merge results
                for error in standard_result.get('errors', []):
                    result.add_error(error, 'STANDARD_VALIDATION_ERROR')
                
                for warning in standard_result.get('warnings', []):
                    result.add_warning(warning, 'STANDARD_VALIDATION_WARNING')
                
                for fix in standard_result.get('fixes_applied', []):
                    result.add_fix(fix, 'STANDARD_FIX')
                
                result.quality_metrics.update(standard_result.get('quality_metrics', {}))
                
            except Exception as e:
                result.add_error(
                    f"Standard validation failed: {str(e)}",
                    "STANDARD_VALIDATION_FAILURE"
                )
            
            # 6. Custom validators
            for custom_validator in self.config.custom_validators:
                try:
                    custom_result = custom_validator(adata)
                    if hasattr(custom_result, 'errors'):
                        for error in custom_result.errors:
                            result.add_error(error.get('message', str(error)), 'CUSTOM_VALIDATION_ERROR')
                except Exception as e:
                    result.add_warning(
                        f"Custom validator failed: {str(e)}",
                        "CUSTOM_VALIDATION_ERROR"
                    )
            
            # 7. Recovery attempts for errors
            if result.errors and self.config.auto_fix:
                self._attempt_error_recovery(adata, result)
            
            # 8. Final validation status
            result.is_valid = not result.errors
            
            # Performance statistics
            end_time = time.time()
            result.performance_stats = {
                'validation_duration_seconds': end_time - start_time,
                'errors_count': len(result.errors),
                'warnings_count': len(result.warnings),
                'fixes_count': len(result.fixes_applied),
                'data_size_mb': adata.X.nbytes / (1024**2) if hasattr(adata.X, 'nbytes') else 0
            }
            
            result.validation_metadata['validation_end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            result.add_error(
                f"Validation framework error: {str(e)}",
                "FRAMEWORK_ERROR"
            )
            logger.exception("Validation framework error")
        
        return result
    
    def _attempt_error_recovery(self, data: AnnData, result: ValidationResult) -> None:
        """Attempt to recover from validation errors."""
        recovered_errors = []
        
        for error in result.errors[:]:  # Copy list as we may modify it
            for strategy in self.recovery_strategies:
                try:
                    if strategy.can_recover(error):
                        recovered_data, success = strategy.apply_recovery(data, error)
                        if success:
                            result.add_fix(
                                f"Recovered from error using {strategy.name}",
                                "ERROR_RECOVERY",
                                error.get('error_code')
                            )
                            recovered_errors.append(error)
                            break
                except Exception as e:
                    logger.warning(f"Recovery strategy {strategy.name} failed: {e}")
        
        # Remove recovered errors
        for error in recovered_errors:
            if error in result.errors:
                result.errors.remove(error)


def timeout_decorator(timeout_seconds: int):
    """Decorator to add timeout to validation functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    raise ValidationException(
                        f"Function {func.__name__} timeout after {timeout_seconds} seconds",
                        "FUNCTION_TIMEOUT"
                    )
        return wrapper
    return decorator


def validate_model_config_robust(config: Dict[str, Any], schema: Optional[Dict] = None) -> ValidationResult:
    """Robust validation for model configurations."""
    result = ValidationResult(is_valid=True)
    
    try:
        # Required fields validation
        required_fields = ['num_genes', 'hidden_dim']
        for field in required_fields:
            if field not in config:
                result.add_error(f"Missing required field: {field}", "MISSING_REQUIRED_FIELD")
        
        # Type validation
        type_checks = {
            'num_genes': int,
            'hidden_dim': int,
            'num_layers': int,
            'num_heads': int,
            'dropout': float
        }
        
        for field, expected_type in type_checks.items():
            if field in config and not isinstance(config[field], expected_type):
                result.add_error(
                    f"Field {field} must be {expected_type.__name__}, got {type(config[field]).__name__}",
                    "INVALID_TYPE"
                )
        
        # Range validation
        if 'hidden_dim' in config:
            if config['hidden_dim'] <= 0:
                result.add_error("hidden_dim must be positive", "INVALID_RANGE")
            elif config['hidden_dim'] % 64 != 0:
                result.add_recommendation(
                    "hidden_dim should be divisible by 64 for optimal performance",
                    "optimize_hidden_dim",
                    "medium"
                )
        
        if 'dropout' in config:
            if not (0 <= config['dropout'] <= 1):
                result.add_error("dropout must be between 0 and 1", "INVALID_RANGE")
        
        # Schema validation if provided
        if schema:
            # Add schema validation logic here
            pass
        
        result.is_valid = not result.errors
        
    except Exception as e:
        result.add_error(f"Config validation error: {str(e)}", "CONFIG_VALIDATION_ERROR")
    
    return result
"""
Input validation and sanitization utilities.

Provides validation for user inputs, file paths, parameters,
and system resource requirements.
"""

import os
import re
import psutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Exception for input validation failures."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        
        # Safe file extensions for different data types
        self.safe_extensions = {
            'expression': {'.h5', '.h5ad', '.csv', '.tsv', '.txt', '.gz', '.loom', '.xlsx'},
            'spatial': {'.csv', '.tsv', '.txt', '.json'},
            'config': {'.yaml', '.yml', '.json', '.toml', '.ini'},
            'image': {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.pdf', '.svg'},
            'archive': {'.zip', '.tar', '.tar.gz', '.tar.bz2'}
        }
        
        # Dangerous file patterns to avoid
        self.dangerous_patterns = [
            r'\.\./',  # Directory traversal
            r'~/',     # Home directory access
            r'/etc/',  # System directories
            r'/bin/',
            r'/usr/',
            r'/var/',
            r'/tmp/',
            r'\\\\',   # UNC paths
            r'\$\{',   # Variable substitution
            r'`',      # Command substitution
        ]
    
    def sanitize_file_path(
        self, 
        file_path: Union[str, Path],
        allowed_extensions: Optional[List[str]] = None,
        max_path_length: int = 1000,
        allow_create: bool = False
    ) -> Path:
        """
        Sanitize and validate file paths.
        
        Args:
            file_path: Input file path
            allowed_extensions: List of allowed file extensions
            max_path_length: Maximum allowed path length
            allow_create: Whether to allow non-existent files
            
        Returns:
            Validated Path object
            
        Raises:
            InputValidationError: If path is invalid or unsafe
        """
        try:
            # Convert to Path object
            if isinstance(file_path, str):
                path = Path(file_path)
            else:
                path = file_path
            
            # Basic validation
            path_str = str(path)
            
            # Check path length
            if len(path_str) > max_path_length:
                raise InputValidationError(f"Path too long: {len(path_str)} > {max_path_length}")
            
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    raise InputValidationError(f"Dangerous path pattern detected: {pattern}")
            
            # Resolve and normalize path
            try:
                resolved_path = path.resolve()
            except (OSError, RuntimeError) as e:
                raise InputValidationError(f"Cannot resolve path: {e}")
            
            # Check if path exists (if required)
            if not allow_create and not resolved_path.exists():
                raise InputValidationError(f"File does not exist: {resolved_path}")
            
            # Check file extension if specified
            if allowed_extensions:
                file_ext = resolved_path.suffix.lower()
                if file_ext not in [ext.lower() for ext in allowed_extensions]:
                    raise InputValidationError(f"Invalid file extension: {file_ext}. Allowed: {allowed_extensions}")
            
            # Check permissions
            if resolved_path.exists():
                if not os.access(resolved_path, os.R_OK):
                    raise InputValidationError(f"No read permission for: {resolved_path}")
                
                if resolved_path.is_dir():
                    if not any(allowed_extensions) or 'directory' not in str(allowed_extensions):
                        raise InputValidationError(f"Expected file, got directory: {resolved_path}")
            
            # Check parent directory permissions (for file creation)
            elif allow_create:
                parent_dir = resolved_path.parent
                if not parent_dir.exists():
                    raise InputValidationError(f"Parent directory does not exist: {parent_dir}")
                
                if not os.access(parent_dir, os.W_OK):
                    raise InputValidationError(f"No write permission for parent directory: {parent_dir}")
            
            logger.debug(f"Path validated: {resolved_path}")
            return resolved_path
            
        except Exception as e:
            if isinstance(e, InputValidationError):
                raise
            raise InputValidationError(f"Path validation failed: {e}")
    
    def validate_parameters(
        self,
        params: Dict[str, Any],
        param_schema: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate parameters against a schema.
        
        Args:
            params: Dictionary of parameters to validate
            param_schema: Schema defining expected parameters and constraints
            
        Returns:
            Validated and potentially modified parameters
            
        Raises:
            InputValidationError: If validation fails
        """
        validated_params = {}
        
        for param_name, param_info in param_schema.items():
            value = params.get(param_name)
            
            # Check required parameters
            if param_info.get('required', False) and value is None:
                raise InputValidationError(f"Required parameter missing: {param_name}")
            
            # Use default if not provided
            if value is None:
                if 'default' in param_info:
                    value = param_info['default']
                else:
                    continue
            
            # Type validation
            expected_type = param_info.get('type')
            if expected_type and not isinstance(value, expected_type):
                try:
                    # Attempt type conversion
                    if expected_type in (int, float, str, bool):
                        value = expected_type(value)
                    else:
                        raise InputValidationError(f"Cannot convert {param_name} to {expected_type}")
                except (ValueError, TypeError):
                    raise InputValidationError(f"Invalid type for {param_name}: expected {expected_type}, got {type(value)}")
            
            # Range validation for numeric types
            if isinstance(value, (int, float)):
                min_val = param_info.get('min')
                max_val = param_info.get('max')
                
                if min_val is not None and value < min_val:
                    raise InputValidationError(f"{param_name} ({value}) below minimum ({min_val})")
                
                if max_val is not None and value > max_val:
                    raise InputValidationError(f"{param_name} ({value}) above maximum ({max_val})")
            
            # Choice validation
            choices = param_info.get('choices')
            if choices and value not in choices:
                raise InputValidationError(f"{param_name} ({value}) not in allowed choices: {choices}")
            
            # String validation
            if isinstance(value, str):
                min_len = param_info.get('min_length', 0)
                max_len = param_info.get('max_length', float('inf'))
                
                if len(value) < min_len:
                    raise InputValidationError(f"{param_name} too short: {len(value)} < {min_len}")
                
                if len(value) > max_len:
                    raise InputValidationError(f"{param_name} too long: {len(value)} > {max_len}")
                
                # Pattern validation
                pattern = param_info.get('pattern')
                if pattern and not re.match(pattern, value):
                    raise InputValidationError(f"{param_name} does not match pattern: {pattern}")
            
            # List validation
            if isinstance(value, list):
                min_items = param_info.get('min_items', 0)
                max_items = param_info.get('max_items', float('inf'))
                
                if len(value) < min_items:
                    raise InputValidationError(f"{param_name} has too few items: {len(value)} < {min_items}")
                
                if len(value) > max_items:
                    raise InputValidationError(f"{param_name} has too many items: {len(value)} > {max_items}")
            
            validated_params[param_name] = value
        
        # Check for unexpected parameters
        unexpected = set(params.keys()) - set(param_schema.keys())
        if unexpected and self.strict_mode:
            raise InputValidationError(f"Unexpected parameters: {unexpected}")
        
        return validated_params
    
    def check_memory_requirements(
        self,
        data_size_bytes: int,
        operation_type: str = "general",
        safety_factor: float = 2.0
    ) -> Dict[str, Any]:
        """
        Check if system has sufficient memory for operation.
        
        Args:
            data_size_bytes: Size of data in bytes
            operation_type: Type of operation (affects memory multiplier)
            safety_factor: Additional safety factor for memory estimation
            
        Returns:
            Dictionary with memory check results
        """
        memory_info = psutil.virtual_memory()
        
        # Operation-specific memory multipliers
        operation_multipliers = {
            "loading": 1.5,      # Data loading
            "preprocessing": 2.0,  # Data preprocessing
            "training": 3.0,     # Model training
            "inference": 1.2,    # Model inference
            "visualization": 1.8, # Data visualization
            "general": 2.0       # General operations
        }
        
        multiplier = operation_multipliers.get(operation_type, 2.0)
        estimated_memory = data_size_bytes * multiplier * safety_factor
        
        available_memory = memory_info.available
        total_memory = memory_info.total
        
        results = {
            "sufficient_memory": estimated_memory <= available_memory,
            "estimated_memory_gb": estimated_memory / (1024**3),
            "available_memory_gb": available_memory / (1024**3),
            "total_memory_gb": total_memory / (1024**3),
            "memory_usage_percent": (estimated_memory / total_memory) * 100,
            "operation_type": operation_type,
            "safety_factor": safety_factor,
            "recommendations": []
        }
        
        # Add recommendations
        if not results["sufficient_memory"]:
            shortage_gb = (estimated_memory - available_memory) / (1024**3)
            results["recommendations"].extend([
                f"Insufficient memory: need {shortage_gb:.2f} GB more",
                "Consider: 1) Reducing data size, 2) Using batch processing, 3) Adding more RAM"
            ])
        
        if results["memory_usage_percent"] > 80:
            results["recommendations"].append("High memory usage expected - monitor system performance")
        
        return results
    
    def validate_array_dimensions(
        self,
        array: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_shape: Optional[Tuple[int, ...]] = None,
        max_shape: Optional[Tuple[int, ...]] = None,
        expected_ndim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate numpy array dimensions and shape.
        
        Args:
            array: Input array to validate
            expected_shape: Exact expected shape (optional)
            min_shape: Minimum shape constraints (optional)
            max_shape: Maximum shape constraints (optional) 
            expected_ndim: Expected number of dimensions (optional)
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "array_info": {
                "shape": array.shape,
                "ndim": array.ndim,
                "size": array.size,
                "dtype": str(array.dtype),
                "memory_mb": array.nbytes / (1024**2)
            }
        }
        
        try:
            # Check expected dimensions
            if expected_ndim is not None and array.ndim != expected_ndim:
                results["errors"].append(f"Expected {expected_ndim}D array, got {array.ndim}D")
                results["valid"] = False
            
            # Check exact shape
            if expected_shape is not None:
                if array.shape != expected_shape:
                    results["errors"].append(f"Expected shape {expected_shape}, got {array.shape}")
                    results["valid"] = False
            
            # Check minimum shape constraints
            if min_shape is not None:
                if len(array.shape) != len(min_shape):
                    results["errors"].append(f"Shape dimension mismatch: {len(array.shape)} vs {len(min_shape)}")
                    results["valid"] = False
                else:
                    for i, (actual, minimum) in enumerate(zip(array.shape, min_shape)):
                        if actual < minimum:
                            results["errors"].append(f"Dimension {i} too small: {actual} < {minimum}")
                            results["valid"] = False
            
            # Check maximum shape constraints
            if max_shape is not None:
                if len(array.shape) != len(max_shape):
                    results["errors"].append(f"Shape dimension mismatch: {len(array.shape)} vs {len(max_shape)}")
                    results["valid"] = False
                else:
                    for i, (actual, maximum) in enumerate(zip(array.shape, max_shape)):
                        if actual > maximum:
                            results["errors"].append(f"Dimension {i} too large: {actual} > {maximum}")
                            results["valid"] = False
            
            # Size warnings
            if array.size > 10**8:  # > 100M elements
                results["warnings"].append(f"Very large array: {array.size:,} elements")
            
            if array.nbytes > 1024**3:  # > 1GB
                results["warnings"].append(f"High memory usage: {array.nbytes / (1024**3):.2f} GB")
        
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Array validation error: {e}")
        
        return results


# Convenience functions
def sanitize_file_path(file_path: Union[str, Path], **kwargs) -> Path:
    """Convenience function for file path sanitization."""
    validator = InputValidator()
    return validator.sanitize_file_path(file_path, **kwargs)


def validate_parameters(params: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function for parameter validation."""
    validator = InputValidator()
    return validator.validate_parameters(params, schema)


def check_memory_requirements(data_size_bytes: int, **kwargs) -> Dict[str, Any]:
    """Convenience function for memory checking."""
    validator = InputValidator()
    return validator.check_memory_requirements(data_size_bytes, **kwargs)
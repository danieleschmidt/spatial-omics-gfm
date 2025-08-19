"""
Simple security utilities that don't require heavy dependencies.
Provides basic input sanitization and validation for Generation 2 robustness.
"""

import re
import logging
from typing import Dict, Any, Optional, Union, Set
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_user_input(user_input: str, context: str = "general") -> str:
    """
    Simple user input sanitization without external dependencies.
    
    Args:
        user_input: String to sanitize
        context: Context of the string (filename, metadata, etc.)
        
    Returns:
        Sanitized string
    """
    if not isinstance(user_input, str):
        raise ValueError(f"Expected string input, got {type(user_input)}")
    
    # Length check
    max_length = 1000
    if len(user_input) > max_length:
        logger.warning(f"String truncated from {len(user_input)} to {max_length} characters")
        user_input = user_input[:max_length]
    
    # Remove control characters
    sanitized = ''.join(char for char in user_input if ord(char) >= 32 or char in '\t\n\r')
    
    # Context-specific sanitization
    if context == "filename":
        # Remove dangerous filename characters
        sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', sanitized)
        # Check for path traversal
        dangerous_patterns = [r'\.\.[\\/]', r'^[\\/]', r'~[\\/]']
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized):
                raise ValueError(f"Path traversal pattern detected: {pattern}")
    
    elif context == "metadata":
        # Check for code injection patterns
        dangerous_patterns = [
            r'__[a-zA-Z_]+__',  # Python magic methods
            r'eval\s*\(',       # eval function
            r'exec\s*\(',       # exec function
            r'import\s+',       # import statements
            r'subprocess\.',    # subprocess calls
            r'os\.',           # os module calls
            r'[;&|`$]',        # Command separators
            r'\$\(',           # Command substitution
            r'`.*`',           # Backtick execution
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected and removed: {pattern}")
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized


def validate_file_path(file_path: Union[str, Path], allowed_extensions: Optional[Set[str]] = None) -> bool:
    """
    Simple file path validation without external dependencies.
    
    Args:
        file_path: File path to validate
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Validated Path object
    """
    path = Path(file_path)
    
    # Default allowed extensions
    if allowed_extensions is None:
        allowed_extensions = {'.h5ad', '.h5', '.csv', '.tsv', '.zarr', '.json', '.yaml', '.yml'}
    
    # Convert to string for pattern matching
    path_str = str(path)
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'\.\.[\\/]',      # Directory traversal
        r'^[\\/]',         # Absolute paths (sometimes dangerous)
        r'[<>:"|?*]',      # Windows illegal characters
        r'[\x00-\x1f]',    # Control characters
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, path_str):
            raise ValueError(f"Dangerous path pattern detected: {pattern}")
    
    # Resolve path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {path}: {e}")
    
    # Check file extension
    if resolved_path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"File extension {resolved_path.suffix} not allowed")
    
    return True


def is_safe_filename(filename: str) -> bool:
    """
    Check if a filename is safe (no path traversal, etc.).
    
    Args:
        filename: Filename to check
        
    Returns:
        True if safe, False otherwise
    """
    # Check for path traversal patterns
    dangerous_patterns = [
        r'\.\.[\\/]',      # Directory traversal
        r'^[\\/]',         # Absolute paths
        r'[<>:"|?*]',      # Windows illegal characters
        r'[\x00-\x1f]',    # Control characters
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, filename):
            return False
    
    return True


def validate_data_integrity_simple(data: Any, shape_hint: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Simple data integrity validation.
    
    Args:
        data: Data to validate (numpy array, pandas DataFrame, etc.)
        shape_hint: Expected shape hint
        
    Returns:
        Validation result dictionary
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Basic shape validation
        if hasattr(data, 'shape'):
            result['metrics']['shape'] = data.shape
            
            if shape_hint and data.shape != shape_hint:
                result['warnings'].append(f"Shape mismatch: expected {shape_hint}, got {data.shape}")
            
            # Check for empty data
            if any(dim == 0 for dim in data.shape):
                result['errors'].append("Data contains zero-sized dimensions")
                result['is_valid'] = False
        
        # Check for NaN/inf values
        if hasattr(data, 'values'):  # DataFrame
            values = data.values
        elif hasattr(data, '__array__'):  # Array-like
            values = data
        else:
            values = None
        
        if values is not None:
            try:
                import numpy as np
                if np.any(np.isnan(values)):
                    result['warnings'].append("Data contains NaN values")
                if np.any(np.isinf(values)):
                    result['warnings'].append("Data contains infinite values")
            except ImportError:
                # numpy not available, skip NaN/inf checks
                pass
        
        # Check data size (memory safety)
        total_elements = 1
        if hasattr(data, 'shape'):
            for dim in data.shape:
                total_elements *= dim
        
        # Warn for very large arrays (>100M elements)
        if total_elements > 100_000_000:
            result['warnings'].append(f"Very large data array: {total_elements} elements")
        
        result['metrics']['total_elements'] = total_elements
        
    except Exception as e:
        result['errors'].append(f"Validation error: {str(e)}")
        result['is_valid'] = False
    
    return result


def create_simple_security_context() -> Dict[str, Any]:
    """Create a simple security context without heavy dependencies."""
    return {
        'sanitize_input': sanitize_user_input,
        'validate_path': validate_file_path,
        'validate_data': validate_data_integrity_simple,
        'version': '1.0.0'
    }


# Simple validation class
class SimpleDataValidator:
    """Simple data validator without external dependencies."""
    
    def __init__(self):
        self.validation_count = 0
        self.errors = []
        self.warnings = []
    
    def validate_basic_data(self, data: Any) -> Dict[str, Any]:
        """Validate basic data properties."""
        self.validation_count += 1
        result = validate_data_integrity_simple(data)
        
        self.errors.extend(result['errors'])
        self.warnings.extend(result['warnings'])
        
        return result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        return {
            'validation_count': self.validation_count,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }
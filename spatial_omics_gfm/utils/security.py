"""
Security utilities for Spatial-Omics GFM.
Implements input sanitization, secure file handling, and protection against various attacks.
"""

import logging
import hashlib
import hmac
import os
import tempfile
import json
from typing import Dict, Any, Optional, Union, List, Set
from pathlib import Path
import secrets
import re
from datetime import datetime, timedelta
from contextlib import contextmanager
import torch
import numpy as np
from anndata import AnnData
import pickle

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Configuration for security measures."""
    
    def __init__(
        self,
        allowed_file_extensions: Optional[Set[str]] = None,
        max_file_size_mb: int = 1000,
        safe_temp_directory: Optional[str] = None,
        enable_checksum_validation: bool = True,
        enable_model_signing: bool = True,
        max_string_length: int = 1000,
        allowed_pickle_modules: Optional[Set[str]] = None
    ):
        self.allowed_file_extensions = allowed_file_extensions or {
            '.h5ad', '.h5', '.csv', '.tsv', '.zarr', '.json', '.yaml', '.yml'
        }
        self.max_file_size_mb = max_file_size_mb
        self.safe_temp_directory = safe_temp_directory
        self.enable_checksum_validation = enable_checksum_validation
        self.enable_model_signing = enable_model_signing
        self.max_string_length = max_string_length
        self.allowed_pickle_modules = allowed_pickle_modules or {
            'numpy', 'torch', 'anndata', 'scipy', 'pandas'
        }


class InputSanitizer:
    """Comprehensive input sanitization for various data types."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.dangerous_patterns = {
            'code_injection': [
                r'__[a-zA-Z_]+__',  # Python magic methods
                r'eval\s*\(',       # eval function
                r'exec\s*\(',       # exec function
                r'import\s+',       # import statements
                r'subprocess\.',    # subprocess calls
                r'os\.',           # os module calls
            ],
            'path_traversal': [
                r'\.\.[\\/]',      # Directory traversal
                r'^[\\/]',         # Absolute paths
                r'~[\\/]',         # Home directory
            ],
            'command_injection': [
                r'[;&|`$]',        # Command separators
                r'\$\(',           # Command substitution
                r'`.*`',           # Backtick execution
            ]
        }
    
    def sanitize_string(self, input_str: str, context: str = "general") -> str:
        """
        Sanitize string input based on context.
        
        Args:
            input_str: String to sanitize
            context: Context of the string (filename, metadata, etc.)
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            raise ValueError(f"Expected string input, got {type(input_str)}")
        
        # Length check
        if len(input_str) > self.config.max_string_length:
            logger.warning(f"String truncated from {len(input_str)} to {self.config.max_string_length} characters")
            input_str = input_str[:self.config.max_string_length]
        
        # Remove control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        
        # Context-specific sanitization
        if context == "filename":
            # Remove dangerous filename characters
            sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', sanitized)
            # Check for path traversal
            for pattern in self.dangerous_patterns['path_traversal']:
                if re.search(pattern, sanitized):
                    raise ValueError(f"Path traversal pattern detected: {pattern}")
        
        elif context == "metadata":
            # Check for code injection patterns
            for pattern_type, patterns in self.dangerous_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sanitized, re.IGNORECASE):
                        logger.warning(f"Potentially dangerous pattern detected and removed: {pattern}")
                        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def sanitize_file_path(self, file_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
        """
        Sanitize and validate file path.
        
        Args:
            file_path: File path to sanitize
            base_dir: Base directory to restrict access to
            
        Returns:
            Sanitized Path object
        """
        path = Path(file_path)
        
        # Convert to string for pattern matching
        path_str = str(path)
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns['path_traversal']:
            if re.search(pattern, path_str):
                raise ValueError(f"Dangerous path pattern detected: {pattern}")
        
        # Resolve path
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path {path}: {e}")
        
        # Check if within base directory
        if base_dir:
            base_dir = base_dir.resolve()
            try:
                resolved_path.relative_to(base_dir)
            except ValueError:
                raise ValueError(f"Path {resolved_path} is outside base directory {base_dir}")
        
        # Check file extension
        if resolved_path.suffix.lower() not in self.config.allowed_file_extensions:
            raise ValueError(f"File extension {resolved_path.suffix} not allowed")
        
        return resolved_path
    
    def sanitize_dict(self, data_dict: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary data.
        
        Args:
            data_dict: Dictionary to sanitize
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            logger.warning("Maximum recursion depth reached during dict sanitization")
            return {}
        
        sanitized = {}
        for key, value in data_dict.items():
            # Sanitize key
            if isinstance(key, str):
                clean_key = self.sanitize_string(key, context="metadata")
            else:
                clean_key = str(key)[:100]  # Limit key length
            
            # Sanitize value
            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(value, context="metadata")
            elif isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, (list, tuple)):
                sanitized[clean_key] = [
                    self.sanitize_string(item, context="metadata") if isinstance(item, str) else item
                    for item in value[:100]  # Limit list length
                ]
            else:
                sanitized[clean_key] = value
        
        return sanitized


class SecureFileHandler:
    """Secure file operations with validation and sandboxing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sanitizer = InputSanitizer(config)
        self.temp_dir = Path(config.safe_temp_directory) if config.safe_temp_directory else None
    
    @contextmanager
    def secure_temp_file(self, suffix: str = ".tmp", prefix: str = "spatial_gfm_"):
        """Create a secure temporary file."""
        if self.temp_dir:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix, prefix=prefix, dir=self.temp_dir, delete=False
            )
        else:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix, prefix=prefix, delete=False
            )
        
        try:
            yield Path(temp_file.name)
        finally:
            temp_path = Path(temp_file.name)
            if temp_path.exists():
                temp_path.unlink()
    
    def validate_file_size(self, file_path: Path) -> bool:
        """Validate file size against limits."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(f"File size {file_size_mb:.1f}MB exceeds limit of {self.config.max_file_size_mb}MB")
        
        return True
    
    def compute_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Compute secure hash of file."""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def verify_file_integrity(self, file_path: Path, expected_checksum: str, algorithm: str = "sha256") -> bool:
        """Verify file integrity using checksum."""
        actual_checksum = self.compute_file_checksum(file_path, algorithm)
        return hmac.compare_digest(actual_checksum, expected_checksum)
    
    def load_data_securely(self, file_path: Union[str, Path], base_dir: Optional[Path] = None) -> AnnData:
        """
        Securely load data file with validation.
        
        Args:
            file_path: Path to data file
            base_dir: Base directory to restrict access to
            
        Returns:
            Loaded AnnData object
        """
        # Sanitize and validate path
        safe_path = self.sanitizer.sanitize_file_path(file_path, base_dir)
        
        # Validate file size
        self.validate_file_size(safe_path)
        
        # Load based on file type
        if safe_path.suffix == '.h5ad':
            return self._load_h5ad_securely(safe_path)
        elif safe_path.suffix == '.h5':
            return self._load_h5_securely(safe_path)
        elif safe_path.suffix in ['.csv', '.tsv']:
            return self._load_csv_securely(safe_path)
        else:
            raise ValueError(f"Unsupported file format: {safe_path.suffix}")
    
    def _load_h5ad_securely(self, file_path: Path) -> AnnData:
        """Securely load H5AD file."""
        try:
            # Use backed mode to avoid loading large files entirely into memory
            adata = AnnData.read_h5ad(file_path, backed='r')
            
            # Validate and sanitize metadata
            if hasattr(adata, 'uns') and adata.uns:
                adata.uns = self.sanitizer.sanitize_dict(dict(adata.uns))
            
            return adata
        except Exception as e:
            logger.error(f"Failed to load H5AD file {file_path}: {e}")
            raise ValueError(f"Invalid or corrupted H5AD file: {e}")
    
    def _load_h5_securely(self, file_path: Path) -> AnnData:
        """Securely load H5 file."""
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # Basic validation of file structure
                if 'X' not in f and 'matrix' not in f:
                    raise ValueError("No expression matrix found in H5 file")
                
                # Load minimal structure
                X = f['X'][:] if 'X' in f else f['matrix'][:]
                adata = AnnData(X)
                
                # Load spatial coordinates if available
                if 'obsm/spatial' in f:
                    adata.obsm['spatial'] = f['obsm/spatial'][:]
                elif 'spatial' in f:
                    adata.obsm['spatial'] = f['spatial'][:]
                
                return adata
        except Exception as e:
            logger.error(f"Failed to load H5 file {file_path}: {e}")
            raise ValueError(f"Invalid or corrupted H5 file: {e}")
    
    def _load_csv_securely(self, file_path: Path) -> AnnData:
        """Securely load CSV file."""
        try:
            import pandas as pd
            
            # Read with size limits
            df = pd.read_csv(file_path, nrows=100000)  # Limit rows for security
            
            # Basic validation
            if df.empty:
                raise ValueError("Empty CSV file")
            
            # Sanitize column names
            df.columns = [self.sanitizer.sanitize_string(col, "metadata") for col in df.columns]
            
            # Create AnnData (assuming genes as columns, cells as rows)
            adata = AnnData(df.select_dtypes(include=[np.number]).values)
            
            return adata
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise ValueError(f"Invalid or corrupted CSV file: {e}")


class ModelSecurity:
    """Security measures for model loading and saving."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.model_signatures = {}
    
    def sign_model(self, model_state: Dict[str, Any], secret_key: Optional[str] = None) -> str:
        """Create cryptographic signature for model."""
        if not self.config.enable_model_signing:
            return ""
        
        # Use provided secret key or generate one
        if secret_key is None:
            secret_key = os.environ.get('MODEL_SIGNING_KEY')
            if secret_key is None:
                logger.warning("No signing key provided, generating temporary key")
                secret_key = secrets.token_hex(32)
        
        # Serialize model state deterministically
        model_bytes = json.dumps(
            {k: str(v) for k, v in model_state.items()}, 
            sort_keys=True
        ).encode()
        
        # Create HMAC signature
        signature = hmac.new(
            secret_key.encode(),
            model_bytes,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_model_signature(
        self, 
        model_state: Dict[str, Any], 
        signature: str, 
        secret_key: Optional[str] = None
    ) -> bool:
        """Verify model signature."""
        if not self.config.enable_model_signing:
            return True  # Skip verification if not enabled
        
        if secret_key is None:
            secret_key = os.environ.get('MODEL_SIGNING_KEY')
            if secret_key is None:
                logger.warning("No signing key provided for verification")
                return False
        
        expected_signature = self.sign_model(model_state, secret_key)
        return hmac.compare_digest(signature, expected_signature)
    
    def save_model_securely(
        self, 
        model: torch.nn.Module, 
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Securely save model with signature and metadata.
        
        Returns:
            Dictionary with file path and signature
        """
        file_path = Path(file_path)
        
        # Create model state
        model_state = {
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Generate signature
        signature = self.sign_model(model_state)
        
        # Save with signature
        save_dict = {
            'model_state': model_state,
            'signature': signature,
            'version': '1.0'
        }
        
        torch.save(save_dict, file_path)
        
        # Log save operation
        logger.info(f"Model saved securely to {file_path}")
        
        return {
            'file_path': str(file_path),
            'signature': signature,
            'checksum': self._compute_file_checksum(file_path)
        }
    
    def load_model_securely(
        self, 
        file_path: Union[str, Path],
        model_class: torch.nn.Module,
        strict: bool = True
    ) -> torch.nn.Module:
        """
        Securely load model with signature verification.
        
        Args:
            file_path: Path to model file
            model_class: Model class to instantiate
            strict: Whether to enforce strict signature verification
            
        Returns:
            Loaded model instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            # Load model data
            save_dict = torch.load(file_path, map_location='cpu')
            
            # Verify structure
            required_keys = ['model_state', 'signature', 'version']
            for key in required_keys:
                if key not in save_dict:
                    raise ValueError(f"Missing required key in model file: {key}")
            
            # Verify signature
            model_state = save_dict['model_state']
            signature = save_dict['signature']
            
            if self.config.enable_model_signing:
                if not self.verify_model_signature(model_state, signature):
                    if strict:
                        raise ValueError("Model signature verification failed")
                    else:
                        logger.warning("Model signature verification failed, proceeding anyway")
            
            # Create model instance
            model = model_class()
            
            # Load state dict
            model.load_state_dict(model_state['state_dict'])
            
            # Log successful load
            logger.info(f"Model loaded securely from {file_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {file_path}: {e}")
            raise ValueError(f"Failed to load model: {e}")
    
    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class SafePickleLoader:
    """Safe pickle loader with restricted imports."""
    
    def __init__(self, allowed_modules: Optional[Set[str]] = None):
        self.allowed_modules = allowed_modules or {
            'numpy', 'torch', 'anndata', 'scipy', 'pandas', 'sklearn'
        }
    
    def restricted_loads(self, data: bytes) -> Any:
        """Load pickle data with restricted imports."""
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow specific modules
                if module.split('.')[0] not in self.allowed_modules:
                    raise pickle.UnpicklingError(f"Module {module} not allowed")
                return super().find_class(module, name)
        
        import io
        return RestrictedUnpickler(io.BytesIO(data)).load()


class SecurityAuditLogger:
    """Logger for security-related events."""
    
    def __init__(self):
        self.security_logger = logging.getLogger('spatial_omics_gfm.security')
        self.security_events = []
    
    def log_security_event(
        self, 
        event_type: str, 
        description: str, 
        severity: str = "INFO",
        context: Optional[Dict] = None
    ):
        """Log security event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'context': context or {}
        }
        
        self.security_events.append(event)
        
        # Log to standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.security_logger.log(log_level, f"{event_type}: {description}", extra=event)
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get security report for recent events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event['timestamp']) > cutoff_time
        ]
        
        # Categorize events
        event_summary = {}
        for event in recent_events:
            event_type = event['event_type']
            event_summary[event_type] = event_summary.get(event_type, 0) + 1
        
        return {
            'time_range_hours': hours,
            'total_events': len(recent_events),
            'event_summary': event_summary,
            'events': recent_events
        }


# Global security instances
default_security_config = SecurityConfig()
global_security_logger = SecurityAuditLogger()


def sanitize_user_input(user_input: str, context: str = "general") -> str:
    """Convenient function to sanitize user input."""
    sanitizer = InputSanitizer(default_security_config)
    return sanitizer.sanitize_string(user_input, context)


def secure_file_operation(file_path: Union[str, Path], operation: str = "read") -> Path:
    """Convenient function for secure file operations."""
    handler = SecureFileHandler(default_security_config)
    return handler.sanitizer.sanitize_file_path(file_path)


def create_security_context(
    allowed_extensions: Optional[Set[str]] = None,
    max_file_size_mb: int = 1000,
    enable_signing: bool = True
) -> Dict[str, Any]:
    """Create a security context with specified settings."""
    config = SecurityConfig(
        allowed_file_extensions=allowed_extensions,
        max_file_size_mb=max_file_size_mb,
        enable_model_signing=enable_signing
    )
    
    return {
        'config': config,
        'sanitizer': InputSanitizer(config),
        'file_handler': SecureFileHandler(config),
        'model_security': ModelSecurity(config),
        'audit_logger': SecurityAuditLogger()
    }
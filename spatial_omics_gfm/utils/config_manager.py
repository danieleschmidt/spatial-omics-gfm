"""
Robust configuration management system for Spatial-Omics GFM.
Supports YAML/JSON configs, environment variables, validation, and runtime updates.
"""

import logging
import json
import yaml
import os
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints
from pathlib import Path
from dataclasses import dataclass, field, fields, is_dataclass
import copy
from datetime import datetime
import threading
from contextlib import contextmanager
import re
from pydantic import BaseModel, ValidationError, Field, validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigUpdateError(Exception):
    """Exception raised when configuration update fails."""
    pass


@pydantic_dataclass
class ModelConfig:
    """Configuration for model parameters."""
    num_genes: int = Field(..., gt=0, description="Number of genes in the dataset")
    hidden_dim: int = Field(1024, gt=0, description="Hidden dimension size")
    num_layers: int = Field(24, gt=0, le=100, description="Number of transformer layers")
    num_heads: int = Field(16, gt=0, description="Number of attention heads")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    max_neighbors: int = Field(6, gt=0, description="Maximum number of spatial neighbors")
    spatial_dim: int = Field(2, ge=2, le=3, description="Spatial dimension (2D or 3D)")
    use_positional_encoding: bool = Field(True, description="Whether to use positional encoding")
    attention_bias_type: str = Field("spatial", description="Type of attention bias")
    
    @validator('hidden_dim')
    def validate_hidden_dim(cls, v, values):
        if 'num_heads' in values and v % values['num_heads'] != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        return v


@pydantic_dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = Field(32, gt=0, description="Training batch size")
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(1e-5, ge=0, description="Weight decay")
    max_epochs: int = Field(100, gt=0, description="Maximum training epochs")
    patience: int = Field(10, gt=0, description="Early stopping patience")
    gradient_clip_val: float = Field(1.0, gt=0, description="Gradient clipping value")
    accumulate_grad_batches: int = Field(1, gt=0, description="Gradient accumulation steps")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    checkpoint_every_n_epochs: int = Field(10, gt=0, description="Checkpoint frequency")
    validate_every_n_epochs: int = Field(1, gt=0, description="Validation frequency")


@pydantic_dataclass
class DataConfig:
    """Configuration for data processing."""
    min_genes_per_cell: int = Field(100, ge=0, description="Minimum genes per cell")
    min_cells_per_gene: int = Field(3, ge=0, description="Minimum cells per gene")
    max_genes_per_cell: int = Field(10000, gt=0, description="Maximum genes per cell")
    min_counts_per_cell: int = Field(500, ge=0, description="Minimum counts per cell")
    max_mitochondrial_pct: float = Field(20.0, ge=0, le=100, description="Max mitochondrial %")
    normalize: bool = Field(True, description="Normalize expression data")
    log_transform: bool = Field(True, description="Log transform data")
    scale_coordinates: bool = Field(True, description="Scale spatial coordinates")
    graph_construction_method: str = Field("knn", description="Graph construction method")
    k_neighbors: int = Field(6, gt=0, description="Number of neighbors for k-NN graph")


@pydantic_dataclass
class SystemConfig:
    """Configuration for system resources."""
    device: str = Field("auto", description="Computing device (auto, cpu, cuda)")
    num_workers: int = Field(4, ge=0, description="Number of data loading workers")
    memory_limit_gb: float = Field(16.0, gt=0, description="Memory limit in GB")
    enable_memory_monitoring: bool = Field(True, description="Enable memory monitoring")
    temp_directory: Optional[str] = Field(None, description="Temporary directory path")
    log_level: str = Field("INFO", description="Logging level")
    enable_profiling: bool = Field(False, description="Enable performance profiling")


@pydantic_dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_input_validation: bool = Field(True, description="Enable input validation")
    enable_model_signing: bool = Field(True, description="Enable model signing")
    max_file_size_mb: int = Field(1000, gt=0, description="Maximum file size in MB")
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ['.h5ad', '.h5', '.csv', '.json', '.yaml'],
        description="Allowed file extensions"
    )
    enable_checksum_validation: bool = Field(True, description="Enable checksum validation")


@pydantic_dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    version: str = Field("1.0.0", description="Configuration version")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Custom parameters")


class ConfigManager:
    """Robust configuration manager with validation and runtime updates."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[ExperimentConfig] = None
        self.config_history: List[ExperimentConfig] = []
        self.watchers: List[callable] = []
        self._lock = threading.Lock()
        self.env_prefix = "SPATIAL_GFM_"
        
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        override_with_env: bool = True
    ) -> ExperimentConfig:
        """
        Load configuration from file with optional environment variable overrides.
        
        Args:
            config_path: Path to configuration file
            override_with_env: Whether to override with environment variables
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path or self.config_path)
        
        if not config_path or not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config_dict = {}
        else:
            config_dict = self._load_config_file(config_path)
        
        # Apply environment variable overrides
        if override_with_env:
            config_dict = self._apply_env_overrides(config_dict)
        
        # Validate and create configuration
        try:
            config = ExperimentConfig(**config_dict)
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        
        # Store configuration
        with self._lock:
            if self.config is not None:
                self.config_history.append(copy.deepcopy(self.config))
            self.config = config
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        self._notify_watchers()
        
        return config
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigValidationError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load config file {config_path}: {e}")
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_overrides = {}
        
        # Collect environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                
                # Convert nested keys (e.g., MODEL_HIDDEN_DIM -> model.hidden_dim)
                key_parts = config_key.split('_')
                
                # Try to convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set nested value
                current_dict = env_overrides
                for part in key_parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[key_parts[-1]] = converted_value
        
        # Deep merge environment overrides
        return self._deep_merge(config_dict, env_overrides)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value and 'e' not in value.lower():
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None, format: str = "yaml") -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        if self.config is None:
            raise ConfigUpdateError("No configuration loaded")
        
        config_path = Path(config_path or self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = self._config_to_dict(self.config)
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ConfigValidationError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigUpdateError(f"Failed to save config to {config_path}: {e}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        if hasattr(config, '__dict__'):
            return {k: self._serialize_value(v) for k, v in config.__dict__.items()}
        else:
            # Handle pydantic dataclass
            return config.dict()
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize configuration value for JSON/YAML output."""
        if is_dataclass(value):
            return self._config_to_dict(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return value
    
    def update_config(
        self, 
        updates: Dict[str, Any], 
        validate: bool = True,
        save_to_file: bool = False
    ) -> None:
        """
        Update configuration at runtime.
        
        Args:
            updates: Dictionary of updates to apply
            validate: Whether to validate the updated configuration
            save_to_file: Whether to save changes to file
        """
        if self.config is None:
            raise ConfigUpdateError("No configuration loaded")
        
        with self._lock:
            # Backup current configuration
            backup_config = copy.deepcopy(self.config)
            
            try:
                # Apply updates
                config_dict = self._config_to_dict(self.config)
                updated_dict = self._deep_merge(config_dict, updates)
                
                # Validate if requested
                if validate:
                    new_config = ExperimentConfig(**updated_dict)
                else:
                    new_config = ExperimentConfig.construct(**updated_dict)
                
                # Store old config in history
                self.config_history.append(backup_config)
                
                # Update current config
                self.config = new_config
                
                # Save to file if requested
                if save_to_file and self.config_path:
                    self.save_config()
                
                logger.info("Configuration updated successfully")
                self._notify_watchers()
                
            except Exception as e:
                # Restore backup on error
                self.config = backup_config
                raise ConfigUpdateError(f"Failed to update configuration: {e}")
    
    def get_config(self) -> Optional[ExperimentConfig]:
        """Get current configuration."""
        return self.config
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "model.hidden_dim")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self.config is None:
            return default
        
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def register_watcher(self, callback: callable) -> None:
        """Register a callback to be called when configuration changes."""
        with self._lock:
            self.watchers.append(callback)
    
    def _notify_watchers(self) -> None:
        """Notify all registered watchers of configuration changes."""
        for watcher in self.watchers:
            try:
                watcher(self.config)
            except Exception as e:
                logger.warning(f"Configuration watcher failed: {e}")
    
    def validate_config(self, config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
        """
        Validate configuration and return validation report.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            Validation report
        """
        config = config or self.config
        if config is None:
            return {'valid': False, 'errors': ['No configuration loaded']}
        
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Pydantic validation is automatic, but we can add custom checks
            
            # Check model configuration consistency
            if config.model.hidden_dim % config.model.num_heads != 0:
                validation_report['errors'].append(
                    "Model hidden_dim must be divisible by num_heads"
                )
            
            # Check system resource limits
            if config.system.memory_limit_gb < 1.0:
                validation_report['warnings'].append(
                    "Very low memory limit may cause performance issues"
                )
            
            # Check training configuration
            if config.training.batch_size > 256:
                validation_report['warnings'].append(
                    "Large batch size may require significant memory"
                )
            
            # Check data configuration
            if config.data.min_genes_per_cell > config.data.max_genes_per_cell:
                validation_report['errors'].append(
                    "min_genes_per_cell cannot be greater than max_genes_per_cell"
                )
            
            # Performance recommendations
            if config.model.hidden_dim % 64 != 0:
                validation_report['recommendations'].append(
                    "Consider using hidden_dim divisible by 64 for optimal GPU performance"
                )
            
            validation_report['valid'] = not validation_report['errors']
            
        except ValidationError as e:
            validation_report['valid'] = False
            validation_report['errors'].extend([str(error) for error in e.errors()])
        
        return validation_report
    
    @contextmanager
    def temporary_config(self, updates: Dict[str, Any]):
        """Context manager for temporary configuration changes."""
        if self.config is None:
            raise ConfigUpdateError("No configuration loaded")
        
        original_config = copy.deepcopy(self.config)
        
        try:
            self.update_config(updates, validate=False, save_to_file=False)
            yield self.config
        finally:
            with self._lock:
                self.config = original_config
    
    def get_config_history(self, limit: int = 10) -> List[ExperimentConfig]:
        """Get configuration history."""
        with self._lock:
            return self.config_history[-limit:] if limit > 0 else self.config_history[:]
    
    def rollback_config(self, steps: int = 1) -> None:
        """Rollback configuration to previous version."""
        with self._lock:
            if len(self.config_history) < steps:
                raise ConfigUpdateError(f"Cannot rollback {steps} steps, only {len(self.config_history)} versions available")
            
            # Get the configuration from history
            target_config = self.config_history[-(steps)]
            
            # Remove the configurations we're rolling back from
            self.config_history = self.config_history[:-(steps)]
            
            self.config = target_config
            
        logger.info(f"Configuration rolled back {steps} steps")
        self._notify_watchers()


def create_default_config(name: str = "default_experiment") -> ExperimentConfig:
    """Create a default configuration."""
    return ExperimentConfig(
        name=name,
        description="Default configuration for Spatial-Omics GFM"
    )


def load_config_from_file(file_path: Union[str, Path]) -> ExperimentConfig:
    """Convenient function to load configuration from file."""
    manager = ConfigManager()
    return manager.load_config(file_path)


def validate_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate configuration file without loading it into a manager."""
    manager = ConfigManager()
    config = manager.load_config(file_path)
    return manager.validate_config(config)


# Environment variable documentation
ENV_VARS_DOC = """
Environment Variables for Configuration Override:

Model Configuration:
- SPATIAL_GFM_MODEL_NUM_GENES: Number of genes
- SPATIAL_GFM_MODEL_HIDDEN_DIM: Hidden dimension
- SPATIAL_GFM_MODEL_NUM_LAYERS: Number of layers
- SPATIAL_GFM_MODEL_NUM_HEADS: Number of attention heads
- SPATIAL_GFM_MODEL_DROPOUT: Dropout rate

Training Configuration:
- SPATIAL_GFM_TRAINING_BATCH_SIZE: Batch size
- SPATIAL_GFM_TRAINING_LEARNING_RATE: Learning rate
- SPATIAL_GFM_TRAINING_MAX_EPOCHS: Maximum epochs

System Configuration:
- SPATIAL_GFM_SYSTEM_DEVICE: Device (cpu/cuda)
- SPATIAL_GFM_SYSTEM_NUM_WORKERS: Number of workers
- SPATIAL_GFM_SYSTEM_MEMORY_LIMIT_GB: Memory limit in GB
- SPATIAL_GFM_SYSTEM_LOG_LEVEL: Logging level

Security Configuration:
- SPATIAL_GFM_SECURITY_ENABLE_MODEL_SIGNING: Enable model signing
- SPATIAL_GFM_SECURITY_MAX_FILE_SIZE_MB: Max file size in MB
"""
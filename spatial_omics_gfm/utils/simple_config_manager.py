"""
Simple configuration manager without YAML dependencies.
Provides basic configuration management using JSON format.
"""

import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    dropout: float = 0.1
    max_neighbors: int = 10
    spatial_encoding_dim: int = 64


@dataclass
class DataConfig:
    """Data processing configuration."""
    batch_size: int = 32
    num_workers: int = 4
    normalize: bool = True
    log_transform: bool = True
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 5000
    min_cells_per_gene: int = 10


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    warmup_steps: int = 1000


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    random_seed: int = 42
    device: str = "auto"
    output_dir: str = "./results"
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        data: Optional[DataConfig] = None,
        training: Optional[TrainingConfig] = None,
        **kwargs
    ):
        self.model = model or ModelConfig()
        self.data = data or DataConfig()
        self.training = training or TrainingConfig()
        
        # Set other attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ConfigManager:
    """Simple configuration manager using JSON."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = None
    
    def load_config(self, config_data: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        """
        Load configuration from data or file.
        
        Args:
            config_data: Configuration dictionary (optional)
            
        Returns:
            ExperimentConfig object
        """
        if config_data is not None:
            # Load from provided data
            self.config = self._dict_to_config(config_data)
        elif self.config_path and self.config_path.exists():
            # Load from file
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = self._dict_to_config(config_dict)
        else:
            # Use default configuration
            self.config = create_default_config()
        
        return self.config
    
    def save_config(self, config: ExperimentConfig, path: Optional[Union[str, Path]] = None):
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        if save_path is None:
            raise ValueError("No save path specified")
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        if self.config is None:
            self.config = create_default_config()
        
        return asdict(self.config.model)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration as dictionary."""
        if self.config is None:
            self.config = create_default_config()
        
        return asdict(self.config.data)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        if self.config is None:
            self.config = create_default_config()
        
        return asdict(self.config.training)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if self.config is None:
            self.config = create_default_config()
        
        # Update model config
        if 'model' in updates:
            for key, value in updates['model'].items():
                if hasattr(self.config.model, key):
                    setattr(self.config.model, key, value)
        
        # Update data config
        if 'data' in updates:
            for key, value in updates['data'].items():
                if hasattr(self.config.data, key):
                    setattr(self.config.data, key, value)
        
        # Update training config
        if 'training' in updates:
            for key, value in updates['training'].items():
                if hasattr(self.config.training, key):
                    setattr(self.config.training, key, value)
        
        # Update top-level attributes
        for key, value in updates.items():
            if key not in ['model', 'data', 'training'] and hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        model_dict = config_dict.get('model', {})
        data_dict = config_dict.get('data', {})
        training_dict = config_dict.get('training', {})
        
        model_config = ModelConfig(**model_dict)
        data_config = DataConfig(**data_dict)
        training_config = TrainingConfig(**training_dict)
        
        # Get other top-level attributes
        other_attrs = {
            k: v for k, v in config_dict.items() 
            if k not in ['model', 'data', 'training']
        }
        
        return ExperimentConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            **other_attrs
        )
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return {
            'model': asdict(config.model),
            'data': asdict(config.data),
            'training': asdict(config.training),
            'random_seed': config.random_seed,
            'device': config.device,
            'output_dir': config.output_dir
        }


def create_default_config() -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        random_seed=42,
        device="auto",
        output_dir="./results"
    )


def load_config_from_file(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from JSON file."""
    manager = ConfigManager(config_path)
    return manager.load_config()


def save_config_to_file(config: ExperimentConfig, config_path: Union[str, Path]):
    """Save configuration to JSON file."""
    manager = ConfigManager()
    manager.save_config(config, config_path)


def create_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Create configuration from dictionary."""
    manager = ConfigManager()
    return manager.load_config(config_dict)


if __name__ == "__main__":
    # Test configuration system
    print("=== Testing Simple Configuration Manager ===")
    
    # Create default config
    config = create_default_config()
    print(f"Default model hidden_dim: {config.model.hidden_dim}")
    print(f"Default data batch_size: {config.data.batch_size}")
    print(f"Default training learning_rate: {config.training.learning_rate}")
    
    # Test config manager
    manager = ConfigManager()
    loaded_config = manager.load_config()
    
    # Test config access methods
    model_config = manager.get_model_config()
    print(f"Model config keys: {list(model_config.keys())}")
    
    # Test config updates
    updates = {
        'model': {'hidden_dim': 512, 'num_layers': 8},
        'data': {'batch_size': 16},
        'training': {'learning_rate': 2e-4}
    }
    manager.update_config(updates)
    
    print(f"Updated model hidden_dim: {manager.config.model.hidden_dim}")
    print(f"Updated data batch_size: {manager.config.data.batch_size}")
    print(f"Updated training learning_rate: {manager.config.training.learning_rate}")
    
    # Test saving to file
    test_config_path = "/tmp/test_config.json"
    manager.save_config(manager.config, test_config_path)
    print(f"Config saved to {test_config_path}")
    
    # Test loading from file
    loaded_from_file = load_config_from_file(test_config_path)
    print(f"Loaded from file - hidden_dim: {loaded_from_file.model.hidden_dim}")
    
    print("\n✅ Simple configuration manager working")
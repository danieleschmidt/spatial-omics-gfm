"""
Pre-trained model loading and management.

This module handles loading pre-trained Spatial-Omics GFM models
and provides a model zoo for different scales and applications.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Any
import warnings
import json
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import os

from .graph_transformer import SpatialGraphTransformer, TransformerConfig


# Available pre-trained models
AVAILABLE_MODELS = {
    "spatial-gfm-base": {
        "parameters": "350M",
        "hidden_dim": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "training_data": "10M cells from 50 tissues",
        "tasks": ["cell_typing", "interaction_prediction"],
        "description": "Base model suitable for most spatial transcriptomics tasks",
        "model_id": "spatial-omics/spatial-gfm-base",
        "config_file": "config.json",
        "weights_file": "pytorch_model.bin"
    },
    "spatial-gfm-large": {
        "parameters": "1.3B", 
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "training_data": "100M cells from 200 tissues",
        "tasks": ["all"],
        "description": "Large model with superior performance on complex tasks",
        "model_id": "spatial-omics/spatial-gfm-large",
        "config_file": "config.json",
        "weights_file": "pytorch_model.bin"
    },
    "spatial-gfm-xlarge": {
        "parameters": "3B",
        "hidden_dim": 2048,
        "num_layers": 36,
        "num_heads": 32,
        "training_data": "500M cells from 1000 tissues",
        "tasks": ["all", "zero_shot_transfer"],
        "description": "Extra-large model for research and specialized applications",
        "model_id": "spatial-omics/spatial-gfm-xlarge",
        "config_file": "config.json",
        "weights_file": "pytorch_model.bin"
    },
    "spatial-gfm-tumor": {
        "parameters": "1.3B",
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "training_data": "50M tumor cells from 100 cancer types",
        "tasks": ["tumor_analysis", "immune_profiling"],
        "description": "Specialized model for tumor microenvironment analysis",
        "model_id": "spatial-omics/spatial-gfm-tumor",
        "config_file": "config.json",
        "weights_file": "pytorch_model.bin"
    },
    "spatial-gfm-brain": {
        "parameters": "1.3B",
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "training_data": "80M brain cells from mouse and human",
        "tasks": ["neuronal_typing", "brain_region_mapping"],
        "description": "Specialized model for brain tissue analysis",
        "model_id": "spatial-omics/spatial-gfm-brain",
        "config_file": "config.json",
        "weights_file": "pytorch_model.bin"
    }
}


def load_pretrained_model(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    trust_remote_code: bool = False,
    **kwargs
) -> SpatialGraphTransformer:
    """
    Load a pre-trained Spatial-Omics GFM model.
    
    Args:
        model_name: Name of the pre-trained model
        device: Device to load the model on
        cache_dir: Directory to cache downloaded models
        force_download: Force re-downloading the model
        trust_remote_code: Trust remote code execution
        **kwargs: Additional arguments for model loading
        
    Returns:
        Pre-trained SpatialGraphTransformer model
        
    Raises:
        ValueError: If model_name is not available
        RuntimeError: If model loading fails
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Available models: {available}"
        )
    
    model_info = AVAILABLE_MODELS[model_name]
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/spatial_omics_gfm")
    
    try:
        # Download model files
        config_path = _download_model_file(
            model_info["model_id"],
            model_info["config_file"],
            cache_dir=cache_dir,
            force_download=force_download
        )
        
        weights_path = _download_model_file(
            model_info["model_id"],
            model_info["weights_file"],
            cache_dir=cache_dir,
            force_download=force_download
        )
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model configuration
        config = TransformerConfig(**config_dict)
        
        # Initialize model
        model = SpatialGraphTransformer(config)
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    warnings.warn(
                        f"Shape mismatch for {key}: "
                        f"expected {model_state_dict[key].shape}, "
                        f"got {value.shape}. Skipping."
                    )
            else:
                warnings.warn(f"Unexpected key in state_dict: {key}")
        
        # Load filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Add model metadata
        model._model_name = model_name
        model._model_info = model_info
        
        print(f"Loaded pre-trained model '{model_name}' with {model_info['parameters']} parameters")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")


def _download_model_file(
    model_id: str,
    filename: str,
    cache_dir: str,
    force_download: bool = False
) -> str:
    """
    Download a model file from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID
        filename: Name of the file to download
        cache_dir: Cache directory
        force_download: Force re-downloading
        
    Returns:
        Path to downloaded file
    """
    try:
        file_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download
        )
        return file_path
    except Exception as e:
        # Fallback to local loading if HuggingFace fails
        warnings.warn(
            f"Failed to download from HuggingFace Hub: {e}. "
            "Trying local files..."
        )
        
        local_path = Path(cache_dir) / model_id / filename
        if local_path.exists():
            return str(local_path)
        else:
            raise FileNotFoundError(
                f"Model file not found locally: {local_path}"
            )


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available pre-trained models.
    
    Returns:
        Dictionary of available models with their information
    """
    return AVAILABLE_MODELS.copy()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model is not available
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Available models: {available}"
        )
    
    return AVAILABLE_MODELS[model_name].copy()


def save_model(
    model: SpatialGraphTransformer,
    save_path: Union[str, Path],
    include_config: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None
) -> None:
    """
    Save a model to disk and optionally to HuggingFace Hub.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        include_config: Whether to save configuration
        metadata: Additional metadata to save
        push_to_hub: Whether to push to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    weights_path = save_path / "pytorch_model.bin"
    torch.save(model.state_dict(), weights_path)
    
    if include_config:
        # Save configuration
        config_dict = {
            "num_genes": model.config.num_genes,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "num_heads": model.config.num_heads,
            "dropout": model.config.dropout,
            "spatial_encoding_dim": model.config.spatial_encoding_dim,
            "max_neighbors": model.config.max_neighbors,
            "max_distance": model.config.max_distance,
            "activation": model.config.activation,
            "layer_norm_eps": model.config.layer_norm_eps,
            "use_edge_features": model.config.use_edge_features,
            "use_hierarchical_pooling": model.config.use_hierarchical_pooling
        }
        
        if metadata:
            config_dict.update(metadata)
        
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Save model card
    model_card_path = save_path / "README.md"
    _create_model_card(model, model_card_path, metadata)
    
    print(f"Model saved to {save_path}")
    
    if push_to_hub and repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Upload files
            api.upload_folder(
                folder_path=str(save_path),
                repo_id=repo_id,
                repo_type="model"
            )
            
            print(f"Model pushed to HuggingFace Hub: {repo_id}")
            
        except ImportError:
            warnings.warn(
                "huggingface_hub not available. "
                "Install it to push to HuggingFace Hub."
            )
        except Exception as e:
            warnings.warn(f"Failed to push to Hub: {e}")


def _create_model_card(
    model: SpatialGraphTransformer,
    card_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Create a model card README file."""
    param_count = model.get_parameter_count()
    
    model_card = f"""# Spatial-Omics Graph Foundation Model

## Model Description

This is a Graph Foundation Model for spatial transcriptomics data analysis.
The model treats tissue sections as graphs where cells are nodes and spatial 
proximity defines edges.

## Model Details

- **Parameters**: {param_count['total']:,}
- **Hidden Dimension**: {model.config.hidden_dim}
- **Number of Layers**: {model.config.num_layers}
- **Number of Heads**: {model.config.num_heads}
- **Max Distance**: {model.config.max_distance}μm

## Parameter Breakdown

"""
    
    for component, count in param_count.items():
        if component != "total":
            model_card += f"- **{component}**: {count:,} parameters\n"
    
    model_card += f"""
## Usage

```python
from spatial_omics_gfm import SpatialGraphTransformer

# Load the model
model = SpatialGraphTransformer.from_pretrained('path/to/model')

# Use for inference
embeddings = model.encode(
    gene_expression=gene_expr,
    spatial_coords=coords,
    edge_index=edges
)
```

## Training Data

This model was trained on spatial transcriptomics data from multiple platforms
including Visium, Slide-seq, and Xenium.

## Intended Use

- Cell type prediction with spatial context
- Cell-cell interaction prediction
- Spatially-resolved pathway analysis
- Tissue organization analysis

## Limitations

- Requires spatial coordinate information
- Performance depends on tissue quality and resolution
- May not generalize to very different tissue types without fine-tuning

## Citation

```bibtex
@article{{spatial_omics_gfm,
  title={{Spatial-Omics GFM: A Graph Foundation Model for Spatial Transcriptomics}},
  author={{Your Name}},
  journal={{Nature Methods}},
  year={{2025}}
}}
```
"""
    
    if metadata:
        model_card += "\n## Additional Information\n\n"
        for key, value in metadata.items():
            model_card += f"- **{key}**: {value}\n"
    
    with open(card_path, 'w') as f:
        f.write(model_card)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[TransformerConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> SpatialGraphTransformer:
    """
    Load model from a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration (if not in checkpoint)
        device: Device to load model on
        strict: Whether to strictly enforce state dict loading
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get configuration
    if config is None:
        if "config" in checkpoint:
            config = TransformerConfig(**checkpoint["config"])
        else:
            raise ValueError(
                "No configuration provided and none found in checkpoint"
            )
    
    # Initialize model
    model = SpatialGraphTransformer(config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    
    # Set device
    if device is not None:
        model = model.to(device)
    
    return model
"""
Spatial-Omics GFM: Graph Foundation Model for Spatial Transcriptomics

A billion-parameter Graph Transformer designed for spatial transcriptomics data analysis.
Enables prediction of cell-cell interactions, tissue organization, and pathway activities.
"""

__version__ = "2.0.0"  # Generation 2 with robustness features
__author__ = "Daniel Schmidt"
__email__ = "daniel@spatial-omics.ai"

# Core components
from .models import SpatialGraphTransformer
from .data import VisiumDataset, SlideSeqDataset, XeniumDataset, MERFISHDataset
from .tasks import CellTypeClassifier, InteractionPredictor, PathwayAnalyzer
from .training import FineTuner, DistributedTrainer
from .inference import EfficientInference, BatchInference
from .visualization import SpatialPlotter, InteractiveSpatialViewer

# Robustness features (Generation 2)
from .utils import (
    # Enhanced validation
    RobustValidator, ValidationConfig, ValidationResult,
    # Security features
    SecureFileHandler, SecurityConfig, sanitize_user_input,
    # Configuration management
    ConfigManager, ExperimentConfig, create_default_config,
    # Memory management
    MemoryMonitor, memory_managed_operation
)

__all__ = [
    # Core components
    "SpatialGraphTransformer",
    "VisiumDataset",
    "SlideSeqDataset", 
    "XeniumDataset",
    "MERFISHDataset",
    "CellTypeClassifier",
    "InteractionPredictor",
    "PathwayAnalyzer",
    "FineTuner",
    "DistributedTrainer",
    "EfficientInference", 
    "BatchInference",
    "SpatialPlotter",
    "InteractiveSpatialViewer",
    
    # Robustness features (Generation 2)
    "RobustValidator",
    "ValidationConfig",
    "ValidationResult",
    "SecureFileHandler",
    "SecurityConfig",
    "sanitize_user_input",
    "ConfigManager",
    "ExperimentConfig",
    "create_default_config",
    "MemoryMonitor",
    "memory_managed_operation",
]
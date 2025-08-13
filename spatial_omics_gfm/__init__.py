"""
Spatial-Omics GFM: Graph Foundation Model for Spatial Transcriptomics

A billion-parameter Graph Transformer designed for spatial transcriptomics data analysis.
Enables prediction of cell-cell interactions, tissue organization, and pathway activities.
"""

__version__ = "2.0.0"  # Generation 2 with robustness features
__author__ = "Daniel Schmidt"
__email__ = "daniel@spatial-omics.ai"

# Core components (with optional heavy dependencies)
try:
    from .models import SpatialGraphTransformer
    from .data import VisiumDataset, SlideSeqDataset, XeniumDataset, MERFISHDataset
    from .tasks import CellTypeClassifier, InteractionPredictor, PathwayAnalyzer
    from .training import FineTuner, DistributedTrainer
    from .inference import EfficientInference, BatchInference
    from .visualization import SpatialPlotter, InteractiveSpatialViewer
    FULL_FEATURES_AVAILABLE = True
except ImportError:
    # Fallback when heavy dependencies not available
    FULL_FEATURES_AVAILABLE = False
    SpatialGraphTransformer = None
    VisiumDataset = SlideSeqDataset = XeniumDataset = MERFISHDataset = None
    CellTypeClassifier = InteractionPredictor = PathwayAnalyzer = None
    FineTuner = DistributedTrainer = None
    EfficientInference = BatchInference = None
    SpatialPlotter = InteractiveSpatialViewer = None

# Always available core functionality
from .core import (
    SimpleSpatialData,
    SimpleCellTypePredictor,
    SimpleInteractionPredictor,
    create_demo_data,
    run_basic_analysis
)

# Robustness features (Generation 2) - optional
try:
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
except ImportError:
    # Fallback when utils not fully available
    RobustValidator = ValidationConfig = ValidationResult = None
    SecureFileHandler = SecurityConfig = sanitize_user_input = None
    ConfigManager = ExperimentConfig = create_default_config = None
    MemoryMonitor = memory_managed_operation = None

__all__ = [
    # Always available core functionality
    "SimpleSpatialData",
    "SimpleCellTypePredictor",
    "SimpleInteractionPredictor",
    "create_demo_data",
    "run_basic_analysis",
    "FULL_FEATURES_AVAILABLE",
    
    # Optional components (when dependencies available)
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
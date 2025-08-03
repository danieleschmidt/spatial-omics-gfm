"""
Spatial-Omics GFM: Graph Foundation Model for Spatial Transcriptomics

A billion-parameter Graph Transformer designed for spatial transcriptomics data analysis.
Enables prediction of cell-cell interactions, tissue organization, and pathway activities.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@spatial-omics.ai"

from .models import SpatialGraphTransformer
from .data import VisiumDataset, SlideSeqDataset, XeniumDataset, MERFISHDataset
from .tasks import CellTypeClassifier, InteractionPredictor, PathwayAnalyzer
from .training import FineTuner, DistributedTrainer
from .inference import EfficientInference, BatchInference
from .visualization import SpatialPlotter, InteractiveSpatialViewer

__all__ = [
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
]
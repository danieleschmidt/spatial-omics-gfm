"""
Task-specific modules for spatial transcriptomics analysis.

This module contains implementations for various analysis tasks
including cell typing, interaction prediction, and pathway analysis.
"""

from .cell_typing import CellTypeClassifier, HierarchicalCellTypeClassifier, CellTypeConfig
from .interaction_prediction import InteractionPredictor, LigandReceptorPredictor
from .pathway_analysis import PathwayAnalyzer, SpatialPathwayAnalyzer
from .tissue_segmentation import TissueSegmenter, RegionClassifier
from .base import BaseTask, TaskConfig, ClassificationHead, RegressionHead, AttentionHead, MultiTaskHead, UncertaintyHead

__all__ = [
    # Base components
    "BaseTask",
    "TaskConfig",
    "ClassificationHead",
    "RegressionHead", 
    "AttentionHead",
    "MultiTaskHead",
    "UncertaintyHead",
    
    # Cell typing
    "CellTypeClassifier",
    "HierarchicalCellTypeClassifier",
    "CellTypeConfig",
    
    # Interaction prediction
    "InteractionPredictor",
    "LigandReceptorPredictor",
    
    # Pathway analysis
    "PathwayAnalyzer",
    "SpatialPathwayAnalyzer",
    
    # Tissue segmentation
    "TissueSegmenter",
    "RegionClassifier",
]
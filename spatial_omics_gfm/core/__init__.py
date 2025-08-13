"""Core functionality for Spatial-Omics GFM."""

from .simple_example import (
    SimpleSpatialData,
    SimpleCellTypePredictor, 
    SimpleInteractionPredictor,
    create_demo_data,
    run_basic_analysis
)

__all__ = [
    "SimpleSpatialData",
    "SimpleCellTypePredictor",
    "SimpleInteractionPredictor", 
    "create_demo_data",
    "run_basic_analysis"
]
"""
spatial-omics-gfm
=================
Graph Foundation Model for Spatial Transcriptomics.

Processes spatially-resolved gene expression data (Visium, Slide-seq, MERFISH)
by representing tissue sections as spatial k-NN graphs and learning joint
representations of gene expression and tissue topology.

Public API
----------
SpatialSpot               — single spot with (x, y) + expression vector
SpatialGraph              — k-NN graph built from a collection of spots
SpatialGraphTransformer   — graph transformer: expression + spatial position
CellCommunicationInferer  — L-R co-expression scoring over spatial edges
SyntheticSpatialDataset   — synthetic Visium-like data for testing / demo
"""

from .spot import SpatialSpot
from .graph import SpatialGraph
from .model import SpatialGraphTransformer, SpatialPositionEncoding
from .communication import (
    CellCommunicationInferer,
    LRPair,
    CommunicationResult,
)
from .dataset import SyntheticSpatialDataset

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "SpatialSpot",
    "SpatialGraph",
    "SpatialGraphTransformer",
    "SpatialPositionEncoding",
    "CellCommunicationInferer",
    "LRPair",
    "CommunicationResult",
    "SyntheticSpatialDataset",
]

"""
Research modules for Spatial-Omics GFM.
Contains novel algorithms, benchmarking frameworks, and experimental features.
"""

from .novel_attention import (
    AdaptiveSpatialAttention,
    SpatialRelativePositionEncoder,
    LocalDensityEncoder,
    HierarchicalSpatialAttention,
    ContextualSpatialAttention,
    create_novel_attention_layer,
    NovelAttentionBenchmark
)

from .benchmarking import (
    BenchmarkConfig,
    SyntheticDataGenerator,
    ModelBenchmark
)

__all__ = [
    "AdaptiveSpatialAttention",
    "SpatialRelativePositionEncoder", 
    "LocalDensityEncoder",
    "HierarchicalSpatialAttention",
    "ContextualSpatialAttention",
    "create_novel_attention_layer",
    "NovelAttentionBenchmark",
    "BenchmarkConfig",
    "SyntheticDataGenerator",
    "ModelBenchmark",
]
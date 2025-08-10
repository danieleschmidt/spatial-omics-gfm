"""
Training modules for Spatial-Omics GFM.

This module contains comprehensive training infrastructure including:
- Fine-tuning for task adaptation
- Distributed training for multi-GPU/multi-node scaling
- Curriculum learning for progressive training
- Contrastive learning for self-supervised pre-training
"""

from .fine_tuning import FineTuner, FineTuningConfig
from .distributed_training import DistributedTrainer, DistributedConfig, launch_distributed_training
from .curriculum_learning import CurriculumTrainer, CurriculumConfig, create_adaptive_curriculum
from .contrastive_learning import ContrastiveTrainer, ContrastiveConfig, SpatialContrastiveDataset

__all__ = [
    # Fine-tuning
    "FineTuner",
    "FineTuningConfig",
    
    # Distributed training
    "DistributedTrainer",
    "DistributedConfig", 
    "launch_distributed_training",
    
    # Curriculum learning
    "CurriculumTrainer",
    "CurriculumConfig",
    "create_adaptive_curriculum",
    
    # Contrastive learning
    "ContrastiveTrainer",
    "ContrastiveConfig",
    "SpatialContrastiveDataset",
]
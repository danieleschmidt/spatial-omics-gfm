"""
Contrastive learning for self-supervised pre-training of Spatial-Omics GFM.

This module implements various contrastive learning strategies tailored for
spatial transcriptomics data, including spatial-aware contrastive objectives.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
import warnings

from ..models.graph_transformer import SpatialGraphTransformer
from ..data.base import BaseSpatialDataset
from ..data.augmentation import SpatialAugmentation

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    # Basic contrastive settings
    method: str = "simclr"  # "simclr", "mocov2", "spatial_contrastive", "cell_type_contrastive"
    temperature: float = 0.07  # Temperature for contrastive loss
    projection_dim: int = 256  # Dimension of projection head
    
    # Batch settings
    batch_size: int = 64
    num_negatives: int = None  # Auto-compute from batch size
    memory_bank_size: int = 4096  # For MoCo-style methods
    momentum: float = 0.999  # Momentum for momentum-based methods
    
    # Augmentation settings
    augmentation_strategy: str = "spatial_aware"  # "spatial_aware", "expression_only", "combined"
    augmentation_strength: float = 0.5
    spatial_noise_std: float = 10.0  # Standard deviation for spatial noise
    expression_noise_std: float = 0.1  # Standard deviation for expression noise
    dropout_prob: float = 0.1  # Probability for random gene dropout
    
    # Spatial-specific settings
    spatial_contrastive_radius: float = 100.0  # Spatial radius for positive pairs
    use_spatial_hierarchy: bool = True  # Use hierarchical spatial contrasts
    hierarchy_levels: List[float] = None  # Different spatial scales
    
    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 100
    warmup_epochs: int = 10
    use_mixed_precision: bool = True
    
    # Loss weighting
    contrastive_weight: float = 1.0
    spatial_weight: float = 0.5  # Weight for spatial contrastive loss
    reconstruction_weight: float = 0.1  # Weight for reconstruction loss (if used)


class SpatialContrastiveDataset(Dataset):
    """Dataset wrapper for contrastive learning with spatial augmentations."""
    
    def __init__(
        self,
        base_dataset: BaseSpatialDataset,
        config: ContrastiveConfig,
        augmentation: Optional[SpatialAugmentation] = None
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            base_dataset: Base spatial transcriptomics dataset
            config: Contrastive learning configuration
            augmentation: Spatial augmentation pipeline
        """
        self.base_dataset = base_dataset
        self.config = config
        
        # Setup augmentation pipeline
        if augmentation is None:
            self.augmentation = SpatialAugmentation(
                spatial_noise_std=config.spatial_noise_std,
                expression_noise_std=config.expression_noise_std,
                dropout_prob=config.dropout_prob
            )
        else:
            self.augmentation = augmentation
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get contrastive sample pair."""
        # Get base sample
        sample = self.base_dataset.get(idx)
        
        # Create two augmented views
        view1 = self._create_augmented_view(sample)
        view2 = self._create_augmented_view(sample)
        
        # Create spatial positive pairs if using spatial contrastive learning
        spatial_positives = []
        if self.config.method == "spatial_contrastive":
            spatial_positives = self._create_spatial_positives(sample)
        
        return {
            'view1': view1,
            'view2': view2,
            'original': sample,
            'spatial_positives': spatial_positives,
            'index': idx
        }
    
    def _create_augmented_view(self, sample: Any) -> Dict[str, torch.Tensor]:
        """Create augmented view of sample."""
        # Apply augmentations based on strategy
        if self.config.augmentation_strategy == "spatial_aware":
            augmented = self.augmentation.apply_spatial_augmentations(sample)
        elif self.config.augmentation_strategy == "expression_only":
            augmented = self.augmentation.apply_expression_augmentations(sample)
        else:  # combined
            augmented = self.augmentation.apply_all_augmentations(sample)
        
        return augmented
    
    def _create_spatial_positives(self, sample: Any) -> List[Dict[str, torch.Tensor]]:
        """Create spatial positive pairs for spatial contrastive learning."""
        if not hasattr(sample, 'pos'):
            return []
        
        coords = sample.pos.numpy() if isinstance(sample.pos, torch.Tensor) else sample.pos
        expression = sample.x.numpy() if isinstance(sample.x, torch.Tensor) else sample.x
        
        # Find spatially nearby cells
        nbrs = NearestNeighbors(radius=self.config.spatial_contrastive_radius).fit(coords)
        distances, indices = nbrs.radius_neighbors(coords)
        
        positives = []
        for i, neighbors in enumerate(indices):
            if len(neighbors) > 1:  # Exclude self
                # Sample a few neighbors as positives
                neighbor_sample_size = min(3, len(neighbors) - 1)
                selected_neighbors = np.random.choice(neighbors[1:], neighbor_sample_size, replace=False)
                
                for neighbor_idx in selected_neighbors:
                    positive_sample = {
                        'x': torch.tensor(expression[neighbor_idx:neighbor_idx+1], dtype=torch.float32),
                        'pos': torch.tensor(coords[neighbor_idx:neighbor_idx+1], dtype=torch.float32),
                        'anchor_idx': i,
                        'neighbor_idx': neighbor_idx
                    }
                    positives.append(positive_sample)
        
        return positives


class ContrastiveLoss(nn.Module):
    """Contrastive loss implementations for spatial transcriptomics."""
    
    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
    
    def simclr_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """SimCLR contrastive loss."""
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create positive mask
        positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z.device)
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = 1
            positive_mask[batch_size + i, i] = 1
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim * (1 - torch.eye(2 * batch_size, device=z.device)), dim=1)
        
        positive_exp = torch.sum(exp_sim * positive_mask, dim=1)
        loss = -torch.log(positive_exp / sum_exp)
        
        return loss.mean()
    
    def spatial_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        spatial_coords: torch.Tensor,
        spatial_positives: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Spatial-aware contrastive loss."""
        if not spatial_positives:
            return torch.tensor(0.0, device=embeddings.device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        for positive in spatial_positives:
            anchor_idx = positive['anchor_idx']
            neighbor_idx = positive['neighbor_idx']
            
            # Get anchor and positive embeddings
            anchor_emb = embeddings[anchor_idx:anchor_idx+1]
            
            # Compute positive embedding (would need to encode the positive sample)
            # For simplification, using neighbor embedding from batch
            if neighbor_idx < len(embeddings):
                positive_emb = embeddings[neighbor_idx:neighbor_idx+1]
                positive_emb = F.normalize(positive_emb, dim=1)
                
                # Positive similarity
                pos_sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature
                
                # Negative similarities (all other samples)
                neg_mask = torch.ones(len(embeddings), device=embeddings.device, dtype=torch.bool)
                neg_mask[anchor_idx] = False
                neg_mask[neighbor_idx] = False
                
                if neg_mask.sum() > 0:
                    neg_embeddings = embeddings[neg_mask]
                    neg_sim = torch.matmul(anchor_emb, neg_embeddings.T) / self.temperature
                    
                    # Compute contrastive loss
                    logits = torch.cat([pos_sim.flatten(), neg_sim.flatten()])
                    labels = torch.zeros(len(logits), device=logits.device, dtype=torch.long)
                    
                    loss = F.cross_entropy(logits.unsqueeze(0), labels[:1])
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def hierarchical_spatial_loss(
        self,
        embeddings: torch.Tensor,
        spatial_coords: torch.Tensor,
        hierarchy_levels: List[float]
    ) -> torch.Tensor:
        """Hierarchical spatial contrastive loss."""
        total_loss = 0.0
        
        for level_radius in hierarchy_levels:
            # Create spatial groups at this hierarchy level
            level_loss = self._compute_level_contrastive_loss(
                embeddings, spatial_coords, level_radius
            )
            total_loss += level_loss
        
        return total_loss / len(hierarchy_levels)
    
    def _compute_level_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        spatial_coords: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """Compute contrastive loss for specific spatial hierarchy level."""
        # Simplified implementation - would need proper spatial grouping
        return torch.tensor(0.0, device=embeddings.device)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, projection_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MomentumEncoder(nn.Module):
    """Momentum encoder for MoCo-style contrastive learning."""
    
    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Create momentum encoder
        self.momentum_encoder = self._create_momentum_copy(encoder)
        
        # Initialize momentum encoder with same weights
        self._update_momentum_encoder(momentum=0.0)
    
    def _create_momentum_copy(self, encoder: nn.Module) -> nn.Module:
        """Create momentum copy of encoder."""
        momentum_encoder = type(encoder)(encoder.config)
        return momentum_encoder
    
    @torch.no_grad()
    def _update_momentum_encoder(self, momentum: Optional[float] = None) -> None:
        """Update momentum encoder weights."""
        if momentum is None:
            momentum = self.momentum
        
        for param, momentum_param in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            momentum_param.data = momentum * momentum_param.data + (1 - momentum) * param.data
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders."""
        query = self.encoder(x)
        
        with torch.no_grad():
            key = self.momentum_encoder(x)
        
        return query, key


class ContrastiveTrainer:
    """
    Contrastive learning trainer for self-supervised pre-training.
    
    Features:
    - Multiple contrastive learning methods (SimCLR, MoCo, etc.)
    - Spatial-aware contrastive objectives
    - Hierarchical spatial contrasts
    - Mixed precision training
    - Momentum updates for key encoders
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        config: ContrastiveConfig,
        device: Optional[str] = None
    ):
        """
        Initialize contrastive trainer.
        
        Args:
            model: Base model for contrastive learning
            config: Contrastive learning configuration
            device: Device for training
        """
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        # Setup contrastive components
        self.projection_head = ProjectionHead(
            model.config.hidden_dim,
            config.projection_dim
        ).to(self.device)
        
        # Setup loss function
        self.contrastive_loss = ContrastiveLoss(config)
        
        # Setup momentum encoder if needed
        self.momentum_encoder = None
        if config.method in ["mocov2"]:
            self.momentum_encoder = MomentumEncoder(model, config.momentum)
        
        # Setup optimizer
        parameters = list(self.model.parameters()) + list(self.projection_head.parameters())
        if self.momentum_encoder:
            parameters += list(self.momentum_encoder.parameters())
        
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.epoch = 0
        self.training_history = []
        
        logger.info(f"Initialized ContrastiveTrainer with method: {config.method}")
    
    def train_contrastive(
        self,
        dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model using contrastive learning.
        
        Args:
            dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for checkpoints
            
        Returns:
            Training results
        """
        # Setup output directory
        if output_dir is None:
            output_dir = "./contrastive_training"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting contrastive learning training")
        
        # Create contrastive dataset
        contrastive_dataset = SpatialContrastiveDataset(dataset, self.config)
        
        # Create data loader
        train_loader = DataLoader(
            contrastive_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_contrastive_dataset = SpatialContrastiveDataset(val_dataset, self.config)
            val_loader = DataLoader(
                val_contrastive_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        results = {
            'config': self.config.__dict__,
            'epochs': [],
            'best_loss': float('inf')
        }
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch(train_loader)
            
            # Validation epoch
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record epoch results
            epoch_result = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            results['epochs'].append(epoch_result)
            self.training_history.append(epoch_result)
            
            # Update best loss
            current_loss = val_metrics.get('total_loss', train_metrics.get('total_loss', float('inf')))
            if current_loss < results['best_loss']:
                results['best_loss'] = current_loss
                
                # Save best model
                best_model_path = output_dir / "best_contrastive_model.pt"
                self._save_checkpoint(best_model_path, epoch, train_metrics, val_metrics, is_best=True)
            
            # Logging
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics.get('total_loss', 0):.4f}, "
                f"Val Loss: {val_metrics.get('total_loss', 0):.4f}, "
                f"LR: {epoch_result['learning_rate']:.2e}"
            )
            
            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
                self._save_checkpoint(checkpoint_path, epoch, train_metrics, val_metrics)
        
        # Save final model
        final_model_path = output_dir / "final_contrastive_model.pt"
        self._save_checkpoint(final_model_path, self.config.epochs - 1, train_metrics, val_metrics)
        
        # Save training results
        results_path = output_dir / "contrastive_results.json"
        with open(results_path, 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)
        
        logger.info(f"Contrastive training completed. Results saved to {output_dir}")
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.projection_head.train()
        
        if self.momentum_encoder:
            self.momentum_encoder.train()
        
        total_loss = 0.0
        contrastive_loss_sum = 0.0
        spatial_loss_sum = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch_view1 = batch['view1']
            batch_view2 = batch['view2']
            
            # Move to device
            for key in batch_view1:
                if isinstance(batch_view1[key], torch.Tensor):
                    batch_view1[key] = batch_view1[key].to(self.device)
                    batch_view2[key] = batch_view2[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_contrastive_loss(batch_view1, batch_view2, batch)
            else:
                loss_dict = self._compute_contrastive_loss(batch_view1, batch_view2, batch)
            
            total_batch_loss = loss_dict['total_loss']
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                self.optimizer.step()
            
            # Update momentum encoder if needed
            if self.momentum_encoder:
                self.momentum_encoder._update_momentum_encoder()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            contrastive_loss_sum += loss_dict.get('contrastive_loss', 0)
            spatial_loss_sum += loss_dict.get('spatial_loss', 0)
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': contrastive_loss_sum / num_batches,
            'spatial_loss': spatial_loss_sum / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.projection_head.eval()
        
        if self.momentum_encoder:
            self.momentum_encoder.eval()
        
        total_loss = 0.0
        contrastive_loss_sum = 0.0
        spatial_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_view1 = batch['view1']
                batch_view2 = batch['view2']
                
                # Move to device
                for key in batch_view1:
                    if isinstance(batch_view1[key], torch.Tensor):
                        batch_view1[key] = batch_view1[key].to(self.device)
                        batch_view2[key] = batch_view2[key].to(self.device)
                
                # Forward pass
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss_dict = self._compute_contrastive_loss(batch_view1, batch_view2, batch)
                else:
                    loss_dict = self._compute_contrastive_loss(batch_view1, batch_view2, batch)
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                contrastive_loss_sum += loss_dict.get('contrastive_loss', 0)
                spatial_loss_sum += loss_dict.get('spatial_loss', 0)
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': contrastive_loss_sum / num_batches,
            'spatial_loss': spatial_loss_sum / num_batches
        }
    
    def _compute_contrastive_loss(
        self,
        view1: Dict[str, torch.Tensor],
        view2: Dict[str, torch.Tensor],
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss."""
        # Get embeddings for both views
        embeddings1 = self.model.encode(
            gene_expression=view1['x'],
            spatial_coords=view1['pos'],
            edge_index=view1.get('edge_index'),
            edge_attr=view1.get('edge_attr')
        )
        
        embeddings2 = self.model.encode(
            gene_expression=view2['x'],
            spatial_coords=view2['pos'],
            edge_index=view2.get('edge_index'),
            edge_attr=view2.get('edge_attr')
        )
        
        # Project embeddings
        proj1 = self.projection_head(embeddings1)
        proj2 = self.projection_head(embeddings2)
        
        # Compute main contrastive loss
        if self.config.method == "simclr":
            contrastive_loss = self.contrastive_loss.simclr_loss(proj1, proj2)
        else:
            # Default to SimCLR
            contrastive_loss = self.contrastive_loss.simclr_loss(proj1, proj2)
        
        loss_dict = {
            'contrastive_loss': contrastive_loss,
            'total_loss': contrastive_loss * self.config.contrastive_weight
        }
        
        # Add spatial contrastive loss if applicable
        if (self.config.method == "spatial_contrastive" and 
            'spatial_positives' in batch and 
            len(batch['spatial_positives']) > 0):
            
            spatial_loss = self.contrastive_loss.spatial_contrastive_loss(
                embeddings1, view1['pos'], batch['spatial_positives']
            )
            
            loss_dict['spatial_loss'] = spatial_loss
            loss_dict['total_loss'] += spatial_loss * self.config.spatial_weight
        
        # Add hierarchical spatial loss if configured
        if (self.config.use_spatial_hierarchy and 
            self.config.hierarchy_levels is not None):
            
            hierarchical_loss = self.contrastive_loss.hierarchical_spatial_loss(
                embeddings1, view1['pos'], self.config.hierarchy_levels
            )
            
            loss_dict['hierarchical_loss'] = hierarchical_loss
            loss_dict['total_loss'] += hierarchical_loss * self.config.spatial_weight * 0.5
        
        return loss_dict
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': self.training_history,
            'is_best': is_best
        }
        
        if self.momentum_encoder:
            checkpoint['momentum_encoder_state_dict'] = self.momentum_encoder.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved {'best ' if is_best else ''}checkpoint: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load momentum encoder if present
        if self.momentum_encoder and 'momentum_encoder_state_dict' in checkpoint:
            self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder_state_dict'])
        
        # Load scaler if present
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def extract_representations(
        self,
        dataset: BaseSpatialDataset,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Extract learned representations for downstream tasks.
        
        Args:
            dataset: Dataset to extract representations from
            batch_size: Batch size for extraction
            
        Returns:
            Extracted representations
        """
        logger.info("Extracting learned representations")
        
        self.model.eval()
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                if hasattr(batch, 'x'):
                    batch = batch.to(self.device)
                
                # Extract embeddings
                embeddings = self.model.encode(
                    gene_expression=batch.x,
                    spatial_coords=batch.pos,
                    edge_index=getattr(batch, 'edge_index', None),
                    edge_attr=getattr(batch, 'edge_attr', None),
                    batch=getattr(batch, 'batch', None)
                )
                
                representations.append(embeddings.cpu().numpy())
        
        # Concatenate all representations
        all_representations = np.concatenate(representations, axis=0)
        
        logger.info(f"Extracted representations shape: {all_representations.shape}")
        return all_representations
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj


def create_spatial_hierarchy(
    spatial_coords: np.ndarray,
    num_levels: int = 3,
    base_radius: float = 50.0
) -> List[float]:
    """
    Create spatial hierarchy levels for hierarchical contrastive learning.
    
    Args:
        spatial_coords: Spatial coordinates of cells
        num_levels: Number of hierarchy levels
        base_radius: Base spatial radius
        
    Returns:
        List of spatial radii for different hierarchy levels
    """
    # Compute spatial extent
    spatial_extent = np.max(spatial_coords, axis=0) - np.min(spatial_coords, axis=0)
    max_extent = np.max(spatial_extent)
    
    # Create logarithmic spacing of radii
    min_radius = base_radius
    max_radius = max_extent / 4  # Cover quarter of the spatial extent
    
    if num_levels == 1:
        return [base_radius]
    
    # Logarithmic spacing
    radii = np.logspace(
        np.log10(min_radius),
        np.log10(max_radius),
        num_levels
    )
    
    return radii.tolist()


def evaluate_contrastive_representations(
    representations: np.ndarray,
    labels: np.ndarray,
    method: str = "linear_probe"
) -> Dict[str, float]:
    """
    Evaluate quality of contrastive representations.
    
    Args:
        representations: Learned representations
        labels: Ground truth labels
        method: Evaluation method
        
    Returns:
        Evaluation metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    if method == "linear_probe":
        # Linear probe evaluation
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'linear_probe_accuracy': accuracy,
            'linear_probe_f1': f1
        }
    
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
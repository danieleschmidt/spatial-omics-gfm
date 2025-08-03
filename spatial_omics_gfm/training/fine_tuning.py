"""
Fine-tuning functionality for pre-trained models.

This module provides comprehensive fine-tuning capabilities for
adapting pre-trained models to specific tasks and datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple
import json
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass
import warnings

from ..models import SpatialGraphTransformer
from ..tasks import CellTypeClassifier, InteractionPredictor, PathwayAnalyzer
from ..data.base import BaseSpatialDataset


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    task: str = "cell_typing"
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    freeze_backbone: bool = False
    freeze_layers: Optional[List[int]] = None
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32


class FineTuner:
    """
    Fine-tuning engine for pre-trained Spatial-Omics GFM models.
    
    Supports various fine-tuning strategies including full fine-tuning,
    frozen backbone, layer-specific freezing, and LoRA adaptation.
    """
    
    def __init__(
        self,
        base_model: SpatialGraphTransformer,
        task: str,
        config: Optional[FineTuningConfig] = None,
        device: Optional[str] = None
    ):
        self.base_model = base_model
        self.task = task
        self.config = config or FineTuningConfig(task=task)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize task-specific components
        self.task_head = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.training_history = []
    
    def setup_task_head(
        self,
        dataset: BaseSpatialDataset,
        task_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Setup task-specific head based on dataset and task type.
        
        Args:
            dataset: Training dataset
            task_config: Task-specific configuration
        """
        hidden_dim = self.base_model.config.hidden_dim
        
        if self.task == "cell_typing":
            # Get unique cell types from dataset
            if hasattr(dataset.adata, 'obs') and 'cell_type' in dataset.adata.obs:
                cell_types = dataset.adata.obs['cell_type'].unique().tolist()
            else:
                # Default cell types for demonstration
                cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Macrophage', 'Dendritic', 'Neutrophil']
            
            from ..tasks.cell_typing import CellTypeConfig, CellTypeClassifier
            
            config = CellTypeConfig(
                hidden_dim=hidden_dim,
                num_classes=len(cell_types),
                **(task_config or {})
            )
            
            self.task_head = CellTypeClassifier(config, cell_types)
            
        elif self.task == "interactions":
            from ..tasks.interaction_prediction import InteractionConfig, InteractionPredictor
            
            config = InteractionConfig(
                hidden_dim=hidden_dim,
                **(task_config or {})
            )
            
            self.task_head = InteractionPredictor(config)
            
        elif self.task == "pathways":
            from ..tasks.pathway_analysis import PathwayConfig, PathwayAnalyzer
            
            config = PathwayConfig(
                hidden_dim=hidden_dim,
                **(task_config or {})
            )
            
            self.task_head = PathwayAnalyzer(config)
            
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        # Move task head to device
        self.task_head = self.task_head.to(self.device)
    
    def setup_training(self) -> None:
        """Setup optimizer, scheduler, and training components."""
        # Determine which parameters to optimize
        if self.config.freeze_backbone:
            # Only optimize task head
            parameters = list(self.task_head.parameters())
            self.logger.info("Freezing backbone, only training task head")
            
        elif self.config.freeze_layers:
            # Freeze specific layers
            parameters = []
            
            # Add non-frozen backbone parameters
            for name, param in self.base_model.named_parameters():
                layer_frozen = False
                for freeze_layer in self.config.freeze_layers:
                    if f"layers.{freeze_layer}" in name:
                        param.requires_grad = False
                        layer_frozen = True
                        break
                
                if not layer_frozen:
                    parameters.append(param)
            
            # Add task head parameters
            parameters.extend(list(self.task_head.parameters()))
            
            self.logger.info(f"Frozen layers: {self.config.freeze_layers}")
            
        else:
            # Full fine-tuning
            parameters = list(self.base_model.parameters()) + list(self.task_head.parameters())
            self.logger.info("Full fine-tuning enabled")
        
        # Setup LoRA if requested
        if self.config.use_lora:
            parameters = self._setup_lora()
            self.logger.info(f"LoRA enabled: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        if self.config.use_scheduler:
            if self.config.scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs
                )
            elif self.config.scheduler_type == "linear":
                total_steps = self.config.epochs * 100  # Approximate
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=total_steps
                )
            else:
                warnings.warn(f"Unknown scheduler: {self.config.scheduler_type}")
                self.scheduler = None
    
    def _setup_lora(self) -> List[torch.nn.Parameter]:
        """
        Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning.
        
        Returns:
            List of LoRA parameters to optimize
        """
        lora_parameters = []
        
        # Apply LoRA to attention layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                # Create LoRA matrices
                lora_A = nn.Parameter(
                    torch.randn(self.config.lora_rank, module.in_features) * 0.02
                )
                lora_B = nn.Parameter(
                    torch.zeros(module.out_features, self.config.lora_rank)
                )
                
                # Register as module parameters
                setattr(module, 'lora_A', lora_A)
                setattr(module, 'lora_B', lora_B)
                
                # Freeze original weights
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                # Add LoRA parameters to optimization
                lora_parameters.extend([lora_A, lora_B])
                
                # Monkey patch forward method
                original_forward = module.forward
                
                def new_forward(x, original_forward=original_forward, module=module):
                    # Original computation
                    original_output = original_forward(x)
                    
                    # LoRA computation
                    lora_output = (x @ module.lora_A.T) @ module.lora_B.T
                    
                    # Scale and combine
                    scaling = self.config.lora_alpha / self.config.lora_rank
                    return original_output + scaling * lora_output
                
                module.forward = new_forward
        
        # Add task head parameters
        lora_parameters.extend(list(self.task_head.parameters()))
        
        return lora_parameters
    
    def fine_tune(
        self,
        train_dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset] = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> SpatialGraphTransformer:
        """
        Run fine-tuning process.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory to save checkpoints and final model
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Fine-tuned model
        """
        # Setup output directory
        if output_dir is None:
            output_dir = f"./finetune_{self.task}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup task head if not already done
        if self.task_head is None:
            self.setup_task_head(train_dataset)
        
        # Setup training components
        self.setup_training()
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Create data loaders
        train_loader = train_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = val_dataset.get_dataloader(
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Training loop
        self.logger.info(f"Starting fine-tuning for {self.config.epochs} epochs")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            
            self.training_history.append(epoch_metrics)
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
                f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
                f"LR: {epoch_metrics['lr']:.2e}"
            )
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_steps // 100) == 0:
                checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}.pt"
                self._save_checkpoint(checkpoint_path)
            
            # Save best model
            current_metric = val_metrics.get('loss', train_metrics.get('loss', float('inf')))
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                best_model_path = output_dir / "best_model.pt"
                self._save_model(best_model_path)
        
        # Save final model and training history
        final_model_path = output_dir / "final_model.pt"
        self._save_model(final_model_path)
        
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Fine-tuning completed. Models saved to {output_dir}")
        
        return self._create_complete_model()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.base_model.train()
        self.task_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass through base model
            with torch.set_grad_enabled(True):
                embeddings = self.base_model.encode(
                    gene_expression=batch.x,
                    spatial_coords=batch.pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
                
                # Forward pass through task head
                predictions = self.task_head(
                    embeddings,
                    edge_index=batch.edge_index,
                    spatial_coords=batch.pos
                )
                
                # Compute loss
                if hasattr(batch, 'y'):
                    targets = batch.y
                elif hasattr(batch, 'cell_type'):
                    # Convert cell type names to indices
                    targets = torch.tensor([
                        self.task_head.cell_type_names.index(ct) 
                        if ct in self.task_head.cell_type_names else 0
                        for ct in batch.cell_type
                    ], device=self.device)
                else:
                    # Skip batch if no targets available
                    continue
                
                loss = self.task_head.compute_loss(predictions, targets)
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.base_model.eval()
        self.task_head.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                embeddings = self.base_model.encode(
                    gene_expression=batch.x,
                    spatial_coords=batch.pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
                
                predictions = self.task_head(
                    embeddings,
                    edge_index=batch.edge_index,
                    spatial_coords=batch.pos
                )
                
                # Compute loss
                if hasattr(batch, 'y'):
                    targets = batch.y
                elif hasattr(batch, 'cell_type'):
                    targets = torch.tensor([
                        self.task_head.cell_type_names.index(ct) 
                        if ct in self.task_head.cell_type_names else 0
                        for ct in batch.cell_type
                    ], device=self.device)
                else:
                    continue
                
                loss = self.task_head.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'base_model_state_dict': self.base_model.state_dict(),
            'task_head_state_dict': self.task_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint['training_history']
        
        self.base_model.load_state_dict(checkpoint['base_model_state_dict'])
        self.task_head.load_state_dict(checkpoint['task_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _save_model(self, path: Path) -> None:
        """Save the complete fine-tuned model."""
        complete_model = self._create_complete_model()
        
        model_data = {
            'model_state_dict': complete_model.state_dict(),
            'model_config': complete_model.config.__dict__,
            'task': self.task,
            'task_head_state_dict': self.task_head.state_dict(),
            'fine_tuning_config': self.config.__dict__
        }
        
        torch.save(model_data, path)
    
    def _create_complete_model(self) -> SpatialGraphTransformer:
        """Create a complete model with integrated task head."""
        # For now, return the base model
        # In a full implementation, you might create a wrapper that includes the task head
        return self.base_model
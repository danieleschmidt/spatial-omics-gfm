"""
Base classes for task-specific analysis modules.

This module provides foundation classes that all task-specific
modules inherit from, ensuring consistent interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Base configuration for tasks."""
    hidden_dim: int = 1024
    num_classes: int = 10
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "gelu"
    confidence_threshold: float = 0.5


class BaseTask(nn.Module, ABC):
    """
    Abstract base class for all task-specific modules.
    
    This class defines the common interface that all task modules
    must implement for consistency across different analysis types.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_classes = config.num_classes
        
    @abstractmethod
    def forward(
        self, 
        embeddings: torch.Tensor, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the task module.
        
        Args:
            embeddings: Node embeddings from the foundation model
            **kwargs: Task-specific additional inputs
            
        Returns:
            Dictionary containing task predictions and metadata
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional loss computation arguments
            
        Returns:
            Loss tensor
        """
        pass
    
    def predict(
        self,
        embeddings: torch.Tensor,
        return_probabilities: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions from embeddings.
        
        Args:
            embeddings: Node embeddings
            return_probabilities: Whether to return class probabilities
            **kwargs: Additional prediction arguments
            
        Returns:
            Predictions, optionally with probabilities
        """
        with torch.no_grad():
            output = self.forward(embeddings, **kwargs)
            
            if 'logits' in output:
                probabilities = F.softmax(output['logits'], dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                if return_probabilities:
                    return predictions, probabilities
                return predictions
            
            elif 'predictions' in output:
                if return_probabilities and 'probabilities' in output:
                    return output['predictions'], output['probabilities']
                return output['predictions']
            
            else:
                raise ValueError("Output must contain 'logits' or 'predictions'")
    
    def get_parameter_count(self) -> int:
        """Get the number of parameters in the task module."""
        return sum(p.numel() for p in self.parameters())
    
    def _get_embeddings(
        self, 
        adata, 
        foundation_model=None
    ) -> torch.Tensor:
        """
        Get embeddings from foundation model or use pre-computed embeddings.
        
        Args:
            adata: AnnData object with spatial transcriptomics data
            foundation_model: Optional foundation model for computing embeddings
            
        Returns:
            Cell embeddings tensor
        """
        # If embeddings are already computed and stored
        if hasattr(adata, 'obsm') and 'X_spatial_gfm' in adata.obsm:
            return torch.tensor(adata.obsm['X_spatial_gfm'], dtype=torch.float32)
        
        # If foundation model is provided, compute embeddings
        if foundation_model is not None:
            with torch.no_grad():
                # Get expression data
                if hasattr(adata.X, 'toarray'):
                    x = torch.tensor(adata.X.toarray(), dtype=torch.float32)
                else:
                    x = torch.tensor(adata.X, dtype=torch.float32)
                
                # Get spatial coordinates
                spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32)
                
                # Get spatial graph if available
                edge_index = None
                edge_attr = None
                if 'spatial_graph' in adata.uns:
                    edge_index = torch.tensor(adata.uns['spatial_graph']['edge_index'], dtype=torch.long)
                    if 'edge_attr' in adata.uns['spatial_graph']:
                        edge_attr = torch.tensor(adata.uns['spatial_graph']['edge_attr'], dtype=torch.float32)
                
                # Compute embeddings using foundation model
                embeddings = foundation_model(
                    x=x,
                    spatial_coords=spatial_coords,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
                
                return embeddings
        
        # Fallback: use raw expression as "embeddings"
        if hasattr(adata.X, 'toarray'):
            return torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            return torch.tensor(adata.X, dtype=torch.float32)
    
    def evaluate(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate task performance using specified metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        if metrics is None:
            metrics = ['accuracy']
        
        results = {}
        
        # Get prediction labels
        if 'predictions' in predictions:
            pred_labels = predictions['predictions']
        elif 'logits' in predictions:
            pred_labels = torch.argmax(predictions['logits'], dim=-1)
        else:
            raise ValueError("No predictions or logits found in output")
        
        # Convert to numpy for metric computation
        pred_labels_np = pred_labels.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = float(np.mean(pred_labels_np == targets_np))
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                results[metric] = precision_score(targets_np, pred_labels_np, average='macro', zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                results[metric] = recall_score(targets_np, pred_labels_np, average='macro', zero_division=0)
            elif metric == 'f1':
                from sklearn.metrics import f1_score
                results[metric] = f1_score(targets_np, pred_labels_np, average='macro', zero_division=0)
        
        return results
    
    def save_task_head(self, filepath: str):
        """Save task-specific parameters."""
        torch.save(self.state_dict(), filepath)
    
    def load_task_head(self, filepath: str):
        """Load task-specific parameters."""
        self.load_state_dict(torch.load(filepath, map_location='cpu'))


class ClassificationHead(nn.Module):
    """
    Standard classification head for various tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        return self.classifier(x)


class RegressionHead(nn.Module):
    """
    Standard regression head for continuous predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "gelu",
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "softplus":
            layers.append(nn.Softplus())
        
        self.regressor = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        return self.regressor(x)


class AttentionHead(nn.Module):
    """
    Attention-based head for learning important features.
    """
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0
        
        # Attention projections
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        
        # Output projection
        self.out_proj = nn.Linear(attention_dim, input_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention head.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.query_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        k = self.key_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        v = self.value_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.attention_dim
        )
        
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)  # Average over heads


class MultiTaskHead(nn.Module):
    """
    Multi-task learning head for joint prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict[str, Any]],
        shared_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_names = list(task_configs.keys())
        
        # Shared layers
        if shared_hidden_dims is None:
            shared_hidden_dims = [input_dim]
        
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                head = ClassificationHead(
                    input_dim=prev_dim,
                    num_classes=config['num_classes'],
                    dropout=dropout
                )
            elif config['type'] == 'regression':
                head = RegressionHead(
                    input_dim=prev_dim,
                    output_dim=config.get('output_dim', 1),
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown task type: {config['type']}")
            
            self.task_heads[task_name] = head
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions for each task
        """
        # Shared feature processing
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
        
        return outputs


class UncertaintyHead(nn.Module):
    """
    Head that provides uncertainty estimates with predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        uncertainty_method: str = "evidential",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.uncertainty_method = uncertainty_method
        
        if uncertainty_method == "evidential":
            # Evidential learning - predict Dirichlet parameters
            self.evidence_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, num_classes),
                nn.Softplus()  # Ensure positive evidence
            )
        
        elif uncertainty_method == "ensemble":
            # Multiple prediction heads for ensemble
            self.ensemble_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, num_classes)
                ) for _ in range(5)  # 5 ensemble members
            ])
        
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if self.uncertainty_method == "evidential":
            evidence = self.evidence_head(x)  # Dirichlet parameters
            alpha = evidence + 1  # Add 1 to ensure alpha > 0
            
            # Compute predictions
            dirichlet_strength = alpha.sum(dim=-1, keepdim=True)
            probabilities = alpha / dirichlet_strength
            predictions = torch.argmax(probabilities, dim=-1)
            
            # Compute uncertainty (higher is more uncertain)
            uncertainty = self.num_classes / dirichlet_strength.squeeze()
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainty': uncertainty,
                'evidence': evidence,
                'alpha': alpha
            }
        
        elif self.uncertainty_method == "ensemble":
            # Get predictions from all ensemble members
            ensemble_logits = []
            for head in self.ensemble_heads:
                logits = head(x)
                ensemble_logits.append(logits)
            
            ensemble_logits = torch.stack(ensemble_logits, dim=0)  # [num_ensemble, batch_size, num_classes]
            
            # Compute mean and variance
            mean_logits = ensemble_logits.mean(dim=0)
            var_logits = ensemble_logits.var(dim=0)
            
            # Convert to probabilities
            probabilities = F.softmax(mean_logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            # Uncertainty from variance (average across classes)
            uncertainty = var_logits.mean(dim=-1)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainty': uncertainty,
                'ensemble_logits': ensemble_logits
            }
"""
Cell type classification tasks for spatial transcriptomics.

This module implements various approaches to cell type prediction
with spatial context and hierarchical classification support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from dataclasses import dataclass

from .base import BaseTask, TaskConfig, ClassificationHead, UncertaintyHead


@dataclass
class CellTypeConfig(TaskConfig):
    """Configuration for cell type classification."""
    cell_type_hierarchy: Optional[Dict[str, List[str]]] = None
    use_spatial_context: bool = True
    spatial_context_radius: float = 100.0
    use_uncertainty: bool = True
    uncertainty_method: str = "evidential"
    class_weights: Optional[List[float]] = None


class CellTypeClassifier(BaseTask):
    """
    Standard cell type classifier with spatial context awareness.
    
    This classifier predicts cell types using both gene expression
    embeddings and spatial neighborhood information.
    """
    
    def __init__(
        self,
        config: CellTypeConfig,
        cell_type_names: List[str]
    ):
        super().__init__(config)
        
        self.cell_type_names = cell_type_names
        self.num_cell_types = len(cell_type_names)
        self.config = config
        
        # Main classification head
        if config.use_uncertainty:
            self.classifier = UncertaintyHead(
                input_dim=config.hidden_dim,
                num_classes=self.num_cell_types,
                uncertainty_method=config.uncertainty_method
            )
        else:
            self.classifier = ClassificationHead(
                input_dim=config.hidden_dim,
                num_classes=self.num_cell_types,
                dropout=config.dropout,
                activation=config.activation
            )
        
        # Spatial context encoder (if enabled)
        if config.use_spatial_context:
            self.spatial_context_encoder = SpatialContextEncoder(
                hidden_dim=config.hidden_dim,
                context_radius=config.spatial_context_radius
            )
            
            # Fusion layer for combining individual and context features
            self.context_fusion = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        
        # Class weights for imbalanced datasets
        if config.class_weights is not None:
            self.register_buffer(
                'class_weights',
                torch.FloatTensor(config.class_weights)
            )
        else:
            self.class_weights = None
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cell type classification.
        
        Args:
            embeddings: Cell embeddings [num_cells, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            spatial_coords: Spatial coordinates [num_cells, 2]
            
        Returns:
            Dictionary with predictions and metadata
        """
        features = embeddings
        
        # Add spatial context if enabled
        if self.config.use_spatial_context and edge_index is not None:
            context_features = self.spatial_context_encoder(
                embeddings, edge_index, spatial_coords
            )
            
            # Fuse individual and context features
            combined_features = torch.cat([features, context_features], dim=-1)
            features = self.context_fusion(combined_features)
        
        # Get predictions
        if self.config.use_uncertainty:
            output = self.classifier(features)
            output['cell_type_names'] = self.cell_type_names
        else:
            logits = self.classifier(features)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            output = {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': predictions,
                'cell_type_names': self.cell_type_names
            }
        
        return output
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth cell type indices
            
        Returns:
            Loss tensor
        """
        if self.config.use_uncertainty and 'alpha' in predictions:
            # Evidential loss for uncertainty estimation
            alpha = predictions['alpha']
            dirichlet_strength = alpha.sum(dim=-1)
            
            # Convert targets to one-hot
            targets_one_hot = F.one_hot(targets, num_classes=self.num_cell_types).float()
            
            # Likelihood term
            likelihood = torch.sum(targets_one_hot * (torch.digamma(alpha) - torch.digamma(dirichlet_strength.unsqueeze(-1))), dim=-1)
            
            # KL divergence regularization
            kl_div = torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(dirichlet_strength.unsqueeze(-1))), dim=-1)
            
            # Total loss (negative log-likelihood + KL)
            loss = -likelihood.mean() + 0.1 * kl_div.mean()
            
        else:
            # Standard cross-entropy loss
            logits = predictions['logits']
            
            if self.class_weights is not None:
                loss = F.cross_entropy(logits, targets, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def predict_with_confidence(
        self,
        embeddings: torch.Tensor,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict cell types with confidence filtering.
        
        Args:
            embeddings: Cell embeddings
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Predictions with confidence scores
        """
        with torch.no_grad():
            output = self.forward(embeddings, **kwargs)
            
            if 'uncertainty' in output:
                # Use uncertainty for confidence (lower uncertainty = higher confidence)
                confidence = 1.0 / (1.0 + output['uncertainty'])
            else:
                # Use max probability as confidence
                confidence = torch.max(output['probabilities'], dim=-1)[0]
            
            # Filter predictions by confidence
            high_confidence_mask = confidence >= confidence_threshold
            
            # Create output with confidence information
            result = {
                'predictions': output['predictions'],
                'probabilities': output['probabilities'],
                'confidence': confidence,
                'high_confidence_mask': high_confidence_mask,
                'cell_type_names': self.cell_type_names
            }
            
            if 'uncertainty' in output:
                result['uncertainty'] = output['uncertainty']
            
            return result
    
    def predict_from_adata(
        self,
        adata,
        foundation_model=None,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict cell types from AnnData object.
        
        Args:
            adata: AnnData object with spatial transcriptomics data
            foundation_model: Optional foundation model for computing embeddings
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Cell type predictions with confidence scores
        """
        # Get embeddings
        embeddings = self._get_embeddings(adata, foundation_model)
        
        # Get additional inputs if available
        edge_index = None
        spatial_coords = None
        
        if 'spatial_graph' in adata.uns:
            edge_index = torch.tensor(adata.uns['spatial_graph']['edge_index'], dtype=torch.long)
        
        if 'spatial' in adata.obsm:
            spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32)
        
        # Make predictions
        return self.predict_with_confidence(
            embeddings=embeddings,
            edge_index=edge_index,
            spatial_coords=spatial_coords,
            confidence_threshold=confidence_threshold,
            **kwargs
        )


class HierarchicalCellTypeClassifier(BaseTask):
    """
    Hierarchical cell type classifier that respects cell type taxonomy.
    
    This classifier predicts cell types at multiple levels of the
    cell type hierarchy (e.g., major type -> subtype -> state).
    """
    
    def __init__(
        self,
        config: CellTypeConfig,
        cell_type_hierarchy: Dict[str, Dict[str, List[str]]]
    ):
        super().__init__(config)
        
        self.hierarchy = cell_type_hierarchy
        self.config = config
        
        # Build hierarchy structure
        self._build_hierarchy_structure()
        
        # Create classifiers for each level
        self.level_classifiers = nn.ModuleDict()
        
        for level_name, classes in self.hierarchy_levels.items():
            self.level_classifiers[level_name] = ClassificationHead(
                input_dim=config.hidden_dim,
                num_classes=len(classes),
                dropout=config.dropout,
                activation=config.activation
            )
        
        # Consistency regularization weight
        self.consistency_weight = 0.1
    
    def _build_hierarchy_structure(self):
        """Build internal hierarchy representation."""
        self.hierarchy_levels = {}
        self.level_to_parent = {}
        self.class_to_index = {}
        
        # Extract all unique classes at each level
        for major_type, subtypes_dict in self.hierarchy.items():
            # Major type level
            if 'major' not in self.hierarchy_levels:
                self.hierarchy_levels['major'] = []
            if major_type not in self.hierarchy_levels['major']:
                self.hierarchy_levels['major'].append(major_type)
            
            for subtype, states in subtypes_dict.items():
                # Subtype level
                if 'subtype' not in self.hierarchy_levels:
                    self.hierarchy_levels['subtype'] = []
                if subtype not in self.hierarchy_levels['subtype']:
                    self.hierarchy_levels['subtype'].append(subtype)
                
                # Record parent relationship
                if subtype not in self.level_to_parent:
                    self.level_to_parent[subtype] = major_type
                
                # State level
                if states:
                    if 'state' not in self.hierarchy_levels:
                        self.hierarchy_levels['state'] = []
                    for state in states:
                        if state not in self.hierarchy_levels['state']:
                            self.hierarchy_levels['state'].append(state)
                        
                        # Record parent relationship
                        if state not in self.level_to_parent:
                            self.level_to_parent[state] = subtype
        
        # Create class-to-index mappings
        for level, classes in self.hierarchy_levels.items():
            self.class_to_index[level] = {
                class_name: idx for idx, class_name in enumerate(classes)
            }
    
    def forward(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for hierarchical classification.
        
        Args:
            embeddings: Cell embeddings [num_cells, hidden_dim]
            
        Returns:
            Dictionary with predictions at each hierarchy level
        """
        output = {}
        
        # Get predictions at each level
        for level_name, classifier in self.level_classifiers.items():
            logits = classifier(embeddings)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            output[f'{level_name}_logits'] = logits
            output[f'{level_name}_probabilities'] = probabilities
            output[f'{level_name}_predictions'] = predictions
        
        # Add hierarchy information
        output['hierarchy_levels'] = self.hierarchy_levels
        output['class_to_index'] = self.class_to_index
        
        return output
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute hierarchical classification loss.
        
        Args:
            predictions: Model predictions at each level
            targets: Ground truth targets at each level
            
        Returns:
            Combined loss with consistency regularization
        """
        total_loss = 0.0
        num_levels = 0
        
        # Classification loss at each level
        for level_name in self.hierarchy_levels.keys():
            if f'{level_name}_logits' in predictions and level_name in targets:
                logits = predictions[f'{level_name}_logits']
                level_targets = targets[level_name]
                
                level_loss = F.cross_entropy(logits, level_targets)
                total_loss += level_loss
                num_levels += 1
        
        # Average classification losses
        if num_levels > 0:
            total_loss /= num_levels
        
        # Consistency regularization
        consistency_loss = self._compute_consistency_loss(predictions, targets)
        total_loss += self.consistency_weight * consistency_loss
        
        return total_loss
    
    def _compute_consistency_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute consistency loss to enforce hierarchy constraints.
        
        Args:
            predictions: Model predictions
            targets: Target labels
            
        Returns:
            Consistency loss tensor
        """
        consistency_loss = 0.0
        
        # Example: Ensure subtype predictions are consistent with major type
        if ('major_probabilities' in predictions and 
            'subtype_probabilities' in predictions):
            
            major_probs = predictions['major_probabilities']
            subtype_probs = predictions['subtype_probabilities']
            
            # For each predicted subtype, check if it's consistent with major type
            # This is a simplified example - full implementation would use
            # the actual hierarchy relationships
            
            # Compute KL divergence between aggregated subtype predictions
            # and major type predictions (placeholder implementation)
            consistency_loss = F.kl_div(
                F.log_softmax(major_probs, dim=-1),
                F.softmax(subtype_probs, dim=-1),
                reduction='batchmean'
            )
        
        return consistency_loss
    
    def predict_hierarchical(
        self,
        embeddings: torch.Tensor,
        return_all_levels: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict cell types at all hierarchy levels.
        
        Args:
            embeddings: Cell embeddings
            return_all_levels: Whether to return predictions at all levels
            
        Returns:
            Hierarchical predictions with consistency checking
        """
        with torch.no_grad():
            output = self.forward(embeddings, **kwargs)
            
            result = {}
            
            # Extract predictions at each level
            for level_name in self.hierarchy_levels.keys():
                if f'{level_name}_predictions' in output:
                    predictions = output[f'{level_name}_predictions']
                    probabilities = output[f'{level_name}_probabilities']
                    
                    # Convert indices to class names
                    class_names = [
                        self.hierarchy_levels[level_name][idx.item()]
                        for idx in predictions
                    ]
                    
                    result[level_name] = {
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'class_names': class_names
                    }
            
            # Check consistency across levels
            result['consistency_score'] = self._check_prediction_consistency(result)
            
            return result
    
    def _check_prediction_consistency(
        self,
        predictions: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Check consistency of predictions across hierarchy levels.
        
        Args:
            predictions: Predictions at each level
            
        Returns:
            Consistency score (higher is more consistent)
        """
        # Simplified consistency check
        # Full implementation would verify parent-child relationships
        
        if 'major' in predictions and 'subtype' in predictions:
            major_confidence = torch.max(predictions['major']['probabilities'], dim=-1)[0]
            subtype_confidence = torch.max(predictions['subtype']['probabilities'], dim=-1)[0]
            
            # Simple consistency measure based on confidence correlation
            consistency = torch.mean(torch.abs(major_confidence - subtype_confidence))
            return 1.0 - consistency  # Higher score for more consistent predictions
        
        return torch.tensor(1.0)  # Perfect consistency if only one level
    
    def predict_from_adata(
        self,
        adata,
        foundation_model=None,
        return_all_levels: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict hierarchical cell types from AnnData object.
        
        Args:
            adata: AnnData object with spatial transcriptomics data
            foundation_model: Optional foundation model for computing embeddings
            return_all_levels: Whether to return predictions at all levels
            
        Returns:
            Hierarchical cell type predictions
        """
        # Get embeddings
        embeddings = self._get_embeddings(adata, foundation_model)
        
        # Make hierarchical predictions
        return self.predict_hierarchical(
            embeddings=embeddings,
            return_all_levels=return_all_levels,
            **kwargs
        )


class SpatialContextEncoder(nn.Module):
    """
    Encoder for spatial neighborhood context in cell type prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        context_radius: float = 100.0,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_radius = context_radius
        self.num_attention_heads = num_attention_heads
        
        # Spatial attention for neighborhood aggregation
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Context transformation
        self.context_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode spatial context for each cell.
        
        Args:
            embeddings: Cell embeddings [num_cells, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            spatial_coords: Spatial coordinates [num_cells, 2]
            
        Returns:
            Context-encoded features [num_cells, hidden_dim]
        """
        num_cells = embeddings.size(0)
        device = embeddings.device
        
        # Create neighborhood features for each cell
        context_features = torch.zeros_like(embeddings)
        
        # Process each cell's neighborhood
        for cell_idx in range(num_cells):
            # Find neighbors
            neighbor_mask = (edge_index[0] == cell_idx) | (edge_index[1] == cell_idx)
            
            if neighbor_mask.sum() > 0:
                # Get neighbor indices
                neighbor_edges = edge_index[:, neighbor_mask]
                neighbor_indices = torch.unique(neighbor_edges.flatten())
                neighbor_indices = neighbor_indices[neighbor_indices != cell_idx]
                
                if len(neighbor_indices) > 0:
                    # Get neighbor embeddings
                    neighbor_embeddings = embeddings[neighbor_indices]
                    
                    # Add query (current cell)
                    query = embeddings[cell_idx:cell_idx+1]
                    
                    # Apply attention to aggregate neighborhood
                    context_emb, _ = self.spatial_attention(
                        query,
                        neighbor_embeddings.unsqueeze(0),
                        neighbor_embeddings.unsqueeze(0)
                    )
                    
                    context_features[cell_idx] = context_emb.squeeze(0)
                else:
                    # No neighbors - use self
                    context_features[cell_idx] = embeddings[cell_idx]
            else:
                # No edges - use self
                context_features[cell_idx] = embeddings[cell_idx]
        
        # Transform context features
        context_features = self.context_transform(context_features)
        
        return context_features
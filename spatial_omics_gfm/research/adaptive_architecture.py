"""
Adaptive Architecture Components for Spatial-Omics GFM.

This module implements adaptive and self-optimizing components that automatically
adjust model architecture and parameters based on data characteristics and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

from .novel_attention import AdaptiveSpatialAttention, HierarchicalSpatialAttention
from ..utils.optimization import PerformanceProfiler
from ..utils.memory_management import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive components."""
    
    enable_architecture_search: bool = True
    enable_hyperparameter_optimization: bool = True
    enable_dynamic_attention: bool = True
    enable_adaptive_pooling: bool = True
    
    # Architecture search parameters
    max_layers: int = 24
    min_layers: int = 4
    layer_search_patience: int = 5
    
    # Attention adaptation
    attention_switch_threshold: float = 0.1
    attention_adaptation_frequency: int = 100  # Steps
    
    # Performance monitoring
    performance_window_size: int = 50
    improvement_threshold: float = 0.01
    
    # Resource constraints
    max_memory_mb: float = 8192.0
    max_latency_ms: float = 1000.0


class DynamicAttentionSelector(nn.Module):
    """
    Dynamically selects optimal attention mechanism based on data characteristics.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Different attention mechanisms
        self.attention_mechanisms = nn.ModuleDict({
            'adaptive': AdaptiveSpatialAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                adaptive_radius=True,
                temperature_learning=True
            ),
            'hierarchical': HierarchicalSpatialAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_scales=3
            ),
            'standard': nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )
        })
        
        # Data characteristics analyzer
        self.data_analyzer = DataCharacteristicsAnalyzer(hidden_dim)
        
        # Attention selector network
        self.attention_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.attention_mechanisms)),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking
        self.performance_history = []
        self.current_attention = 'adaptive'  # Default
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic attention selection.
        """
        # Analyze data characteristics
        data_features = self.data_analyzer(x, edge_index, spatial_coords)
        
        # Select attention mechanism
        attention_weights = self.attention_selector(data_features.mean(dim=0))
        selected_attention = self._select_attention_mechanism(attention_weights)
        
        # Apply selected attention
        if selected_attention == 'standard':
            # Convert to format expected by standard attention
            attn_output, _ = self.attention_mechanisms[selected_attention](
                x, x, x
            )
            return attn_output
        else:
            # Use spatial attention mechanisms
            return self.attention_mechanisms[selected_attention](
                x, edge_index, edge_attr, spatial_coords, batch
            )
    
    def _select_attention_mechanism(self, weights: torch.Tensor) -> str:
        """Select attention mechanism based on weights."""
        mechanism_names = list(self.attention_mechanisms.keys())
        selected_idx = torch.argmax(weights).item()
        return mechanism_names[selected_idx]
    
    def update_performance(self, performance_metric: float) -> None:
        """Update performance history for adaptation."""
        self.performance_history.append(performance_metric)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]


class DataCharacteristicsAnalyzer(nn.Module):
    """
    Analyzes data characteristics to inform adaptive decisions.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature extractors for different data aspects
        self.spatial_analyzer = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # Assuming 2D coordinates
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.graph_analyzer = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # Degree, clustering, centrality
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.expression_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Analyze data characteristics."""
        features = []
        
        # Analyze expression patterns
        expr_features = self.expression_analyzer(x)
        features.append(expr_features)
        
        # Analyze spatial characteristics
        if spatial_coords is not None:
            spatial_features = self.spatial_analyzer(spatial_coords)
            features.append(spatial_features)
        else:
            # Use zeros if no spatial coordinates
            spatial_features = torch.zeros(
                x.size(0), self.hidden_dim // 4, 
                device=x.device, dtype=x.dtype
            )
            features.append(spatial_features)
        
        # Analyze graph characteristics
        graph_stats = self._compute_graph_statistics(edge_index, x.size(0))
        graph_features = self.graph_analyzer(graph_stats)
        features.append(graph_features)
        
        # Combine all features
        combined = torch.cat(features, dim=-1)
        return self.feature_combiner(combined)
    
    def _compute_graph_statistics(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute basic graph statistics."""
        device = edge_index.device
        
        # Node degrees
        row, col = edge_index
        degree = torch.zeros(num_nodes, device=device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # Local clustering coefficient (simplified)
        clustering = torch.zeros(num_nodes, device=device)
        
        # Centrality measures (simplified - using degree as proxy)
        centrality = degree / (degree.max() + 1e-8)
        
        # Stack statistics
        stats = torch.stack([degree, clustering, centrality], dim=1)
        
        return stats


class AdaptiveLayerController(nn.Module):
    """
    Dynamically controls the number of layers used based on performance and efficiency.
    """
    
    def __init__(self, max_layers: int = 24, min_layers: int = 4):
        super().__init__()
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.current_layers = min_layers
        
        # Performance tracking
        self.layer_performance = {}
        self.performance_history = []
        
        # Layer usage predictor
        self.layer_predictor = nn.Sequential(
            nn.Linear(128, 64),  # Input: data characteristics
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: fraction of max layers to use
        )
        
    def predict_optimal_layers(self, data_characteristics: torch.Tensor) -> int:
        """Predict optimal number of layers for given data."""
        with torch.no_grad():
            # Get prediction
            layer_fraction = self.layer_predictor(data_characteristics.mean(dim=0))
            
            # Convert to actual layer count
            predicted_layers = int(
                self.min_layers + 
                layer_fraction.item() * (self.max_layers - self.min_layers)
            )
            
            return max(self.min_layers, min(self.max_layers, predicted_layers))
    
    def should_early_stop(self, layer_idx: int, layer_output: torch.Tensor) -> bool:
        """Determine if we should stop processing more layers."""
        # Simple convergence check based on output variance
        if layer_idx < self.min_layers:
            return False
            
        output_variance = torch.var(layer_output).item()
        
        # If output is converging (low variance), consider early stopping
        return output_variance < 1e-6
    
    def update_performance(self, num_layers: int, performance: float) -> None:
        """Update performance tracking for layer optimization."""
        if num_layers not in self.layer_performance:
            self.layer_performance[num_layers] = []
        
        self.layer_performance[num_layers].append(performance)
        
        # Keep only recent performance data
        for layers in self.layer_performance:
            if len(self.layer_performance[layers]) > 20:
                self.layer_performance[layers] = self.layer_performance[layers][-10:]


class AdaptivePoolingStrategy(nn.Module):
    """
    Adaptive pooling that selects optimal pooling strategy based on data characteristics.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Different pooling strategies
        self.pooling_strategies = nn.ModuleDict({
            'mean': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1),
            'attention': AttentionPooling(hidden_dim),
            'hierarchical': HierarchicalPooling(hidden_dim)
        })
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.pooling_strategies)),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply adaptive pooling."""
        # Analyze data to select strategy
        data_summary = x.mean(dim=0)
        strategy_weights = self.strategy_selector(data_summary)
        
        # Select dominant strategy
        selected_strategy = self._select_pooling_strategy(strategy_weights)
        
        # Apply selected pooling
        if selected_strategy in ['mean', 'max']:
            # Standard pooling
            pooled = self.pooling_strategies[selected_strategy](
                x.unsqueeze(0).transpose(1, 2)
            ).squeeze(-1).squeeze(0)
        else:
            # Custom pooling methods
            pooled = self.pooling_strategies[selected_strategy](x, edge_index, batch)
        
        return pooled
    
    def _select_pooling_strategy(self, weights: torch.Tensor) -> str:
        """Select pooling strategy based on weights."""
        strategy_names = list(self.pooling_strategies.keys())
        selected_idx = torch.argmax(weights).item()
        return strategy_names[selected_idx]


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply attention pooling."""
        # Compute attention weights
        attn_weights = self.attention(x)  # [num_nodes, 1]
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Apply weighted pooling
        pooled = torch.sum(x * attn_weights, dim=0)
        
        return pooled


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling for multi-scale analysis."""
    
    def __init__(self, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Level-specific pooling layers
        self.level_poolers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // (2 ** i))
            ) for i in range(num_levels)
        ])
        
        # Combiner
        total_dim = sum(hidden_dim // (2 ** i) for i in range(num_levels))
        self.combiner = nn.Linear(total_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply hierarchical pooling."""
        level_outputs = []
        
        for i, pooler in enumerate(self.level_poolers):
            # Apply level-specific transformation
            level_features = pooler(x)
            
            # Pool at this level (simple mean pooling)
            level_pooled = level_features.mean(dim=0)
            level_outputs.append(level_pooled)
        
        # Combine all levels
        combined = torch.cat(level_outputs, dim=0)
        final_output = self.combiner(combined)
        
        return final_output


class AdaptiveSpatialTransformer(nn.Module):
    """
    Main adaptive transformer that combines all adaptive components.
    """
    
    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 512,
        max_layers: int = 24,
        num_heads: int = 8,
        config: Optional[AdaptiveConfig] = None
    ):
        super().__init__()
        
        self.config = config or AdaptiveConfig()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.max_layers = max_layers
        self.num_heads = num_heads
        
        # Input encoding
        self.gene_encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Adaptive components
        if self.config.enable_dynamic_attention:
            self.attention_selector = DynamicAttentionSelector(hidden_dim, num_heads)
        
        if self.config.enable_architecture_search:
            self.layer_controller = AdaptiveLayerController(max_layers)
        
        if self.config.enable_adaptive_pooling:
            self.adaptive_pooler = AdaptivePoolingStrategy(hidden_dim)
        
        # Data analyzer
        self.data_analyzer = DataCharacteristicsAnalyzer(hidden_dim)
        
        # Performance monitoring
        self.performance_monitor = PerformanceProfiler()
        self.memory_monitor = MemoryMonitor()
        
        # Standard transformer layers (fallback)
        self.standard_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(max_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(f"AdaptiveSpatialTransformer initialized with {self._count_parameters():,} parameters")
    
    def forward(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_adaptations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive components."""
        
        # Start performance monitoring
        if self.config.enable_hyperparameter_optimization:
            self.performance_monitor.start_timing('forward_pass')
            self.memory_monitor.start_monitoring()
        
        # Encode inputs
        gene_emb = self.gene_encoder(gene_expression)
        spatial_emb = self.spatial_encoder(spatial_coords)
        
        # Combine embeddings
        x = gene_emb + spatial_emb
        
        # Analyze data characteristics
        data_chars = self.data_analyzer(x, edge_index, spatial_coords)
        
        adaptations = {}
        
        # Determine optimal number of layers
        if self.config.enable_architecture_search:
            optimal_layers = self.layer_controller.predict_optimal_layers(data_chars)
            adaptations['num_layers_used'] = optimal_layers
        else:
            optimal_layers = self.max_layers
        
        # Process through layers
        for i in range(min(optimal_layers, len(self.standard_layers))):
            if self.config.enable_dynamic_attention:
                # Use adaptive attention
                x = self.attention_selector(
                    x, edge_index, edge_attr, spatial_coords, batch
                )
            else:
                # Use standard transformer layer
                x = self.standard_layers[i](x)
            
            # Check for early stopping
            if self.config.enable_architecture_search:
                if self.layer_controller.should_early_stop(i, x):
                    adaptations['early_stopped_at_layer'] = i
                    break
        
        # Apply adaptive pooling if needed
        if self.config.enable_adaptive_pooling and batch is not None:
            pooled_x = self.adaptive_pooler(x, edge_index, batch)
            adaptations['pooling_applied'] = True
        else:
            pooled_x = x
        
        # Final normalization
        output = self.output_norm(pooled_x if self.config.enable_adaptive_pooling and batch is not None else x)
        
        # Stop performance monitoring
        if self.config.enable_hyperparameter_optimization:
            timing_info = self.performance_monitor.get_timing_info()
            memory_info = self.memory_monitor.get_memory_usage()
            
            adaptations.update({
                'forward_time_ms': timing_info.get('forward_pass', 0) * 1000,
                'memory_usage_mb': memory_info.get('memory_mb', 0)
            })
            
            self.memory_monitor.stop_monitoring()
        
        # Prepare output
        result = {'embeddings': output}
        
        if return_adaptations:
            result['adaptations'] = adaptations
        
        return result
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def optimize_for_dataset(self, dataloader, num_epochs: int = 5) -> Dict[str, Any]:
        """
        Optimize model configuration for a specific dataset.
        
        Args:
            dataloader: DataLoader with training data
            num_epochs: Number of optimization epochs
            
        Returns:
            Optimization results and best configuration
        """
        logger.info("Starting adaptive optimization for dataset")
        
        best_config = None
        best_performance = float('-inf')
        optimization_history = []
        
        # Try different configurations
        configs_to_try = self._generate_config_candidates()
        
        for epoch in range(num_epochs):
            for config_candidate in configs_to_try:
                # Apply configuration
                self._apply_config(config_candidate)
                
                # Evaluate on sample batches
                performance = self._evaluate_configuration(dataloader)
                
                optimization_history.append({
                    'epoch': epoch,
                    'config': config_candidate,
                    'performance': performance
                })
                
                # Update best configuration
                if performance > best_performance:
                    best_performance = performance
                    best_config = config_candidate
                
                logger.info(f"Config evaluation - Performance: {performance:.4f}")
        
        # Apply best configuration
        if best_config:
            self._apply_config(best_config)
            logger.info(f"Applied best configuration with performance: {best_performance:.4f}")
        
        return {
            'best_config': best_config,
            'best_performance': best_performance,
            'optimization_history': optimization_history
        }
    
    def _generate_config_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidate configurations to evaluate."""
        candidates = []
        
        # Different layer counts
        for num_layers in [4, 8, 12, 16, 20, 24]:
            candidates.append({
                'max_layers': num_layers,
                'enable_dynamic_attention': True,
                'enable_adaptive_pooling': True
            })
        
        # Different attention strategies
        for attention_only in [True, False]:
            candidates.append({
                'max_layers': 12,
                'enable_dynamic_attention': attention_only,
                'enable_adaptive_pooling': not attention_only
            })
        
        return candidates[:10]  # Limit candidates for reasonable runtime
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply a configuration to the model."""
        if 'max_layers' in config:
            self.max_layers = config['max_layers']
        
        if 'enable_dynamic_attention' in config:
            self.config.enable_dynamic_attention = config['enable_dynamic_attention']
        
        if 'enable_adaptive_pooling' in config:
            self.config.enable_adaptive_pooling = config['enable_adaptive_pooling']
    
    def _evaluate_configuration(self, dataloader) -> float:
        """
        Evaluate current configuration on sample data.
        
        Returns:
            Performance score (higher is better)
        """
        total_performance = 0.0
        num_batches = 0
        
        self.eval()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if batch_idx >= 5:  # Only evaluate on first 5 batches
                    break
                
                try:
                    # Extract batch data (format depends on your dataloader)
                    if isinstance(batch_data, dict):
                        outputs = self(**batch_data)
                    else:
                        # Assume tuple format
                        gene_expr, spatial_coords, edge_index = batch_data[:3]
                        outputs = self(
                            gene_expression=gene_expr,
                            spatial_coords=spatial_coords,
                            edge_index=edge_index
                        )
                    
                    # Simple performance metric: negative of embedding variance
                    # (encourage stable, informative representations)
                    embeddings = outputs['embeddings']
                    variance = torch.var(embeddings).item()
                    
                    # Performance score (higher variance = more informative)
                    performance = variance
                    
                    total_performance += performance
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Batch evaluation failed: {e}")
                    continue
        
        self.train()
        
        return total_performance / max(num_batches, 1)


# Factory function for creating adaptive models
def create_adaptive_spatial_transformer(
    num_genes: int,
    model_size: str = "base",
    enable_all_adaptations: bool = True
) -> AdaptiveSpatialTransformer:
    """
    Create an adaptive spatial transformer with predefined configurations.
    
    Args:
        num_genes: Number of genes in the dataset
        model_size: Model size ('small', 'base', 'large')
        enable_all_adaptations: Whether to enable all adaptive features
        
    Returns:
        Configured AdaptiveSpatialTransformer
    """
    
    size_configs = {
        'small': {'hidden_dim': 256, 'max_layers': 8, 'num_heads': 4},
        'base': {'hidden_dim': 512, 'max_layers': 16, 'num_heads': 8},
        'large': {'hidden_dim': 1024, 'max_layers': 24, 'num_heads': 16}
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config_params = size_configs[model_size]
    
    # Create adaptive configuration
    adaptive_config = AdaptiveConfig(
        enable_architecture_search=enable_all_adaptations,
        enable_hyperparameter_optimization=enable_all_adaptations,
        enable_dynamic_attention=enable_all_adaptations,
        enable_adaptive_pooling=enable_all_adaptations,
        max_layers=config_params['max_layers']
    )
    
    model = AdaptiveSpatialTransformer(
        num_genes=num_genes,
        hidden_dim=config_params['hidden_dim'],
        max_layers=config_params['max_layers'],
        num_heads=config_params['num_heads'],
        config=adaptive_config
    )
    
    logger.info(f"Created adaptive {model_size} model with {model._count_parameters():,} parameters")
    
    return model

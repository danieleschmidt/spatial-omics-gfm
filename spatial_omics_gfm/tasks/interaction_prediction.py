"""
Cell-cell interaction prediction for spatial transcriptomics data.
Implements ligand-receptor analysis, paracrine signaling prediction,
and spatial communication network reconstruction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from anndata import AnnData

from .base import BaseTask, TaskConfig
from ..models.graph_transformer import SpatialGraphTransformer

logger = logging.getLogger(__name__)


class InteractionPredictor(BaseTask):
    """
    Predicts cell-cell interactions from spatial transcriptomics data.
    
    This class implements various methods for predicting interactions including:
    - Ligand-receptor pair analysis
    - Paracrine and juxtacrine signaling
    - Spatial communication networks
    - Statistical significance testing
    """
    
    def __init__(
        self,
        config: Optional[TaskConfig] = None,
        interaction_database: str = "cellphonedb",
        species: str = "human",
        distance_threshold: float = 200.0,
        min_expression_threshold: float = 0.1,
        significance_threshold: float = 0.05
    ):
        """
        Initialize interaction predictor.
        
        Args:
            config: Task configuration
            interaction_database: Database for L-R pairs ('cellphonedb', 'nichenet', 'connectome')
            species: Species for interaction database
            distance_threshold: Maximum distance for interactions (micrometers)
            min_expression_threshold: Minimum expression for L-R consideration
            significance_threshold: P-value threshold for significance
        """
        if config is None:
            config = TaskConfig(hidden_dim=1024, num_classes=3)  # 3 interaction types
        super().__init__(config)
        
        self.interaction_database = interaction_database
        self.species = species
        self.distance_threshold = distance_threshold
        self.min_expression_threshold = min_expression_threshold
        self.significance_threshold = significance_threshold
        
        # Load interaction database
        self.lr_database = self._load_lr_database()
        
        # Initialize prediction head
        self.interaction_head = InteractionPredictionHead(
            hidden_dim=config.hidden_dim,
            num_interaction_types=3,  # ligand-receptor, paracrine, juxtacrine
            dropout=config.dropout
        )
        
        logger.info(f"Initialized InteractionPredictor with {len(self.lr_database)} L-R pairs")
    
    def _load_lr_database(self) -> pd.DataFrame:
        """Load ligand-receptor interaction database."""
        logger.info(f"Loading {self.interaction_database} database for {self.species}")
        
        # This would typically load from a file or download from a database
        # For now, we'll create a minimal example database
        if self.interaction_database.lower() == "cellphonedb":
            return self._load_cellphonedb()
        elif self.interaction_database.lower() == "nichenet":
            return self._load_nichenet()
        else:
            logger.warning(f"Unknown database {self.interaction_database}, using minimal default")
            return self._create_minimal_database()
    
    def _load_cellphonedb(self) -> pd.DataFrame:
        """Load CellPhoneDB ligand-receptor pairs."""
        # Example L-R pairs for demonstration
        lr_pairs = [
            ("TGFB1", "TGFBR1", "paracrine"),
            ("TNF", "TNFRSF1A", "paracrine"),
            ("IL1B", "IL1R1", "paracrine"),
            ("VEGFA", "FLT1", "paracrine"),
            ("PDGFA", "PDGFRA", "paracrine"),
            ("IGF1", "IGF1R", "paracrine"),
            ("EGF", "EGFR", "paracrine"),
            ("FGF2", "FGFR1", "paracrine"),
            ("WNT3A", "FZD1", "juxtacrine"),
            ("DLL1", "NOTCH1", "juxtacrine"),
            ("JAG1", "NOTCH2", "juxtacrine"),
            ("EFNA1", "EPHA2", "juxtacrine"),
            ("CD274", "PDCD1", "juxtacrine"),
            ("CD80", "CD28", "juxtacrine"),
            ("ICAM1", "ITGAL", "juxtacrine"),
        ]
        
        df = pd.DataFrame(lr_pairs, columns=['ligand', 'receptor', 'interaction_type'])
        df['database'] = 'cellphonedb'
        df['species'] = self.species
        return df
    
    def _load_nichenet(self) -> pd.DataFrame:
        """Load NicheNet ligand-receptor pairs."""
        # Would load NicheNet database
        return self._create_minimal_database()
    
    def _create_minimal_database(self) -> pd.DataFrame:
        """Create minimal L-R database for testing."""
        lr_pairs = [
            ("TGFB1", "TGFBR1", "paracrine"),
            ("TNF", "TNFRSF1A", "paracrine"),
            ("IL1B", "IL1R1", "paracrine"),
            ("WNT3A", "FZD1", "juxtacrine"),
            ("DLL1", "NOTCH1", "juxtacrine"),
        ]
        
        df = pd.DataFrame(lr_pairs, columns=['ligand', 'receptor', 'interaction_type'])
        df['database'] = 'minimal'
        df['species'] = self.species
        return df
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for interaction prediction.
        
        Args:
            embeddings: Node embeddings from foundation model
            edge_index: Graph edge connectivity
            edge_attr: Edge attributes (distances, etc.)
            
        Returns:
            Dictionary containing interaction predictions
        """
        if edge_index is None:
            raise ValueError("edge_index is required for interaction prediction")
        
        if edge_attr is None:
            # Create default edge attributes (just distances)
            edge_attr = torch.ones(edge_index.size(1), 3)  # distance, angle, type placeholder
        
        # Predict interaction types
        interaction_logits = self.interaction_head(embeddings, edge_index, edge_attr)
        interaction_probs = F.softmax(interaction_logits, dim=-1)
        
        return {
            'interaction_logits': interaction_logits,
            'interaction_probabilities': interaction_probs,
            'predictions': torch.argmax(interaction_probs, dim=-1)
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute interaction prediction loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth interaction labels
            
        Returns:
            Loss tensor
        """
        logits = predictions['interaction_logits']
        return F.cross_entropy(logits, targets)
    
    def predict(
        self,
        adata: AnnData,
        foundation_model=None,
        cell_types: Optional[pd.Series] = None,
        return_embeddings: bool = False,
        compute_significance: bool = True
    ) -> Dict[str, Any]:
        """
        Predict cell-cell interactions.
        
        Args:
            adata: Spatial transcriptomics data
            cell_types: Cell type annotations
            return_embeddings: Whether to return embeddings
            compute_significance: Whether to compute statistical significance
            
        Returns:
            Dictionary with interaction predictions and statistics
        """
        logger.info("Predicting cell-cell interactions")
        
        # Get model embeddings
        embeddings = self._get_embeddings(adata, foundation_model)
        
        # Predict interaction probabilities
        interaction_probs = self._predict_interactions(adata, embeddings)
        
        # Perform ligand-receptor analysis
        lr_results = self._analyze_ligand_receptor_pairs(adata, cell_types)
        
        # Find spatial communication networks
        comm_networks = self._find_communication_networks(adata, interaction_probs, lr_results)
        
        # Compute statistical significance
        significance_results = None
        if compute_significance:
            significance_results = self._compute_significance(adata, lr_results)
        
        results = {
            'interaction_probabilities': interaction_probs,
            'ligand_receptor_analysis': lr_results,
            'communication_networks': comm_networks,
            'significance_results': significance_results
        }
        
        if return_embeddings:
            results['embeddings'] = embeddings
        
        return results
    
    def _predict_interactions(self, adata: AnnData, embeddings: torch.Tensor) -> Dict[str, np.ndarray]:
        """Predict interaction probabilities using the neural network."""
        logger.info("Computing interaction probabilities")
        
        # Get spatial graph
        if 'spatial_graph' not in adata.uns:
            raise ValueError("Spatial graph not found. Please compute spatial neighbors first.")
        
        edge_index = torch.tensor(adata.uns['spatial_graph']['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(adata.uns['spatial_graph']['edge_attr'], dtype=torch.float32)
        
        # Predict interactions for each edge
        with torch.no_grad():
            interaction_logits = self.interaction_head(embeddings, edge_index, edge_attr)
            interaction_probs = torch.softmax(interaction_logits, dim=-1)
        
        # Convert to numpy and organize by interaction type
        interaction_probs_np = interaction_probs.cpu().numpy()
        
        results = {
            'ligand_receptor': interaction_probs_np[:, 0],
            'paracrine': interaction_probs_np[:, 1],
            'juxtacrine': interaction_probs_np[:, 2]
        }
        
        return results
    
    def _analyze_ligand_receptor_pairs(
        self,
        adata: AnnData,
        cell_types: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Analyze ligand-receptor pairs in the data."""
        logger.info("Analyzing ligand-receptor pairs")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X
        
        # Get gene names
        gene_names = adata.var_names.tolist()
        
        # Get spatial coordinates and graph
        coords = adata.obsm['spatial']
        edge_index = adata.uns['spatial_graph']['edge_index']
        edge_distances = adata.uns['spatial_graph']['edge_attr'][:, 0]  # First column is distance
        
        results = []
        
        for _, lr_pair in self.lr_database.iterrows():
            ligand = lr_pair['ligand']
            receptor = lr_pair['receptor']
            interaction_type = lr_pair['interaction_type']
            
            # Check if both genes are in the data
            if ligand not in gene_names or receptor not in gene_names:
                continue
            
            ligand_idx = gene_names.index(ligand)
            receptor_idx = gene_names.index(receptor)
            
            # Get expression levels
            ligand_expr = expression[:, ligand_idx]
            receptor_expr = expression[:, receptor_idx]
            
            # Filter edges by distance threshold for this interaction type
            if interaction_type == 'juxtacrine':
                max_dist = min(self.distance_threshold, 50.0)  # Close contact
            else:
                max_dist = self.distance_threshold
            
            valid_edges = edge_distances <= max_dist
            filtered_edges = edge_index[:, valid_edges]
            
            # Calculate interaction scores for valid edges
            for i in range(filtered_edges.shape[1]):
                source_idx = filtered_edges[0, i]
                target_idx = filtered_edges[1, i]
                
                ligand_level = ligand_expr[source_idx]
                receptor_level = receptor_expr[target_idx]
                
                # Only consider if both are expressed above threshold
                if (ligand_level >= self.min_expression_threshold and 
                    receptor_level >= self.min_expression_threshold):
                    
                    # Calculate interaction score (simple product for now)
                    interaction_score = ligand_level * receptor_level
                    distance = edge_distances[valid_edges][i]
                    
                    result = {
                        'ligand': ligand,
                        'receptor': receptor,
                        'interaction_type': interaction_type,
                        'source_cell': source_idx,
                        'target_cell': target_idx,
                        'ligand_expression': ligand_level,
                        'receptor_expression': receptor_level,
                        'interaction_score': interaction_score,
                        'distance': distance
                    }
                    
                    # Add cell type information if available
                    if cell_types is not None:
                        result['source_cell_type'] = cell_types.iloc[source_idx]
                        result['target_cell_type'] = cell_types.iloc[target_idx]
                    
                    results.append(result)
        
        lr_results_df = pd.DataFrame(results)
        
        if len(lr_results_df) > 0:
            # Rank interactions by score
            lr_results_df = lr_results_df.sort_values('interaction_score', ascending=False)
            logger.info(f"Found {len(lr_results_df)} potential L-R interactions")
        else:
            logger.warning("No L-R interactions found")
        
        return lr_results_df
    
    def _find_communication_networks(
        self,
        adata: AnnData,
        interaction_probs: Dict[str, np.ndarray],
        lr_results: pd.DataFrame
    ) -> Dict[str, Any]:
        """Find spatial communication networks."""
        logger.info("Finding communication networks")
        
        edge_index = adata.uns['spatial_graph']['edge_index']
        
        # Create network graphs for each interaction type
        networks = {}
        
        for interaction_type, probs in interaction_probs.items():
            # Threshold probabilities to create binary network
            threshold = 0.5
            strong_interactions = probs > threshold
            
            # Get edges with strong interactions
            strong_edges = edge_index[:, strong_interactions]
            strong_probs = probs[strong_interactions]
            
            networks[interaction_type] = {
                'edges': strong_edges,
                'weights': strong_probs,
                'num_interactions': len(strong_probs)
            }
        
        # Find communication hubs (cells with many interactions)
        all_edges = np.concatenate([net['edges'] for net in networks.values()], axis=1)
        if all_edges.size > 0:
            unique_cells, interaction_counts = np.unique(all_edges, return_counts=True)
            
            # Identify hubs (top 10% by interaction count)
            threshold_count = np.percentile(interaction_counts, 90)
            hub_mask = interaction_counts >= threshold_count
            hubs = unique_cells[hub_mask]
            
            networks['communication_hubs'] = {
                'cell_indices': hubs,
                'interaction_counts': interaction_counts[hub_mask]
            }
        
        return networks
    
    def _compute_significance(self, adata: AnnData, lr_results: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical significance of interactions."""
        logger.info("Computing interaction significance")
        
        if not lr_results:
            return {'p_values': [], 'significant_interactions': pd.DataFrame()}
        
        # Perform permutation test for each L-R pair
        significance_results = []
        
        for lr_pair in lr_results[['ligand', 'receptor']].drop_duplicates().itertuples():
            ligand, receptor = lr_pair.ligand, lr_pair.receptor
            
            # Get observed scores for this L-R pair
            pair_results = lr_results[
                (lr_results['ligand'] == ligand) & 
                (lr_results['receptor'] == receptor)
            ]
            
            if not pair_results:
                continue
            
            observed_scores = pair_results['interaction_score'].values
            observed_mean = np.mean(observed_scores)
            
            # Perform permutation test
            n_permutations = 1000
            permuted_means = []
            
            gene_names = adata.var_names.tolist()
            ligand_idx = gene_names.index(ligand)
            receptor_idx = gene_names.index(receptor)
            
            if hasattr(adata.X, 'toarray'):
                expression = adata.X.toarray()
            else:
                expression = adata.X
            
            ligand_expr = expression[:, ligand_idx]
            receptor_expr = expression[:, receptor_idx]
            
            for _ in range(n_permutations):
                # Permute cell assignments
                permuted_receptor = np.random.permutation(receptor_expr)
                
                # Calculate permuted scores
                permuted_scores = []
                for _, row in pair_results.iterrows():
                    source_idx = row['source_cell']
                    target_idx = row['target_cell']
                    
                    perm_score = ligand_expr[source_idx] * permuted_receptor[target_idx]
                    permuted_scores.append(perm_score)
                
                if permuted_scores:
                    permuted_means.append(np.mean(permuted_scores))
            
            # Calculate p-value
            if permuted_means:
                p_value = np.mean(np.array(permuted_means) >= observed_mean)
            else:
                p_value = 1.0
            
            significance_results.append({
                'ligand': ligand,
                'receptor': receptor,
                'observed_mean_score': observed_mean,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold,
                'num_interactions': len(pair_results)
            })
        
        significance_df = pd.DataFrame(significance_results)
        
        # Apply multiple testing correction
        if len(significance_df) > 0:
            from statsmodels.stats.multitest import multipletests
            
            _, corrected_p_values, _, _ = multipletests(
                significance_df['p_value'], 
                alpha=self.significance_threshold, 
                method='fdr_bh'
            )
            
            significance_df['corrected_p_value'] = corrected_p_values
            significance_df['significant_corrected'] = corrected_p_values < self.significance_threshold
        
        return {
            'significance_results': significance_df,
            'significant_pairs': significance_df[significance_df.get('significant_corrected', False)]
        }
    
    def find_signaling_neighborhoods(
        self,
        adata: AnnData,
        interactions: pd.DataFrame,
        min_size: int = 5,
        enrichment_method: str = "spatial_permutation"
    ) -> Dict[str, Any]:
        """Find spatially enriched signaling neighborhoods."""
        logger.info("Finding signaling neighborhoods")
        
        coords = adata.obsm['spatial']
        
        # Use spatial clustering to find neighborhoods
        from sklearn.cluster import DBSCAN
        
        # Different radii for different neighborhood sizes
        neighborhoods = {}
        
        for radius in [50, 100, 200]:
            clustering = DBSCAN(eps=radius, min_samples=min_size).fit(coords)
            
            neighborhood_interactions = []
            
            for cluster_id in np.unique(clustering.labels_):
                if cluster_id == -1:  # Skip noise
                    continue
                
                cluster_cells = np.where(clustering.labels_ == cluster_id)[0]
                
                # Find interactions within this neighborhood
                cluster_interactions = interactions[
                    interactions['source_cell'].isin(cluster_cells) |
                    interactions['target_cell'].isin(cluster_cells)
                ]
                
                if len(cluster_interactions) >= min_size:
                    # Calculate enrichment statistics
                    total_interactions = len(interactions)
                    cluster_size = len(cluster_cells)
                    total_cells = len(coords)
                    expected_interactions = (cluster_size / total_cells) * total_interactions
                    
                    enrichment_score = len(cluster_interactions) / max(expected_interactions, 1)
                    
                    neighborhood_interactions.append({
                        'neighborhood_id': f"{radius}_{cluster_id}",
                        'radius': radius,
                        'cluster_id': cluster_id,
                        'cells': cluster_cells,
                        'num_cells': len(cluster_cells),
                        'num_interactions': len(cluster_interactions),
                        'enrichment_score': enrichment_score,
                        'interactions': cluster_interactions
                    })
            
            neighborhoods[f'radius_{radius}'] = neighborhood_interactions
        
        return neighborhoods
    
    def plot_interaction_network(
        self,
        adata: AnnData,
        interactions: pd.DataFrame,
        interaction_types: Optional[List[str]] = None,
        min_score_threshold: float = 0.1,
        max_interactions: int = 1000,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """Plot interaction network on spatial coordinates."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            logger.error("Matplotlib required for plotting")
            return
        
        coords = adata.obsm['spatial']
        
        # Filter interactions
        if interaction_types:
            interactions = interactions[interactions['interaction_type'].isin(interaction_types)]
        
        interactions = interactions[interactions['interaction_score'] >= min_score_threshold]
        
        # Subsample if too many interactions
        if len(interactions) > max_interactions:
            interactions = interactions.nlargest(max_interactions, 'interaction_score')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cells
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=20, alpha=0.6)
        
        # Plot interactions as lines
        for _, interaction in interactions.iterrows():
            source_idx = interaction['source_cell']
            target_idx = interaction['target_cell']
            score = interaction['interaction_score']
            
            source_pos = coords[source_idx]
            target_pos = coords[target_idx]
            
            # Line width proportional to score
            linewidth = min(score * 10, 3)
            
            # Color by interaction type
            color_map = {
                'paracrine': 'blue',
                'juxtacrine': 'red',
                'ligand_receptor': 'green'
            }
            color = color_map.get(interaction['interaction_type'], 'black')
            
            ax.plot([source_pos[0], target_pos[0]], 
                   [source_pos[1], target_pos[1]], 
                   color=color, linewidth=linewidth, alpha=0.6)
        
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_title('Cell-Cell Interaction Network')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='Paracrine'),
            plt.Line2D([0], [0], color='red', label='Juxtacrine'),
            plt.Line2D([0], [0], color='green', label='Ligand-Receptor')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()


class InteractionPredictionHead(nn.Module):
    """Neural network head for interaction prediction."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_interaction_types: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_interaction_types = num_interaction_types
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),  # concat source+target+edge_attr
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Interaction type classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_interaction_types)
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for interaction prediction.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Interaction type logits [num_edges, num_interaction_types]
        """
        # Get source and target embeddings
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate source, target, and edge features
        edge_features = torch.cat([
            source_embeddings,
            target_embeddings,
            edge_attr
        ], dim=-1)
        
        # Process edge features
        processed_edges = self.edge_encoder(edge_features)
        
        # Classify interaction types
        interaction_logits = self.classifier(processed_edges)
        
        return interaction_logits


class LigandReceptorPredictor(InteractionPredictor):
    """Specialized predictor focusing on ligand-receptor interactions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initialized specialized LigandReceptorPredictor")
    
    def predict_lr_activity(
        self,
        adata: AnnData,
        cell_types: Optional[pd.Series] = None,
        return_pathway_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Predict ligand-receptor activity with pathway analysis.
        
        Args:
            adata: Spatial transcriptomics data
            cell_types: Cell type annotations
            return_pathway_scores: Whether to compute pathway activity scores
            
        Returns:
            L-R activity predictions with pathway information
        """
        # Get base predictions
        base_results = self.predict(adata, cell_types)
        
        # Add specialized L-R analysis
        lr_results = base_results['ligand_receptor_analysis']
        
        if return_pathway_scores and len(lr_results) > 0:
            # Group by pathways (simplified example)
            pathway_scores = self._compute_pathway_scores(lr_results)
            base_results['pathway_scores'] = pathway_scores
        
        return base_results
    
    def _compute_pathway_scores(self, lr_results: pd.DataFrame) -> Dict[str, float]:
        """Compute pathway activity scores from L-R interactions."""
        # Simplified pathway mapping
        pathway_map = {
            'TGFB1': 'TGF-beta',
            'TNF': 'TNF',
            'IL1B': 'Inflammatory',
            'VEGFA': 'Angiogenesis',
            'PDGFA': 'Growth',
            'WNT3A': 'Wnt',
            'DLL1': 'Notch',
            'JAG1': 'Notch'
        }
        
        pathway_scores = {}
        
        for pathway in set(pathway_map.values()):
            pathway_ligands = [k for k, v in pathway_map.items() if v == pathway]
            
            pathway_interactions = lr_results[
                lr_results['ligand'].isin(pathway_ligands)
            ]
            
            if len(pathway_interactions) > 0:
                pathway_scores[pathway] = pathway_interactions['interaction_score'].mean()
            else:
                pathway_scores[pathway] = 0.0
        
        return pathway_scores
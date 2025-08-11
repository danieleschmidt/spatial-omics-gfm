"""
Spatial pathway analysis for spatial transcriptomics data.
Implements spatially-resolved pathway activity scoring, gradient analysis,
and pathway-pathway communication networks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats, spatial
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from anndata import AnnData

from .base import BaseTask, TaskConfig
from ..models.graph_transformer import SpatialGraphTransformer

logger = logging.getLogger(__name__)


class PathwayAnalyzer(BaseTask):
    """
    Analyzes spatial pathway activity in transcriptomics data.
    
    This class implements various methods for pathway analysis including:
    - Spatially-resolved pathway activity scoring
    - Pathway gradient detection
    - Pathway-pathway interaction networks
    - Spatial coherence analysis
    """
    
    def __init__(
        self,
        config: Optional[TaskConfig] = None,
        pathway_database: str = "kegg",
        species: str = "human",
        min_genes_per_pathway: int = 5,
        spatial_smoothing: bool = True,
        smoothing_sigma: float = 1.5
    ):
        """
        Initialize pathway analyzer.
        
        Args:
            config: Task configuration
            pathway_database: Database for pathways ('kegg', 'reactome', 'go', 'hallmark')
            species: Species for pathway database
            min_genes_per_pathway: Minimum genes required per pathway
            spatial_smoothing: Whether to apply spatial smoothing
            smoothing_sigma: Sigma for Gaussian smoothing
        """
        if config is None:
            config = TaskConfig(hidden_dim=1024, num_classes=50)  # Default number of pathways
        super().__init__(config)
        
        self.pathway_database = pathway_database
        self.species = species
        self.min_genes_per_pathway = min_genes_per_pathway
        self.spatial_smoothing = spatial_smoothing
        self.smoothing_sigma = smoothing_sigma
        
        # Load pathway database
        self.pathways = self._load_pathway_database()
        
        # Initialize pathway scoring head
        self.pathway_head = PathwayScoringHead(
            hidden_dim=config.hidden_dim,
            num_pathways=len(self.pathways),
            dropout=config.dropout
        )
        
        logger.info(f"Initialized PathwayAnalyzer with {len(self.pathways)} pathways")
    
    def _load_pathway_database(self) -> Dict[str, List[str]]:
        """Load pathway gene sets."""
        logger.info(f"Loading {self.pathway_database} pathways for {self.species}")
        
        # This would typically load from MSigDB, KEGG, or other databases
        # For now, we'll create example pathways
        if self.pathway_database.lower() == "kegg":
            return self._load_kegg_pathways()
        elif self.pathway_database.lower() == "reactome":
            return self._load_reactome_pathways()
        elif self.pathway_database.lower() == "go":
            return self._load_go_pathways()
        elif self.pathway_database.lower() == "hallmark":
            return self._load_hallmark_pathways()
        else:
            logger.warning(f"Unknown database {self.pathway_database}, using example pathways")
            return self._create_example_pathways()
    
    def _load_kegg_pathways(self) -> Dict[str, List[str]]:
        """Load KEGG pathway gene sets."""
        # Example KEGG pathways
        pathways = {
            "KEGG_GLYCOLYSIS_GLUCONEOGENESIS": [
                "HK1", "HK2", "GPI", "PFKL", "PFKM", "PFKP", "ALDOA", "ALDOB", "ALDOC",
                "TPI1", "GAPDH", "PGK1", "PGAM1", "ENO1", "PKM", "LDHA", "LDHB"
            ],
            "KEGG_TCA_CYCLE": [
                "CS", "ACO1", "ACO2", "IDH1", "IDH2", "OGDH", "SUCLA2", "SUCLG1",
                "SDHA", "SDHB", "SDHC", "SDHD", "FH", "MDH1", "MDH2"
            ],
            "KEGG_OXIDATIVE_PHOSPHORYLATION": [
                "NDUFA1", "NDUFA2", "NDUFB1", "NDUFB2", "NDUFS1", "NDUFS2",
                "SDHC", "SDHD", "UQCRC1", "UQCRC2", "CYC1", "COX4I1", "COX5A",
                "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E"
            ],
            "KEGG_P53_SIGNALING_PATHWAY": [
                "TP53", "MDM2", "MDM4", "CDKN1A", "CDKN2A", "RB1", "E2F1",
                "BBC3", "BAX", "BAK1", "CASP3", "CASP7", "CASP9"
            ],
            "KEGG_WNT_SIGNALING_PATHWAY": [
                "WNT1", "WNT3A", "WNT5A", "WNT10B", "FZD1", "FZD2", "FZD3",
                "LRP5", "LRP6", "CTNNB1", "APC", "AXIN1", "GSK3B", "TCF7", "LEF1"
            ],
            "KEGG_NOTCH_SIGNALING_PATHWAY": [
                "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "DLL1", "DLL3", "DLL4",
                "JAG1", "JAG2", "RBPJ", "HES1", "HES5", "HEY1", "HEY2"
            ],
            "KEGG_APOPTOSIS": [
                "TNF", "TNFRSF1A", "FADD", "CASP8", "CASP3", "CASP7", "CASP9",
                "BAX", "BAK1", "BCL2", "BCL2L1", "BBC3", "BID", "TP53"
            ],
            "KEGG_CELL_CYCLE": [
                "CCND1", "CCNE1", "CCNA2", "CCNB1", "CDK1", "CDK2", "CDK4", "CDK6",
                "CDKN1A", "CDKN1B", "CDKN2A", "RB1", "E2F1", "E2F3"
            ],
            "KEGG_DNA_REPLICATION": [
                "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA",
                "POLA1", "POLA2", "POLE", "POLD1", "RFC1", "RPA1", "RPA2"
            ],
            "KEGG_MISMATCH_REPAIR": [
                "MSH2", "MSH3", "MSH6", "MLH1", "PMS2", "PCNA", "RFC1",
                "POLD1", "POLE", "LIG1"
            ]
        }
        return pathways
    
    def _load_reactome_pathways(self) -> Dict[str, List[str]]:
        """Load Reactome pathway gene sets."""
        # Example Reactome pathways
        pathways = {
            "REACTOME_IMMUNE_SYSTEM": [
                "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD19", "CD20",
                "IGHM", "IGHA1", "IGHE", "IL2", "IL4", "IL10", "IFNG", "TNF"
            ],
            "REACTOME_SIGNAL_TRANSDUCTION": [
                "EGFR", "PDGFRA", "FGFR1", "IGF1R", "INSR", "MAPK1", "MAPK3",
                "AKT1", "AKT2", "PIK3CA", "PTEN", "MTOR", "RPS6KB1"
            ],
            "REACTOME_METABOLISM": [
                "HK1", "PFK1", "ALDOA", "GAPDH", "PGK1", "PKM", "CS", "IDH1",
                "OGDH", "SDHA", "MDH1", "G6PD", "TALDO1", "TKT"
            ],
            "REACTOME_GENE_EXPRESSION": [
                "POLR2A", "POLR2B", "TBP", "TAF1", "MYC", "JUN", "FOS",
                "SP1", "STAT1", "STAT3", "NFKB1", "RELA", "TP53"
            ],
            "REACTOME_DNA_REPAIR": [
                "ATM", "ATR", "BRCA1", "BRCA2", "RAD51", "RAD52", "XRCC1",
                "XRCC4", "LIG4", "PARP1", "H2AFX", "TP53BP1"
            ]
        }
        return pathways
    
    def _load_go_pathways(self) -> Dict[str, List[str]]:
        """Load Gene Ontology pathway gene sets."""
        pathways = {
            "GO_BIOLOGICAL_PROCESS_APOPTOSIS": [
                "TP53", "BAX", "BCL2", "CASP3", "CASP7", "CASP8", "CASP9",
                "FADD", "BID", "BBC3", "BAK1", "BCL2L1", "MCL1"
            ],
            "GO_MOLECULAR_FUNCTION_KINASE": [
                "CDK1", "CDK2", "CDK4", "AKT1", "MAPK1", "MAPK3", "GSK3B",
                "MTOR", "PIK3CA", "EGFR", "PDGFRA", "IGF1R"
            ],
            "GO_CELLULAR_COMPONENT_NUCLEUS": [
                "TP53", "MYC", "JUN", "FOS", "NFKB1", "STAT1", "STAT3",
                "POLR2A", "TBP", "HIST1H1A", "HIST1H2A", "HIST1H3A"
            ]
        }
        return pathways
    
    def _load_hallmark_pathways(self) -> Dict[str, List[str]]:
        """Load MSigDB Hallmark pathway gene sets."""
        pathways = {
            "HALLMARK_GLYCOLYSIS": [
                "HK1", "HK2", "GPI", "PFKL", "ALDOA", "TPI1", "GAPDH",
                "PGK1", "PGAM1", "ENO1", "PKM", "LDHA", "LDHB", "PDK1"
            ],
            "HALLMARK_OXIDATIVE_PHOSPHORYLATION": [
                "NDUFA1", "NDUFS1", "SDHC", "UQCRC1", "CYC1", "COX4I1",
                "ATP5F1A", "ATP5F1B", "ATP5F1C", "MT-CO1", "MT-ND1"
            ],
            "HALLMARK_MYC_TARGETS_V1": [
                "MYC", "CDK4", "CCND2", "E2F1", "MCM2", "MCM3", "MCM4",
                "PCNA", "RRM1", "RRM2", "TK1", "TYMS", "DHFR"
            ],
            "HALLMARK_P53_PATHWAY": [
                "TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "PUMA", "NOXA",
                "CASP3", "CASP7", "RB1", "E2F1", "GADD45A"
            ],
            "HALLMARK_INFLAMMATORY_RESPONSE": [
                "TNF", "IL1B", "IL6", "NFKB1", "RELA", "JUN", "FOS",
                "CXCL1", "CXCL2", "CCL2", "CCL3", "PTGS2", "NOS2"
            ],
            "HALLMARK_INTERFERON_GAMMA_RESPONSE": [
                "IFNG", "STAT1", "IRF1", "GBP1", "GBP2", "CXCL9", "CXCL10",
                "IDO1", "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1"
            ],
            "HALLMARK_HYPOXIA": [
                "HIF1A", "VEGFA", "EPO", "LDHA", "PDK1", "SLC2A1", "SLC2A3",
                "ENO1", "PKM", "ALDOA", "GAPDH", "CA9", "BNIP3"
            ],
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": [
                "VIM", "FN1", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2",
                "TWIST1", "TGFB1", "TGFBR1", "SMAD2", "SMAD3", "CDH1"
            ]
        }
        return pathways
    
    def _create_example_pathways(self) -> Dict[str, List[str]]:
        """Create example pathways for testing."""
        pathways = {
            "GLYCOLYSIS": ["HK1", "HK2", "PFKL", "ALDOA", "GAPDH", "PKM", "LDHA"],
            "APOPTOSIS": ["TP53", "BAX", "BCL2", "CASP3", "CASP7", "CASP8", "CASP9"],
            "CELL_CYCLE": ["CCND1", "CCNE1", "CDK1", "CDK2", "CDK4", "RB1", "E2F1"],
            "IMMUNE_RESPONSE": ["TNF", "IL1B", "IL6", "NFKB1", "CD3D", "CD4", "CD8A"],
            "WNT_SIGNALING": ["WNT3A", "CTNNB1", "APC", "GSK3B", "TCF7", "LEF1"]
        }
        return pathways
    
    def forward(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pathway activity prediction.
        
        Args:
            embeddings: Node embeddings from foundation model
            
        Returns:
            Dictionary containing pathway activity scores
        """
        pathway_scores = self.pathway_head(embeddings)
        
        return {
            'pathway_scores': pathway_scores,
            'predictions': pathway_scores,  # For compatibility
            'logits': pathway_scores
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute pathway activity prediction loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth pathway activities
            
        Returns:
            Loss tensor
        """
        pathway_scores = predictions['pathway_scores']
        # Use MSE loss for continuous pathway scores
        return F.mse_loss(pathway_scores, targets)
    
    def predict(
        self,
        adata: AnnData,
        foundation_model=None,
        cell_types: Optional[pd.Series] = None,
        return_embeddings: bool = False,
        compute_gradients: bool = True,
        pathway_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict pathway activities and analyze spatial patterns.
        
        Args:
            adata: Spatial transcriptomics data
            cell_types: Cell type annotations
            return_embeddings: Whether to return embeddings
            compute_gradients: Whether to compute pathway gradients
            pathway_subset: Subset of pathways to analyze
            
        Returns:
            Dictionary with pathway analysis results
        """
        logger.info("Analyzing spatial pathway activity")
        
        # Filter pathways if subset provided
        pathways_to_analyze = self.pathways
        if pathway_subset:
            pathways_to_analyze = {k: v for k, v in self.pathways.items() if k in pathway_subset}
        
        # Get model embeddings
        embeddings = self._get_embeddings(adata, foundation_model)
        
        # Compute pathway activity scores
        pathway_scores = self._compute_pathway_activity(adata, pathways_to_analyze)
        
        # Apply spatial smoothing if requested
        if self.spatial_smoothing:
            pathway_scores = self._apply_spatial_smoothing(adata, pathway_scores)
        
        # Find spatially co-regulated pathways
        coregulated_pathways = self._find_coregulated_pathways(pathway_scores)
        
        # Compute pathway gradients
        gradients = None
        if compute_gradients:
            gradients = self._analyze_pathway_gradients(adata, pathway_scores)
        
        # Identify pathway boundaries and zones
        pathway_zones = self._identify_pathway_zones(adata, pathway_scores)
        
        results = {
            'pathway_scores': pathway_scores,
            'coregulated_pathways': coregulated_pathways,
            'pathway_gradients': gradients,
            'pathway_zones': pathway_zones,
            'pathways_analyzed': list(pathways_to_analyze.keys())
        }
        
        if return_embeddings:
            results['embeddings'] = embeddings
        
        return results
    
    def _compute_pathway_activity(self, adata: AnnData, pathways: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute pathway activity scores for each cell."""
        logger.info("Computing pathway activity scores")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X
        
        gene_names = adata.var_names.tolist()
        
        pathway_scores = []
        pathway_names = []
        
        for pathway_name, pathway_genes in pathways.items():
            # Find genes in the dataset
            available_genes = [gene for gene in pathway_genes if gene in gene_names]
            
            if len(available_genes) < self.min_genes_per_pathway:
                logger.warning(f"Pathway {pathway_name} has only {len(available_genes)} genes, skipping")
                continue
            
            # Get gene indices
            gene_indices = [gene_names.index(gene) for gene in available_genes]
            
            # Compute pathway score (mean expression of pathway genes)
            pathway_expr = expression[:, gene_indices]
            
            # Different scoring methods
            pathway_score = self._score_pathway_activity(pathway_expr, method="mean")
            
            pathway_scores.append(pathway_score)
            pathway_names.append(pathway_name)
        
        # Create DataFrame
        pathway_df = pd.DataFrame(
            np.column_stack(pathway_scores),
            columns=pathway_names,
            index=adata.obs.index
        )
        
        return pathway_df
    
    def _score_pathway_activity(self, pathway_expr: np.ndarray, method: str = "mean") -> np.ndarray:
        """Score pathway activity using different methods."""
        if method == "mean":
            return np.mean(pathway_expr, axis=1)
        elif method == "median":
            return np.median(pathway_expr, axis=1)
        elif method == "ssgsea":
            # Simplified single-sample GSEA
            return self._simple_ssgsea(pathway_expr)
        elif method == "z_score":
            # Z-score of mean expression
            scores = np.mean(pathway_expr, axis=1)
            return (scores - np.mean(scores)) / np.std(scores)
        else:
            return np.mean(pathway_expr, axis=1)
    
    def _simple_ssgsea(self, pathway_expr: np.ndarray) -> np.ndarray:
        """Simplified single-sample Gene Set Enrichment Analysis."""
        # Rank genes by expression for each cell
        scores = []
        
        for i in range(pathway_expr.shape[0]):
            cell_expr = pathway_expr[i, :]
            
            # Rank pathway genes
            ranks = stats.rankdata(cell_expr)
            
            # Compute enrichment score (simplified)
            es = np.sum(ranks) / len(ranks)
            scores.append(es)
        
        return np.array(scores)
    
    def _apply_spatial_smoothing(self, adata: AnnData, pathway_scores: pd.DataFrame) -> pd.DataFrame:
        """Apply spatial smoothing to pathway scores."""
        logger.info("Applying spatial smoothing to pathway scores")
        
        coords = adata.obsm['spatial']
        
        smoothed_scores = pathway_scores.copy()
        
        for pathway in pathway_scores.columns:
            scores = pathway_scores[pathway].values
            
            # Create 2D grid for smoothing
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            # Create regular grid
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            grid_size = 100
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Interpolate scores to grid
            from scipy.interpolate import griddata
            
            grid_scores = griddata(
                coords, scores, (xx, yy), 
                method='linear', fill_value=0
            )
            
            # Apply Gaussian smoothing
            smoothed_grid = gaussian_filter(grid_scores, sigma=self.smoothing_sigma)
            
            # Interpolate back to original coordinates
            smoothed_scores[pathway] = griddata(
                np.column_stack([xx.ravel(), yy.ravel()]),
                smoothed_grid.ravel(),
                coords,
                method='linear'
            )
        
        # Fill any NaN values
        smoothed_scores = smoothed_scores.fillna(pathway_scores)
        
        return smoothed_scores
    
    def _find_coregulated_pathways(
        self,
        pathway_scores: pd.DataFrame,
        correlation_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Find spatially co-regulated pathways."""
        logger.info("Finding co-regulated pathways")
        
        # Compute correlation matrix
        correlation_matrix = pathway_scores.corr()
        
        # Find highly correlated pathway pairs
        coregulated_pairs = []
        
        for i, pathway1 in enumerate(pathway_scores.columns):
            for j, pathway2 in enumerate(pathway_scores.columns):
                if i < j:  # Avoid duplicates
                    correlation = correlation_matrix.loc[pathway1, pathway2]
                    
                    if abs(correlation) >= correlation_threshold:
                        coregulated_pairs.append({
                            'pathway1': pathway1,
                            'pathway2': pathway2,
                            'correlation': correlation,
                            'correlation_type': 'positive' if correlation > 0 else 'negative'
                        })
        
        # Cluster pathways based on correlation
        pathway_clusters = self._cluster_pathways(correlation_matrix, threshold=correlation_threshold)
        
        return {
            'correlation_matrix': correlation_matrix,
            'coregulated_pairs': pd.DataFrame(coregulated_pairs),
            'pathway_clusters': pathway_clusters
        }
    
    def _cluster_pathways(self, correlation_matrix: pd.DataFrame, threshold: float) -> Dict[int, List[str]]:
        """Cluster pathways based on correlation."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert correlation to distance
        distance_matrix = 1 - abs(correlation_matrix)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            linkage='average',
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group pathways by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(correlation_matrix.index[i])
        
        return clusters
    
    def _analyze_pathway_gradients(
        self,
        adata: AnnData,
        pathway_scores: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze pathway gradients in space."""
        logger.info("Analyzing pathway gradients")
        
        coords = adata.obsm['spatial']
        gradients = {}
        
        for pathway in pathway_scores.columns:
            scores = pathway_scores[pathway].values
            
            # Compute spatial gradients
            gradient_x, gradient_y = self._compute_spatial_gradient(coords, scores)
            
            # Find gradient magnitude and direction
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_direction = np.arctan2(gradient_y, gradient_x)
            
            # Identify regions with strong gradients
            gradient_threshold = np.percentile(gradient_magnitude, 75)
            strong_gradient_regions = gradient_magnitude > gradient_threshold
            
            gradients[pathway] = {
                'gradient_x': gradient_x,
                'gradient_y': gradient_y,
                'magnitude': gradient_magnitude,
                'direction': gradient_direction,
                'strong_gradient_cells': np.where(strong_gradient_regions)[0],
                'mean_magnitude': np.mean(gradient_magnitude)
            }
        
        return gradients
    
    def _compute_spatial_gradient(self, coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial gradient using finite differences."""
        from scipy.spatial import cKDTree
        
        # Build spatial tree
        tree = cKDTree(coords)
        
        gradient_x = np.zeros_like(values)
        gradient_y = np.zeros_like(values)
        
        for i, coord in enumerate(coords):
            # Find nearest neighbors
            distances, indices = tree.query(coord, k=7)  # Include self + 6 neighbors
            
            # Exclude self
            neighbor_indices = indices[1:]
            neighbor_coords = coords[neighbor_indices]
            neighbor_values = values[neighbor_indices]
            
            if len(neighbor_indices) >= 3:
                # Compute gradient using least squares
                A = np.column_stack([
                    neighbor_coords[:, 0] - coord[0],
                    neighbor_coords[:, 1] - coord[1],
                    np.ones(len(neighbor_indices))
                ])
                b = neighbor_values - values[i]
                
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    gradient_x[i] = coeffs[0]
                    gradient_y[i] = coeffs[1]
                except np.linalg.LinAlgError:
                    gradient_x[i] = 0
                    gradient_y[i] = 0
        
        return gradient_x, gradient_y
    
    def _identify_pathway_zones(
        self,
        adata: AnnData,
        pathway_scores: pd.DataFrame
    ) -> Dict[str, Any]:
        """Identify distinct pathway activity zones."""
        logger.info("Identifying pathway zones")
        
        from sklearn.cluster import KMeans
        
        coords = adata.obsm['spatial']
        zones = {}
        
        for pathway in pathway_scores.columns:
            scores = pathway_scores[pathway].values
            
            # Combine spatial and score information for clustering
            features = np.column_stack([
                coords,
                scores.reshape(-1, 1)
            ])
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Cluster into zones
            n_zones = 3  # Low, medium, high activity zones
            kmeans = KMeans(n_clusters=n_zones, random_state=42)
            zone_labels = kmeans.fit_predict(features_scaled)
            
            # Characterize zones
            zone_info = {}
            for zone_id in range(n_zones):
                zone_mask = zone_labels == zone_id
                zone_coords = coords[zone_mask]
                zone_scores = scores[zone_mask]
                
                zone_info[zone_id] = {
                    'cell_indices': np.where(zone_mask)[0],
                    'num_cells': np.sum(zone_mask),
                    'mean_score': np.mean(zone_scores),
                    'std_score': np.std(zone_scores),
                    'centroid': np.mean(zone_coords, axis=0),
                    'area': self._compute_zone_area(zone_coords)
                }
            
            zones[pathway] = {
                'zone_labels': zone_labels,
                'zone_info': zone_info,
                'n_zones': n_zones
            }
        
        return zones
    
    def _compute_zone_area(self, coords: np.ndarray) -> float:
        """Compute approximate area of a zone using convex hull."""
        if len(coords) < 3:
            return 0.0
        
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(coords)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def plot_pathway_activity(
        self,
        adata: AnnData,
        pathway_scores: pd.DataFrame,
        pathways: Optional[List[str]] = None,
        ncols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        cmap: str = 'viridis'
    ) -> None:
        """Plot spatial pathway activity."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib required for plotting")
            return
        
        coords = adata.obsm['spatial']
        
        if pathways is None:
            pathways = pathway_scores.columns[:6]  # Plot first 6 pathways
        
        nrows = (len(pathways) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows > 1 else [axes]
        
        for i, pathway in enumerate(pathways):
            if i >= len(axes):
                break
            
            ax = axes[i]
            scores = pathway_scores[pathway]
            
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=scores, cmap=cmap, s=20, alpha=0.8
            )
            
            ax.set_title(pathway.replace('_', ' '), fontsize=10)
            ax.set_xlabel('Spatial X')
            ax.set_ylabel('Spatial Y')
            
            plt.colorbar(scatter, ax=ax)
        
        # Hide unused subplots
        for i in range(len(pathways), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pathway_gradients(
        self,
        adata: AnnData,
        gradients: Dict[str, Any],
        pathways: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Plot pathway gradients as vector fields."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib required for plotting")
            return
        
        coords = adata.obsm['spatial']
        
        if pathways is None:
            pathways = list(gradients.keys())[:4]  # Plot first 4 pathways
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, pathway in enumerate(pathways):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            gradient_info = gradients[pathway]
            magnitude = gradient_info['magnitude']
            gradient_x = gradient_info['gradient_x']
            gradient_y = gradient_info['gradient_y']
            
            # Plot magnitude as background
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=magnitude, cmap='viridis', s=20, alpha=0.6
            )
            
            # Subsample for vector field
            subsample = slice(None, None, 10)  # Every 10th point
            ax.quiver(
                coords[subsample, 0], coords[subsample, 1],
                gradient_x[subsample], gradient_y[subsample],
                angles='xy', scale_units='xy', scale=1, alpha=0.7
            )
            
            ax.set_title(f'{pathway.replace("_", " ")} Gradient')
            ax.set_xlabel('Spatial X')
            ax.set_ylabel('Spatial Y')
            
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.show()


class PathwayScoringHead(nn.Module):
    """Neural network head for pathway activity scoring."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_pathways: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_pathways = num_pathways
        
        # Pathway scoring network
        self.pathway_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_pathways),
            nn.Sigmoid()  # Pathway scores between 0 and 1
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pathway scoring.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            
        Returns:
            Pathway scores [num_nodes, num_pathways]
        """
        pathway_scores = self.pathway_scorer(node_embeddings)
        return pathway_scores


class SpatialPathwayAnalyzer(PathwayAnalyzer):
    """Specialized analyzer focusing on spatial aspects of pathway activity."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initialized specialized SpatialPathwayAnalyzer")
    
    def analyze_pathway_communication(
        self,
        adata: AnnData,
        pathway_scores: pd.DataFrame,
        interaction_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Analyze pathway-pathway communication networks.
        
        Args:
            adata: Spatial transcriptomics data
            pathway_scores: Pathway activity scores
            interaction_threshold: Threshold for pathway interactions
            
        Returns:
            Pathway communication network analysis
        """
        logger.info("Analyzing pathway communication networks")
        
        # Get spatial graph
        if 'spatial_graph' not in adata.uns:
            raise ValueError("Spatial graph not found")
        
        edge_index = adata.uns['spatial_graph']['edge_index']
        
        # Find pathway transitions along edges
        pathway_transitions = []
        
        for i in range(edge_index.shape[1]):
            source_idx = edge_index[0, i]
            target_idx = edge_index[1, i]
            
            source_pathways = pathway_scores.iloc[source_idx]
            target_pathways = pathway_scores.iloc[target_idx]
            
            # Find dominant pathways for each cell
            source_dominant = source_pathways.idxmax()
            target_dominant = target_pathways.idxmax()
            
            if (source_pathways[source_dominant] > interaction_threshold and
                target_pathways[target_dominant] > interaction_threshold and
                source_dominant != target_dominant):
                
                pathway_transitions.append({
                    'source_cell': source_idx,
                    'target_cell': target_idx,
                    'source_pathway': source_dominant,
                    'target_pathway': target_dominant,
                    'source_score': source_pathways[source_dominant],
                    'target_score': target_pathways[target_dominant]
                })
        
        transitions_df = pd.DataFrame(pathway_transitions)
        
        # Analyze pathway boundary zones
        boundary_analysis = self._analyze_pathway_boundaries(transitions_df)
        
        return {
            'pathway_transitions': transitions_df,
            'boundary_analysis': boundary_analysis
        }
    
    def _analyze_pathway_boundaries(self, transitions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pathway boundary zones."""
        if not transitions_df:
            return {'boundary_zones': [], 'transition_frequencies': pd.DataFrame()}
        
        # Count transition frequencies between pathway pairs
        transition_counts = transitions_df.groupby(
            ['source_pathway', 'target_pathway']
        ).size().reset_index(name='frequency')
        
        # Identify major boundary zones
        boundary_zones = []
        
        for _, row in transition_counts.iterrows():
            if row['frequency'] >= 5:  # Minimum 5 transitions
                boundary_zones.append({
                    'pathway_pair': f"{row['source_pathway']} - {row['target_pathway']}",
                    'frequency': row['frequency'],
                    'bidirectional': self._check_bidirectional_transition(
                        transition_counts, row['source_pathway'], row['target_pathway']
                    )
                })
        
        return {
            'boundary_zones': boundary_zones,
            'transition_frequencies': transition_counts
        }
    
    def _check_bidirectional_transition(
        self,
        transition_counts: pd.DataFrame,
        pathway1: str,
        pathway2: str
    ) -> bool:
        """Check if transition is bidirectional."""
        reverse_transition = transition_counts[
            (transition_counts['source_pathway'] == pathway2) &
            (transition_counts['target_pathway'] == pathway1)
        ]
        
        return len(reverse_transition) > 0
"""
Spatial graph construction utilities for building connectivity graphs
from spatial transcriptomics data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from scipy.spatial import cKDTree, Voronoi, distance_matrix
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import networkx as nx
from anndata import AnnData

from .base import SpatialDataConfig

logger = logging.getLogger(__name__)


class SpatialGraphBuilder:
    """
    Build spatial connectivity graphs from spatial transcriptomics data.
    
    Supports multiple graph construction methods including k-NN, radius-based,
    Delaunay triangulation, and Voronoi tessellation.
    """
    
    def __init__(self, config: Optional[SpatialDataConfig] = None):
        """Initialize graph builder with configuration."""
        self.config = config or SpatialDataConfig()
        
    def build_spatial_graph(
        self,
        coordinates: np.ndarray,
        method: str = 'knn',
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Build spatial graph from coordinates.
        
        Args:
            coordinates: Array of shape (n_cells, 2) with spatial coordinates
            method: Graph construction method
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (edge_index, edge_attr, graph_info)
        """
        logger.info(f"Building spatial graph with method: {method}")
        
        if coordinates.shape[1] not in [2, 3]:
            raise ValueError("Coordinates must be 2D or 3D")
            
        if method == 'knn':
            return self._build_knn_graph(coordinates, **kwargs)
        elif method == 'radius':
            return self._build_radius_graph(coordinates, **kwargs)
        elif method == 'delaunay':
            return self._build_delaunay_graph(coordinates, **kwargs)
        elif method == 'voronoi':
            return self._build_voronoi_graph(coordinates, **kwargs)
        elif method == 'hybrid':
            return self._build_hybrid_graph(coordinates, **kwargs)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
    
    def _build_knn_graph(
        self,
        coordinates: np.ndarray,
        k: int = 6,
        include_self: bool = False,
        metric: str = 'euclidean'
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build k-nearest neighbors graph."""
        logger.info(f"Building k-NN graph with k={k}")
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Remove self-connections unless specified
        if not include_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        
        # Build edge list
        n_cells = coordinates.shape[0]
        source_nodes = np.repeat(np.arange(n_cells), k)
        target_nodes = indices.flatten()
        edge_distances = distances.flatten()
        
        # Create edge tensors
        edge_index = torch.tensor(
            [source_nodes, target_nodes], 
            dtype=torch.long
        )
        edge_attr = self._compute_edge_features(
            coordinates, edge_index, edge_distances
        )
        
        graph_info = {
            'method': 'knn',
            'k': k,
            'n_edges': edge_index.shape[1],
            'avg_degree': edge_index.shape[1] / n_cells,
            'is_directed': True
        }
        
        return edge_index, edge_attr, graph_info
    
    def _build_radius_graph(
        self,
        coordinates: np.ndarray,
        radius: float = 100.0,
        max_neighbors: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build radius-based graph."""
        logger.info(f"Building radius graph with radius={radius}")
        
        # Build radius graph
        adj_matrix = radius_neighbors_graph(
            coordinates,
            radius=radius,
            mode='distance',
            include_self=False,
            n_jobs=-1
        )
        
        # Convert to COO format
        coo = adj_matrix.tocoo()
        edge_distances = coo.data
        
        # Limit neighbors if specified
        if max_neighbors is not None:
            edge_index, edge_distances = self._limit_neighbors(
                coo.row, coo.col, edge_distances, max_neighbors
            )
        else:
            edge_index = np.vstack([coo.row, coo.col])
            
        # Create tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = self._compute_edge_features(
            coordinates, edge_index, edge_distances
        )
        
        graph_info = {
            'method': 'radius',
            'radius': radius,
            'n_edges': edge_index.shape[1],
            'avg_degree': edge_index.shape[1] / coordinates.shape[0],
            'is_directed': False
        }
        
        return edge_index, edge_attr, graph_info
    
    def _build_delaunay_graph(
        self,
        coordinates: np.ndarray,
        max_distance: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build Delaunay triangulation graph."""
        if coordinates.shape[1] != 2:
            raise ValueError("Delaunay triangulation requires 2D coordinates")
            
        logger.info("Building Delaunay triangulation graph")
        
        from scipy.spatial import Delaunay
        
        # Compute Delaunay triangulation
        tri = Delaunay(coordinates)
        
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        # Convert to arrays
        edges = np.array(list(edges))
        source_nodes = edges[:, 0]
        target_nodes = edges[:, 1]
        
        # Compute distances
        edge_distances = np.linalg.norm(
            coordinates[source_nodes] - coordinates[target_nodes],
            axis=1
        )
        
        # Filter by distance if specified
        if max_distance is not None:
            valid_mask = edge_distances <= max_distance
            source_nodes = source_nodes[valid_mask]
            target_nodes = target_nodes[valid_mask] 
            edge_distances = edge_distances[valid_mask]
        
        # Make undirected
        edge_index = np.vstack([
            np.concatenate([source_nodes, target_nodes]),
            np.concatenate([target_nodes, source_nodes])
        ])
        edge_distances = np.concatenate([edge_distances, edge_distances])
        
        # Create tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = self._compute_edge_features(
            coordinates, edge_index, edge_distances
        )
        
        graph_info = {
            'method': 'delaunay',
            'max_distance': max_distance,
            'n_edges': edge_index.shape[1],
            'avg_degree': edge_index.shape[1] / coordinates.shape[0],
            'is_directed': False
        }
        
        return edge_index, edge_attr, graph_info
    
    def _build_voronoi_graph(
        self,
        coordinates: np.ndarray,
        max_distance: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build Voronoi tessellation graph."""
        if coordinates.shape[1] != 2:
            raise ValueError("Voronoi tessellation requires 2D coordinates")
            
        logger.info("Building Voronoi tessellation graph")
        
        # Compute Voronoi diagram
        vor = Voronoi(coordinates)
        
        # Extract edges from Voronoi diagram
        edges = []
        for ridge in vor.ridge_points:
            if len(ridge) == 2:
                edges.append(ridge)
                
        if not edges:
            logger.warning("No Voronoi edges found, falling back to Delaunay")
            return self._build_delaunay_graph(coordinates, max_distance)
        
        edges = np.array(edges)
        source_nodes = edges[:, 0]
        target_nodes = edges[:, 1]
        
        # Compute distances
        edge_distances = np.linalg.norm(
            coordinates[source_nodes] - coordinates[target_nodes],
            axis=1
        )
        
        # Filter by distance if specified
        if max_distance is not None:
            valid_mask = edge_distances <= max_distance
            source_nodes = source_nodes[valid_mask]
            target_nodes = target_nodes[valid_mask]
            edge_distances = edge_distances[valid_mask]
            
        # Make undirected
        edge_index = np.vstack([
            np.concatenate([source_nodes, target_nodes]),
            np.concatenate([target_nodes, source_nodes])
        ])
        edge_distances = np.concatenate([edge_distances, edge_distances])
        
        # Create tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = self._compute_edge_features(
            coordinates, edge_index, edge_distances
        )
        
        graph_info = {
            'method': 'voronoi',
            'max_distance': max_distance,
            'n_edges': edge_index.shape[1],
            'avg_degree': edge_index.shape[1] / coordinates.shape[0],
            'is_directed': False
        }
        
        return edge_index, edge_attr, graph_info
    
    def _build_hybrid_graph(
        self,
        coordinates: np.ndarray,
        methods: List[str] = ['knn', 'radius'],
        weights: List[float] = [0.7, 0.3],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build hybrid graph combining multiple methods."""
        logger.info(f"Building hybrid graph with methods: {methods}")
        
        if len(methods) != len(weights):
            raise ValueError("Number of methods must match number of weights")
            
        all_edges = []
        all_attrs = []
        
        for method, weight in zip(methods, weights):
            edge_index, edge_attr, _ = self.build_spatial_graph(
                coordinates, method=method, **kwargs
            )
            
            # Weight the edge attributes
            edge_attr = edge_attr * weight
            
            all_edges.append(edge_index)
            all_attrs.append(edge_attr)
        
        # Combine all edges
        combined_edge_index = torch.cat(all_edges, dim=1)
        combined_edge_attr = torch.cat(all_attrs, dim=0)
        
        # Remove duplicate edges and aggregate attributes
        edge_index, edge_attr = self._aggregate_duplicate_edges(
            combined_edge_index, combined_edge_attr
        )
        
        graph_info = {
            'method': 'hybrid',
            'methods': methods,
            'weights': weights,
            'n_edges': edge_index.shape[1],
            'avg_degree': edge_index.shape[1] / coordinates.shape[0],
            'is_directed': False
        }
        
        return edge_index, edge_attr, graph_info
    
    def _compute_edge_features(
        self,
        coordinates: np.ndarray,
        edge_index: torch.Tensor,
        distances: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Compute edge features including distance and direction.
        
        Args:
            coordinates: Spatial coordinates
            edge_index: Edge connectivity
            distances: Pre-computed distances (optional)
            
        Returns:
            Edge feature tensor
        """
        source_coords = coordinates[edge_index[0].numpy()]
        target_coords = coordinates[edge_index[1].numpy()]
        
        # Compute distances if not provided
        if distances is None:
            distances = np.linalg.norm(
                source_coords - target_coords, axis=1
            )
        
        # Compute direction vectors (normalized)
        directions = target_coords - source_coords
        direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
        direction_norms = np.where(direction_norms == 0, 1, direction_norms)
        directions = directions / direction_norms
        
        # Compute angles (for 2D coordinates)
        if coordinates.shape[1] == 2:
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            
            # Convert to sin/cos for circular encoding
            sin_angles = np.sin(angles)
            cos_angles = np.cos(angles)
            
            edge_features = np.column_stack([
                distances,
                directions,
                sin_angles,
                cos_angles
            ])
        else:
            edge_features = np.column_stack([
                distances,
                directions
            ])
        
        return torch.tensor(edge_features, dtype=torch.float32)
    
    def _limit_neighbors(
        self,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        distances: np.ndarray,
        max_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Limit the number of neighbors per node."""
        unique_sources = np.unique(source_nodes)
        limited_edges = []
        limited_distances = []
        
        for source in unique_sources:
            # Find edges from this source
            source_mask = source_nodes == source
            source_targets = target_nodes[source_mask]
            source_distances = distances[source_mask]
            
            # Sort by distance and take top k
            if len(source_targets) > max_neighbors:
                sorted_indices = np.argsort(source_distances)[:max_neighbors]
                source_targets = source_targets[sorted_indices]
                source_distances = source_distances[sorted_indices]
            
            # Add to limited edges
            limited_edges.extend(list(zip([source] * len(source_targets), source_targets)))
            limited_distances.extend(source_distances)
        
        limited_edges = np.array(limited_edges).T
        limited_distances = np.array(limited_distances)
        
        return limited_edges, limited_distances
    
    def _aggregate_duplicate_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate duplicate edges by averaging attributes."""
        # Convert to numpy for processing
        edges = edge_index.numpy().T
        attrs = edge_attr.numpy()
        
        # Create unique edge identifier
        edge_ids = edges[:, 0] * 1000000 + edges[:, 1]
        unique_ids, inverse_indices = np.unique(edge_ids, return_inverse=True)
        
        # Aggregate attributes for duplicate edges
        n_unique = len(unique_ids)
        n_features = attrs.shape[1]
        aggregated_attrs = np.zeros((n_unique, n_features))
        
        for i in range(n_unique):
            mask = inverse_indices == i
            aggregated_attrs[i] = np.mean(attrs[mask], axis=0)
        
        # Create unique edge index
        unique_edges = np.zeros((n_unique, 2), dtype=int)
        for i, edge_id in enumerate(unique_ids):
            unique_edges[i] = [edge_id // 1000000, edge_id % 1000000]
        
        return (
            torch.tensor(unique_edges.T, dtype=torch.long),
            torch.tensor(aggregated_attrs, dtype=torch.float32)
        )
    
    def add_graph_to_adata(
        self,
        adata: AnnData,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_info: Dict[str, Any],
        coord_key: str = 'spatial'
    ) -> AnnData:
        """
        Add graph information to AnnData object.
        
        Args:
            adata: AnnData object
            edge_index: Graph edge indices
            edge_attr: Graph edge attributes
            graph_info: Graph metadata
            coord_key: Key for spatial coordinates
            
        Returns:
            AnnData with graph information added
        """
        logger.info("Adding graph information to AnnData")
        
        # Store graph in uns
        adata.uns[f'{coord_key}_graph'] = {
            'edge_index': edge_index.numpy(),
            'edge_attr': edge_attr.numpy(),
            'info': graph_info
        }
        
        # Create adjacency matrix for compatibility
        n_cells = adata.n_obs
        adj_matrix = csr_matrix(
            (np.ones(edge_index.shape[1]), 
             (edge_index[0].numpy(), edge_index[1].numpy())),
            shape=(n_cells, n_cells)
        )
        
        adata.obsp[f'{coord_key}_connectivities'] = adj_matrix
        
        # Store distances as weights
        if edge_attr.shape[1] > 0:
            distance_matrix = csr_matrix(
                (edge_attr[:, 0].numpy(),
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(n_cells, n_cells)
            )
            adata.obsp[f'{coord_key}_distances'] = distance_matrix
        
        return adata
    
    def create_pytorch_geometric_data(
        self,
        adata: AnnData,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coord_key: str = 'spatial'
    ) -> Data:
        """
        Create PyTorch Geometric Data object.
        
        Args:
            adata: AnnData object
            edge_index: Graph edges
            edge_attr: Edge attributes
            coord_key: Key for coordinates
            
        Returns:
            PyTorch Geometric Data object
        """
        # Convert expression data to tensor
        if hasattr(adata.X, 'toarray'):
            x = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            x = torch.tensor(adata.X, dtype=torch.float32)
        
        # Get coordinates
        pos = torch.tensor(adata.obsm[coord_key], dtype=torch.float32)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )
        
        # Add metadata
        if adata.obs.shape[1] > 0:
            for col in adata.obs.columns:
                if adata.obs[col].dtype.kind in 'biufc':  # Numeric columns
                    data[f'obs_{col}'] = torch.tensor(
                        adata.obs[col].values, dtype=torch.float32
                    )
        
        return data
    
    def validate_graph(
        self,
        edge_index: torch.Tensor,
        n_nodes: int,
        check_connectivity: bool = True
    ) -> Dict[str, Any]:
        """
        Validate graph structure and properties.
        
        Args:
            edge_index: Graph edges
            n_nodes: Number of nodes
            check_connectivity: Whether to check graph connectivity
            
        Returns:
            Validation report
        """
        logger.info("Validating graph structure")
        
        report = {
            'n_nodes': n_nodes,
            'n_edges': edge_index.shape[1],
            'is_undirected': self._is_undirected(edge_index),
            'has_self_loops': self._has_self_loops(edge_index),
            'avg_degree': edge_index.shape[1] / n_nodes
        }
        
        # Check for isolated nodes
        unique_nodes = torch.unique(edge_index).numpy()
        isolated_nodes = set(range(n_nodes)) - set(unique_nodes)
        report['n_isolated_nodes'] = len(isolated_nodes)
        
        # Check connectivity if requested
        if check_connectivity and not isolated_nodes:
            adj_matrix = csr_matrix(
                (np.ones(edge_index.shape[1]),
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(n_nodes, n_nodes)
            )
            
            # Convert to NetworkX graph
            G = nx.from_scipy_sparse_array(adj_matrix)
            report['is_connected'] = nx.is_connected(G)
            report['n_components'] = nx.number_connected_components(G)
            
            if report['is_connected']:
                report['diameter'] = nx.diameter(G)
                report['avg_clustering'] = nx.average_clustering(G)
        
        logger.info(f"Graph validation: {report['n_edges']} edges, avg degree: {report['avg_degree']:.2f}")
        
        return report
    
    def _is_undirected(self, edge_index: torch.Tensor) -> bool:
        """Check if graph is undirected."""
        # Sort edges to compare
        sorted_edges = torch.sort(edge_index, dim=0)[0]
        forward_edges = set(map(tuple, sorted_edges.T.numpy()))
        
        # Check if all edges have reverse counterparts
        for i, j in edge_index.T.numpy():
            if (min(i,j), max(i,j)) not in forward_edges:
                return False
        return True
    
    def _has_self_loops(self, edge_index: torch.Tensor) -> bool:
        """Check if graph has self loops."""
        return (edge_index[0] == edge_index[1]).any().item()
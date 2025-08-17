"""
Helper functions for common operations.

This module provides utility functions for spatial computations,
data transformations, and other common operations.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

import numpy as np
from typing import Tuple, Union, Optional, List
import warnings


def compute_spatial_distance(
    coords1: Union[np.ndarray, 'torch.Tensor'],
    coords2: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
    metric: str = "euclidean"
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Compute spatial distances between coordinates.
    
    Args:
        coords1: First set of coordinates [N, 2]
        coords2: Second set of coordinates [M, 2] (optional)
        metric: Distance metric ('euclidean', 'manhattan')
        
    Returns:
        Distance matrix [N, M] or [N, N] if coords2 is None
    """
    if coords2 is None:
        coords2 = coords1
    
    # Convert to numpy if torch not available
    if not TORCH_AVAILABLE:
        if hasattr(coords1, 'detach'):  # torch tensor
            coords1 = coords1.detach().cpu().numpy()
        if hasattr(coords2, 'detach'):  # torch tensor
            coords2 = coords2.detach().cpu().numpy()
        
        # Use numpy implementation
        return _compute_distance_numpy(coords1, coords2, metric)
    
    # Use torch implementation if available
    if isinstance(coords1, np.ndarray):
        coords1 = torch.from_numpy(coords1).float()
    if isinstance(coords2, np.ndarray):
        coords2 = torch.from_numpy(coords2).float()
    
    # Expand dimensions for broadcasting
    coords1_expanded = coords1.unsqueeze(1)  # [N, 1, 2]
    coords2_expanded = coords2.unsqueeze(0)  # [1, M, 2]
    
    if metric == "euclidean":
        distances = torch.norm(coords1_expanded - coords2_expanded, dim=2)
    elif metric == "manhattan":
        distances = torch.sum(torch.abs(coords1_expanded - coords2_expanded), dim=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def _compute_distance_numpy(coords1: np.ndarray, coords2: np.ndarray, metric: str) -> np.ndarray:
    """Numpy-based distance computation fallback."""
    # Expand dimensions for broadcasting
    coords1_expanded = coords1[:, np.newaxis, :]  # [N, 1, 2]
    coords2_expanded = coords2[np.newaxis, :, :]  # [1, M, 2]
    
    if metric == "euclidean":
        distances = np.linalg.norm(coords1_expanded - coords2_expanded, axis=2)
    elif metric == "manhattan":
        distances = np.sum(np.abs(coords1_expanded - coords2_expanded), axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def normalize_coordinates(
    coords: Union[np.ndarray, 'torch.Tensor'],
    method: str = "min_max"
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Normalize spatial coordinates.
    
    Args:
        coords: Coordinates to normalize [N, 2]
        method: Normalization method ('min_max', 'z_score', 'unit_circle')
        
    Returns:
        Normalized coordinates
    """
    is_numpy = isinstance(coords, np.ndarray)
    
    # Use numpy implementation if torch not available
    if not TORCH_AVAILABLE:
        return _normalize_coordinates_numpy(coords, method)
    
    if is_numpy:
        coords_tensor = torch.from_numpy(coords).float()
    else:
        coords_tensor = coords.float()
    
    if method == "min_max":
        # Scale to [0, 1]
        min_vals = torch.min(coords_tensor, dim=0)[0]
        max_vals = torch.max(coords_tensor, dim=0)[0]
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
        
        normalized = (coords_tensor - min_vals) / ranges
        
    elif method == "z_score":
        # Standard normalization
        mean_vals = torch.mean(coords_tensor, dim=0)
        std_vals = torch.std(coords_tensor, dim=0)
        
        # Avoid division by zero
        std_vals = torch.where(std_vals == 0, torch.ones_like(std_vals), std_vals)
        
        normalized = (coords_tensor - mean_vals) / std_vals
        
    elif method == "unit_circle":
        # Scale to unit circle
        center = torch.mean(coords_tensor, dim=0)
        centered = coords_tensor - center
        
        max_distance = torch.max(torch.norm(centered, dim=1))
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_numpy:
        return normalized.numpy()
    return normalized


def _normalize_coordinates_numpy(coords: np.ndarray, method: str) -> np.ndarray:
    """Numpy-based coordinate normalization fallback."""
    if method == "min_max":
        # Scale to [0, 1]
        min_vals = np.min(coords, axis=0)
        max_vals = np.max(coords, axis=0)
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1.0, ranges)
        
        normalized = (coords - min_vals) / ranges
        
    elif method == "z_score":
        # Standard normalization
        mean_vals = np.mean(coords, axis=0)
        std_vals = np.std(coords, axis=0)
        
        # Avoid division by zero
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        
        normalized = (coords - mean_vals) / std_vals
        
    elif method == "unit_circle":
        # Scale to unit circle
        center = np.mean(coords, axis=0)
        centered = coords - center
        
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def _create_spatial_graph_numpy(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 6,
    radius: Optional[float] = None,
    include_self: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-based spatial graph creation fallback."""
    num_nodes = coords.shape[0]
    
    # Compute distance matrix
    distances = _compute_distance_numpy(coords, coords, "euclidean")
    
    if method == "knn":
        # For each node, find k nearest neighbors
        edge_list = []
        edge_weights = []
        
        for i in range(num_nodes):
            node_distances = distances[i]
            if not include_self:
                node_distances[i] = np.inf  # Exclude self
            
            # Get k nearest neighbors
            nearest_indices = np.argsort(node_distances)[:k]
            
            for j in nearest_indices:
                if node_distances[j] != np.inf:  # Valid neighbor
                    edge_list.append([i, j])
                    edge_weights.append(node_distances[j])
        
        edge_index = np.array(edge_list).T if edge_list else np.array([[], []])
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        
    elif method == "radius":
        if radius is None:
            radius = 1.0
        
        edge_list = []
        edge_weights = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j and not include_self:
                    continue
                
                if distances[i, j] <= radius:
                    edge_list.append([i, j])
                    edge_weights.append(distances[i, j])
        
        edge_index = np.array(edge_list).T if edge_list else np.array([[], []])
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        
    else:
        raise ValueError(f"Unknown graph method: {method}")
    
    return edge_index, edge_weights


def create_spatial_graph(
    coords: Union[np.ndarray, 'torch.Tensor'],
    method: str = "knn",
    k: int = 6,
    radius: Optional[float] = None,
    include_self: bool = False
) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Create spatial graph from coordinates.
    
    Args:
        coords: Spatial coordinates [N, 2]
        method: Graph construction method ('knn', 'radius')
        k: Number of neighbors for knn
        radius: Radius for radius graph
        include_self: Whether to include self-loops
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    if not TORCH_AVAILABLE:
        # Use numpy fallback
        return _create_spatial_graph_numpy(coords, method, k, radius, include_self)
    
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).float()
    
    num_nodes = coords.size(0)
    
    # Compute distance matrix
    distances = compute_spatial_distance(coords, coords)
    
    if method == "knn":
        # K-nearest neighbors
        _, indices = torch.topk(distances, k + 1, dim=1, largest=False)
        
        # Remove self-loops if not requested
        if not include_self:
            indices = indices[:, 1:]  # Skip first column (self)
        else:
            indices = indices[:, :k]
        
        # Create edge indices
        source_nodes = torch.arange(num_nodes).unsqueeze(1).expand(-1, indices.size(1))
        edge_index = torch.stack([source_nodes.flatten(), indices.flatten()], dim=0)
        
        # Get edge weights (distances)
        edge_weights = distances[source_nodes.flatten(), indices.flatten()]
        
    elif method == "radius":
        if radius is None:
            raise ValueError("Radius must be specified for radius graph")
        
        # Find all pairs within radius
        within_radius = distances <= radius
        
        if not include_self:
            # Remove self-loops
            within_radius.fill_diagonal_(False)
        
        # Get edge indices
        edge_index = torch.nonzero(within_radius, as_tuple=False).t()
        edge_weights = distances[within_radius]
        
    else:
        raise ValueError(f"Unknown graph method: {method}")
    
    return edge_index, edge_weights


def batch_process(
    data: Union[np.ndarray, 'torch.Tensor'],
    batch_size: int,
    process_fn: callable,
    **kwargs
) -> List[Union[np.ndarray, 'torch.Tensor']]:
    """
    Process data in batches.
    
    Args:
        data: Data to process [N, ...]
        batch_size: Size of each batch
        process_fn: Function to apply to each batch
        **kwargs: Additional arguments for process_fn
        
    Returns:
        List of processed batches
    """
    num_samples = data.shape[0]
    results = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = data[i:end_idx]
        
        result = process_fn(batch, **kwargs)
        results.append(result)
    
    return results


def filter_low_count_features(
    expression_matrix: Union[np.ndarray, 'torch.Tensor'],
    min_counts: int = 1,
    min_cells: int = 3
) -> Tuple[Union[np.ndarray, 'torch.Tensor'], np.ndarray]:
    """
    Filter genes with low counts.
    
    Args:
        expression_matrix: Gene expression matrix [cells, genes]
        min_counts: Minimum total counts per gene
        min_cells: Minimum number of cells expressing gene
        
    Returns:
        Tuple of (filtered_matrix, gene_mask)
    """
    is_numpy = isinstance(expression_matrix, np.ndarray)
    
    if not is_numpy:
        expr_np = expression_matrix.numpy()
    else:
        expr_np = expression_matrix
    
    # Calculate gene statistics
    total_counts = np.sum(expr_np, axis=0)
    num_cells_expressing = np.sum(expr_np > 0, axis=0)
    
    # Create filter mask
    gene_mask = (total_counts >= min_counts) & (num_cells_expressing >= min_cells)
    
    # Apply filter
    filtered_matrix = expr_np[:, gene_mask]
    
    if not is_numpy:
        filtered_matrix = torch.from_numpy(filtered_matrix)
    
    return filtered_matrix, gene_mask


def compute_highly_variable_genes(
    expression_matrix: Union[np.ndarray, 'torch.Tensor'],
    n_top_genes: int = 3000,
    flavor: str = "seurat"
) -> np.ndarray:
    """
    Identify highly variable genes.
    
    Args:
        expression_matrix: Gene expression matrix [cells, genes]
        n_top_genes: Number of top variable genes to select
        flavor: Method for variance calculation ('seurat', 'cell_ranger')
        
    Returns:
        Boolean mask for highly variable genes
    """
    if TORCH_AVAILABLE and hasattr(expression_matrix, 'numpy'):
        expr_np = expression_matrix.numpy()
    else:
        expr_np = expression_matrix
    
    # Compute mean and variance
    mean_expr = np.mean(expr_np, axis=0)
    var_expr = np.var(expr_np, axis=0)
    
    if flavor == "seurat":
        # Seurat method: variance/mean ratio
        # Avoid division by zero
        mean_expr = np.where(mean_expr == 0, 1e-8, mean_expr)
        dispersion = var_expr / mean_expr
        
    elif flavor == "cell_ranger":
        # Cell Ranger method: coefficient of variation
        std_expr = np.sqrt(var_expr)
        mean_expr = np.where(mean_expr == 0, 1e-8, mean_expr)
        dispersion = std_expr / mean_expr
        
    else:
        raise ValueError(f"Unknown flavor: {flavor}")
    
    # Select top variable genes
    top_gene_indices = np.argsort(dispersion)[-n_top_genes:]
    
    # Create boolean mask
    hvg_mask = np.zeros(len(dispersion), dtype=bool)
    hvg_mask[top_gene_indices] = True
    
    return hvg_mask


def safe_log_transform(
    data: Union[np.ndarray, 'torch.Tensor'],
    pseudocount: float = 1.0
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Apply safe log transformation.
    
    Args:
        data: Data to transform
        pseudocount: Pseudocount to add before log
        
    Returns:
        Log-transformed data
    """
    if TORCH_AVAILABLE and hasattr(data, 'numpy'):
        return torch.log(data + pseudocount)
    else:
        return np.log(data + pseudocount)


def standardize_features(
    data: Union[np.ndarray, 'torch.Tensor'],
    axis: int = 0
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        data: Data to standardize
        axis: Axis along which to compute statistics
        
    Returns:
        Standardized data
    """
    if TORCH_AVAILABLE and hasattr(data, 'numpy'):
        mean = torch.mean(data, dim=axis, keepdim=True)
        std = torch.std(data, dim=axis, keepdim=True)
        
        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        return (data - mean) / std
    else:
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return (data - mean) / std
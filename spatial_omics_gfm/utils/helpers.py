"""
Helper functions for common operations.

This module provides utility functions for spatial computations,
data transformations, and other common operations.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional, List
import warnings


def compute_spatial_distance(
    coords1: Union[torch.Tensor, np.ndarray],
    coords2: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metric: str = "euclidean"
) -> Union[torch.Tensor, np.ndarray]:
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


def normalize_coordinates(
    coords: Union[torch.Tensor, np.ndarray],
    method: str = "min_max"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize spatial coordinates.
    
    Args:
        coords: Coordinates to normalize [N, 2]
        method: Normalization method ('min_max', 'z_score', 'unit_circle')
        
    Returns:
        Normalized coordinates
    """
    is_numpy = isinstance(coords, np.ndarray)
    
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


def create_spatial_graph(
    coords: Union[torch.Tensor, np.ndarray],
    method: str = "knn",
    k: int = 6,
    radius: Optional[float] = None,
    include_self: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int,
    process_fn: callable,
    **kwargs
) -> List[Union[torch.Tensor, np.ndarray]]:
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
    expression_matrix: Union[torch.Tensor, np.ndarray],
    min_counts: int = 1,
    min_cells: int = 3
) -> Tuple[Union[torch.Tensor, np.ndarray], np.ndarray]:
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
    expression_matrix: Union[torch.Tensor, np.ndarray],
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
    if isinstance(expression_matrix, torch.Tensor):
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
    data: Union[torch.Tensor, np.ndarray],
    pseudocount: float = 1.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply safe log transformation.
    
    Args:
        data: Data to transform
        pseudocount: Pseudocount to add before log
        
    Returns:
        Log-transformed data
    """
    if isinstance(data, torch.Tensor):
        return torch.log(data + pseudocount)
    else:
        return np.log(data + pseudocount)


def standardize_features(
    data: Union[torch.Tensor, np.ndarray],
    axis: int = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        data: Data to standardize
        axis: Axis along which to compute statistics
        
    Returns:
        Standardized data
    """
    if isinstance(data, torch.Tensor):
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
"""
CUDA Kernel Optimizations for Spatial Operations.

This module implements optimized CUDA kernels for spatial transcriptomics operations:
- Spatial neighbor computation with GPU acceleration
- Distance matrix calculations with memory optimization
- Graph construction kernels for large-scale data
- Attention computation optimizations
"""

import os
import sys
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

logger = __import__('logging').getLogger(__name__)


# CUDA kernel source code
SPATIAL_KERNELS_CUDA = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

// CUDA kernel for computing spatial neighbors
__global__ void spatial_neighbors_kernel(
    const float* __restrict__ coords,
    int* __restrict__ neighbors,
    float* __restrict__ distances,
    const int n_points,
    const int spatial_dim,
    const int max_neighbors,
    const float max_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    float x = coords[idx * spatial_dim];
    float y = coords[idx * spatial_dim + 1];
    float z = (spatial_dim > 2) ? coords[idx * spatial_dim + 2] : 0.0f;
    
    int neighbor_count = 0;
    
    for (int i = 0; i < n_points && neighbor_count < max_neighbors; i++) {
        if (i == idx) continue;
        
        float dx = coords[i * spatial_dim] - x;
        float dy = coords[i * spatial_dim + 1] - y;
        float dz = (spatial_dim > 2) ? coords[i * spatial_dim + 2] - z : 0.0f;
        
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        
        if (dist <= max_distance) {
            neighbors[idx * max_neighbors + neighbor_count] = i;
            distances[idx * max_neighbors + neighbor_count] = dist;
            neighbor_count++;
        }
    }
    
    // Fill remaining slots with -1
    for (int i = neighbor_count; i < max_neighbors; i++) {
        neighbors[idx * max_neighbors + i] = -1;
        distances[idx * max_neighbors + i] = -1.0f;
    }
}

// CUDA kernel for computing pairwise distances
__global__ void pairwise_distances_kernel(
    const float* __restrict__ coords1,
    const float* __restrict__ coords2,
    float* __restrict__ distances,
    const int n_points1,
    const int n_points2,
    const int spatial_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= n_points1 || idy >= n_points2) return;
    
    float dist_squared = 0.0f;
    
    for (int d = 0; d < spatial_dim; d++) {
        float diff = coords1[idx * spatial_dim + d] - coords2[idy * spatial_dim + d];
        dist_squared += diff * diff;
    }
    
    distances[idx * n_points2 + idy] = sqrtf(dist_squared);
}

// CUDA kernel for k-nearest neighbors search
__global__ void knn_search_kernel(
    const float* __restrict__ coords,
    int* __restrict__ indices,
    float* __restrict__ distances,
    const int n_points,
    const int spatial_dim,
    const int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    float query_coord[3];  // Support up to 3D
    for (int d = 0; d < spatial_dim; d++) {
        query_coord[d] = coords[idx * spatial_dim + d];
    }
    
    // Simple insertion sort for k neighbors
    float max_dist[64];  // Assuming k <= 64
    int max_idx[64];
    
    int effective_k = min(k, n_points - 1);
    
    // Initialize with large distances
    for (int i = 0; i < effective_k; i++) {
        max_dist[i] = INFINITY;
        max_idx[i] = -1;
    }
    
    for (int i = 0; i < n_points; i++) {
        if (i == idx) continue;
        
        float dist_squared = 0.0f;
        for (int d = 0; d < spatial_dim; d++) {
            float diff = coords[i * spatial_dim + d] - query_coord[d];
            dist_squared += diff * diff;
        }
        float dist = sqrtf(dist_squared);
        
        // Insert into sorted list if closer than farthest
        if (dist < max_dist[effective_k - 1]) {
            max_dist[effective_k - 1] = dist;
            max_idx[effective_k - 1] = i;
            
            // Bubble up to maintain sorted order
            for (int j = effective_k - 1; j > 0 && max_dist[j] < max_dist[j-1]; j--) {
                float temp_dist = max_dist[j];
                int temp_idx = max_idx[j];
                max_dist[j] = max_dist[j-1];
                max_idx[j] = max_idx[j-1];
                max_dist[j-1] = temp_dist;
                max_idx[j-1] = temp_idx;
            }
        }
    }
    
    // Copy results
    for (int i = 0; i < effective_k; i++) {
        indices[idx * k + i] = max_idx[i];
        distances[idx * k + i] = max_dist[i];
    }
    
    // Fill remaining with -1
    for (int i = effective_k; i < k; i++) {
        indices[idx * k + i] = -1;
        distances[idx * k + i] = -1.0f;
    }
}

// CUDA kernel for spatial attention weights
__global__ void spatial_attention_kernel(
    const float* __restrict__ spatial_coords,
    const int* __restrict__ edge_index,
    float* __restrict__ spatial_bias,
    const int n_edges,
    const int spatial_dim,
    const float temperature
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx >= n_edges) return;
    
    int src = edge_index[edge_idx];
    int dst = edge_index[edge_idx + n_edges];
    
    float dist_squared = 0.0f;
    for (int d = 0; d < spatial_dim; d++) {
        float diff = spatial_coords[src * spatial_dim + d] - spatial_coords[dst * spatial_dim + d];
        dist_squared += diff * diff;
    }
    
    float distance = sqrtf(dist_squared);
    spatial_bias[edge_idx] = expf(-distance / temperature);
}

// CUDA kernel for graph construction
__global__ void construct_graph_kernel(
    const float* __restrict__ coords,
    const int* __restrict__ neighbors,
    const float* __restrict__ distances,
    int* __restrict__ edge_index,
    float* __restrict__ edge_attr,
    const int n_points,
    const int max_neighbors,
    const int feature_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    for (int i = 0; i < max_neighbors; i++) {
        int neighbor = neighbors[idx * max_neighbors + i];
        if (neighbor == -1) break;
        
        int edge_idx = idx * max_neighbors + i;
        
        // Set edge connectivity
        edge_index[edge_idx] = idx;
        edge_index[edge_idx + n_points * max_neighbors] = neighbor;
        
        // Set edge features
        float dist = distances[idx * max_neighbors + i];
        
        if (feature_dim >= 1) edge_attr[edge_idx * feature_dim] = dist;
        
        if (feature_dim >= 2) {
            // Direction vector
            float dx = coords[neighbor * 2] - coords[idx * 2];
            float dy = coords[neighbor * 2 + 1] - coords[idx * 2 + 1];
            edge_attr[edge_idx * feature_dim + 1] = atan2f(dy, dx);
        }
        
        if (feature_dim >= 3) {
            // Normalized distance
            edge_attr[edge_idx * feature_dim + 2] = dist / 100.0f;  // Assuming typical scale
        }
    }
}

// C++ interface functions
torch::Tensor spatial_neighbors_cuda(
    torch::Tensor coords,
    int max_neighbors,
    float max_distance
) {
    const auto n_points = coords.size(0);
    const auto spatial_dim = coords.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto neighbors = torch::full({n_points, max_neighbors}, -1, options);
    auto distances = torch::full({n_points, max_neighbors}, -1.0f, 
                                coords.options());
    
    const int threads = 256;
    const int blocks = (n_points + threads - 1) / threads;
    
    spatial_neighbors_kernel<<<blocks, threads>>>(
        coords.data_ptr<float>(),
        neighbors.data_ptr<int>(),
        distances.data_ptr<float>(),
        n_points,
        spatial_dim,
        max_neighbors,
        max_distance
    );
    
    return neighbors;
}

torch::Tensor pairwise_distances_cuda(
    torch::Tensor coords1,
    torch::Tensor coords2
) {
    const auto n_points1 = coords1.size(0);
    const auto n_points2 = coords2.size(0);
    const auto spatial_dim = coords1.size(1);
    
    auto distances = torch::zeros({n_points1, n_points2}, coords1.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((n_points1 + threads.x - 1) / threads.x,
                      (n_points2 + threads.y - 1) / threads.y);
    
    pairwise_distances_kernel<<<blocks, threads>>>(
        coords1.data_ptr<float>(),
        coords2.data_ptr<float>(),
        distances.data_ptr<float>(),
        n_points1,
        n_points2,
        spatial_dim
    );
    
    return distances;
}

std::tuple<torch::Tensor, torch::Tensor> knn_search_cuda(
    torch::Tensor coords,
    int k
) {
    const auto n_points = coords.size(0);
    const auto spatial_dim = coords.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto indices = torch::full({n_points, k}, -1, options);
    auto distances = torch::full({n_points, k}, -1.0f, coords.options());
    
    const int threads = 256;
    const int blocks = (n_points + threads - 1) / threads;
    
    knn_search_kernel<<<blocks, threads>>>(
        coords.data_ptr<float>(),
        indices.data_ptr<int>(),
        distances.data_ptr<float>(),
        n_points,
        spatial_dim,
        k
    );
    
    return std::make_tuple(indices, distances);
}

torch::Tensor spatial_attention_cuda(
    torch::Tensor spatial_coords,
    torch::Tensor edge_index,
    float temperature
) {
    const auto n_edges = edge_index.size(1);
    const auto spatial_dim = spatial_coords.size(1);
    
    auto spatial_bias = torch::zeros({n_edges}, spatial_coords.options());
    
    const int threads = 256;
    const int blocks = (n_edges + threads - 1) / threads;
    
    spatial_attention_kernel<<<blocks, threads>>>(
        spatial_coords.data_ptr<float>(),
        edge_index.data_ptr<int>(),
        spatial_bias.data_ptr<float>(),
        n_edges,
        spatial_dim,
        temperature
    );
    
    return spatial_bias;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_neighbors", &spatial_neighbors_cuda, "Spatial neighbors CUDA");
    m.def("pairwise_distances", &pairwise_distances_cuda, "Pairwise distances CUDA");
    m.def("knn_search", &knn_search_cuda, "KNN search CUDA");
    m.def("spatial_attention", &spatial_attention_cuda, "Spatial attention CUDA");
}
"""

# C++ CPU fallback implementations
SPATIAL_KERNELS_CPP = """
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

torch::Tensor spatial_neighbors_cpu(
    torch::Tensor coords,
    int max_neighbors,
    float max_distance
) {
    const auto n_points = coords.size(0);
    const auto spatial_dim = coords.size(1);
    
    auto neighbors = torch::full({n_points, max_neighbors}, -1, torch::kInt32);
    auto coords_a = coords.accessor<float, 2>();
    auto neighbors_a = neighbors.accessor<int, 2>();
    
    for (int i = 0; i < n_points; i++) {
        std::vector<std::pair<float, int>> candidates;
        
        for (int j = 0; j < n_points; j++) {
            if (i == j) continue;
            
            float dist_sq = 0;
            for (int d = 0; d < spatial_dim; d++) {
                float diff = coords_a[i][d] - coords_a[j][d];
                dist_sq += diff * diff;
            }
            float dist = std::sqrt(dist_sq);
            
            if (dist <= max_distance) {
                candidates.emplace_back(dist, j);
            }
        }
        
        // Sort by distance and take top k
        std::sort(candidates.begin(), candidates.end());
        
        int count = std::min(static_cast<int>(candidates.size()), max_neighbors);
        for (int k = 0; k < count; k++) {
            neighbors_a[i][k] = candidates[k].second;
        }
    }
    
    return neighbors;
}

torch::Tensor pairwise_distances_cpu(
    torch::Tensor coords1,
    torch::Tensor coords2
) {
    const auto n_points1 = coords1.size(0);
    const auto n_points2 = coords2.size(0);
    const auto spatial_dim = coords1.size(1);
    
    auto distances = torch::zeros({n_points1, n_points2}, coords1.options());
    auto coords1_a = coords1.accessor<float, 2>();
    auto coords2_a = coords2.accessor<float, 2>();
    auto distances_a = distances.accessor<float, 2>();
    
    for (int i = 0; i < n_points1; i++) {
        for (int j = 0; j < n_points2; j++) {
            float dist_sq = 0;
            for (int d = 0; d < spatial_dim; d++) {
                float diff = coords1_a[i][d] - coords2_a[j][d];
                dist_sq += diff * diff;
            }
            distances_a[i][j] = std::sqrt(dist_sq);
        }
    }
    
    return distances;
}

std::tuple<torch::Tensor, torch::Tensor> knn_search_cpu(
    torch::Tensor coords,
    int k
) {
    const auto n_points = coords.size(0);
    const auto spatial_dim = coords.size(1);
    
    auto indices = torch::full({n_points, k}, -1, torch::kInt32);
    auto distances = torch::full({n_points, k}, -1.0f, coords.options());
    
    auto coords_a = coords.accessor<float, 2>();
    auto indices_a = indices.accessor<int, 2>();
    auto distances_a = distances.accessor<float, 2>();
    
    for (int i = 0; i < n_points; i++) {
        std::vector<std::pair<float, int>> candidates;
        
        for (int j = 0; j < n_points; j++) {
            if (i == j) continue;
            
            float dist_sq = 0;
            for (int d = 0; d < spatial_dim; d++) {
                float diff = coords_a[i][d] - coords_a[j][d];
                dist_sq += diff * diff;
            }
            candidates.emplace_back(std::sqrt(dist_sq), j);
        }
        
        std::sort(candidates.begin(), candidates.end());
        
        int actual_k = std::min(k, static_cast<int>(candidates.size()));
        for (int j = 0; j < actual_k; j++) {
            indices_a[i][j] = candidates[j].second;
            distances_a[i][j] = candidates[j].first;
        }
    }
    
    return std::make_tuple(indices, distances);
}

torch::Tensor spatial_attention_cpu(
    torch::Tensor spatial_coords,
    torch::Tensor edge_index,
    float temperature
) {
    const auto n_edges = edge_index.size(1);
    const auto spatial_dim = spatial_coords.size(1);
    
    auto spatial_bias = torch::zeros({n_edges}, spatial_coords.options());
    auto coords_a = spatial_coords.accessor<float, 2>();
    auto edge_a = edge_index.accessor<int, 2>();
    auto bias_a = spatial_bias.accessor<float, 1>();
    
    for (int e = 0; e < n_edges; e++) {
        int src = edge_a[0][e];
        int dst = edge_a[1][e];
        
        float dist_sq = 0;
        for (int d = 0; d < spatial_dim; d++) {
            float diff = coords_a[src][d] - coords_a[dst][d];
            dist_sq += diff * diff;
        }
        
        float distance = std::sqrt(dist_sq);
        bias_a[e] = std::exp(-distance / temperature);
    }
    
    return spatial_bias;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_neighbors", &spatial_neighbors_cpu, "Spatial neighbors CPU");
    m.def("pairwise_distances", &pairwise_distances_cpu, "Pairwise distances CPU");
    m.def("knn_search", &knn_search_cpu, "KNN search CPU");
    m.def("spatial_attention", &spatial_attention_cpu, "Spatial attention CPU");
}
"""


class CUDAKernelManager:
    """Manager for CUDA kernel compilation and execution."""
    
    def __init__(self, enable_cuda: bool = True):
        self.enable_cuda = enable_cuda and torch.cuda.is_available()
        self.kernels_compiled = False
        self.cuda_kernels = None
        self.cpu_kernels = None
        
        self._compile_kernels()
        
        logger.info(f"CUDAKernelManager initialized (CUDA: {self.enable_cuda})")
    
    def _compile_kernels(self) -> None:
        """Compile CUDA and CPU kernels."""
        try:
            # Compile CUDA kernels if available
            if self.enable_cuda:
                self.cuda_kernels = self._compile_cuda_kernels()
            
            # Always compile CPU fallback
            self.cpu_kernels = self._compile_cpu_kernels()
            
            self.kernels_compiled = True
            logger.info("Spatial kernels compiled successfully")
            
        except Exception as e:
            logger.warning(f"Kernel compilation failed: {e}")
            self.kernels_compiled = False
    
    def _compile_cuda_kernels(self):
        """Compile CUDA kernels."""
        # Create temporary files for compilation
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(SPATIAL_KERNELS_CUDA)
            cuda_file = f.name
        
        try:
            # Compile CUDA extension
            cuda_kernels = load(
                name='spatial_kernels_cuda',
                sources=[cuda_file],
                verbose=False,
                with_cuda=True
            )
            
            return cuda_kernels
            
        finally:
            # Clean up temporary file
            os.unlink(cuda_file)
    
    def _compile_cpu_kernels(self):
        """Compile CPU fallback kernels."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(SPATIAL_KERNELS_CPP)
            cpp_file = f.name
        
        try:
            # Compile CPU extension
            cpu_kernels = load(
                name='spatial_kernels_cpu',
                sources=[cpp_file],
                verbose=False,
                with_cuda=False
            )
            
            return cpu_kernels
            
        finally:
            # Clean up temporary file
            os.unlink(cpp_file)
    
    def get_kernels(self):
        """Get appropriate kernels (CUDA or CPU)."""
        if not self.kernels_compiled:
            raise RuntimeError("Kernels not compiled")
        
        if self.enable_cuda and self.cuda_kernels:
            return self.cuda_kernels
        elif self.cpu_kernels:
            return self.cpu_kernels
        else:
            raise RuntimeError("No kernels available")


# Global kernel manager instance
_kernel_manager = None


def get_kernel_manager() -> CUDAKernelManager:
    """Get global kernel manager instance."""
    global _kernel_manager
    
    if _kernel_manager is None:
        _kernel_manager = CUDAKernelManager()
    
    return _kernel_manager


class OptimizedSpatialOps:
    """Optimized spatial operations using CUDA kernels."""
    
    def __init__(self, enable_cuda: bool = True):
        self.kernel_manager = CUDAKernelManager(enable_cuda)
    
    def spatial_neighbors(
        self,
        coords: torch.Tensor,
        max_neighbors: int = 10,
        max_distance: float = 100.0
    ) -> torch.Tensor:
        """
        Find spatial neighbors using optimized kernels.
        
        Args:
            coords: Spatial coordinates [N, spatial_dim]
            max_neighbors: Maximum number of neighbors per point
            max_distance: Maximum distance for neighbors
            
        Returns:
            Neighbor indices [N, max_neighbors]
        """
        if not self.kernel_manager.kernels_compiled:
            # Fallback to PyTorch implementation
            return self._spatial_neighbors_pytorch(coords, max_neighbors, max_distance)
        
        try:
            kernels = self.kernel_manager.get_kernels()
            return kernels.spatial_neighbors(coords.contiguous(), max_neighbors, max_distance)
        except Exception as e:
            logger.warning(f"Kernel execution failed, falling back to PyTorch: {e}")
            return self._spatial_neighbors_pytorch(coords, max_neighbors, max_distance)
    
    def pairwise_distances(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise distances using optimized kernels.
        
        Args:
            coords1: First set of coordinates [N, spatial_dim]
            coords2: Second set of coordinates [M, spatial_dim]
            
        Returns:
            Distance matrix [N, M]
        """
        if not self.kernel_manager.kernels_compiled:
            return self._pairwise_distances_pytorch(coords1, coords2)
        
        try:
            kernels = self.kernel_manager.get_kernels()
            return kernels.pairwise_distances(coords1.contiguous(), coords2.contiguous())
        except Exception as e:
            logger.warning(f"Kernel execution failed, falling back to PyTorch: {e}")
            return self._pairwise_distances_pytorch(coords1, coords2)
    
    def knn_search(
        self,
        coords: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        K-nearest neighbor search using optimized kernels.
        
        Args:
            coords: Spatial coordinates [N, spatial_dim]
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (indices [N, k], distances [N, k])
        """
        if not self.kernel_manager.kernels_compiled:
            return self._knn_search_pytorch(coords, k)
        
        try:
            kernels = self.kernel_manager.get_kernels()
            return kernels.knn_search(coords.contiguous(), k)
        except Exception as e:
            logger.warning(f"Kernel execution failed, falling back to PyTorch: {e}")
            return self._knn_search_pytorch(coords, k)
    
    def spatial_attention_bias(
        self,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute spatial attention bias using optimized kernels.
        
        Args:
            spatial_coords: Spatial coordinates [N, spatial_dim]
            edge_index: Edge connectivity [2, E]
            temperature: Temperature parameter for attention
            
        Returns:
            Spatial bias values [E]
        """
        if not self.kernel_manager.kernels_compiled:
            return self._spatial_attention_pytorch(spatial_coords, edge_index, temperature)
        
        try:
            kernels = self.kernel_manager.get_kernels()
            return kernels.spatial_attention(
                spatial_coords.contiguous(),
                edge_index.contiguous(),
                temperature
            )
        except Exception as e:
            logger.warning(f"Kernel execution failed, falling back to PyTorch: {e}")
            return self._spatial_attention_pytorch(spatial_coords, edge_index, temperature)
    
    def build_spatial_graph(
        self,
        coords: torch.Tensor,
        max_neighbors: int = 10,
        max_distance: float = 100.0,
        include_features: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build spatial graph with optimized operations.
        
        Args:
            coords: Spatial coordinates [N, spatial_dim]
            max_neighbors: Maximum neighbors per node
            max_distance: Maximum edge distance
            include_features: Whether to compute edge features
            
        Returns:
            Tuple of (edge_index [2, E], edge_attr [E, F] or None)
        """
        # Find neighbors
        neighbors = self.spatial_neighbors(coords, max_neighbors, max_distance)
        
        # Build edge list
        edge_list = []
        edge_features = []
        
        n_points = coords.size(0)
        spatial_dim = coords.size(1)
        
        for i in range(n_points):
            for j in range(max_neighbors):
                neighbor = neighbors[i, j].item()
                if neighbor == -1:
                    break
                
                edge_list.append([i, neighbor])
                
                if include_features:
                    # Compute edge features
                    src_coord = coords[i]
                    dst_coord = coords[neighbor]
                    
                    # Distance
                    dist = torch.norm(src_coord - dst_coord).item()
                    
                    # Direction (angle for 2D)
                    if spatial_dim == 2:
                        dx = dst_coord[0] - src_coord[0]
                        dy = dst_coord[1] - src_coord[1]
                        angle = torch.atan2(dy, dx).item()
                        edge_features.append([dist, angle])
                    else:
                        edge_features.append([dist])
        
        if not edge_list:
            # Return empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
            edge_attr = None if not include_features else torch.zeros((0, 1), device=coords.device)
            return edge_index, edge_attr
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=coords.device).t()
        
        if include_features and edge_features:
            edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=coords.device)
        else:
            edge_attr = None
        
        return edge_index, edge_attr
    
    # PyTorch fallback implementations
    def _spatial_neighbors_pytorch(
        self,
        coords: torch.Tensor,
        max_neighbors: int,
        max_distance: float
    ) -> torch.Tensor:
        """PyTorch implementation of spatial neighbors."""
        n_points = coords.size(0)
        device = coords.device
        
        # Compute pairwise distances
        coords_expanded = coords.unsqueeze(1)  # [N, 1, D]
        coords_tiled = coords.unsqueeze(0)     # [1, N, D]
        
        distances = torch.norm(coords_expanded - coords_tiled, dim=2)  # [N, N]
        
        # Mask self-distances
        distances.fill_diagonal_(float('inf'))
        
        # Apply distance threshold
        distances = torch.where(distances <= max_distance, distances, float('inf'))
        
        # Get top-k neighbors
        _, indices = torch.topk(distances, min(max_neighbors, n_points-1), dim=1, largest=False)
        
        # Mask invalid neighbors (distance was inf)
        valid_mask = distances.gather(1, indices) != float('inf')
        indices = torch.where(valid_mask, indices, -1)
        
        return indices
    
    def _pairwise_distances_pytorch(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch implementation of pairwise distances."""
        coords1_expanded = coords1.unsqueeze(1)  # [N, 1, D]
        coords2_expanded = coords2.unsqueeze(0)  # [1, M, D]
        
        distances = torch.norm(coords1_expanded - coords2_expanded, dim=2)  # [N, M]
        
        return distances
    
    def _knn_search_pytorch(
        self,
        coords: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch implementation of KNN search."""
        n_points = coords.size(0)
        
        # Compute pairwise distances
        distances = self._pairwise_distances_pytorch(coords, coords)
        
        # Mask self-distances
        distances.fill_diagonal_(float('inf'))
        
        # Get top-k
        top_distances, indices = torch.topk(distances, min(k, n_points-1), dim=1, largest=False)
        
        # Replace inf distances with -1
        valid_mask = top_distances != float('inf')
        indices = torch.where(valid_mask, indices, -1)
        top_distances = torch.where(valid_mask, top_distances, -1.0)
        
        return indices, top_distances
    
    def _spatial_attention_pytorch(
        self,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """PyTorch implementation of spatial attention bias."""
        src_coords = spatial_coords[edge_index[0]]  # [E, D]
        dst_coords = spatial_coords[edge_index[1]]  # [E, D]
        
        distances = torch.norm(src_coords - dst_coords, dim=1)  # [E]
        spatial_bias = torch.exp(-distances / temperature)
        
        return spatial_bias


class OptimizedGraphConstruction:
    """Optimized graph construction for spatial transcriptomics data."""
    
    def __init__(self, enable_cuda: bool = True):
        self.spatial_ops = OptimizedSpatialOps(enable_cuda)
    
    def construct_spatial_graph(
        self,
        spatial_coords: torch.Tensor,
        method: str = "knn",
        k: int = 10,
        radius: float = 100.0,
        include_features: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct spatial graph with various methods.
        
        Args:
            spatial_coords: Spatial coordinates [N, spatial_dim]
            method: Graph construction method ('knn', 'radius', 'delaunay')
            k: Number of neighbors for KNN
            radius: Radius for radius-based graphs
            include_features: Whether to include edge features
            
        Returns:
            Tuple of (edge_index [2, E], edge_attr [E, F] or None)
        """
        if method == "knn":
            return self._construct_knn_graph(spatial_coords, k, include_features)
        elif method == "radius":
            return self._construct_radius_graph(spatial_coords, radius, include_features)
        elif method == "delaunay":
            return self._construct_delaunay_graph(spatial_coords, include_features)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
    
    def _construct_knn_graph(
        self,
        coords: torch.Tensor,
        k: int,
        include_features: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Construct k-nearest neighbor graph."""
        indices, distances = self.spatial_ops.knn_search(coords, k)
        
        edge_list = []
        edge_features = []
        
        n_points = coords.size(0)
        
        for i in range(n_points):
            for j in range(k):
                neighbor = indices[i, j].item()
                if neighbor == -1:
                    break
                
                edge_list.append([i, neighbor])
                
                if include_features:
                    dist = distances[i, j].item()
                    edge_features.append([dist])
        
        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
            edge_attr = None
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=coords.device).t()
            
            if include_features and edge_features:
                edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=coords.device)
            else:
                edge_attr = None
        
        return edge_index, edge_attr
    
    def _construct_radius_graph(
        self,
        coords: torch.Tensor,
        radius: float,
        include_features: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Construct radius-based graph."""
        neighbors = self.spatial_ops.spatial_neighbors(coords, max_neighbors=50, max_distance=radius)
        
        edge_list = []
        edge_features = []
        
        n_points = coords.size(0)
        
        for i in range(n_points):
            for j in range(50):  # max_neighbors
                neighbor = neighbors[i, j].item()
                if neighbor == -1:
                    break
                
                edge_list.append([i, neighbor])
                
                if include_features:
                    dist = torch.norm(coords[i] - coords[neighbor]).item()
                    edge_features.append([dist])
        
        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
            edge_attr = None
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=coords.device).t()
            
            if include_features and edge_features:
                edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=coords.device)
            else:
                edge_attr = None
        
        return edge_index, edge_attr
    
    def _construct_delaunay_graph(
        self,
        coords: torch.Tensor,
        include_features: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Construct Delaunay triangulation graph."""
        # For 2D coordinates only
        if coords.size(1) != 2:
            raise ValueError("Delaunay triangulation only supported for 2D coordinates")
        
        try:
            from scipy.spatial import Delaunay
            
            # Convert to numpy for scipy
            coords_np = coords.cpu().numpy()
            tri = Delaunay(coords_np)
            
            # Extract edges from triangulation
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)
            
            # Convert to tensors
            edge_list = list(edges)
            
            if not edge_list:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
                edge_attr = None
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=coords.device).t()
                
                if include_features:
                    # Compute edge features
                    edge_features = []
                    for i, j in edge_list:
                        dist = torch.norm(coords[i] - coords[j]).item()
                        edge_features.append([dist])
                    
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=coords.device)
                else:
                    edge_attr = None
            
            return edge_index, edge_attr
            
        except ImportError:
            logger.warning("scipy not available, falling back to KNN graph")
            return self._construct_knn_graph(coords, 6, include_features)


# Global instances
_spatial_ops = None
_graph_constructor = None


def get_spatial_ops(enable_cuda: bool = True) -> OptimizedSpatialOps:
    """Get global spatial operations instance."""
    global _spatial_ops
    
    if _spatial_ops is None:
        _spatial_ops = OptimizedSpatialOps(enable_cuda)
    
    return _spatial_ops


def get_graph_constructor(enable_cuda: bool = True) -> OptimizedGraphConstruction:
    """Get global graph constructor instance."""
    global _graph_constructor
    
    if _graph_constructor is None:
        _graph_constructor = OptimizedGraphConstruction(enable_cuda)
    
    return _graph_constructor


# Convenience functions
def spatial_neighbors(
    coords: torch.Tensor,
    max_neighbors: int = 10,
    max_distance: float = 100.0,
    enable_cuda: bool = True
) -> torch.Tensor:
    """Find spatial neighbors with CUDA acceleration."""
    ops = get_spatial_ops(enable_cuda)
    return ops.spatial_neighbors(coords, max_neighbors, max_distance)


def knn_search(
    coords: torch.Tensor,
    k: int = 10,
    enable_cuda: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-nearest neighbor search with CUDA acceleration."""
    ops = get_spatial_ops(enable_cuda)
    return ops.knn_search(coords, k)


def construct_spatial_graph(
    coords: torch.Tensor,
    method: str = "knn",
    k: int = 10,
    radius: float = 100.0,
    enable_cuda: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct spatial graph with CUDA acceleration."""
    constructor = get_graph_constructor(enable_cuda)
    return constructor.construct_spatial_graph(coords, method, k, radius)


def benchmark_kernels() -> Dict[str, Dict[str, float]]:
    """Benchmark CUDA kernels vs PyTorch implementations."""
    # Create test data
    n_points = 10000
    spatial_dim = 2
    k = 10
    
    coords = torch.randn(n_points, spatial_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Test CUDA kernels
    cuda_ops = OptimizedSpatialOps(enable_cuda=True)
    
    # Benchmark KNN search
    import time
    
    # CUDA benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        _ = cuda_ops.knn_search(coords, k)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / 10
    
    # CPU benchmark
    cpu_ops = OptimizedSpatialOps(enable_cuda=False)
    coords_cpu = coords.cpu()
    
    start_time = time.time()
    for _ in range(10):
        _ = cpu_ops.knn_search(coords_cpu, k)
    cpu_time = (time.time() - start_time) / 10
    
    results['knn_search'] = {
        'cuda_time': cuda_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time / cuda_time if cuda_time > 0 else 0
    }
    
    logger.info(f"KNN search benchmark: CUDA {cuda_time:.4f}s, CPU {cpu_time:.4f}s, Speedup: {results['knn_search']['speedup']:.2f}x")
    
    return results
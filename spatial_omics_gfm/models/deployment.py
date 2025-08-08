"""
Production deployment configurations and serving utilities for Spatial-Omics GFM.
Supports multiple deployment scenarios: cloud inference, edge deployment, and batch processing.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData
from dataclasses import dataclass, asdict
import pickle
import warnings

from .graph_transformer import SpatialGraphTransformer, TransformerConfig
from ..utils.optimization import ModelOptimizer, BatchProcessor, optimize_for_production

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_type: str = "spatial_graph_transformer"
    model_size: str = "base"
    device: str = "auto"
    batch_size: int = 32
    max_sequence_length: int = 10000
    use_compilation: bool = True
    use_quantization: bool = False
    use_onnx: bool = False
    enable_amp: bool = True
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 300
    cache_size: int = 100


class ModelServer:
    """
    Production model server for spatial transcriptomics inference.
    
    Supports:
    - Synchronous and asynchronous inference
    - Multiple model variants (compiled, quantized, ONNX)
    - Automatic batching and memory management
    - Request caching and rate limiting
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[DeploymentConfig] = None,
        warm_start: bool = True
    ):
        """
        Initialize model server.
        
        Args:
            model_path: Path to saved model or model directory
            config: Deployment configuration
            warm_start: Whether to warm up the model on initialization
        """
        self.model_path = Path(model_path)
        self.config = config or DeploymentConfig()
        
        # Device setup
        self.device = self._setup_device()
        
        # Load and optimize model
        self.models = self._load_and_optimize_models()
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            model=self.models['primary'],
            device=self.device,
            max_batch_size=self.config.batch_size,
            enable_amp=self.config.enable_amp
        )
        
        # Request cache
        self.cache = {} if self.config.cache_size > 0 else None
        self.cache_keys = []
        
        # Performance tracking
        self.request_count = 0
        self.total_inference_time = 0.0
        
        if warm_start:
            self._warm_up_models()
        
        logger.info(f"ModelServer initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _load_and_optimize_models(self) -> Dict[str, nn.Module]:
        """Load and create optimized model variants."""
        logger.info("Loading and optimizing models")
        
        # Load base model
        base_model = self._load_base_model()
        
        # Create sample input for optimization
        sample_input = self._create_sample_input()
        
        # Apply optimizations
        optimization_config = {
            'enable_compilation': self.config.use_compilation,
            'enable_quantization': self.config.use_quantization,
            'export_onnx': self.config.use_onnx,
            'profile_performance': False  # Skip profiling during deployment
        }
        
        optimized_results = optimize_for_production(
            base_model, sample_input, optimization_config
        )
        
        models = {'primary': base_model}
        
        # Add optimized variants
        if 'compiled_model' in optimized_results:
            models['compiled'] = optimized_results['compiled_model']
            models['primary'] = optimized_results['compiled_model']  # Use compiled as primary
        
        if 'quantized_model' in optimized_results:
            models['quantized'] = optimized_results['quantized_model']
        
        if 'memory_optimized_model' in optimized_results:
            models['memory_optimized'] = optimized_results['memory_optimized_model']
        
        logger.info(f"Loaded model variants: {list(models.keys())}")
        return models
    
    def _load_base_model(self) -> SpatialGraphTransformer:
        """Load the base model from saved state."""
        if self.model_path.is_file():
            # Load from single file
            if self.model_path.suffix == '.pth':
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Extract config and model state
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    config = TransformerConfig(**checkpoint['config'])
                    model = SpatialGraphTransformer(config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Assume direct model save
                    raise ValueError("Model file must contain config and state_dict")
            else:
                raise ValueError(f"Unsupported model file format: {self.model_path.suffix}")
        else:
            # Load from directory
            config_path = self.model_path / "config.json"
            model_path = self.model_path / "pytorch_model.bin"
            
            if not config_path.exists() or not model_path.exists():
                raise FileNotFoundError("Model directory must contain config.json and pytorch_model.bin")
            
            # Load config
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = TransformerConfig(**config_dict)
            
            # Load model
            model = SpatialGraphTransformer(config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _create_sample_input(self) -> Dict[str, torch.Tensor]:
        """Create sample input for model optimization."""
        # Get model config for dimensions
        primary_model = list(self.models.values())[0] if hasattr(self, 'models') else None
        
        if primary_model and hasattr(primary_model, 'config'):
            num_genes = primary_model.config.num_genes
        else:
            num_genes = 3000  # Default
        
        # Create dummy spatial data
        num_nodes = 1000
        
        sample_input = {
            'gene_expression': torch.randn(num_nodes, num_genes, device=self.device),
            'spatial_coords': torch.randn(num_nodes, 2, device=self.device),
            'edge_index': torch.randint(0, num_nodes, (2, num_nodes * 6), device=self.device)
        }
        
        return sample_input
    
    def _warm_up_models(self) -> None:
        """Warm up models with dummy inference."""
        logger.info("Warming up models")
        
        sample_input = self._create_sample_input()
        
        for model_name, model in self.models.items():
            try:
                with torch.no_grad():
                    _ = model(**sample_input)
                logger.info(f"Warmed up {model_name} model")
            except Exception as e:
                logger.warning(f"Failed to warm up {model_name} model: {e}")
    
    def predict(
        self,
        adata: AnnData,
        return_embeddings: bool = True,
        use_cache: bool = True,
        model_variant: str = "primary"
    ) -> Dict[str, Any]:
        """
        Run inference on spatial transcriptomics data.
        
        Args:
            adata: Spatial transcriptomics data
            return_embeddings: Whether to return embeddings
            use_cache: Whether to use response cache
            model_variant: Which model variant to use
            
        Returns:
            Inference results
        """
        import time
        start_time = time.time()
        
        # Generate cache key
        cache_key = None
        if use_cache and self.cache is not None:
            cache_key = self._generate_cache_key(adata, return_embeddings)
            if cache_key in self.cache:
                logger.debug("Cache hit for inference request")
                return self.cache[cache_key]
        
        # Validate input
        self._validate_input(adata)
        
        # Prepare model input
        model_input = self._prepare_model_input(adata)
        
        # Select model
        if model_variant not in self.models:
            logger.warning(f"Model variant '{model_variant}' not available, using primary")
            model_variant = "primary"
        
        model = self.models[model_variant]
        
        # Run inference
        with torch.no_grad():
            if self.config.enable_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(**model_input, return_embeddings=return_embeddings)
            else:
                outputs = model(**model_input, return_embeddings=return_embeddings)
        
        # Prepare response
        response = self._prepare_response(outputs, adata)
        
        # Cache response
        if cache_key and self.cache is not None:
            self._update_cache(cache_key, response)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.request_count += 1
        self.total_inference_time += inference_time
        
        logger.debug(f"Inference completed in {inference_time:.3f}s")
        
        return response
    
    async def predict_async(
        self,
        adata: AnnData,
        return_embeddings: bool = True,
        use_cache: bool = True,
        model_variant: str = "primary"
    ) -> Dict[str, Any]:
        """
        Asynchronous inference.
        
        Args:
            adata: Spatial transcriptomics data
            return_embeddings: Whether to return embeddings
            use_cache: Whether to use response cache
            model_variant: Which model variant to use
            
        Returns:
            Inference results
        """
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.predict,
            adata,
            return_embeddings,
            use_cache,
            model_variant
        )
        return result
    
    def predict_batch(
        self,
        adata_list: List[AnnData],
        return_embeddings: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch inference on multiple datasets.
        
        Args:
            adata_list: List of spatial transcriptomics datasets
            return_embeddings: Whether to return embeddings
            progress_callback: Progress callback function
            
        Returns:
            List of inference results
        """
        logger.info(f"Starting batch inference on {len(adata_list)} datasets")
        
        results = []
        
        for i, adata in enumerate(adata_list):
            try:
                result = self.predict(adata, return_embeddings=return_embeddings, use_cache=False)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(adata_list))
                    
            except Exception as e:
                logger.error(f"Failed to process dataset {i}: {e}")
                results.append({"error": str(e)})
        
        logger.info("Batch inference completed")
        return results
    
    def _validate_input(self, adata: AnnData) -> None:
        """Validate input data."""
        if not isinstance(adata, AnnData):
            raise ValueError("Input must be an AnnData object")
        
        if 'spatial' not in adata.obsm:
            raise ValueError("Input must contain spatial coordinates in adata.obsm['spatial']")
        
        if adata.n_obs > self.config.max_sequence_length:
            raise ValueError(f"Input too large: {adata.n_obs} > {self.config.max_sequence_length}")
        
        # Check memory requirements
        estimated_memory = self._estimate_memory_usage(adata)
        if estimated_memory > self.config.memory_limit_gb:
            raise ValueError(f"Input would exceed memory limit: {estimated_memory:.1f}GB > {self.config.memory_limit_gb}GB")
    
    def _estimate_memory_usage(self, adata: AnnData) -> float:
        """Estimate memory usage for inference."""
        # Rough estimation based on data size and model parameters
        data_memory = adata.X.nbytes / (1024**3)  # Convert to GB
        
        # Estimate model memory (rough approximation)
        model_params = sum(p.numel() for p in self.models['primary'].parameters())
        model_memory = model_params * 4 / (1024**3)  # 4 bytes per float32 parameter
        
        # Factor in intermediate computations (rough estimate)
        computation_memory = data_memory * 3
        
        total_memory = data_memory + model_memory + computation_memory
        return total_memory
    
    def _prepare_model_input(self, adata: AnnData) -> Dict[str, torch.Tensor]:
        """Prepare input tensors for the model."""
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = torch.from_numpy(adata.X.toarray()).float()
        else:
            expression = torch.from_numpy(adata.X).float()
        
        # Get spatial coordinates
        spatial_coords = torch.from_numpy(adata.obsm['spatial']).float()
        
        # Create simple k-NN graph if not available
        if 'spatial_graph' in adata.uns:
            edge_index = torch.from_numpy(adata.uns['spatial_graph']['edge_index']).long()
            edge_attr = torch.from_numpy(adata.uns['spatial_graph']['edge_attr']).float()
        else:
            # Create simple k-NN graph
            from sklearn.neighbors import NearestNeighbors
            
            k = min(6, adata.n_obs - 1)
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
            distances, indices = nbrs.kneighbors(spatial_coords)
            
            # Build edge index
            edges = []
            edge_distances = []
            
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):  # Skip self
                    edges.append([i, indices[i][j]])
                    edge_distances.append(distances[i][j])
            
            edge_index = torch.tensor(edges).T.long()
            edge_attr = torch.tensor(edge_distances).float().unsqueeze(1)
        
        # Move to device
        model_input = {
            'gene_expression': expression.to(self.device),
            'spatial_coords': spatial_coords.to(self.device),
            'edge_index': edge_index.to(self.device),
            'edge_attr': edge_attr.to(self.device)
        }
        
        return model_input
    
    def _prepare_response(self, outputs: Dict[str, torch.Tensor], adata: AnnData) -> Dict[str, Any]:
        """Prepare response from model outputs."""
        response = {}
        
        # Convert tensors to numpy arrays
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                response[key] = value.cpu().numpy()
            else:
                response[key] = value
        
        # Add metadata
        response['metadata'] = {
            'num_cells': adata.n_obs,
            'num_genes': adata.n_vars,
            'model_variant': 'primary',
            'device': str(self.device)
        }
        
        return response
    
    def _generate_cache_key(self, adata: AnnData, return_embeddings: bool) -> str:
        """Generate cache key for request."""
        # Use hash of expression data and coordinates
        import hashlib
        
        if hasattr(adata.X, 'toarray'):
            data_bytes = adata.X.toarray().tobytes()
        else:
            data_bytes = adata.X.tobytes()
        
        coords_bytes = adata.obsm['spatial'].tobytes()
        
        combined_bytes = data_bytes + coords_bytes + str(return_embeddings).encode()
        cache_key = hashlib.md5(combined_bytes).hexdigest()
        
        return cache_key
    
    def _update_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Update response cache."""
        if len(self.cache_keys) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
        self.cache_keys.append(cache_key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_inference_time = self.total_inference_time / max(self.request_count, 1)
        
        stats = {
            'total_requests': self.request_count,
            'avg_inference_time_sec': avg_inference_time,
            'total_inference_time_sec': self.total_inference_time,
            'cache_hit_rate': len(self.cache) / max(self.request_count, 1) if self.cache else 0,
            'available_models': list(self.models.keys()),
            'device': str(self.device)
        }
        
        # Memory stats if CUDA
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache is not None:
            self.cache.clear()
            self.cache_keys.clear()
            logger.info("Response cache cleared")
    
    def save_deployment_info(self, output_path: Union[str, Path]) -> None:
        """Save deployment configuration and model info."""
        deployment_info = {
            'config': asdict(self.config),
            'model_path': str(self.model_path),
            'device': str(self.device),
            'available_models': list(self.models.keys()),
            'performance_stats': self.get_performance_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Deployment info saved to: {output_path}")


class EdgeDeployment:
    """
    Optimized deployment for edge devices with limited resources.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization_method: str = "dynamic",
        max_memory_mb: int = 512
    ):
        self.model_path = Path(model_path)
        self.quantization_method = quantization_method
        self.max_memory_mb = max_memory_mb
        
        # Setup for CPU-only inference
        self.device = torch.device("cpu")
        
        # Load and optimize model for edge
        self.model = self._prepare_edge_model()
        
        logger.info(f"EdgeDeployment initialized with max memory: {max_memory_mb}MB")
    
    def _prepare_edge_model(self) -> nn.Module:
        """Prepare model optimized for edge deployment."""
        # Load base model
        server = ModelServer(self.model_path, config=DeploymentConfig(device="cpu"))
        base_model = server.models['primary']
        
        # Apply aggressive optimizations
        optimizer = ModelOptimizer()
        
        # Quantize for reduced memory and faster inference
        sample_input = server._create_sample_input()
        quantized_model = optimizer.quantize_model(
            base_model, sample_input, quantization_method=self.quantization_method
        )
        
        # Memory optimization
        optimized_model = optimizer.optimize_memory_usage(quantized_model)
        
        return optimized_model
    
    def predict_lightweight(
        self,
        gene_expression: np.ndarray,
        spatial_coords: np.ndarray,
        max_cells: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Lightweight prediction for edge devices.
        
        Args:
            gene_expression: Gene expression matrix [n_cells, n_genes]
            spatial_coords: Spatial coordinates [n_cells, 2]
            max_cells: Maximum number of cells to process at once
            
        Returns:
            Prediction results
        """
        # Subsample if too many cells
        if len(gene_expression) > max_cells:
            indices = np.random.choice(len(gene_expression), max_cells, replace=False)
            gene_expression = gene_expression[indices]
            spatial_coords = spatial_coords[indices]
        
        # Create simple k-NN graph
        from sklearn.neighbors import NearestNeighbors
        
        k = min(4, len(spatial_coords) - 1)  # Reduced neighbors for edge
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        # Build edge tensors
        edges = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                edges.append([i, indices[i][j]])
        
        edge_index = torch.tensor(edges).T.long()
        edge_attr = torch.tensor([[1.0]] * len(edges)).float()
        
        # Prepare model input
        model_input = {
            'gene_expression': torch.from_numpy(gene_expression).float(),
            'spatial_coords': torch.from_numpy(spatial_coords).float(),
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**model_input)
        
        # Convert to numpy
        results = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.numpy()
        
        return results


def create_deployment_package(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    include_onnx: bool = True,
    include_quantized: bool = True,
    include_edge: bool = True
) -> Dict[str, Path]:
    """
    Create complete deployment package with multiple model variants.
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory for deployment package
        include_onnx: Whether to include ONNX export
        include_quantized: Whether to include quantized models
        include_edge: Whether to include edge-optimized models
        
    Returns:
        Dictionary mapping variant names to file paths
    """
    logger.info("Creating deployment package")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model server
    config = DeploymentConfig(
        use_compilation=True,
        use_quantization=include_quantized,
        use_onnx=include_onnx
    )
    
    server = ModelServer(model_path, config=config, warm_start=False)
    
    deployment_files = {}
    
    # Save base model
    base_path = output_dir / "base_model.pth"
    torch.save({
        'model_state_dict': server.models['primary'].state_dict(),
        'config': asdict(server.models['primary'].config)
    }, base_path)
    deployment_files['base'] = base_path
    
    # Save optimized variants
    for variant_name, model in server.models.items():
        if variant_name != 'primary':
            variant_path = output_dir / f"{variant_name}_model.pth"
            torch.save(model.state_dict(), variant_path)
            deployment_files[variant_name] = variant_path
    
    # Create edge deployment
    if include_edge:
        edge_deployment = EdgeDeployment(model_path)
        edge_path = output_dir / "edge_model.pth"
        torch.save(edge_deployment.model.state_dict(), edge_path)
        deployment_files['edge'] = edge_path
    
    # Save deployment configuration
    config_path = output_dir / "deployment_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Save server configuration
    server.save_deployment_info(output_dir / "deployment_info.json")
    
    logger.info(f"Deployment package created at: {output_dir}")
    logger.info(f"Available variants: {list(deployment_files.keys())}")
    
    return deployment_files
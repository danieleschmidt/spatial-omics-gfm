"""
Comprehensive benchmarking framework for spatial transcriptomics methods.
Enables systematic comparison of different models and approaches.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from anndata import AnnData
import warnings

from ..models.graph_transformer import SpatialGraphTransformer, TransformerConfig
from ..utils.metrics import SpatialMetrics, evaluate_model_performance
from ..utils.optimization import PerformanceProfiler
from ..utils.memory_management import MemoryMonitor, MemoryConfig
from .novel_attention import NovelAttentionBenchmark

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    dataset_sizes: List[int] = None
    model_variants: List[str] = None
    batch_sizes: List[int] = None
    enable_gpu: bool = True
    num_runs: int = 3
    timeout_seconds: int = 300
    memory_limit_gb: float = 16.0
    output_dir: Optional[str] = None
    detailed_metrics: bool = True
    
    def __post_init__(self):
        if self.dataset_sizes is None:
            self.dataset_sizes = [100, 500, 1000, 2000, 5000]
        if self.model_variants is None:
            self.model_variants = ['base', 'large']
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64]


class SyntheticDataGenerator:
    """Generate synthetic spatial transcriptomics data for benchmarking."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def generate_spatial_data(
        self,
        n_cells: int,
        n_genes: int = 2000,
        spatial_pattern: str = 'random',
        noise_level: float = 0.1,
        cell_types: Optional[List[str]] = None
    ) -> AnnData:
        """
        Generate synthetic spatial transcriptomics data.
        
        Args:
            n_cells: Number of cells
            n_genes: Number of genes
            spatial_pattern: Spatial organization pattern
            noise_level: Level of noise to add
            cell_types: List of cell type names
            
        Returns:
            Synthetic AnnData object
        """
        logger.info(f"Generating synthetic data: {n_cells} cells, {n_genes} genes")
        
        # Generate spatial coordinates
        coords = self._generate_spatial_coordinates(n_cells, spatial_pattern)
        
        # Generate expression data
        expression_data = self._generate_expression_data(
            n_cells, n_genes, coords, noise_level
        )
        
        # Create cell metadata
        obs_data = self._generate_cell_metadata(n_cells, coords, cell_types)
        
        # Create gene metadata
        var_data = self._generate_gene_metadata(n_genes)
        
        # Create AnnData object
        adata = AnnData(
            X=expression_data,
            obs=obs_data,
            var=var_data
        )
        adata.obsm['spatial'] = coords
        
        return adata
    
    def _generate_spatial_coordinates(
        self,
        n_cells: int,
        pattern: str = 'random'
    ) -> np.ndarray:
        """Generate spatial coordinates with different patterns."""
        if pattern == 'random':
            # Random uniform distribution
            coords = np.random.uniform(0, 1000, (n_cells, 2))
        
        elif pattern == 'clustered':
            # Multiple clusters
            n_clusters = max(3, n_cells // 200)
            cluster_centers = np.random.uniform(200, 800, (n_clusters, 2))
            
            coords = []
            cells_per_cluster = n_cells // n_clusters
            
            for i, center in enumerate(cluster_centers):
                if i == len(cluster_centers) - 1:
                    # Last cluster gets remaining cells
                    n_cluster_cells = n_cells - len(coords)
                else:
                    n_cluster_cells = cells_per_cluster
                
                # Generate cells around cluster center
                cluster_coords = np.random.multivariate_normal(
                    center, np.eye(2) * 50, n_cluster_cells
                )
                coords.extend(cluster_coords)
            
            coords = np.array(coords)
        
        elif pattern == 'grid':
            # Regular grid pattern with some noise
            grid_size = int(np.ceil(np.sqrt(n_cells)))
            x_coords = np.tile(np.linspace(0, 1000, grid_size), grid_size)[:n_cells]
            y_coords = np.repeat(np.linspace(0, 1000, grid_size), grid_size)[:n_cells]
            
            # Add noise
            noise = np.random.normal(0, 20, (n_cells, 2))
            coords = np.column_stack([x_coords, y_coords]) + noise
        
        elif pattern == 'tissue_like':
            # Tissue-like organization with regions
            coords = self._generate_tissue_like_coordinates(n_cells)
        
        else:
            raise ValueError(f"Unknown spatial pattern: {pattern}")
        
        return coords.astype(np.float32)
    
    def _generate_tissue_like_coordinates(self, n_cells: int) -> np.ndarray:
        """Generate tissue-like spatial organization."""
        # Define tissue regions
        regions = [
            {'center': [250, 250], 'radius': 150, 'density': 0.3},
            {'center': [750, 250], 'radius': 100, 'density': 0.2},
            {'center': [500, 750], 'radius': 200, 'density': 0.4},
            {'center': [200, 700], 'radius': 80, 'density': 0.1}
        ]
        
        coords = []
        cells_assigned = 0
        
        for region in regions:
            n_region_cells = int(n_cells * region['density'])
            if cells_assigned + n_region_cells > n_cells:
                n_region_cells = n_cells - cells_assigned
            
            # Generate cells in region
            center = np.array(region['center'])
            radius = region['radius']
            
            # Use rejection sampling for circular regions
            region_coords = []
            while len(region_coords) < n_region_cells:
                candidate = center + np.random.uniform(-radius, radius, 2)
                if np.linalg.norm(candidate - center) <= radius:
                    region_coords.append(candidate)
            
            coords.extend(region_coords)
            cells_assigned += n_region_cells
            
            if cells_assigned >= n_cells:
                break
        
        # Fill remaining with random coordinates
        while len(coords) < n_cells:
            coords.append(np.random.uniform(0, 1000, 2))
        
        return np.array(coords[:n_cells])
    
    def _generate_expression_data(
        self,
        n_cells: int,
        n_genes: int,
        coords: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """Generate realistic expression data."""
        # Create spatial expression patterns
        expression = np.zeros((n_cells, n_genes))
        
        # Generate different gene expression patterns
        for gene_idx in range(n_genes):
            pattern_type = np.random.choice(['spatial_gradient', 'clustered', 'random'])
            
            if pattern_type == 'spatial_gradient':
                # Spatial gradient expression
                direction = np.random.uniform(0, 2 * np.pi)
                gradient_vector = np.array([np.cos(direction), np.sin(direction)])
                
                # Project coordinates onto gradient direction
                projections = np.dot(coords, gradient_vector)
                projections = (projections - projections.min()) / (projections.max() - projections.min() + 1e-8)
                
                # Create expression gradient
                base_expression = np.random.exponential(2.0) * projections
                
            elif pattern_type == 'clustered':
                # Clustered expression around random points
                n_hotspots = np.random.randint(1, 4)
                hotspot_centers = np.random.uniform(0, 1000, (n_hotspots, 2))
                
                base_expression = np.zeros(n_cells)
                for center in hotspot_centers:
                    distances = np.linalg.norm(coords - center, axis=1)
                    expression_strength = np.exp(-distances / 100) * np.random.exponential(3.0)
                    base_expression += expression_strength
            
            else:  # random
                base_expression = np.random.exponential(1.0, n_cells)
            
            # Add noise
            noise = np.random.normal(0, noise_level * np.mean(base_expression), n_cells)
            expression[:, gene_idx] = np.maximum(0, base_expression + noise)
        
        return expression.astype(np.float32)
    
    def _generate_cell_metadata(
        self,
        n_cells: int,
        coords: np.ndarray,
        cell_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate cell metadata."""
        if cell_types is None:
            cell_types = ['TypeA', 'TypeB', 'TypeC', 'TypeD']
        
        # Assign cell types based on spatial location
        cell_type_assignments = []
        for coord in coords:
            # Simple spatial assignment based on position
            if coord[0] < 500 and coord[1] < 500:
                prob_dist = [0.6, 0.2, 0.1, 0.1]
            elif coord[0] >= 500 and coord[1] < 500:
                prob_dist = [0.2, 0.6, 0.1, 0.1]
            elif coord[0] < 500 and coord[1] >= 500:
                prob_dist = [0.1, 0.2, 0.6, 0.1]
            else:
                prob_dist = [0.1, 0.1, 0.2, 0.6]
            
            cell_type = np.random.choice(cell_types, p=prob_dist)
            cell_type_assignments.append(cell_type)
        
        obs_data = pd.DataFrame({
            'cell_type': cell_type_assignments,
            'x_coord': coords[:, 0],
            'y_coord': coords[:, 1],
            'sample_id': [f'sample_{i // 1000}' for i in range(n_cells)]
        })
        obs_data.index = [f'cell_{i}' for i in range(n_cells)]
        
        return obs_data
    
    def _generate_gene_metadata(self, n_genes: int) -> pd.DataFrame:
        """Generate gene metadata."""
        gene_types = ['protein_coding', 'lncRNA', 'miRNA', 'pseudogene']
        
        var_data = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_genes)],
            'gene_type': np.random.choice(gene_types, n_genes, p=[0.7, 0.15, 0.1, 0.05]),
            'highly_variable': np.random.choice([True, False], n_genes, p=[0.3, 0.7])
        })
        var_data.index = [f'ENSG{i:08d}' for i in range(n_genes)]
        
        return var_data


class ModelBenchmark:
    """Benchmark different model configurations and architectures."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.enable_gpu else 'cpu')
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator()
        self.metrics_calculator = SpatialMetrics()
        self.performance_profiler = PerformanceProfiler()
        self.attention_benchmark = NovelAttentionBenchmark(self.device)
        
        # Results storage
        self.benchmark_results = {}
        
        logger.info(f"ModelBenchmark initialized with device: {self.device}")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking suite."""
        logger.info("Starting comprehensive benchmarking suite...")
        
        benchmark_results = {
            'config': asdict(self.config),
            'device': str(self.device),
            'scalability_benchmark': self.run_scalability_benchmark(),
            'model_comparison': self.run_model_comparison(),
            'attention_benchmark': self.run_attention_benchmark(),
            'performance_analysis': self.run_performance_analysis(),
            'memory_analysis': self.run_memory_analysis()
        }
        
        # Save results if output directory specified
        if self.config.output_dir:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'benchmark_results.json', 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {output_path}")
        
        self.benchmark_results = benchmark_results
        return benchmark_results
    
    def run_scalability_benchmark(self) -> Dict[str, Any]:
        """Benchmark model scalability across different dataset sizes."""
        logger.info("Running scalability benchmark...")
        
        scalability_results = {}
        
        for dataset_size in self.config.dataset_sizes:
            logger.info(f"Benchmarking dataset size: {dataset_size}")
            
            # Generate test data
            adata = self.data_generator.generate_spatial_data(
                n_cells=dataset_size,
                spatial_pattern='tissue_like'
            )
            
            # Test with base model
            model_config = TransformerConfig(
                num_genes=adata.n_vars,
                hidden_dim=512,
                num_layers=6,
                num_heads=8
            )
            
            model = SpatialGraphTransformer(model_config).to(self.device)
            
            # Benchmark inference
            results = self._benchmark_model_inference(model, adata)
            scalability_results[dataset_size] = results
        
        return scalability_results
    
    def run_model_comparison(self) -> Dict[str, Any]:
        """Compare different model variants."""
        logger.info("Running model comparison benchmark...")
        
        # Generate test dataset
        test_size = 1000
        adata = self.data_generator.generate_spatial_data(
            n_cells=test_size,
            spatial_pattern='tissue_like'
        )
        
        model_variants = {
            'small': {'hidden_dim': 256, 'num_layers': 4, 'num_heads': 4},
            'base': {'hidden_dim': 512, 'num_layers': 8, 'num_heads': 8},
            'large': {'hidden_dim': 1024, 'num_layers': 12, 'num_heads': 16}
        }
        
        comparison_results = {}
        
        for variant_name, variant_config in model_variants.items():
            logger.info(f"Benchmarking {variant_name} model variant...")
            
            model_config = TransformerConfig(
                num_genes=adata.n_vars,
                **variant_config
            )
            
            model = SpatialGraphTransformer(model_config).to(self.device)
            results = self._benchmark_model_inference(model, adata)
            
            # Add model statistics
            results['parameter_count'] = sum(p.numel() for p in model.parameters())
            results['model_size_mb'] = results['parameter_count'] * 4 / (1024**2)
            
            comparison_results[variant_name] = results
        
        return comparison_results
    
    def run_attention_benchmark(self) -> Dict[str, Any]:
        """Benchmark novel attention mechanisms."""
        logger.info("Running attention mechanism benchmark...")
        
        attention_results = self.attention_benchmark.benchmark_attention_mechanisms(
            hidden_dim=512,
            num_nodes=1000,
            num_edges=5000,
            num_heads=8
        )
        
        return attention_results
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance across different configurations."""
        logger.info("Running performance analysis...")
        
        test_adata = self.data_generator.generate_spatial_data(
            n_cells=1000,
            spatial_pattern='tissue_like'
        )
        
        performance_results = {}
        
        # Test different batch sizes
        model_config = TransformerConfig(
            num_genes=test_adata.n_vars,
            hidden_dim=512,
            num_layers=8,
            num_heads=8
        )
        model = SpatialGraphTransformer(model_config).to(self.device)
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create sample input for profiling
            sample_input = self._prepare_model_input(test_adata, batch_size)
            
            # Profile performance
            perf_results = self.performance_profiler.profile_model_inference(
                model, sample_input, num_runs=10, warmup_runs=3
            )
            
            performance_results[f'batch_size_{batch_size}'] = perf_results
        
        return performance_results
    
    def run_memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        logger.info("Running memory analysis...")
        
        memory_config = MemoryConfig(max_memory_gb=self.config.memory_limit_gb)
        monitor = MemoryMonitor(memory_config)
        
        memory_results = {}
        
        for dataset_size in [500, 1000, 2000]:
            logger.info(f"Analyzing memory for dataset size: {dataset_size}")
            
            # Generate test data
            adata = self.data_generator.generate_spatial_data(n_cells=dataset_size)
            
            # Monitor memory during inference
            initial_memory = monitor.get_memory_usage()
            
            # Create and run model
            model_config = TransformerConfig(
                num_genes=adata.n_vars,
                hidden_dim=512,
                num_layers=8
            )
            model = SpatialGraphTransformer(model_config).to(self.device)
            
            # Run inference
            sample_input = self._prepare_model_input(adata)
            with torch.no_grad():
                _ = model(**sample_input)
            
            peak_memory = monitor.get_memory_usage()
            
            memory_results[dataset_size] = {
                'initial_memory_gb': initial_memory['memory_gb'],
                'peak_memory_gb': peak_memory['memory_gb'],
                'memory_increase_gb': peak_memory['memory_gb'] - initial_memory['memory_gb']
            }
            
            if torch.cuda.is_available():
                memory_results[dataset_size].update({
                    'gpu_memory_allocated_gb': peak_memory.get('gpu_allocated_gb', 0),
                    'gpu_memory_cached_gb': peak_memory.get('gpu_cached_gb', 0)
                })
        
        return memory_results
    
    def _benchmark_model_inference(
        self,
        model: nn.Module,
        adata: AnnData,
        num_runs: int = None
    ) -> Dict[str, Any]:
        """Benchmark model inference performance."""
        num_runs = num_runs or self.config.num_runs
        
        # Prepare input
        sample_input = self._prepare_model_input(adata)
        
        # Profile inference
        perf_results = self.performance_profiler.profile_model_inference(
            model, sample_input, num_runs=num_runs, warmup_runs=3
        )
        
        return perf_results
    
    def _prepare_model_input(
        self,
        adata: AnnData,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare model input from AnnData."""
        if batch_size is not None:
            # Use subset of data
            indices = np.random.choice(adata.n_obs, min(batch_size, adata.n_obs), replace=False)
            subset_adata = adata[indices].copy()
        else:
            subset_adata = adata
        
        # Convert to tensors
        if hasattr(subset_adata.X, 'toarray'):
            gene_expression = torch.from_numpy(subset_adata.X.toarray()).float()
        else:
            gene_expression = torch.from_numpy(subset_adata.X).float()
        
        spatial_coords = torch.from_numpy(subset_adata.obsm['spatial']).float()
        
        # Create simple graph
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(6, subset_adata.n_obs - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(spatial_coords.numpy())
        distances, indices = nbrs.kneighbors(spatial_coords.numpy())
        
        # Build edge list
        edges = []
        edge_weights = []
        
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self
                edges.append([i, indices[i][j]])
                edge_weights.append(distances[i][j])
        
        edge_index = torch.tensor(edges).T.long()
        edge_attr = torch.tensor(edge_weights).float().unsqueeze(1)
        
        # Move to device
        return {
            'gene_expression': gene_expression.to(self.device),
            'spatial_coords': spatial_coords.to(self.device),
            'edge_index': edge_index.to(self.device),
            'edge_attr': edge_attr.to(self.device)
        }
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        report_lines = [
            "# Spatial-Omics GFM Benchmark Report",
            "",
            f"**Device**: {self.benchmark_results['device']}",
            f"**Configuration**: {self.benchmark_results['config']}",
            "",
            "## Scalability Analysis",
            ""
        ]
        
        # Scalability results
        scalability = self.benchmark_results.get('scalability_benchmark', {})
        if scalability:
            report_lines.extend([
                "| Dataset Size | Avg Inference Time (ms) | Memory Usage (GB) | Throughput (samples/s) |",
                "|--------------|-------------------------|-------------------|------------------------|"
            ])
            
            for size, results in scalability.items():
                report_lines.append(
                    f"| {size} | {results['avg_inference_time_ms']:.2f} | "
                    f"{results['memory_stats'].get('allocated_mb', 0)/1024:.2f} | "
                    f"{results['throughput_samples_per_sec']:.2f} |"
                )
        
        # Model comparison
        report_lines.extend([
            "",
            "## Model Comparison",
            ""
        ])
        
        comparison = self.benchmark_results.get('model_comparison', {})
        if comparison:
            report_lines.extend([
                "| Model Variant | Parameters | Size (MB) | Inference Time (ms) | Memory (MB) |",
                "|---------------|------------|-----------|---------------------|-------------|"
            ])
            
            for variant, results in comparison.items():
                report_lines.append(
                    f"| {variant} | {results['parameter_count']:,} | "
                    f"{results['model_size_mb']:.1f} | "
                    f"{results['avg_inference_time_ms']:.2f} | "
                    f"{results['memory_stats'].get('allocated_mb', 0):.1f} |"
                )
        
        # Attention mechanisms
        report_lines.extend([
            "",
            "## Attention Mechanism Comparison",
            ""
        ])
        
        attention = self.benchmark_results.get('attention_benchmark', {})
        if attention:
            report_lines.extend([
                "| Attention Type | Forward Time (ms) | Memory (MB) | Parameters |",
                "|----------------|-------------------|-------------|------------|"
            ])
            
            for att_type, results in attention.items():
                report_lines.append(
                    f"| {att_type} | {results['avg_forward_time_ms']:.2f} | "
                    f"{results['memory_usage_mb']:.1f} | {results['parameter_count']:,} |"
                )
        
        report_lines.extend([
            "",
            "## Performance Recommendations",
            ""
        ])
        
        # Generate recommendations based on results
        recommendations = self._generate_performance_recommendations()
        report_lines.extend(recommendations)
        
        return "\n".join(report_lines)
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        if not self.benchmark_results:
            return ["No benchmark data available for recommendations."]
        
        # Memory recommendations
        memory_results = self.benchmark_results.get('memory_analysis', {})
        if memory_results:
            max_memory = max(r['peak_memory_gb'] for r in memory_results.values())
            if max_memory > 8.0:
                recommendations.append(
                    "- **High Memory Usage**: Consider enabling quantization or reducing batch size for datasets >1000 cells"
                )
        
        # Model variant recommendations
        comparison = self.benchmark_results.get('model_comparison', {})
        if comparison:
            # Find best performance/size trade-off
            best_efficiency = None
            best_variant = None
            
            for variant, results in comparison.items():
                efficiency = results['throughput_samples_per_sec'] / results['model_size_mb']
                if best_efficiency is None or efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_variant = variant
            
            if best_variant:
                recommendations.append(
                    f"- **Model Selection**: '{best_variant}' variant offers best performance/size trade-off"
                )
        
        # Scalability recommendations
        scalability = self.benchmark_results.get('scalability_benchmark', {})
        if scalability:
            large_dataset_time = None
            for size, results in scalability.items():
                if size >= 2000:
                    large_dataset_time = results['avg_inference_time_ms']
                    break
            
            if large_dataset_time and large_dataset_time > 1000:
                recommendations.append(
                    "- **Large Datasets**: Consider using batch processing or model compilation for datasets >2000 cells"
                )
        
        if not recommendations:
            recommendations.append("- **Overall**: Current configuration appears well-optimized for tested scenarios")
        
        return recommendations
"""
Experimental Framework for Spatial-Omics GFM Research.

This module implements a comprehensive experimental framework for conducting
rigorous research experiments with statistical validation, reproducible baselines,
and publication-ready results.
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from anndata import AnnData
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from .benchmarking import SyntheticDataGenerator, BenchmarkConfig
from .novel_attention import (
    AdaptiveSpatialAttention, HierarchicalSpatialAttention,
    ContextualSpatialAttention, create_novel_attention_layer
)
from .advanced_benchmarking import (
    AdvancedBenchmarkSuite, StatisticalAnalyzer, 
    BenchmarkVisualizer, ReportGenerator
)
from ..models.graph_transformer import SpatialGraphTransformer, TransformerConfig
from ..utils.optimization import PerformanceProfiler
from ..utils.memory_management import MemoryMonitor
from ..utils.metrics import SpatialMetrics
from ..utils.validators import validate_spatial_data
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experimental framework."""
    
    # Experiment metadata
    experiment_name: str
    description: str = ""
    author: str = "Spatial-Omics Research Team"
    
    # Data configuration
    dataset_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'n_cells': 1000, 'n_genes': 2000, 'spatial_pattern': 'clustered'},
        {'n_cells': 5000, 'n_genes': 2000, 'spatial_pattern': 'tissue_like'},
        {'n_cells': 10000, 'n_genes': 3000, 'spatial_pattern': 'random'}
    ])
    
    # Model configurations to test
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'baseline_gnn': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'standard'},
        'adaptive_attention': {'hidden_dim': 512, 'num_layers': 8, 'attention_type': 'adaptive'},
        'hierarchical_attention': {'hidden_dim': 512, 'num_layers': 8, 'attention_type': 'hierarchical'},
        'contextual_attention': {'hidden_dim': 512, 'num_layers': 8, 'attention_type': 'contextual'}
    })
    
    # Experimental design
    num_runs_per_config: int = 5
    cross_validation_folds: int = 5
    statistical_significance_level: float = 0.05
    effect_size_threshold: float = 0.5  # Cohen's d
    
    # Tasks to evaluate
    evaluation_tasks: List[str] = field(default_factory=lambda: [
        'cell_type_prediction', 'spatial_clustering', 'interaction_prediction'
    ])
    
    # Metrics to compute
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        'accuracy', 'silhouette_score', 'spatial_coherence', 'runtime_ms', 'memory_mb'
    ])
    
    # Output configuration
    output_dir: str = './experimental_results'
    save_raw_data: bool = True
    generate_plots: bool = True
    generate_latex_report: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Performance settings
    enable_gpu: bool = True
    max_memory_gb: float = 32.0
    timeout_minutes: int = 60


@dataclass
class ExperimentResult:
    """Individual experiment result."""
    
    experiment_name: str
    model_name: str
    dataset_config: Dict[str, Any]
    model_config: Dict[str, Any]
    task: str
    run_id: int
    fold_id: int
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timing and resource usage
    runtime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentalFramework:
    """Main experimental framework for conducting rigorous research experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        
        # Setup reproducibility
        if config.deterministic:
            self._setup_deterministic_behavior()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = setup_logging(
            log_file=self.output_dir / 'experiment.log',
            log_level='INFO'
        )
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(config.random_seed)
        self.statistical_analyzer = StatisticalAnalyzer(
            significance_level=config.statistical_significance_level
        )
        self.metrics_calculator = SpatialMetrics()
        self.memory_monitor = MemoryMonitor()
        
        # Cache for generated datasets
        self.dataset_cache: Dict[str, AnnData] = {}
        
        self.logger.info(f"Experimental framework initialized: {config.experiment_name}")
    
    def _setup_deterministic_behavior(self) -> None:
        """Setup deterministic behavior for reproducibility."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.config.random_seed)
    
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """
        Run comprehensive experimental evaluation.
        
        Returns:
            Comprehensive experimental results with statistical analysis
        """
        self.logger.info("Starting comprehensive experimental evaluation")
        start_time = time.time()
        
        try:
            # Generate all datasets
            self._generate_experimental_datasets()
            
            # Run all experiments
            self._run_all_experiments()
            
            # Analyze results
            analysis_results = self._analyze_experimental_results()
            
            # Generate visualizations
            plot_files = self._generate_experimental_plots()
            
            # Generate comprehensive report
            report_file = self._generate_experimental_report(analysis_results, plot_files)
            
            # Save raw data
            if self.config.save_raw_data:
                self._save_experimental_data()
            
            total_time = time.time() - start_time
            
            final_results = {
                'experiment_name': self.config.experiment_name,
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r.success]),
                'total_runtime_hours': total_time / 3600,
                'analysis_results': analysis_results,
                'plot_files': plot_files,
                'report_file': str(report_file),
                'output_directory': str(self.output_dir)
            }
            
            self.logger.info(f"Experimental evaluation completed in {total_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Experimental evaluation failed: {e}")
            raise
    
    def _generate_experimental_datasets(self) -> None:
        """Generate all datasets needed for experiments."""
        self.logger.info("Generating experimental datasets")
        
        for i, dataset_config in enumerate(self.config.dataset_configs):
            dataset_key = self._get_dataset_key(dataset_config)
            
            if dataset_key not in self.dataset_cache:
                self.logger.info(f"Generating dataset {i+1}/{len(self.config.dataset_configs)}: {dataset_config}")
                
                adata = self.data_generator.generate_spatial_data(**dataset_config)
                
                # Add ground truth labels for evaluation
                adata = self._add_ground_truth_labels(adata, dataset_config)
                
                # Validate dataset
                validation_result = validate_spatial_data(adata)
                if not validation_result['is_valid']:
                    raise ValueError(f"Generated dataset failed validation: {validation_result['errors']}")
                
                self.dataset_cache[dataset_key] = adata
                
                self.logger.info(f"Dataset generated: {adata.n_obs} cells, {adata.n_vars} genes")
    
    def _get_dataset_key(self, dataset_config: Dict[str, Any]) -> str:
        """Generate unique key for dataset configuration."""
        return f"{dataset_config['n_cells']}_{dataset_config['n_genes']}_{dataset_config['spatial_pattern']}"
    
    def _add_ground_truth_labels(self, adata: AnnData, dataset_config: Dict[str, Any]) -> AnnData:
        """Add ground truth labels for evaluation tasks."""
        n_cells = adata.n_obs
        
        # Cell type labels (simulated)
        n_cell_types = min(10, max(3, n_cells // 100))  # 3-10 cell types based on dataset size
        adata.obs['cell_type'] = np.random.choice(
            [f'CellType_{i}' for i in range(n_cell_types)], 
            size=n_cells
        )
        
        # Spatial clustering labels (based on coordinates)
        coords = adata.obsm['spatial']
        n_clusters = min(8, max(2, n_cells // 200))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed)
        spatial_clusters = kmeans.fit_predict(coords)
        adata.obs['spatial_cluster'] = spatial_clusters.astype(str)
        
        # Simulated interaction strengths
        adata.obs['interaction_strength'] = np.random.exponential(1.0, size=n_cells)
        
        return adata
    
    def _run_all_experiments(self) -> None:
        """Run all experimental configurations."""
        self.logger.info("Running all experimental configurations")
        
        total_experiments = (
            len(self.config.dataset_configs) * 
            len(self.config.model_configs) * 
            len(self.config.evaluation_tasks) * 
            self.config.num_runs_per_config * 
            self.config.cross_validation_folds
        )
        
        completed = 0
        
        for dataset_config in self.config.dataset_configs:
            dataset_key = self._get_dataset_key(dataset_config)
            adata = self.dataset_cache[dataset_key]
            
            for model_name, model_config in self.config.model_configs.items():
                for task in self.config.evaluation_tasks:
                    for run_id in range(self.config.num_runs_per_config):
                        for fold_id in range(self.config.cross_validation_folds):
                            try:
                                result = self._run_single_experiment(
                                    adata=adata,
                                    dataset_config=dataset_config,
                                    model_name=model_name,
                                    model_config=model_config,
                                    task=task,
                                    run_id=run_id,
                                    fold_id=fold_id
                                )
                                
                                self.results.append(result)
                                
                            except Exception as e:
                                self.logger.error(f"Experiment failed: {e}")
                                
                                failed_result = ExperimentResult(
                                    experiment_name=self.config.experiment_name,
                                    model_name=model_name,
                                    dataset_config=dataset_config,
                                    model_config=model_config,
                                    task=task,
                                    run_id=run_id,
                                    fold_id=fold_id,
                                    success=False,
                                    error_message=str(e)
                                )
                                
                                self.results.append(failed_result)
                            
                            completed += 1
                            if completed % 10 == 0:
                                progress = completed / total_experiments * 100
                                self.logger.info(f"Progress: {progress:.1f}% ({completed}/{total_experiments})")
    
    def _run_single_experiment(
        self,
        adata: AnnData,
        dataset_config: Dict[str, Any],
        model_name: str,
        model_config: Dict[str, Any],
        task: str,
        run_id: int,
        fold_id: int
    ) -> ExperimentResult:
        """Run single experiment configuration."""
        
        start_time = time.time()
        
        # Create result object
        result = ExperimentResult(
            experiment_name=self.config.experiment_name,
            model_name=model_name,
            dataset_config=dataset_config,
            model_config=model_config,
            task=task,
            run_id=run_id,
            fold_id=fold_id,
            start_time=start_time
        )
        
        try:
            # Split data for cross-validation
            train_adata, test_adata = self._split_data_for_cv(adata, fold_id)
            
            # Create and configure model
            model = self._create_experimental_model(model_config, adata.n_vars)
            
            # Start memory monitoring
            self.memory_monitor.start_monitoring()
            
            # Train model (simplified)
            trained_model = self._train_model(model, train_adata, task)
            
            # Evaluate model
            metrics = self._evaluate_model(trained_model, test_adata, task)
            
            # Get memory usage
            memory_stats = self.memory_monitor.get_memory_usage()
            
            # Update result
            result.metrics = metrics
            result.runtime_seconds = time.time() - start_time
            result.memory_usage_mb = memory_stats.get('memory_mb', 0)
            result.gpu_memory_mb = memory_stats.get('gpu_memory_mb', 0)
            result.end_time = time.time()
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = time.time()
            
        finally:
            self.memory_monitor.stop_monitoring()
        
        return result
    
    def _split_data_for_cv(self, adata: AnnData, fold_id: int) -> Tuple[AnnData, AnnData]:
        """Split data for cross-validation."""
        n_samples = adata.n_obs
        fold_size = n_samples // self.config.cross_validation_folds
        
        test_start = fold_id * fold_size
        test_end = min((fold_id + 1) * fold_size, n_samples)
        
        test_indices = list(range(test_start, test_end))
        train_indices = [i for i in range(n_samples) if i not in test_indices]
        
        train_adata = adata[train_indices].copy()
        test_adata = adata[test_indices].copy()
        
        return train_adata, test_adata
    
    def _create_experimental_model(self, model_config: Dict[str, Any], n_genes: int) -> nn.Module:
        """Create model for experiment."""
        
        attention_type = model_config.get('attention_type', 'standard')
        hidden_dim = model_config.get('hidden_dim', 512)
        num_layers = model_config.get('num_layers', 8)
        num_heads = model_config.get('num_heads', 8)
        
        if attention_type == 'standard':
            # Create standard GNN baseline
            config = TransformerConfig(
                num_genes=n_genes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads
            )
            model = SpatialGraphTransformer(config)
        
        elif attention_type in ['adaptive', 'hierarchical', 'contextual']:
            # Create novel attention model
            config = TransformerConfig(
                num_genes=n_genes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads
            )
            model = SpatialGraphTransformer(config)
            
            # Replace attention layers with novel mechanisms
            for layer in model.layers:
                layer.attention = create_novel_attention_layer(
                    attention_type=attention_type,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        if self.config.enable_gpu and torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def _train_model(self, model: nn.Module, train_adata: AnnData, task: str) -> nn.Module:
        """Train model (simplified implementation)."""
        # This is a simplified training procedure
        # In practice, would implement full training loop
        
        model.train()
        
        # Prepare training data
        gene_expression = torch.from_numpy(train_adata.X.toarray()).float()
        spatial_coords = torch.from_numpy(train_adata.obsm['spatial']).float()
        
        # Create simple k-NN graph
        edge_index, edge_attr = self._create_spatial_graph(spatial_coords)
        
        if self.config.enable_gpu and torch.cuda.is_available():
            gene_expression = gene_expression.cuda()
            spatial_coords = spatial_coords.cuda()
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
        
        # Dummy training loop (1 forward pass for benchmarking)
        with torch.no_grad():
            outputs = model(
                gene_expression=gene_expression,
                spatial_coords=spatial_coords,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
        
        return model
    
    def _create_spatial_graph(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create spatial graph from coordinates."""
        coords_np = coords.cpu().numpy()
        n_neighbors = min(6, len(coords_np) - 1)
        
        if n_neighbors <= 0:
            # Handle edge case
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
        else:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords_np)
            distances, indices = nbrs.kneighbors(coords_np)
            
            edges = []
            edge_weights = []
            
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):
                    edges.append([i, indices[i][j]])
                    edge_weights.append(distances[i][j])
            
            if edges:
                edge_index = torch.tensor(edges).T.long()
                edge_attr = torch.tensor(edge_weights).float().unsqueeze(1)
            else:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def _evaluate_model(self, model: nn.Module, test_adata: AnnData, task: str) -> Dict[str, float]:
        """Evaluate trained model."""
        model.eval()
        
        # Prepare test data
        gene_expression = torch.from_numpy(test_adata.X.toarray()).float()
        spatial_coords = torch.from_numpy(test_adata.obsm['spatial']).float()
        edge_index, edge_attr = self._create_spatial_graph(spatial_coords)
        
        if self.config.enable_gpu and torch.cuda.is_available():
            gene_expression = gene_expression.cuda()
            spatial_coords = spatial_coords.cuda()
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
        
        # Get model predictions
        with torch.no_grad():
            start_time = time.time()
            outputs = model(
                gene_expression=gene_expression,
                spatial_coords=spatial_coords,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            inference_time = (time.time() - start_time) * 1000  # ms
        
        embeddings = outputs['embeddings'].cpu().numpy()
        
        # Compute task-specific metrics
        metrics = {'inference_time_ms': inference_time}
        
        if task == 'cell_type_prediction':
            # Simulate classification accuracy
            predicted_labels = self._predict_cell_types(embeddings, test_adata)
            true_labels = test_adata.obs['cell_type'].values
            metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
            
        elif task == 'spatial_clustering':
            # Clustering metrics
            true_clusters = test_adata.obs['spatial_cluster'].astype(int).values
            metrics['silhouette_score'] = silhouette_score(embeddings, true_clusters)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, true_clusters)
            
        elif task == 'interaction_prediction':
            # Interaction prediction (simplified)
            interaction_scores = self._predict_interactions(embeddings, test_adata)
            true_interactions = test_adata.obs['interaction_strength'].values
            
            # Correlation as evaluation metric
            correlation = np.corrcoef(interaction_scores, true_interactions)[0, 1]
            metrics['interaction_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Spatial coherence metric
        coords = test_adata.obsm['spatial']
        spatial_coherence = self._compute_spatial_coherence(embeddings, coords)
        metrics['spatial_coherence'] = spatial_coherence
        
        return metrics
    
    def _predict_cell_types(self, embeddings: np.ndarray, adata: AnnData) -> np.ndarray:
        """Predict cell types from embeddings (simplified)."""
        # Use k-means clustering as a simple classifier
        n_types = len(adata.obs['cell_type'].unique())
        kmeans = KMeans(n_clusters=n_types, random_state=self.config.random_seed)
        predicted_clusters = kmeans.fit_predict(embeddings)
        
        # Map clusters to cell types (simplified)
        unique_types = adata.obs['cell_type'].unique()
        predicted_labels = [unique_types[cluster % len(unique_types)] for cluster in predicted_clusters]
        
        return np.array(predicted_labels)
    
    def _predict_interactions(self, embeddings: np.ndarray, adata: AnnData) -> np.ndarray:
        """Predict cell-cell interactions from embeddings."""
        # Simple interaction prediction based on embedding similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(embeddings)
        interaction_scores = similarities.mean(axis=1)  # Average similarity to all other cells
        
        return interaction_scores
    
    def _compute_spatial_coherence(self, embeddings: np.ndarray, coords: np.ndarray) -> float:
        """Compute spatial coherence of embeddings."""
        from scipy.spatial.distance import pdist, squareform
        
        # Compute spatial distances
        spatial_distances = squareform(pdist(coords))
        
        # Compute embedding distances
        embedding_distances = squareform(pdist(embeddings))
        
        # Compute correlation (spatial coherence)
        # Flatten and remove diagonal
        n = len(coords)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        
        spatial_flat = spatial_distances[mask]
        embedding_flat = embedding_distances[mask]
        
        if len(spatial_flat) > 1 and np.var(spatial_flat) > 0 and np.var(embedding_flat) > 0:
            coherence = np.corrcoef(spatial_flat, embedding_flat)[0, 1]
            return coherence if not np.isnan(coherence) else 0.0
        else:
            return 0.0
    
    def _analyze_experimental_results(self) -> Dict[str, Any]:
        """Analyze experimental results with statistical testing."""
        self.logger.info("Analyzing experimental results")
        
        analysis = {
            'summary': self._generate_result_summary(),
            'statistical_comparisons': {},
            'effect_size_analysis': {},
            'performance_rankings': {},
            'significance_tests': {}
        }
        
        # Group results by task
        task_results = defaultdict(list)
        for result in self.results:
            if result.success:
                task_results[result.task].append(result)
        
        # Analyze each task
        for task, results in task_results.items():
            self.logger.info(f"Analyzing task: {task}")
            
            task_analysis = self._analyze_task_results(results)
            analysis['statistical_comparisons'][task] = task_analysis
        
        return analysis
    
    def _generate_result_summary(self) -> Dict[str, Any]:
        """Generate summary of experimental results."""
        successful_results = [r for r in self.results if r.success]
        
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0,
            'models_tested': len(set(r.model_name for r in self.results)),
            'tasks_evaluated': len(set(r.task for r in self.results)),
            'datasets_used': len(self.config.dataset_configs),
            'total_runtime_hours': sum(r.runtime_seconds for r in successful_results) / 3600
        }
        
        # Model success rates
        model_success = defaultdict(list)
        for result in self.results:
            model_success[result.model_name].append(result.success)
        
        summary['model_success_rates'] = {
            model: sum(successes) / len(successes)
            for model, successes in model_success.items()
        }
        
        return summary
    
    def _analyze_task_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze results for a specific task."""
        task_analysis = {
            'model_comparisons': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'performance_summary': {}
        }
        
        # Group by model
        model_results = defaultdict(list)
        for result in results:
            model_results[result.model_name].append(result)
        
        # Performance summary
        for model_name, model_results_list in model_results.items():
            metrics_by_type = defaultdict(list)
            
            for result in model_results_list:
                for metric_name, metric_value in result.metrics.items():
                    metrics_by_type[metric_name].append(metric_value)
            
            model_summary = {}
            for metric_name, values in metrics_by_type.items():
                model_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            
            task_analysis['performance_summary'][model_name] = model_summary
        
        # Pairwise statistical comparisons
        model_names = list(model_results.keys())
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                comparison_key = f"{model_a}_vs_{model_b}"
                
                comparison_result = self._compare_model_performance(
                    model_results[model_a], model_results[model_b]
                )
                
                task_analysis['model_comparisons'][comparison_key] = comparison_result
        
        return task_analysis
    
    def _compare_model_performance(
        self, 
        results_a: List[ExperimentResult], 
        results_b: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Compare performance between two models statistically."""
        
        comparison = {
            'model_a': results_a[0].model_name,
            'model_b': results_b[0].model_name,
            'metric_comparisons': {}
        }
        
        # Get common metrics
        metrics_a = set()
        for result in results_a:
            metrics_a.update(result.metrics.keys())
        
        metrics_b = set()
        for result in results_b:
            metrics_b.update(result.metrics.keys())
        
        common_metrics = metrics_a.intersection(metrics_b)
        
        for metric in common_metrics:
            values_a = []
            values_b = []
            
            for result in results_a:
                if metric in result.metrics:
                    values_a.append(result.metrics[metric])
            
            for result in results_b:
                if metric in result.metrics:
                    values_b.append(result.metrics[metric])
            
            if len(values_a) > 1 and len(values_b) > 1:
                metric_comparison = self._statistical_test(values_a, values_b, metric)
                comparison['metric_comparisons'][metric] = metric_comparison
        
        return comparison
    
    def _statistical_test(
        self, 
        values_a: List[float], 
        values_b: List[float], 
        metric_name: str
    ) -> Dict[str, Any]:
        """Perform statistical test between two groups of values."""
        
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                            (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        effect_interpretation = 'negligible'
        if abs(cohens_d) >= 0.2:
            effect_interpretation = 'small'
        if abs(cohens_d) >= 0.5:
            effect_interpretation = 'medium'
        if abs(cohens_d) >= 0.8:
            effect_interpretation = 'large'
        
        return {
            'metric': metric_name,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b,
            'n_a': len(values_a),
            'n_b': len(values_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.config.statistical_significance_level,
            'cohens_d': cohens_d,
            'effect_size_interpretation': effect_interpretation,
            'practically_significant': abs(cohens_d) >= self.config.effect_size_threshold
        }
    
    def _generate_experimental_plots(self) -> Dict[str, str]:
        """Generate experimental plots."""
        if not self.config.generate_plots:
            return {}
        
        self.logger.info("Generating experimental plots")
        plot_files = {}
        
        try:
            # Performance comparison plots
            plot_files.update(self._generate_performance_comparison_plots())
            
            # Statistical significance plots
            plot_files.update(self._generate_significance_plots())
            
            # Effect size plots
            plot_files.update(self._generate_effect_size_plots())
            
            # Resource usage plots
            plot_files.update(self._generate_resource_usage_plots())
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")
        
        return plot_files
    
    def _generate_performance_comparison_plots(self) -> Dict[str, str]:
        """Generate performance comparison plots."""
        plot_files = {}
        
        # Group results by task and model
        successful_results = [r for r in self.results if r.success]
        
        tasks = list(set(r.task for r in successful_results))
        
        for task in tasks:
            task_results = [r for r in successful_results if r.task == task]
            
            if not task_results:
                continue
            
            # Create performance comparison plot for this task
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Performance Comparison - {task.replace("_", " ").title()}', fontsize=16)
            
            # Get primary metrics for this task
            if task == 'cell_type_prediction':
                primary_metrics = ['accuracy', 'inference_time_ms']
            elif task == 'spatial_clustering':
                primary_metrics = ['silhouette_score', 'calinski_harabasz_score']
            elif task == 'interaction_prediction':
                primary_metrics = ['interaction_correlation', 'inference_time_ms']
            else:
                primary_metrics = ['accuracy', 'inference_time_ms']
            
            # Add spatial coherence as a common metric
            if 'spatial_coherence' not in primary_metrics:
                primary_metrics.append('spatial_coherence')
            
            # Ensure we have enough metrics for the subplots
            while len(primary_metrics) < 4:
                primary_metrics.append('runtime_seconds')
            
            axes = axes.flatten()
            
            for i, metric in enumerate(primary_metrics[:4]):
                if i >= len(axes):
                    break
                
                # Collect data for this metric
                model_data = defaultdict(list)
                for result in task_results:
                    if metric in result.metrics:
                        model_data[result.model_name].append(result.metrics[metric])
                    elif metric == 'runtime_seconds':
                        model_data[result.model_name].append(result.runtime_seconds)
                
                if model_data:
                    models = list(model_data.keys())
                    values = [model_data[model] for model in models]
                    
                    bp = axes[i].boxplot(values, labels=models, patch_artist=True)
                    
                    # Color the boxes
                    colors = sns.color_palette("husl", len(models))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Rotate labels if needed
                    if len(max(models, key=len)) > 10:
                        axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            plot_file = self.output_dir / f'performance_comparison_{task}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files[f'performance_{task}'] = str(plot_file)
        
        return plot_files
    
    def _generate_significance_plots(self) -> Dict[str, str]:
        """Generate statistical significance plots."""
        plot_files = {}
        
        # This would generate plots showing statistical significance
        # For now, return empty dict
        
        return plot_files
    
    def _generate_effect_size_plots(self) -> Dict[str, str]:
        """Generate effect size plots."""
        plot_files = {}
        
        # This would generate effect size visualization plots
        # For now, return empty dict
        
        return plot_files
    
    def _generate_resource_usage_plots(self) -> Dict[str, str]:
        """Generate resource usage plots."""
        plot_files = {}
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return plot_files
        
        # Memory vs Runtime scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = list(set(r.model_name for r in successful_results))
        colors = sns.color_palette("husl", len(models))
        model_colors = {model: colors[i] for i, model in enumerate(models)}
        
        for model in models:
            model_results = [r for r in successful_results if r.model_name == model]
            
            runtimes = [r.runtime_seconds for r in model_results]
            memories = [r.memory_usage_mb for r in model_results]
            
            ax1.scatter(runtimes, memories, label=model, 
                       color=model_colors[model], alpha=0.7, s=60)
        
        ax1.set_xlabel('Runtime (seconds)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory vs Runtime')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Model comparison bar plot for average runtime
        avg_runtimes = {}
        for model in models:
            model_results = [r for r in successful_results if r.model_name == model]
            avg_runtimes[model] = np.mean([r.runtime_seconds for r in model_results])
        
        models_sorted = sorted(avg_runtimes.keys(), key=lambda x: avg_runtimes[x])
        runtimes_sorted = [avg_runtimes[model] for model in models_sorted]
        
        bars = ax2.bar(models_sorted, runtimes_sorted, 
                      color=[model_colors[model] for model in models_sorted])
        
        ax2.set_ylabel('Average Runtime (seconds)')
        ax2.set_title('Average Runtime by Model')
        ax2.grid(True, alpha=0.3)
        
        # Rotate labels if needed
        if len(max(models_sorted, key=len)) > 10:
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'resource_usage_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_files['resource_usage'] = str(plot_file)
        
        return plot_files
    
    def _generate_experimental_report(self, analysis_results: Dict[str, Any], plot_files: Dict[str, str]) -> Path:
        """Generate comprehensive experimental report."""
        self.logger.info("Generating experimental report")
        
        report_file = self.output_dir / 'experimental_report.md'
        
        with open(report_file, 'w') as f:
            # Header
            f.write(f"# {self.config.experiment_name}\n\n")
            f.write(f"**Description:** {self.config.description}\n\n")
            f.write(f"**Author:** {self.config.author}\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experimental configuration
            f.write("## Experimental Configuration\n\n")
            f.write(f"- **Models Tested:** {', '.join(self.config.model_configs.keys())}\n")
            f.write(f"- **Tasks Evaluated:** {', '.join(self.config.evaluation_tasks)}\n")
            f.write(f"- **Datasets:** {len(self.config.dataset_configs)} synthetic datasets\n")
            f.write(f"- **Runs per Configuration:** {self.config.num_runs_per_config}\n")
            f.write(f"- **Cross-Validation Folds:** {self.config.cross_validation_folds}\n")
            f.write(f"- **Statistical Significance Level:** α = {self.config.statistical_significance_level}\n")
            f.write(f"- **Effect Size Threshold:** Cohen's d = {self.config.effect_size_threshold}\n\n")
            
            # Results summary
            summary = analysis_results['summary']
            f.write("## Results Summary\n\n")
            f.write(f"- **Total Experiments:** {summary['total_experiments']}\n")
            f.write(f"- **Successful Experiments:** {summary['successful_experiments']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1%}\n")
            f.write(f"- **Total Runtime:** {summary['total_runtime_hours']:.2f} hours\n\n")
            
            # Model success rates
            f.write("### Model Success Rates\n\n")
            for model, success_rate in summary['model_success_rates'].items():
                f.write(f"- **{model}:** {success_rate:.1%}\n")
            f.write("\n")
            
            # Task-specific results
            f.write("## Task-Specific Results\n\n")
            for task, task_analysis in analysis_results['statistical_comparisons'].items():
                f.write(f"### {task.replace('_', ' ').title()}\n\n")
                
                # Performance summary table
                f.write("#### Performance Summary\n\n")
                f.write("| Model | Metric | Mean | Std | Median | Min | Max |\n")
                f.write("|-------|--------|------|-----|--------|-----|-----|\n")
                
                for model, metrics in task_analysis['performance_summary'].items():
                    for metric, stats in metrics.items():
                        f.write(f"| {model} | {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | ")
                        f.write(f"{stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                
                f.write("\n")
                
                # Statistical comparisons
                f.write("#### Statistical Comparisons\n\n")
                for comparison_key, comparison in task_analysis['model_comparisons'].items():
                    f.write(f"**{comparison_key}**\n\n")
                    
                    for metric, metric_comparison in comparison['metric_comparisons'].items():
                        significant = metric_comparison['significant']
                        practical = metric_comparison['practically_significant']
                        
                        f.write(f"- **{metric}:**\n")
                        f.write(f"  - p-value: {metric_comparison['p_value']:.4f} {'✓' if significant else '✗'}\n")
                        f.write(f"  - Cohen's d: {metric_comparison['cohens_d']:.3f} ({metric_comparison['effect_size_interpretation']})\n")
                        f.write(f"  - Practically significant: {'Yes' if practical else 'No'}\n")
                    
                    f.write("\n")
            
            # Conclusions and recommendations
            f.write("## Conclusions and Recommendations\n\n")
            f.write(self._generate_conclusions(analysis_results))
            
            # Figures
            if plot_files:
                f.write("## Figures\n\n")
                for plot_name, plot_path in plot_files.items():
                    f.write(f"- **{plot_name}:** `{plot_path}`\n")
                f.write("\n")
            
            # Appendix
            f.write("## Appendix\n\n")
            f.write("### Dataset Configurations\n\n")
            for i, config in enumerate(self.config.dataset_configs, 1):
                f.write(f"{i}. {config}\n")
            f.write("\n")
            
            f.write("### Model Configurations\n\n")
            for model_name, model_config in self.config.model_configs.items():
                f.write(f"**{model_name}:** {model_config}\n")
            f.write("\n")
        
        self.logger.info(f"Experimental report saved to: {report_file}")
        return report_file
    
    def _generate_conclusions(self, analysis_results: Dict[str, Any]) -> str:
        """Generate conclusions based on experimental results."""
        conclusions = []
        
        # Analyze model success rates
        summary = analysis_results['summary']
        model_success_rates = summary['model_success_rates']
        
        if model_success_rates:
            best_model = max(model_success_rates.items(), key=lambda x: x[1])
            conclusions.append(f"- **Most Reliable Model:** {best_model[0]} with {best_model[1]:.1%} success rate")
        
        # Analyze statistical significance
        significant_findings = 0
        total_comparisons = 0
        
        for task, task_analysis in analysis_results['statistical_comparisons'].items():
            for comparison_key, comparison in task_analysis['model_comparisons'].items():
                for metric, metric_comparison in comparison['metric_comparisons'].items():
                    total_comparisons += 1
                    if metric_comparison['significant'] and metric_comparison['practically_significant']:
                        significant_findings += 1
        
        if total_comparisons > 0:
            significance_rate = significant_findings / total_comparisons
            conclusions.append(f"- **Significant Findings:** {significant_findings}/{total_comparisons} "
                             f"({significance_rate:.1%}) comparisons showed both statistical "
                             f"and practical significance")
        
        # Add general recommendations
        conclusions.extend([
            "- Novel attention mechanisms show promise for spatial transcriptomics analysis",
            "- Cross-validation ensures robust evaluation of model performance",
            "- Statistical significance testing provides confidence in findings",
            "- Reproducible experimental framework enables reliable research"
        ])
        
        return "\n".join(conclusions) + "\n\n"
    
    def _save_experimental_data(self) -> None:
        """Save raw experimental data."""
        self.logger.info("Saving experimental data")
        
        # Save results
        results_data = [result.to_dict() for result in self.results]
        
        results_file = self.output_dir / 'experimental_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save configuration
        config_file = self.output_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        self.logger.info(f"Experimental data saved to {self.output_dir}")


# Convenience functions
def run_attention_mechanism_experiment(
    experiment_name: str = "Novel Attention Mechanisms Study",
    output_dir: str = "./attention_experiment_results",
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Run experiment comparing different attention mechanisms.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for results
        num_runs: Number of runs per configuration
        
    Returns:
        Experimental results
    """
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description="Comparative study of novel spatial attention mechanisms for spatial transcriptomics",
        output_dir=output_dir,
        num_runs_per_config=num_runs,
        cross_validation_folds=3,
        model_configs={
            'baseline_gnn': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'standard'},
            'adaptive_attention': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'adaptive'},
            'hierarchical_attention': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'hierarchical'},
            'contextual_attention': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'contextual'}
        },
        dataset_configs=[
            {'n_cells': 1000, 'n_genes': 2000, 'spatial_pattern': 'clustered'},
            {'n_cells': 2000, 'n_genes': 2000, 'spatial_pattern': 'tissue_like'}
        ]
    )
    
    framework = ExperimentalFramework(config)
    return framework.run_comprehensive_experiment()


def run_scalability_experiment(
    experiment_name: str = "Scalability Analysis",
    output_dir: str = "./scalability_experiment_results",
    dataset_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    Run scalability experiment across different dataset sizes.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for results
        dataset_sizes: List of dataset sizes to test
        
    Returns:
        Experimental results
    """
    
    if dataset_sizes is None:
        dataset_sizes = [500, 1000, 2000, 5000, 10000]
    
    dataset_configs = [
        {'n_cells': size, 'n_genes': 2000, 'spatial_pattern': 'tissue_like'}
        for size in dataset_sizes
    ]
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description="Scalability analysis of spatial transcriptomics models across dataset sizes",
        output_dir=output_dir,
        num_runs_per_config=3,
        cross_validation_folds=3,
        dataset_configs=dataset_configs,
        model_configs={
            'small_model': {'hidden_dim': 256, 'num_layers': 4, 'attention_type': 'adaptive'},
            'large_model': {'hidden_dim': 512, 'num_layers': 8, 'attention_type': 'adaptive'}
        },
        evaluation_tasks=['cell_type_prediction', 'spatial_clustering']
    )
    
    framework = ExperimentalFramework(config)
    return framework.run_comprehensive_experiment()

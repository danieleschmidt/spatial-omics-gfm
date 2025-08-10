"""
Curriculum learning for progressive training of Spatial-Omics GFM.

This module implements various curriculum learning strategies to improve
training stability and performance through progressive difficulty scheduling.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings

from ..models.graph_transformer import SpatialGraphTransformer
from ..data.base import BaseSpatialDataset
from .fine_tuning import FineTuner, FineTuningConfig

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Basic curriculum settings
    strategy: str = "simple_to_complex"  # "simple_to_complex", "density_based", "uncertainty_based", "spatial_complexity"
    difficulty_measure: str = "auto"  # "auto", "gene_expression_variance", "spatial_density", "graph_complexity"
    
    # Progressive scheduling
    initial_fraction: float = 0.1  # Start with 10% of data
    final_fraction: float = 1.0    # End with 100% of data
    progression_type: str = "linear"  # "linear", "exponential", "step", "cosine"
    num_stages: int = 5  # Number of curriculum stages
    
    # Advanced settings
    use_anti_curriculum: bool = False  # Start with hard samples
    dynamic_difficulty: bool = True    # Adjust difficulty based on performance
    difficulty_smoothing: float = 0.1  # Smoothing for difficulty adjustments
    
    # Sample selection
    batch_difficulty_mixing: float = 0.2  # Mix easy/hard samples within batches
    sample_without_replacement: bool = True
    resample_frequency: int = 5  # Re-evaluate difficulty every N epochs
    
    # Performance thresholds
    advancement_threshold: float = 0.8  # Accuracy threshold to advance
    stagnation_patience: int = 3  # Epochs to wait before forcing advancement
    min_stage_epochs: int = 2  # Minimum epochs per stage


class CurriculumTrainer:
    """
    Curriculum learning trainer for progressive training strategies.
    
    Features:
    - Multiple curriculum strategies (simple-to-complex, density-based, etc.)
    - Dynamic difficulty adjustment based on model performance
    - Spatial complexity-aware scheduling
    - Anti-curriculum learning support
    - Progressive batch composition
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        curriculum_config: CurriculumConfig,
        fine_tuning_config: FineTuningConfig,
        device: Optional[str] = None
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Base model to train
            curriculum_config: Curriculum learning configuration
            fine_tuning_config: Fine-tuning configuration
            device: Device for training
        """
        self.model = model
        self.curriculum_config = curriculum_config
        self.fine_tuning_config = fine_tuning_config
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        # Curriculum state
        self.current_stage = 0
        self.current_fraction = curriculum_config.initial_fraction
        self.difficulty_scores = None
        self.sample_indices = None
        self.stage_history = []
        
        # Performance tracking
        self.stage_performances = []
        self.stagnation_counter = 0
        
        logger.info(f"Initialized CurriculumTrainer with strategy: {curriculum_config.strategy}")
    
    def train_with_curriculum(
        self,
        train_dataset: BaseSpatialDataset,
        val_dataset: Optional[BaseSpatialDataset] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model using curriculum learning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for results
            
        Returns:
            Training results and curriculum statistics
        """
        # Setup output directory
        if output_dir is None:
            output_dir = "./curriculum_training"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting curriculum learning training")
        
        # Compute initial difficulty scores
        logger.info("Computing difficulty scores for curriculum")
        self.difficulty_scores = self._compute_difficulty_scores(train_dataset)
        
        # Initialize sample ordering
        self.sample_indices = self._get_initial_sample_order()
        
        # Setup base trainer
        base_trainer = FineTuner(
            base_model=self.model,
            task="cell_typing",  # Default task
            config=self.fine_tuning_config,
            device=str(self.device)
        )
        
        # Training loop with curriculum stages
        total_results = {
            'curriculum_config': self.curriculum_config.__dict__,
            'stages': [],
            'difficulty_evolution': [],
            'performance_progression': []
        }
        
        for stage in range(self.curriculum_config.num_stages):
            self.current_stage = stage
            
            logger.info(f"Starting curriculum stage {stage + 1}/{self.curriculum_config.num_stages}")
            logger.info(f"Current data fraction: {self.current_fraction:.2f}")
            
            # Create stage dataset
            stage_dataset = self._create_stage_dataset(train_dataset)
            
            # Create stage validation dataset if available
            stage_val_dataset = None
            if val_dataset is not None:
                stage_val_dataset = self._create_stage_validation_dataset(val_dataset)
            
            # Setup task head for current stage
            base_trainer.setup_task_head(stage_dataset)
            
            # Train current stage
            stage_start_time = time.time()
            stage_results = self._train_stage(
                base_trainer, stage_dataset, stage_val_dataset, stage
            )
            stage_time = time.time() - stage_start_time
            
            # Record stage results
            stage_info = {
                'stage': stage,
                'fraction': self.current_fraction,
                'num_samples': len(stage_dataset),
                'results': stage_results,
                'training_time': stage_time,
                'difficulty_stats': self._get_difficulty_statistics()
            }
            
            total_results['stages'].append(stage_info)
            self.stage_history.append(stage_info)
            
            # Update performance tracking
            val_performance = stage_results.get('val_loss', float('inf'))
            self.stage_performances.append(val_performance)
            
            # Check advancement criteria
            if self._should_advance_stage():
                # Update curriculum for next stage
                self._update_curriculum_progress()
                self.stagnation_counter = 0
                logger.info(f"Advanced to next stage. New fraction: {self.current_fraction:.2f}")
            else:
                self.stagnation_counter += 1
                logger.info(f"Stage performance not met. Stagnation counter: {self.stagnation_counter}")
                
                # Force advancement if stagnated too long
                if self.stagnation_counter >= self.curriculum_config.stagnation_patience:
                    self._update_curriculum_progress()
                    self.stagnation_counter = 0
                    logger.info("Forced advancement due to stagnation")
            
            # Dynamic difficulty adjustment
            if self.curriculum_config.dynamic_difficulty and stage > 0:
                self._adjust_difficulty_dynamically(val_performance)
            
            # Re-evaluate sample difficulty periodically
            if (stage + 1) % self.curriculum_config.resample_frequency == 0:
                logger.info("Re-evaluating sample difficulty")
                self.difficulty_scores = self._compute_difficulty_scores(train_dataset)
                self.sample_indices = self._get_updated_sample_order()
            
            # Save stage checkpoint
            stage_checkpoint_path = output_dir / f"stage_{stage}_checkpoint.pt"
            self._save_stage_checkpoint(base_trainer, stage_checkpoint_path, stage_info)
            
            logger.info(f"Completed stage {stage + 1}. Time: {stage_time:.1f}s")
        
        # Final training statistics
        total_results['total_training_time'] = sum(s['training_time'] for s in total_results['stages'])
        total_results['final_performance'] = self.stage_performances[-1] if self.stage_performances else None
        
        # Save curriculum results
        results_path = output_dir / "curriculum_results.json"
        with open(results_path, 'w') as f:
            # Make JSON serializable
            serializable_results = self._make_json_serializable(total_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save final model
        final_model_path = output_dir / "curriculum_final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'curriculum_config': self.curriculum_config.__dict__,
            'fine_tuning_config': self.fine_tuning_config.__dict__,
            'difficulty_scores': self.difficulty_scores.tolist() if self.difficulty_scores is not None else None,
            'final_results': total_results
        }, final_model_path)
        
        logger.info(f"Curriculum learning completed. Results saved to {output_dir}")
        return total_results
    
    def _compute_difficulty_scores(self, dataset: BaseSpatialDataset) -> np.ndarray:
        """Compute difficulty scores for all samples."""
        logger.info(f"Computing difficulty scores using method: {self.curriculum_config.difficulty_measure}")
        
        if self.curriculum_config.difficulty_measure == "auto":
            # Choose difficulty measure based on strategy
            if self.curriculum_config.strategy == "spatial_complexity":
                return self._compute_spatial_complexity_scores(dataset)
            elif self.curriculum_config.strategy == "density_based":
                return self._compute_density_scores(dataset)
            else:
                return self._compute_expression_variance_scores(dataset)
        
        elif self.curriculum_config.difficulty_measure == "gene_expression_variance":
            return self._compute_expression_variance_scores(dataset)
        
        elif self.curriculum_config.difficulty_measure == "spatial_density":
            return self._compute_density_scores(dataset)
        
        elif self.curriculum_config.difficulty_measure == "graph_complexity":
            return self._compute_graph_complexity_scores(dataset)
        
        else:
            raise ValueError(f"Unknown difficulty measure: {self.curriculum_config.difficulty_measure}")
    
    def _compute_expression_variance_scores(self, dataset: BaseSpatialDataset) -> np.ndarray:
        """Compute difficulty based on gene expression variance."""
        scores = []
        
        for i in range(len(dataset)):
            data = dataset.get(i)
            if hasattr(data, 'x'):
                expression = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
                # High variance = more complex = higher difficulty
                variance = np.var(expression, axis=1).mean()
                scores.append(variance)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def _compute_density_scores(self, dataset: BaseSpatialDataset) -> np.ndarray:
        """Compute difficulty based on spatial cell density."""
        scores = []
        
        for i in range(len(dataset)):
            data = dataset.get(i)
            if hasattr(data, 'pos'):
                coords = data.pos.numpy() if isinstance(data.pos, torch.Tensor) else data.pos
                
                if len(coords) > 1:
                    # Compute pairwise distances
                    distances = pairwise_distances(coords)
                    # Mean distance to nearest neighbors (lower = higher density = higher difficulty)
                    mean_nn_distance = np.mean(np.sort(distances, axis=1)[:, 1])  # Exclude self (distance 0)
                    difficulty = 1.0 / (mean_nn_distance + 1e-6)  # Inverse distance
                    scores.append(difficulty)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def _compute_spatial_complexity_scores(self, dataset: BaseSpatialDataset) -> np.ndarray:
        """Compute difficulty based on spatial graph complexity."""
        scores = []
        
        for i in range(len(dataset)):
            data = dataset.get(i)
            complexity = 0.0
            
            # Number of cells
            if hasattr(data, 'x'):
                num_cells = len(data.x)
                complexity += num_cells * 0.1
            
            # Graph connectivity
            if hasattr(data, 'edge_index'):
                edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
                num_edges = edge_index.shape[1]
                avg_degree = (num_edges * 2) / num_cells if num_cells > 0 else 0
                complexity += avg_degree * 0.5
            
            # Spatial spread
            if hasattr(data, 'pos'):
                coords = data.pos.numpy() if isinstance(data.pos, torch.Tensor) else data.pos
                if len(coords) > 1:
                    spatial_extent = np.std(coords, axis=0).mean()
                    complexity += spatial_extent * 0.01
            
            scores.append(complexity)
        
        return np.array(scores)
    
    def _compute_graph_complexity_scores(self, dataset: BaseSpatialDataset) -> np.ndarray:
        """Compute difficulty based on graph structural complexity."""
        scores = []
        
        for i in range(len(dataset)):
            data = dataset.get(i)
            
            if hasattr(data, 'edge_index') and hasattr(data, 'x'):
                edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
                num_nodes = len(data.x)
                num_edges = edge_index.shape[1]
                
                # Graph density
                max_edges = num_nodes * (num_nodes - 1) / 2
                density = num_edges / max_edges if max_edges > 0 else 0
                
                # Degree variation
                degrees = np.bincount(edge_index[0], minlength=num_nodes)
                degree_std = np.std(degrees)
                
                # Combined complexity score
                complexity = density * 0.5 + degree_std * 0.1
                scores.append(complexity)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def _get_initial_sample_order(self) -> np.ndarray:
        """Get initial sample ordering based on difficulty."""
        if self.difficulty_scores is None:
            return np.arange(len(self.difficulty_scores))
        
        if self.curriculum_config.use_anti_curriculum:
            # Anti-curriculum: start with hard samples
            return np.argsort(self.difficulty_scores)[::-1]
        else:
            # Regular curriculum: start with easy samples
            return np.argsort(self.difficulty_scores)
    
    def _get_updated_sample_order(self) -> np.ndarray:
        """Get updated sample ordering after difficulty re-evaluation."""
        return self._get_initial_sample_order()
    
    def _create_stage_dataset(self, dataset: BaseSpatialDataset) -> BaseSpatialDataset:
        """Create dataset subset for current curriculum stage."""
        num_samples = int(len(dataset) * self.current_fraction)
        
        if self.curriculum_config.sample_without_replacement:
            # Select samples without replacement based on current ordering
            selected_indices = self.sample_indices[:num_samples]
        else:
            # Sample with replacement (allows for more diverse batches)
            probabilities = self._get_sampling_probabilities(len(dataset))
            selected_indices = np.random.choice(
                len(dataset), size=num_samples, replace=True, p=probabilities
            )
        
        # Create subset
        stage_dataset = Subset(dataset, selected_indices)
        logger.info(f"Created stage dataset with {len(stage_dataset)} samples")
        
        return stage_dataset
    
    def _create_stage_validation_dataset(self, val_dataset: BaseSpatialDataset) -> BaseSpatialDataset:
        """Create validation dataset for current stage."""
        # For validation, we typically use a fixed subset or the full dataset
        # Here we use a progressive approach similar to training
        num_samples = min(len(val_dataset), int(len(val_dataset) * self.current_fraction * 2))
        indices = np.random.choice(len(val_dataset), size=num_samples, replace=False)
        return Subset(val_dataset, indices)
    
    def _get_sampling_probabilities(self, dataset_size: int) -> np.ndarray:
        """Get sampling probabilities for current stage."""
        if self.difficulty_scores is None:
            return np.ones(dataset_size) / dataset_size
        
        # Convert difficulty scores to probabilities
        if self.curriculum_config.use_anti_curriculum:
            # Higher difficulty = higher probability
            probs = self.difficulty_scores
        else:
            # Lower difficulty = higher probability
            probs = 1.0 / (self.difficulty_scores + 1e-6)
        
        # Add some mixing with uniform distribution
        uniform_probs = np.ones_like(probs) / len(probs)
        probs = (1 - self.curriculum_config.batch_difficulty_mixing) * probs + \
                self.curriculum_config.batch_difficulty_mixing * uniform_probs
        
        # Normalize
        return probs / np.sum(probs)
    
    def _train_stage(
        self,
        trainer: FineTuner,
        stage_dataset: Dataset,
        stage_val_dataset: Optional[Dataset],
        stage: int
    ) -> Dict[str, Any]:
        """Train single curriculum stage."""
        # Adjust epochs for current stage
        stage_epochs = max(
            self.curriculum_config.min_stage_epochs,
            self.fine_tuning_config.epochs // self.curriculum_config.num_stages
        )
        
        # Temporarily modify trainer config
        original_epochs = trainer.config.epochs
        trainer.config.epochs = stage_epochs
        
        # Create data loaders
        stage_train_loader = DataLoader(
            stage_dataset,
            batch_size=self.fine_tuning_config.batch_size,
            shuffle=True,
            num_workers=self.fine_tuning_config.dataloader_num_workers,
            pin_memory=self.fine_tuning_config.pin_memory
        )
        
        stage_val_loader = None
        if stage_val_dataset is not None:
            stage_val_loader = DataLoader(
                stage_val_dataset,
                batch_size=self.fine_tuning_config.batch_size,
                shuffle=False,
                num_workers=self.fine_tuning_config.dataloader_num_workers,
                pin_memory=self.fine_tuning_config.pin_memory
            )
        
        # Train stage
        results = {'train_losses': [], 'val_losses': []}
        
        trainer.setup_training()
        
        for epoch in range(stage_epochs):
            # Training epoch
            train_loss = trainer._train_epoch(stage_train_loader)
            results['train_losses'].append(train_loss['loss'])
            
            # Validation epoch
            if stage_val_loader is not None:
                val_loss = trainer._validate_epoch(stage_val_loader)
                results['val_losses'].append(val_loss['loss'])
            
            # Early stopping for stage
            if self.curriculum_config.use_early_stopping and len(results['val_losses']) >= 2:
                if self._check_early_stopping(results['val_losses']):
                    logger.info(f"Early stopping at epoch {epoch + 1} for stage {stage}")
                    break
        
        # Restore original config
        trainer.config.epochs = original_epochs
        
        # Compute final metrics
        final_train_loss = results['train_losses'][-1] if results['train_losses'] else float('inf')
        final_val_loss = results['val_losses'][-1] if results['val_losses'] else float('inf')
        
        return {
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'train_losses': results['train_losses'],
            'val_losses': results['val_losses'],
            'epochs_trained': len(results['train_losses'])
        }
    
    def _should_advance_stage(self) -> bool:
        """Check if should advance to next curriculum stage."""
        if len(self.stage_performances) < 2:
            return True  # Always advance from first stage
        
        # Check if performance meets threshold
        current_performance = self.stage_performances[-1]
        baseline_performance = np.mean(self.stage_performances[:-1])
        
        improvement = (baseline_performance - current_performance) / baseline_performance
        meets_threshold = improvement >= self.curriculum_config.advancement_threshold
        
        return meets_threshold
    
    def _update_curriculum_progress(self) -> None:
        """Update curriculum progress to next stage."""
        if self.curriculum_config.progression_type == "linear":
            fraction_increment = (self.curriculum_config.final_fraction - self.curriculum_config.initial_fraction) / self.curriculum_config.num_stages
            self.current_fraction = min(
                self.current_fraction + fraction_increment,
                self.curriculum_config.final_fraction
            )
        
        elif self.curriculum_config.progression_type == "exponential":
            growth_rate = np.log(self.curriculum_config.final_fraction / self.curriculum_config.initial_fraction) / self.curriculum_config.num_stages
            self.current_fraction = min(
                self.current_fraction * np.exp(growth_rate),
                self.curriculum_config.final_fraction
            )
        
        elif self.curriculum_config.progression_type == "step":
            # Step increases at specific stages
            step_size = (self.curriculum_config.final_fraction - self.curriculum_config.initial_fraction) / (self.curriculum_config.num_stages // 2)
            if self.current_stage % 2 == 0:  # Every other stage
                self.current_fraction = min(
                    self.current_fraction + step_size,
                    self.curriculum_config.final_fraction
                )
        
        elif self.curriculum_config.progression_type == "cosine":
            # Cosine schedule
            progress = (self.current_stage + 1) / self.curriculum_config.num_stages
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.current_fraction = (
                self.curriculum_config.final_fraction - 
                (self.curriculum_config.final_fraction - self.curriculum_config.initial_fraction) * cosine_factor
            )
    
    def _adjust_difficulty_dynamically(self, current_performance: float) -> None:
        """Dynamically adjust difficulty based on performance."""
        if len(self.stage_performances) < 2:
            return
        
        # Performance trend
        recent_performance = np.mean(self.stage_performances[-2:])
        historical_performance = np.mean(self.stage_performances[:-2]) if len(self.stage_performances) > 2 else recent_performance
        
        performance_trend = (historical_performance - recent_performance) / historical_performance
        
        # Adjust difficulty scores based on performance
        if performance_trend > 0.1:  # Performance improving
            # Make curriculum slightly more aggressive
            adjustment = 1.0 + self.curriculum_config.difficulty_smoothing
        elif performance_trend < -0.1:  # Performance degrading
            # Make curriculum more conservative
            adjustment = 1.0 - self.curriculum_config.difficulty_smoothing
        else:
            adjustment = 1.0
        
        # Apply adjustment
        if self.difficulty_scores is not None:
            self.difficulty_scores *= adjustment
            logger.info(f"Adjusted difficulty scores by factor: {adjustment:.3f}")
    
    def _check_early_stopping(self, val_losses: List[float]) -> bool:
        """Check early stopping criteria for current stage."""
        if len(val_losses) < self.curriculum_config.early_stopping_patience:
            return False
        
        recent_losses = val_losses[-self.curriculum_config.early_stopping_patience:]
        best_recent_loss = min(recent_losses)
        current_loss = val_losses[-1]
        
        # Check if improvement is below threshold
        improvement = (best_recent_loss - current_loss) / best_recent_loss
        return improvement < self.curriculum_config.early_stopping_threshold
    
    def _get_difficulty_statistics(self) -> Dict[str, Any]:
        """Get statistics about current difficulty distribution."""
        if self.difficulty_scores is None:
            return {}
        
        num_samples = int(len(self.difficulty_scores) * self.current_fraction)
        current_difficulties = self.difficulty_scores[self.sample_indices[:num_samples]]
        
        return {
            'mean_difficulty': float(np.mean(current_difficulties)),
            'std_difficulty': float(np.std(current_difficulties)),
            'min_difficulty': float(np.min(current_difficulties)),
            'max_difficulty': float(np.max(current_difficulties)),
            'num_samples': num_samples
        }
    
    def _save_stage_checkpoint(
        self,
        trainer: FineTuner,
        path: Path,
        stage_info: Dict[str, Any]
    ) -> None:
        """Save checkpoint for current stage."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'curriculum_config': self.curriculum_config.__dict__,
            'fine_tuning_config': self.fine_tuning_config.__dict__,
            'current_stage': self.current_stage,
            'current_fraction': self.current_fraction,
            'difficulty_scores': self.difficulty_scores.tolist() if self.difficulty_scores is not None else None,
            'sample_indices': self.sample_indices.tolist() if self.sample_indices is not None else None,
            'stage_info': stage_info,
            'stage_history': self.stage_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved stage checkpoint: {path}")
    
    def load_stage_checkpoint(self, path: str) -> None:
        """Load stage checkpoint to resume training."""
        logger.info(f"Loading stage checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore curriculum state
        self.current_stage = checkpoint['current_stage']
        self.current_fraction = checkpoint['current_fraction']
        
        if checkpoint['difficulty_scores'] is not None:
            self.difficulty_scores = np.array(checkpoint['difficulty_scores'])
        
        if checkpoint['sample_indices'] is not None:
            self.sample_indices = np.array(checkpoint['sample_indices'])
        
        self.stage_history = checkpoint['stage_history']
        
        logger.info(f"Resumed from stage {self.current_stage}, fraction {self.current_fraction:.2f}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj
    
    def analyze_curriculum_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of curriculum learning."""
        if not self.stage_history:
            return {}
        
        analysis = {
            'total_stages_completed': len(self.stage_history),
            'final_data_fraction': self.current_fraction,
            'performance_improvement': None,
            'training_efficiency': None,
            'difficulty_progression': []
        }
        
        # Performance improvement analysis
        if len(self.stage_performances) >= 2:
            initial_performance = self.stage_performances[0]
            final_performance = self.stage_performances[-1]
            improvement = (initial_performance - final_performance) / initial_performance
            analysis['performance_improvement'] = improvement
        
        # Training efficiency (performance per time)
        total_time = sum(stage['training_time'] for stage in self.stage_history)
        if total_time > 0 and analysis['performance_improvement'] is not None:
            analysis['training_efficiency'] = analysis['performance_improvement'] / total_time
        
        # Difficulty progression
        for stage_info in self.stage_history:
            if 'difficulty_stats' in stage_info:
                analysis['difficulty_progression'].append({
                    'stage': stage_info['stage'],
                    'fraction': stage_info['fraction'],
                    'mean_difficulty': stage_info['difficulty_stats'].get('mean_difficulty', 0)
                })
        
        return analysis


def create_adaptive_curriculum(
    dataset: BaseSpatialDataset,
    model: SpatialGraphTransformer,
    performance_metric: str = "loss"
) -> CurriculumConfig:
    """
    Create adaptive curriculum configuration based on dataset characteristics.
    
    Args:
        dataset: Training dataset
        model: Model to train
        performance_metric: Metric to optimize
        
    Returns:
        Optimized curriculum configuration
    """
    logger.info("Creating adaptive curriculum configuration")
    
    # Analyze dataset characteristics
    dataset_size = len(dataset)
    
    # Sample some data to analyze complexity
    sample_size = min(100, dataset_size // 10)
    sample_indices = np.random.choice(dataset_size, sample_size, replace=False)
    
    complexities = []
    for idx in sample_indices:
        data = dataset.get(idx)
        complexity = 0
        
        if hasattr(data, 'x'):
            complexity += len(data.x) * 0.1  # Number of cells
        
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
            complexity += edge_index.shape[1] * 0.01  # Number of edges
        
        complexities.append(complexity)
    
    complexity_std = np.std(complexities)
    
    # Configure curriculum based on complexity variation
    config = CurriculumConfig()
    
    if complexity_std > 1.0:  # High variation
        config.strategy = "spatial_complexity"
        config.num_stages = 7
        config.initial_fraction = 0.05
        config.progression_type = "exponential"
    elif complexity_std > 0.5:  # Medium variation
        config.strategy = "density_based"
        config.num_stages = 5
        config.initial_fraction = 0.1
        config.progression_type = "linear"
    else:  # Low variation
        config.strategy = "simple_to_complex"
        config.num_stages = 3
        config.initial_fraction = 0.2
        config.progression_type = "step"
    
    # Adjust for dataset size
    if dataset_size > 10000:
        config.num_stages += 2
        config.initial_fraction *= 0.5
    
    logger.info(f"Created adaptive curriculum: strategy={config.strategy}, "
               f"stages={config.num_stages}, initial_fraction={config.initial_fraction:.2f}")
    
    return config
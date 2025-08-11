"""
Tests for training modules in Spatial-Omics GFM.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from pathlib import Path


class TestDistributedTraining(unittest.TestCase):
    """Test distributed training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=nn.Module)
        self.mock_dataset = Mock()
        self.mock_dataset.__len__ = Mock(return_value=1000)
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from spatial_omics_gfm.training.distributed_training import DistributedTrainer
        
        trainer = DistributedTrainer(
            model=self.mock_model,
            dataset=self.mock_dataset,
            batch_size=32
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.batch_size, 32)
    
    @patch('torch.distributed.init_process_group')
    def test_distributed_setup(self, mock_init):
        """Test distributed training setup."""
        from spatial_omics_gfm.training.distributed_training import DistributedTrainer
        
        trainer = DistributedTrainer(
            model=self.mock_model,
            dataset=self.mock_dataset
        )
        
        # Mock distributed environment
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'RANK': '0'}):
            trainer.setup_distributed()
            mock_init.assert_called_once()
    
    def test_fine_tuning_config(self):
        """Test fine-tuning configuration."""
        from spatial_omics_gfm.training.fine_tuning import FineTuner
        
        fine_tuner = FineTuner(
            base_model=self.mock_model,
            task="cell_typing"
        )
        
        self.assertIsNotNone(fine_tuner)
        self.assertEqual(fine_tuner.task, "cell_typing")


class TestFineTuning(unittest.TestCase):
    """Test fine-tuning functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=nn.Module)
        self.mock_dataset = Mock()
        
    def test_lora_adaptation(self):
        """Test LoRA fine-tuning."""
        from spatial_omics_gfm.training.fine_tuning import FineTuner
        
        fine_tuner = FineTuner(
            base_model=self.mock_model,
            task="interaction_prediction"
        )
        
        # Test LoRA configuration
        lora_config = fine_tuner.get_lora_config(rank=16, alpha=32)
        self.assertEqual(lora_config['rank'], 16)
        self.assertEqual(lora_config['alpha'], 32)
    
    def test_progressive_fine_tuning(self):
        """Test progressive fine-tuning strategy."""
        from spatial_omics_gfm.training.fine_tuning import FineTuner
        
        fine_tuner = FineTuner(
            base_model=self.mock_model,
            task="pathway_analysis"
        )
        
        datasets = [Mock(), Mock(), Mock()]
        epochs_per_stage = [5, 3, 2]
        learning_rates = [1e-4, 5e-5, 1e-5]
        
        # Mock the progressive fine-tuning
        with patch.object(fine_tuner, 'train_stage') as mock_train:
            fine_tuner.progressive_finetune(datasets, epochs_per_stage, learning_rates)
            self.assertEqual(mock_train.call_count, 3)


class TestCurriculumLearning(unittest.TestCase):
    """Test curriculum learning functionality."""
    
    def test_difficulty_scheduler(self):
        """Test difficulty scheduling in curriculum learning."""
        from spatial_omics_gfm.training.curriculum_learning import CurriculumScheduler
        
        scheduler = CurriculumScheduler(
            initial_difficulty=0.3,
            max_difficulty=1.0,
            warmup_steps=1000
        )
        
        # Test difficulty progression
        difficulty_0 = scheduler.get_difficulty(step=0)
        difficulty_500 = scheduler.get_difficulty(step=500)
        difficulty_1000 = scheduler.get_difficulty(step=1000)
        
        self.assertLessEqual(difficulty_0, difficulty_500)
        self.assertLessEqual(difficulty_500, difficulty_1000)
    
    def test_sample_selection(self):
        """Test curriculum-based sample selection."""
        from spatial_omics_gfm.training.curriculum_learning import CurriculumDataLoader
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        loader = CurriculumDataLoader(
            dataset=mock_dataset,
            batch_size=16,
            curriculum_strategy="easy_to_hard"
        )
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.batch_size, 16)


class TestContrastiveLearning(unittest.TestCase):
    """Test contrastive learning functionality."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        from spatial_omics_gfm.training.contrastive_learning import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(temperature=0.1)
        
        # Mock embeddings
        embeddings = torch.randn(32, 128)  # batch_size x embedding_dim
        labels = torch.randint(0, 10, (32,))  # batch_size
        
        loss = loss_fn(embeddings, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_data_augmentation(self):
        """Test data augmentation for contrastive learning."""
        from spatial_omics_gfm.training.contrastive_learning import SpatialAugmentation
        
        augmenter = SpatialAugmentation(
            rotation_range=30,
            noise_std=0.1,
            dropout_rate=0.1
        )
        
        # Mock spatial data
        mock_data = Mock()
        mock_data.X = torch.randn(100, 50)
        mock_data.obsm = {'spatial': torch.randn(100, 2)}
        
        augmented = augmenter.augment(mock_data)
        
        self.assertIsNotNone(augmented)


class TestTrainingMetrics(unittest.TestCase):
    """Test training metrics and monitoring."""
    
    def test_metric_collection(self):
        """Test training metrics collection."""
        from spatial_omics_gfm.training.metrics import TrainingMetrics
        
        metrics = TrainingMetrics()
        
        # Record some metrics
        metrics.record('loss', 0.5, step=100)
        metrics.record('accuracy', 0.85, step=100)
        metrics.record('learning_rate', 1e-4, step=100)
        
        # Check metrics were recorded
        self.assertIn('loss', metrics.get_metric_names())
        self.assertIn('accuracy', metrics.get_metric_names())
        self.assertIn('learning_rate', metrics.get_metric_names())
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        from spatial_omics_gfm.training.early_stopping import EarlyStopping
        
        early_stop = EarlyStopping(patience=3, min_delta=0.01)
        
        # Simulate training progress
        self.assertFalse(early_stop.should_stop(1.0))  # First epoch
        self.assertFalse(early_stop.should_stop(0.9))  # Improvement
        self.assertFalse(early_stop.should_stop(0.89)) # Small improvement
        self.assertFalse(early_stop.should_stop(0.88)) # Within patience
        self.assertTrue(early_stop.should_stop(0.87))  # Should stop


class TestModelCheckpointing(unittest.TestCase):
    """Test model checkpointing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=nn.Module)
        self.mock_optimizer = Mock()
        
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        from spatial_omics_gfm.training.checkpointing import ModelCheckpoint
        
        checkpoint_manager = ModelCheckpoint(
            checkpoint_dir="/tmp/test_checkpoints",
            save_top_k=3
        )
        
        # Mock checkpoint data
        checkpoint_data = {
            'model_state_dict': {'param1': torch.randn(10, 10)},
            'optimizer_state_dict': {'param_groups': []},
            'epoch': 5,
            'step': 1000,
            'loss': 0.5
        }
        
        # Test save
        checkpoint_path = checkpoint_manager.save_checkpoint(
            checkpoint_data, 
            metric_value=0.5,
            metric_name='val_loss'
        )
        
        self.assertIsInstance(checkpoint_path, Path)
    
    def test_best_checkpoint_tracking(self):
        """Test best checkpoint tracking."""
        from spatial_omics_gfm.training.checkpointing import ModelCheckpoint
        
        checkpoint_manager = ModelCheckpoint(
            checkpoint_dir="/tmp/test_checkpoints",
            monitor='val_accuracy',
            mode='max'
        )
        
        # Test best checkpoint logic
        self.assertTrue(checkpoint_manager.is_best_checkpoint(0.9, 'val_accuracy'))
        self.assertFalse(checkpoint_manager.is_best_checkpoint(0.8, 'val_accuracy'))
        self.assertTrue(checkpoint_manager.is_best_checkpoint(0.95, 'val_accuracy'))


if __name__ == '__main__':
    unittest.main()
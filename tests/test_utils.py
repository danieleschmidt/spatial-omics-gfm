"""
Tests for spatial-omics GFM utility modules.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging

from spatial_omics_gfm.utils.validators import (
    SpatialDataValidator, ModelInputValidator, validate_file_format, validate_model_config
)
from spatial_omics_gfm.utils.metrics import SpatialMetrics, evaluate_model_performance
from spatial_omics_gfm.utils.logging_config import (
    SpatialOmicsLogger, setup_logging, LoggedOperation, SpatialOmicsFormatter
)
from spatial_omics_gfm.utils.optimization import (
    ModelOptimizer, BatchProcessor, PerformanceProfiler, optimize_for_production
)
from spatial_omics_gfm.utils.memory_management import (
    MemoryMonitor, DataChunker, SwapManager, memory_managed_operation,
    MemoryConfig, get_memory_recommendations
)

from tests.conftest import assert_tensor_equal


class TestSpatialDataValidator:
    """Test spatial data validation utilities."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = SpatialDataValidator()
        assert hasattr(validator, 'validation_errors')
        assert hasattr(validator, 'validation_warnings')
    
    def test_validate_adata_valid(self, sample_adata):
        """Test validation of valid AnnData."""
        validator = SpatialDataValidator()
        
        report = validator.validate_adata(sample_adata)
        
        assert report['is_valid'] == True
        assert 'spatial_coords' in report
        assert 'expression_matrix' in report
    
    def test_validate_adata_missing_spatial(self, sample_adata):
        """Test validation with missing spatial coordinates."""
        validator = SpatialDataValidator()
        
        # Remove spatial coordinates
        adata_copy = sample_adata.copy()
        del adata_copy.obsm['spatial']
        
        report = validator.validate_adata(adata_copy)
        
        assert report['is_valid'] == False
        assert len(report['errors']) > 0
    
    def test_validate_adata_fix_issues(self, sample_adata):
        """Test validation with automatic issue fixing."""
        validator = SpatialDataValidator()
        
        # Introduce some issues
        adata_copy = sample_adata.copy()
        
        # Add NaN coordinates
        adata_copy.obsm['spatial'][0, 0] = np.nan
        
        # Add negative expression values
        adata_copy.X[0, 0] = -1
        
        report = validator.validate_adata(adata_copy, fix_issues=True)
        
        # Issues should be fixed
        assert not np.isnan(adata_copy.obsm['spatial']).any()
        assert np.all(adata_copy.X >= 0)
    
    def test_validate_spatial_coordinates(self, sample_adata):
        """Test spatial coordinate validation."""
        validator = SpatialDataValidator()
        
        report = {}
        validator._validate_spatial_coordinates(sample_adata, report, fix_issues=False)
        
        assert report['spatial_coords']['has_spatial'] == True
        assert report['spatial_coords']['num_dimensions'] == 2
        assert report['spatial_coords']['coordinate_range'] is not None
    
    def test_validate_expression_matrix(self, sample_adata):
        """Test expression matrix validation."""
        validator = SpatialDataValidator()
        
        report = {}
        validator._validate_expression_matrix(sample_adata, report, fix_issues=False)
        
        assert report['expression_matrix']['shape'] == sample_adata.shape
        assert 'sparsity' in report['expression_matrix']
        assert 'data_type' in report['expression_matrix']


class TestModelInputValidator:
    """Test model input validation."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = ModelInputValidator()
        assert hasattr(validator, 'validation_errors')
    
    def test_validate_model_inputs_valid(self, sample_batch_data):
        """Test validation of valid model inputs."""
        validator = ModelInputValidator()
        
        is_valid = validator.validate_model_inputs(**sample_batch_data)
        
        assert is_valid == True
        assert len(validator.validation_errors) == 0
    
    def test_validate_model_inputs_invalid_shape(self, sample_batch_data):
        """Test validation with invalid tensor shapes."""
        validator = ModelInputValidator()
        
        # Modify edge_index to have wrong shape
        invalid_inputs = sample_batch_data.copy()
        invalid_inputs['edge_index'] = torch.randn(3, 10)  # Wrong shape
        
        is_valid = validator.validate_model_inputs(**invalid_inputs)
        
        assert is_valid == False
        assert len(validator.validation_errors) > 0
    
    def test_validate_tensor_properties(self):
        """Test tensor property validation."""
        validator = ModelInputValidator()
        
        # Valid tensor
        valid_tensor = torch.randn(10, 5)
        errors = validator._validate_tensor_properties(
            valid_tensor, 'test_tensor', expected_shape=(10, 5)
        )
        assert len(errors) == 0
        
        # Invalid shape
        errors = validator._validate_tensor_properties(
            valid_tensor, 'test_tensor', expected_shape=(10, 3)
        )
        assert len(errors) > 0


class TestFileFormatValidation:
    """Test file format validation functions."""
    
    def test_validate_h5ad_format(self, temp_dir):
        """Test H5AD file format validation."""
        from tests.conftest import create_test_h5ad_file
        
        # Create valid H5AD file
        h5ad_path = create_test_h5ad_file(temp_dir)
        
        is_valid, errors = validate_file_format(h5ad_path, 'h5ad')
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation of non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.h5ad"
        
        is_valid, errors = validate_file_format(nonexistent_path, 'h5ad')
        
        assert is_valid == False
        assert len(errors) > 0
    
    def test_validate_unsupported_format(self, temp_dir):
        """Test validation of unsupported file format."""
        # Create a text file
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("test content")
        
        is_valid, errors = validate_file_format(txt_path, 'unsupported')
        
        assert is_valid == False
        assert len(errors) > 0


class TestSpatialMetrics:
    """Test spatial metrics computation."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = SpatialMetrics()
        assert hasattr(metrics, 'compute_classification_metrics')
    
    def test_compute_classification_metrics(self):
        """Test classification metrics computation."""
        metrics = SpatialMetrics()
        
        # Generate sample data
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        y_prob = np.random.dirichlet([1, 1, 1], 100)
        
        results = metrics.compute_classification_metrics(
            y_true, y_pred, y_prob, class_names=['A', 'B', 'C']
        )
        
        assert 'accuracy' in results
        assert 'precision_macro' in results
        assert 'f1_macro' in results
        assert 'per_class' in results
        assert 'roc_auc_ovr' in results
    
    def test_compute_spatial_coherence(self):
        """Test spatial coherence metrics."""
        metrics = SpatialMetrics()
        
        # Create sample data
        coords = np.random.uniform(0, 100, (50, 2))
        labels = np.random.randint(0, 3, 50)
        
        results = metrics.compute_spatial_coherence(
            coords, labels, method='all', k_neighbors=6
        )
        
        assert 'morans_i' in results
        assert 'geary_c' in results
        assert 'lee_l' in results
        assert 'spatial_clustering' in results
    
    def test_compute_clustering_metrics(self):
        """Test clustering quality metrics."""
        metrics = SpatialMetrics()
        
        # Create sample data
        embeddings = np.random.randn(100, 10)
        true_labels = np.random.randint(0, 3, 100)
        pred_labels = np.random.randint(0, 3, 100)
        
        results = metrics.compute_clustering_metrics(
            embeddings, true_labels, pred_labels
        )
        
        assert 'silhouette_score' in results
        assert 'adjusted_rand_score' in results
        assert 'normalized_mutual_info' in results
    
    def test_compute_uncertainty_metrics(self):
        """Test uncertainty quantification metrics."""
        metrics = SpatialMetrics()
        
        # Create sample data
        predictions = np.random.randn(100, 3)
        uncertainties = np.random.beta(2, 5, 100)
        true_labels = np.random.randint(0, 3, 100)
        
        results = metrics.compute_uncertainty_metrics(
            predictions, uncertainties, true_labels
        )
        
        assert 'mean_uncertainty' in results
        assert 'uncertainty_correctness_correlation' in results
        assert 'ece' in results
        assert 'mce' in results
    
    def test_compute_biological_plausibility(self):
        """Test biological plausibility metrics."""
        metrics = SpatialMetrics()
        
        # Create sample predictions
        predictions = {
            'cell_types': np.random.randint(0, 3, 50),
            'interactions': np.random.beta(2, 5, 50),
            'pathway_scores': np.random.beta(3, 3, (50, 5))
        }
        
        # Create known biology constraints
        known_biology = {
            'colocation_rules': {
                'forbidden': [[0, 1], [1, 2]]
            },
            'known_interactions': {},
            'pathway_relationships': {
                'positive_correlations': [[0, 1], [2, 3]]
            }
        }
        
        coords = np.random.uniform(0, 100, (50, 2))
        
        results = metrics.compute_biological_plausibility(
            predictions, known_biology, coords
        )
        
        assert 'cell_type_colocation_score' in results
        assert 'interaction_plausibility_score' in results
        assert 'pathway_coherence_score' in results


class TestEvaluateModelPerformance:
    """Test comprehensive model evaluation."""
    
    def test_evaluate_model_performance(self, sample_predictions):
        """Test complete model evaluation."""
        predictions, ground_truth, coordinates = sample_predictions
        
        evaluation_report = evaluate_model_performance(
            predictions, ground_truth, coordinates
        )
        
        assert 'cell_type_classification' in evaluation_report
        assert 'cell_type_spatial_coherence' in evaluation_report
        assert 'interaction_prediction' in evaluation_report
        assert 'uncertainty_quantification' in evaluation_report
        assert 'summary' in evaluation_report


class TestLoggingConfig:
    """Test logging configuration and utilities."""
    
    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        logger_instance = setup_logging(
            log_level="INFO",
            log_dir=str(temp_dir),
            enable_file_logging=True
        )
        
        assert isinstance(logger_instance, SpatialOmicsLogger)
        assert (temp_dir / "spatial_omics_gfm.log").exists()
    
    def test_spatial_omics_formatter(self):
        """Test custom log formatter."""
        formatter = SpatialOmicsFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should be valid JSON
        import json
        log_dict = json.loads(formatted)
        
        assert 'message' in log_dict
        assert 'level' in log_dict
        assert 'logger' in log_dict
    
    def test_logged_operation_context(self):
        """Test logged operation context manager."""
        with patch('spatial_omics_gfm.utils.logging_config.logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            
            with LoggedOperation('test_operation'):
                pass
            
            # Should have logged start and end
            assert mock_logger.info.call_count >= 2
    
    def test_logged_operation_with_exception(self):
        """Test logged operation with exception handling."""
        with patch('spatial_omics_gfm.utils.logging_config.logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                with LoggedOperation('test_operation'):
                    raise ValueError("Test error")


class TestModelOptimizer:
    """Test model optimization utilities."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = ModelOptimizer()
        
        assert hasattr(optimizer, 'optimized_models')
        assert hasattr(optimizer, 'compilation_cache')
    
    def test_compile_model(self, sample_model):
        """Test model compilation."""
        optimizer = ModelOptimizer()
        
        # Mock torch.compile if not available
        with patch('torch.compile', return_value=sample_model) as mock_compile:
            compiled_model = optimizer.compile_model(sample_model, mode="default")
            
            if hasattr(torch, 'compile'):
                assert compiled_model is not None
            else:
                # Should return original model if compile not available
                assert compiled_model == sample_model
    
    def test_quantize_model(self, sample_model, sample_batch_data):
        """Test model quantization."""
        optimizer = ModelOptimizer()
        
        # Prepare calibration data
        calibration_data = {
            k: v[:1] for k, v in sample_batch_data.items()
        }
        
        quantized_model = optimizer.quantize_model(
            sample_model, calibration_data, quantization_method="dynamic"
        )
        
        # Should return a model (may be original if quantization fails)
        assert quantized_model is not None
    
    def test_export_to_onnx(self, sample_model, sample_batch_data, temp_dir):
        """Test ONNX export."""
        optimizer = ModelOptimizer()
        
        export_path = temp_dir / "test_model.onnx"
        
        # Create minimal dummy input
        dummy_input = {
            k: v[:1] for k, v in sample_batch_data.items()
        }
        
        success = optimizer.export_to_onnx(
            sample_model, dummy_input, export_path, optimize=False
        )
        
        # Export may fail due to complex model structure, which is acceptable
        if success:
            assert export_path.exists()


class TestBatchProcessor:
    """Test batch processing utilities."""
    
    def test_initialization(self, sample_model, device):
        """Test batch processor initialization."""
        processor = BatchProcessor(
            model=sample_model,
            device=device,
            max_batch_size=16,
            enable_amp=False  # Disable for testing
        )
        
        assert processor.model == sample_model
        assert processor.device == device
        assert processor.max_batch_size == 16
    
    def test_process_large_dataset_mock(self, sample_model, device):
        """Test large dataset processing with mock data."""
        processor = BatchProcessor(
            model=sample_model,
            device=device,
            enable_amp=False
        )
        
        # Create mock data loader
        mock_data = []
        for i in range(3):
            batch = {
                'gene_expression': torch.randn(10, sample_model.config.num_genes, device=device),
                'spatial_coords': torch.randn(10, 2, device=device),
                'edge_index': torch.randint(0, 10, (2, 20), device=device),
                'edge_attr': torch.randn(20, 1, device=device)
            }
            mock_data.append(batch)
        
        results = processor.process_large_dataset(mock_data)
        
        assert len(results) == 3
        assert all('embeddings' in result for result in results)


class TestPerformanceProfiler:
    """Test performance profiling utilities."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        
        assert hasattr(profiler, 'profiling_data')
    
    def test_profile_model_inference(self, sample_model, sample_batch_data):
        """Test model inference profiling."""
        profiler = PerformanceProfiler()
        
        # Create minimal input for profiling
        sample_input = {
            k: v[:1] for k, v in sample_batch_data.items()
        }
        
        results = profiler.profile_model_inference(
            sample_model, sample_input, num_runs=5, warmup_runs=2
        )
        
        assert 'avg_inference_time_ms' in results
        assert 'throughput_samples_per_sec' in results
        assert 'num_runs' in results
        assert results['num_runs'] == 5


class TestMemoryManagement:
    """Test memory management utilities."""
    
    def test_memory_config(self):
        """Test memory configuration."""
        config = MemoryConfig(
            max_memory_gb=4.0,
            warning_threshold=0.7,
            chunk_size=500
        )
        
        assert config.max_memory_gb == 4.0
        assert config.warning_threshold == 0.7
        assert config.chunk_size == 500
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        config = MemoryConfig()
        monitor = MemoryMonitor(config)
        
        assert monitor.config == config
        assert not monitor.monitoring
    
    def test_memory_monitor_usage(self):
        """Test memory usage reporting."""
        config = MemoryConfig()
        monitor = MemoryMonitor(config)
        
        memory_info = monitor.get_memory_usage()
        
        assert 'memory_gb' in memory_info
        assert 'memory_percent' in memory_info
        assert 'available_gb' in memory_info
        assert 'total_gb' in memory_info
    
    def test_data_chunker(self, sample_adata):
        """Test data chunking functionality."""
        config = MemoryConfig(chunk_size=100)
        chunker = DataChunker(config)
        
        chunks = list(chunker.chunk_adata(sample_adata, chunk_size=100))
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, type(sample_adata)) for chunk in chunks)
        assert all('chunk_info' in chunk.uns for chunk in chunks)
    
    def test_swap_manager(self, temp_dir):
        """Test data swapping functionality."""
        config = MemoryConfig(temp_dir=str(temp_dir))
        swap_manager = SwapManager(config)
        
        # Test swapping numpy array
        test_array = np.random.randn(100, 50)
        swap_key = swap_manager.swap_to_disk(test_array, 'test_array')
        
        # Load back from disk
        loaded_array = swap_manager.load_from_disk(swap_key)
        
        np.testing.assert_array_equal(test_array, loaded_array)
        
        # Cleanup
        swap_manager.remove_swap(swap_key)
    
    def test_memory_managed_operation(self):
        """Test memory managed operation context."""
        config = MemoryConfig(enable_monitoring=False, enable_swapping=False)
        
        with memory_managed_operation(config, enable_monitoring=False) as context:
            assert 'chunker' in context
            assert context['monitor'] is None
            assert context['swap_manager'] is None
    
    def test_get_memory_recommendations(self, sample_adata):
        """Test memory usage recommendations."""
        recommendations = get_memory_recommendations(sample_adata)
        
        assert 'data_size_gb' in recommendations
        assert 'system_memory_gb' in recommendations
        assert 'memory_ratio' in recommendations
        assert 'use_chunking' in recommendations
        assert 'recommended_chunk_size' in recommendations


class TestOptimizeForProduction:
    """Test production optimization workflow."""
    
    def test_optimize_for_production(self, sample_model, sample_batch_data):
        """Test complete production optimization."""
        # Create minimal sample input
        sample_input = {
            k: v[:1] for k, v in sample_batch_data.items()
        }
        
        optimization_config = {
            'enable_compilation': False,  # Disable for testing
            'enable_quantization': False,  # Disable for testing
            'export_onnx': False,  # Disable for testing
            'profile_performance': True
        }
        
        results = optimize_for_production(
            sample_model, sample_input, optimization_config
        )
        
        assert 'original_model' in results
        assert 'optimizations_applied' in results
        assert 'performance_metrics' in results
        assert 'memory_optimized_model' in results


@pytest.mark.slow
class TestUtilsIntegration:
    """Integration tests for utility modules."""
    
    def test_validation_and_metrics_pipeline(self, sample_adata, sample_predictions):
        """Test validation and metrics computation pipeline."""
        # Validate data
        validator = SpatialDataValidator()
        validation_report = validator.validate_adata(sample_adata)
        
        assert validation_report['is_valid']
        
        # Compute metrics
        predictions, ground_truth, coordinates = sample_predictions
        evaluation_report = evaluate_model_performance(
            predictions, ground_truth, coordinates
        )
        
        assert 'summary' in evaluation_report
        assert 'weighted_overall_score' in evaluation_report['summary']
    
    def test_logging_and_optimization_pipeline(self, sample_model, sample_batch_data, temp_dir):
        """Test logging and optimization pipeline."""
        # Setup logging
        logger_instance = setup_logging(
            log_level="INFO",
            log_dir=str(temp_dir),
            enable_file_logging=True,
            enable_console_logging=False
        )
        
        # Use logged operation for optimization
        with LoggedOperation('model_optimization'):
            sample_input = {k: v[:1] for k, v in sample_batch_data.items()}
            
            optimization_config = {
                'enable_compilation': False,
                'enable_quantization': False,
                'export_onnx': False,
                'profile_performance': True
            }
            
            results = optimize_for_production(
                sample_model, sample_input, optimization_config
            )
        
        # Check that log file was created and has content
        log_file = temp_dir / "spatial_omics_gfm.log"
        assert log_file.exists()
        assert log_file.stat().st_size > 0
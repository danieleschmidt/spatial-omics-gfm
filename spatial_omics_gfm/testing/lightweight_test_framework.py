"""
Lightweight Testing Framework for Spatial-Omics GFM

This framework enables comprehensive testing without heavy ML dependencies,
ensuring the codebase remains testable in various deployment environments.
"""

import sys
import traceback
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class LightweightTestRunner:
    """Test runner that works without heavy dependencies."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results: List[TestResult] = []
        
    def run_test(self, test_func: Callable, test_name: str, **kwargs) -> TestResult:
        """Run a single test function and capture results."""
        if self.verbose:
            print(f"Running {test_name}...")
        
        start_time = time.time()
        
        try:
            # Execute test function
            result = test_func(**kwargs)
            execution_time = time.time() - start_time
            
            # Check if test passed
            passed = True
            error_message = None
            metadata = {}
            
            # If result is a dict, extract test info
            if isinstance(result, dict):
                passed = result.get('passed', True)
                error_message = result.get('error_message')
                metadata = result.get('metadata', {})
            
            test_result = TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                error_message=error_message,
                metadata=metadata
            )
            
            if self.verbose:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} ({execution_time:.3f}s)")
                if error_message:
                    print(f"  Error: {error_message}")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"
            
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=error_message
            )
            
            if self.verbose:
                print(f"  ❌ FAIL ({execution_time:.3f}s)")
                print(f"  Error: {error_message}")
                if self.verbose:
                    traceback.print_exc()
            
            return test_result
    
    def run_test_suite(self, test_functions: Dict[str, Callable], **kwargs) -> Dict[str, Any]:
        """Run a complete test suite."""
        if self.verbose:
            print(f"\n🧪 Running test suite with {len(test_functions)} tests...\n")
        
        self.test_results = []
        
        for test_name, test_func in test_functions.items():
            result = self.run_test(test_func, test_name, **kwargs)
            self.test_results.append(result)
        
        # Generate summary
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.test_results)
        
        summary = {
            'total_tests': len(self.test_results),
            'passed': len(passed_tests),
            'failed': len(failed_tests),
            'success_rate': len(passed_tests) / len(self.test_results) * 100,
            'total_execution_time': total_time,
            'failed_tests': [r.test_name for r in failed_tests],
            'test_results': [
                {
                    'name': r.test_name,
                    'passed': r.passed,
                    'execution_time': r.execution_time,
                    'error_message': r.error_message,
                    'metadata': r.metadata
                }
                for r in self.test_results
            ]
        }
        
        if self.verbose:
            print(f"\n📊 Test Summary:")
            print(f"  Total: {summary['total_tests']}")
            print(f"  Passed: {summary['passed']}")
            print(f"  Failed: {summary['failed']}")
            print(f"  Success Rate: {summary['success_rate']:.1f}%")
            print(f"  Total Time: {summary['total_execution_time']:.3f}s")
            
            if failed_tests:
                print(f"\n❌ Failed Tests:")
                for test in failed_tests:
                    print(f"  - {test.test_name}: {test.error_message}")
        
        return summary


# Core functionality tests
def test_core_data_structures():
    """Test basic data structures without ML dependencies."""
    try:
        from spatial_omics_gfm.core.simple_example import SimpleSpatialData, create_demo_data
        
        # Test data creation
        data = create_demo_data(n_cells=100, n_genes=50)
        
        # Validate data structure
        assert data.n_cells == 100, f"Expected 100 cells, got {data.n_cells}"
        assert data.n_genes == 50, f"Expected 50 genes, got {data.n_genes}"
        assert data.expression_matrix.shape == (100, 50), f"Invalid matrix shape: {data.expression_matrix.shape}"
        assert data.coordinates.shape == (100, 2), f"Invalid coordinates shape: {data.coordinates.shape}"
        
        # Test normalization
        original_sum = np.sum(data.expression_matrix)
        data.normalize_expression()
        normalized_sum = np.sum(data.expression_matrix)
        
        # Should have changed after normalization
        assert abs(original_sum - normalized_sum) > 1, "Normalization didn't change data"
        
        # Test spatial neighbors
        neighbors = data.find_spatial_neighbors(k=5)
        assert len(neighbors) == 100, f"Expected 100 neighbor lists, got {len(neighbors)}"
        assert all(len(n) == 5 for n in neighbors.values()), "All cells should have 5 neighbors"
        
        return {'passed': True, 'metadata': {'cells_tested': 100, 'genes_tested': 50}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_cell_type_prediction():
    """Test cell type prediction functionality."""
    try:
        from spatial_omics_gfm.core.simple_example import (
            SimpleCellTypePredictor, create_demo_data
        )
        
        # Create test data
        data = create_demo_data(n_cells=50, n_genes=100)
        data.normalize_expression()
        
        # Test prediction
        predictor = SimpleCellTypePredictor()
        predictions = predictor.predict_cell_types(data)
        
        # Validate predictions
        expected_cell_types = ['T_cell', 'B_cell', 'Macrophage', 'Fibroblast', 'Epithelial']
        assert set(predictions.keys()) == set(expected_cell_types), "Missing cell type predictions"
        
        for cell_type, scores in predictions.items():
            assert len(scores) == 50, f"Expected 50 scores for {cell_type}, got {len(scores)}"
            assert all(0 <= score <= 1 for score in scores), f"Scores for {cell_type} not in [0,1] range"
        
        return {'passed': True, 'metadata': {'cell_types_tested': len(expected_cell_types)}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_interaction_prediction():
    """Test cell-cell interaction prediction."""
    try:
        from spatial_omics_gfm.core.simple_example import (
            SimpleInteractionPredictor, create_demo_data
        )
        
        # Create test data
        data = create_demo_data(n_cells=50, n_genes=100)
        data.normalize_expression()
        
        # Test interaction prediction
        predictor = SimpleInteractionPredictor()
        interactions = predictor.predict_interactions(data, max_distance=100.0)
        
        # Validate interactions
        assert isinstance(interactions, list), "Interactions should be a list"
        
        if interactions:  # Only test if interactions found
            interaction = interactions[0]
            required_keys = ['sender_cell', 'receiver_cell', 'ligand', 'receptor', 'interaction_score', 'distance']
            assert all(key in interaction for key in required_keys), "Missing required keys in interaction"
            
            assert 0 <= interaction['sender_cell'] < 50, "Invalid sender cell ID"
            assert 0 <= interaction['receiver_cell'] < 50, "Invalid receiver cell ID"
            assert interaction['interaction_score'] > 0, "Interaction score should be positive"
            assert interaction['distance'] <= 100.0, "Distance should be <= max_distance"
        
        return {
            'passed': True, 
            'metadata': {
                'interactions_found': len(interactions),
                'has_interactions': len(interactions) > 0
            }
        }
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_configuration_system():
    """Test configuration management system."""
    try:
        from spatial_omics_gfm.utils.config_manager import ConfigManager, create_default_config
        
        # Test default config creation
        default_config = create_default_config()
        assert hasattr(default_config, 'model'), "Config should have model attribute"
        assert hasattr(default_config, 'data'), "Config should have data attribute"
        assert hasattr(default_config, 'training'), "Config should have training attribute"
        
        # Test config manager
        config_manager = ConfigManager()
        config_manager.load_config()
        
        # Test config access
        model_config = config_manager.get_model_config()
        assert isinstance(model_config, dict), "Model config should be a dict"
        
        return {'passed': True, 'metadata': {'config_sections': 3}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_security_features():
    """Test security and validation features."""
    try:
        from spatial_omics_gfm.utils.simple_security import (
            sanitize_user_input, validate_file_path, is_safe_filename
        )
        
        # Test input sanitization
        clean_input = sanitize_user_input("normal_input")
        assert clean_input == "normal_input", "Normal input should pass through"
        
        # Test dangerous input with metadata context (triggers sanitization)
        dangerous_input = "import os; os.system('rm -rf /')"
        clean_dangerous = sanitize_user_input(dangerous_input, "metadata")
        assert clean_dangerous != dangerous_input, "Dangerous input should be sanitized"
        
        # Test file path validation (use relative path)
        safe_path = "safe/path/file.h5ad" 
        assert validate_file_path(safe_path), "Safe path should be valid"
        
        # Test filename validation
        assert is_safe_filename("normal_file.txt"), "Normal filename should be safe"
        assert not is_safe_filename("../../../etc/passwd"), "Path traversal should be unsafe"
        
        return {'passed': True, 'metadata': {'security_checks': 4}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    try:
        from spatial_omics_gfm.utils.memory_management import MemoryMonitor
        
        # Test memory monitoring
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()
        
        assert 'total_memory' in stats, "Memory stats should include total memory"
        assert 'available_memory' in stats, "Memory stats should include available memory"
        assert 'memory_percent' in stats, "Memory stats should include memory percentage"
        
        # Test with memory operation
        @monitor.memory_managed_operation
        def test_operation():
            # Simulate some work
            data = np.random.rand(1000, 1000)
            return np.sum(data)
        
        result = test_operation()
        assert isinstance(result, (int, float)), "Memory managed operation should return result"
        
        return {'passed': True, 'metadata': {'memory_stats_keys': len(stats)}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_data_validation():
    """Test data validation framework."""
    try:
        from spatial_omics_gfm.validation.data_validation import (
            DataValidator, ValidationConfig
        )
        
        # Test validator creation
        config = ValidationConfig()
        validator = DataValidator(config)
        
        # Test with valid data
        valid_data = np.random.rand(100, 50)
        valid_coords = np.random.rand(100, 2)
        
        is_valid, errors = validator.validate_spatial_data_simple(valid_data, valid_coords)
        assert is_valid, f"Valid data should pass validation: {errors}"
        
        # Test with invalid data (mismatched shapes)
        invalid_coords = np.random.rand(50, 2)  # Wrong number of cells
        is_invalid, errors = validator.validate_spatial_data_simple(valid_data, invalid_coords)
        assert not is_invalid, "Invalid data should fail validation"
        assert len(errors) > 0, "Should have validation errors"
        
        return {'passed': True, 'metadata': {'validation_checks': 2}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def test_globalization_features():
    """Test internationalization and globalization."""
    try:
        from spatial_omics_gfm.globalization.simple_i18n import SimpleI18n
        
        # Test i18n system
        i18n = SimpleI18n()
        
        # Test default language (English)
        msg_en = i18n.translate('analysis.complete')
        assert isinstance(msg_en, str), "Should return string message"
        assert len(msg_en) > 0, "Message should not be empty"
        
        # Test language switching
        i18n.set_locale('es')  # Spanish
        msg_es = i18n.translate('analysis.complete')
        assert isinstance(msg_es, str), "Should return string message in Spanish"
        
        # Test fallback
        unknown_msg = i18n.translate('unknown_key')
        assert isinstance(unknown_msg, str), "Should return fallback for unknown key"
        
        return {'passed': True, 'metadata': {'languages_tested': 2}}
        
    except Exception as e:
        return {'passed': False, 'error_message': str(e)}


def run_comprehensive_tests():
    """Run all lightweight tests."""
    # Define test suite
    test_suite = {
        'core_data_structures': test_core_data_structures,
        'cell_type_prediction': test_cell_type_prediction,
        'interaction_prediction': test_interaction_prediction,
        'configuration_system': test_configuration_system,
        'security_features': test_security_features,
        'performance_monitoring': test_performance_monitoring,
        'data_validation': test_data_validation,
        'globalization_features': test_globalization_features,
    }
    
    # Run tests
    runner = LightweightTestRunner(verbose=True)
    results = runner.run_test_suite(test_suite)
    
    # Save results
    output_file = Path("/root/repo/lightweight_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_tests()
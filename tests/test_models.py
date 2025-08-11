"""
Tests for spatial-omics GFM models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from spatial_omics_gfm.models.graph_transformer import (
    SpatialGraphTransformer, TransformerConfig, SpatialPositionEncoding,
    SpatialTransformerLayer, create_model_config
)
from spatial_omics_gfm.models.spatial_attention import SpatialAttention
from spatial_omics_gfm.models.hierarchical_pooling import HierarchicalPooling
from spatial_omics_gfm.models.deployment import ModelServer, DeploymentConfig, EdgeDeployment

from tests.conftest import assert_tensor_equal, cuda_required


class TestTransformerConfig:
    """Test TransformerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TransformerConfig(num_genes=1000)
        
        assert config.num_genes == 1000
        assert config.hidden_dim == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.dropout == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TransformerConfig(
            num_genes=500,
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            dropout=0.2
        )
        
        assert config.num_genes == 500
        assert config.hidden_dim == 512
        assert config.num_layers == 12
        assert config.num_heads == 8
        assert config.dropout == 0.2


class TestSpatialPositionEncoding:
    """Test spatial position encoding module."""
    
    def test_initialization(self):
        """Test module initialization."""
        encoder = SpatialPositionEncoding(
            spatial_dim=2,
            encoding_dim=64,
            max_position=1000
        )
        
        assert encoder.spatial_dim == 2
        assert encoder.encoding_dim == 64
        assert encoder.max_position == 1000
    
    def test_forward_pass(self, device):
        """Test forward pass."""
        encoder = SpatialPositionEncoding(spatial_dim=2, encoding_dim=64)
        encoder.to(device)
        
        coords = torch.randn(100, 2, device=device)
        output = encoder(coords)
        
        assert output.shape == (100, 64)
        assert output.device == device
    
    def test_different_spatial_dims(self, device):
        """Test with different spatial dimensions."""
        for spatial_dim in [2, 3]:
            encoder = SpatialPositionEncoding(spatial_dim=spatial_dim, encoding_dim=64)
            encoder.to(device)
            
            coords = torch.randn(50, spatial_dim, device=device)
            output = encoder(coords)
            
            assert output.shape == (50, 64)


class TestSpatialTransformerLayer:
    """Test spatial transformer layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = SpatialTransformerLayer(
            hidden_dim=256,
            num_heads=8,
            dropout=0.1,
            activation="gelu"
        )
        
        assert layer.hidden_dim == 256
        assert layer.num_heads == 8
    
    def test_forward_pass(self, device):
        """Test forward pass."""
        layer = SpatialTransformerLayer(hidden_dim=64, num_heads=4)
        layer.to(device)
        
        num_nodes = 100
        x = torch.randn(num_nodes, 64, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 200), device=device)
        edge_attr = torch.randn(200, 1, device=device)
        
        output = layer(x, edge_index, edge_attr)
        
        assert output.shape == x.shape
        assert output.device == device
    
    def test_without_edge_features(self, device):
        """Test forward pass without edge features."""
        layer = SpatialTransformerLayer(hidden_dim=64, num_heads=4, use_edge_features=False)
        layer.to(device)
        
        num_nodes = 100
        x = torch.randn(num_nodes, 64, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 200), device=device)
        
        output = layer(x, edge_index)
        
        assert output.shape == x.shape


class TestSpatialGraphTransformer:
    """Test main spatial graph transformer model."""
    
    def test_initialization(self, transformer_config):
        """Test model initialization."""
        model = SpatialGraphTransformer(transformer_config)
        
        assert len(model.layers) == transformer_config.num_layers
        assert hasattr(model, 'gene_encoder')
        assert hasattr(model, 'spatial_encoder')
    
    def test_forward_pass(self, sample_model, sample_batch_data, device):
        """Test complete forward pass."""
        outputs = sample_model(**sample_batch_data)
        
        assert 'embeddings' in outputs
        embeddings = outputs['embeddings']
        
        expected_shape = (sample_batch_data['gene_expression'].shape[0], sample_model.config.hidden_dim)
        assert embeddings.shape == expected_shape
        assert embeddings.device == device
    
    def test_encode_method(self, sample_model, sample_batch_data):
        """Test encode method."""
        embeddings = sample_model.encode(**sample_batch_data)
        
        expected_shape = (sample_batch_data['gene_expression'].shape[0], sample_model.config.hidden_dim)
        assert embeddings.shape == expected_shape
    
    def test_return_embeddings(self, sample_model, sample_batch_data):
        """Test returning intermediate embeddings."""
        outputs = sample_model(**sample_batch_data, return_embeddings=True)
        
        assert 'embeddings' in outputs
        assert 'layer_embeddings' in outputs
        
        layer_embeddings = outputs['layer_embeddings']
        assert len(layer_embeddings) == sample_model.config.num_layers
    
    def test_hierarchical_pooling(self, transformer_config, device):
        """Test hierarchical pooling functionality."""
        config = transformer_config
        config.use_hierarchical_pooling = True
        
        model = SpatialGraphTransformer(config)
        model.to(device)
        
        # Create sample data
        num_nodes = 100
        gene_expression = torch.randn(num_nodes, config.num_genes, device=device)
        spatial_coords = torch.randn(num_nodes, 2, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 200), device=device)
        
        outputs = model(
            gene_expression=gene_expression,
            spatial_coords=spatial_coords,
            edge_index=edge_index
        )
        
        assert 'hierarchical_embeddings' in outputs
    
    def test_parameter_count(self, sample_model):
        """Test parameter counting."""
        param_counts = sample_model.get_parameter_count()
        
        assert 'total' in param_counts
        assert 'gene_encoder' in param_counts
        assert 'spatial_encoder' in param_counts
        assert 'transformer_layers' in param_counts
        
        assert param_counts['total'] > 0
        assert param_counts['total'] == sum(
            count for key, count in param_counts.items() if key != 'total'
        )
    
    def test_gradient_checkpointing(self, sample_model):
        """Test gradient checkpointing functionality."""
        # Enable gradient checkpointing
        sample_model.enable_gradient_checkpointing()
        
        # Verify it's enabled (this is implementation dependent)
        for layer in sample_model.layers:
            if hasattr(layer, 'gradient_checkpointing'):
                assert layer.gradient_checkpointing
        
        # Disable gradient checkpointing
        sample_model.disable_gradient_checkpointing()
        
        for layer in sample_model.layers:
            if hasattr(layer, 'gradient_checkpointing'):
                assert not layer.gradient_checkpointing
    
    @cuda_required
    def test_cuda_compatibility(self, transformer_config):
        """Test CUDA compatibility."""
        model = SpatialGraphTransformer(transformer_config)
        model.cuda()
        
        # Create CUDA tensors
        num_nodes = 50
        gene_expression = torch.randn(num_nodes, transformer_config.num_genes).cuda()
        spatial_coords = torch.randn(num_nodes, 2).cuda()
        edge_index = torch.randint(0, num_nodes, (2, 100)).cuda()
        
        outputs = model(
            gene_expression=gene_expression,
            spatial_coords=spatial_coords,
            edge_index=edge_index
        )
        
        assert outputs['embeddings'].is_cuda
    
    def test_different_input_sizes(self, transformer_config, device):
        """Test model with different input sizes."""
        model = SpatialGraphTransformer(transformer_config)
        model.to(device)
        
        sizes = [10, 50, 100, 200]
        
        for size in sizes:
            gene_expression = torch.randn(size, transformer_config.num_genes, device=device)
            spatial_coords = torch.randn(size, 2, device=device)
            edge_index = torch.randint(0, size, (2, size * 2), device=device)
            
            outputs = model(
                gene_expression=gene_expression,
                spatial_coords=spatial_coords,
                edge_index=edge_index
            )
            
            assert outputs['embeddings'].shape[0] == size


class TestCreateModelConfig:
    """Test model configuration creation utility."""
    
    def test_base_config(self):
        """Test base model configuration."""
        config = create_model_config(num_genes=1000, model_size="base")
        
        assert config.num_genes == 1000
        assert config.hidden_dim == 1024
        assert config.num_layers == 12
        assert config.num_heads == 16
    
    def test_large_config(self):
        """Test large model configuration."""
        config = create_model_config(num_genes=2000, model_size="large")
        
        assert config.num_genes == 2000
        assert config.hidden_dim == 1536
        assert config.num_layers == 24
        assert config.num_heads == 24
    
    def test_xlarge_config(self):
        """Test extra-large model configuration."""
        config = create_model_config(num_genes=3000, model_size="xlarge")
        
        assert config.num_genes == 3000
        assert config.hidden_dim == 2048
        assert config.num_layers == 36
        assert config.num_heads == 32
    
    def test_invalid_size(self):
        """Test invalid model size."""
        with pytest.raises(ValueError, match="Unknown model size"):
            create_model_config(num_genes=1000, model_size="invalid")


class TestModelServer:
    """Test model deployment server."""
    
    @pytest.fixture
    def saved_model_path(self, sample_model, temp_dir):
        """Create a saved model for testing."""
        model_path = temp_dir / "test_model.pth"
        
        # Save model
        torch.save({
            'config': {
                'num_genes': sample_model.config.num_genes,
                'hidden_dim': sample_model.config.hidden_dim,
                'num_layers': sample_model.config.num_layers,
                'num_heads': sample_model.config.num_heads,
                'dropout': sample_model.config.dropout,
                'spatial_encoding_dim': sample_model.config.spatial_encoding_dim,
                'max_neighbors': sample_model.config.max_neighbors,
                'max_distance': sample_model.config.max_distance,
                'activation': sample_model.config.activation,
                'layer_norm_eps': sample_model.config.layer_norm_eps,
                'use_edge_features': sample_model.config.use_edge_features,
                'use_hierarchical_pooling': sample_model.config.use_hierarchical_pooling
            },
            'model_state_dict': sample_model.state_dict()
        }, model_path)
        
        return model_path
    
    def test_server_initialization(self, saved_model_path):
        """Test model server initialization."""
        config = DeploymentConfig(
            batch_size=16,
            use_compilation=False,  # Disable for testing
            use_quantization=False,
            use_onnx=False
        )
        
        server = ModelServer(saved_model_path, config=config, warm_start=False)
        
        assert server.device is not None
        assert 'primary' in server.models
        assert server.config.batch_size == 16
    
    def test_predict(self, saved_model_path, sample_adata):
        """Test model prediction."""
        config = DeploymentConfig(
            use_compilation=False,
            use_quantization=False,
            use_onnx=False
        )
        
        server = ModelServer(saved_model_path, config=config, warm_start=False)
        
        # Make prediction
        response = server.predict(sample_adata, return_embeddings=True)
        
        assert 'embeddings' in response
        assert 'metadata' in response
        assert response['metadata']['num_cells'] == sample_adata.n_obs
    
    def test_predict_with_cache(self, saved_model_path, sample_adata):
        """Test prediction with caching."""
        config = DeploymentConfig(
            cache_size=10,
            use_compilation=False,
            use_quantization=False,
            use_onnx=False
        )
        
        server = ModelServer(saved_model_path, config=config, warm_start=False)
        
        # First prediction
        response1 = server.predict(sample_adata, use_cache=True)
        
        # Second prediction (should use cache)
        response2 = server.predict(sample_adata, use_cache=True)
        
        # Results should be identical
        np.testing.assert_array_equal(response1['embeddings'], response2['embeddings'])
    
    def test_performance_stats(self, saved_model_path, sample_adata):
        """Test performance statistics."""
        config = DeploymentConfig(
            use_compilation=False,
            use_quantization=False,
            use_onnx=False
        )
        
        server = ModelServer(saved_model_path, config=config, warm_start=False)
        
        # Make some predictions
        for _ in range(3):
            server.predict(sample_adata)
        
        stats = server.get_performance_stats()
        
        assert stats['total_requests'] == 3
        assert 'avg_inference_time_sec' in stats
        assert 'available_models' in stats
    
    def test_clear_cache(self, saved_model_path, sample_adata):
        """Test cache clearing."""
        config = DeploymentConfig(
            cache_size=10,
            use_compilation=False,
            use_quantization=False,
            use_onnx=False
        )
        
        server = ModelServer(saved_model_path, config=config, warm_start=False)
        
        # Make cached prediction
        server.predict(sample_adata, use_cache=True)
        assert len(server.cache) > 0
        
        # Clear cache
        server.clear_cache()
        assert not server.cache


class TestEdgeDeployment:
    """Test edge deployment functionality."""
    
    def test_edge_initialization(self, saved_model_path):
        """Test edge deployment initialization."""
        edge_deployment = EdgeDeployment(
            saved_model_path,
            quantization_method="dynamic",
            max_memory_mb=256
        )
        
        assert edge_deployment.device == torch.device("cpu")
        assert edge_deployment.model is not None
    
    def test_lightweight_prediction(self, saved_model_path):
        """Test lightweight prediction for edge devices."""
        edge_deployment = EdgeDeployment(saved_model_path)
        
        # Create lightweight input
        gene_expression = np.random.randn(50, 100).astype(np.float32)
        spatial_coords = np.random.randn(50, 2).astype(np.float32)
        
        results = edge_deployment.predict_lightweight(
            gene_expression, spatial_coords, max_cells=100
        )
        
        assert 'embeddings' in results
        assert isinstance(results['embeddings'], np.ndarray)


@pytest.mark.slow
class TestModelIntegration:
    """Integration tests for complete model workflows."""
    
    def test_end_to_end_training_simulation(self, transformer_config, sample_batch_data, device):
        """Test end-to-end training simulation."""
        model = SpatialGraphTransformer(transformer_config)
        model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Simulate training step
        outputs = model(**sample_batch_data, return_embeddings=True)
        
        # Dummy loss (MSE on embeddings)
        target = torch.randn_like(outputs['embeddings'])
        loss = torch.nn.functional.mse_loss(outputs['embeddings'], target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert loss.item() > 0
    
    def test_model_saving_loading(self, sample_model, temp_dir):
        """Test model saving and loading."""
        save_path = temp_dir / "model_checkpoint.pth"
        
        # Save model
        torch.save({
            'model_state_dict': sample_model.state_dict(),
            'config': {
                'num_genes': sample_model.config.num_genes,
                'hidden_dim': sample_model.config.hidden_dim,
                'num_layers': sample_model.config.num_layers,
                'num_heads': sample_model.config.num_heads,
                'dropout': sample_model.config.dropout,
                'spatial_encoding_dim': sample_model.config.spatial_encoding_dim,
                'max_neighbors': sample_model.config.max_neighbors,
                'max_distance': sample_model.config.max_distance,
                'activation': sample_model.config.activation,
                'layer_norm_eps': sample_model.config.layer_norm_eps,
                'use_edge_features': sample_model.config.use_edge_features,
                'use_hierarchical_pooling': sample_model.config.use_hierarchical_pooling
            }
        }, save_path)
        
        # Load model
        checkpoint = torch.load(save_path, map_location='cpu')
        config = TransformerConfig(**checkpoint['config'])
        loaded_model = SpatialGraphTransformer(config)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Compare parameter counts
        original_params = sample_model.get_parameter_count()
        loaded_params = loaded_model.get_parameter_count()
        
        assert original_params == loaded_params
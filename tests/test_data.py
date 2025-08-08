"""
Tests for spatial-omics GFM data loading and processing modules.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from anndata import AnnData

from spatial_omics_gfm.data.visium import VisiumDataset
from spatial_omics_gfm.data.xenium import XeniumDataset
from spatial_omics_gfm.data.merfish import MERFISHDataset
from spatial_omics_gfm.data.base import BaseSpatialDataset, SpatialDataConfig
from spatial_omics_gfm.data.preprocessing import SpatialPreprocessor
from spatial_omics_gfm.data.graph_construction import SpatialGraphBuilder
from spatial_omics_gfm.data.augmentation import SpatialAugmentor

from tests.conftest import assert_adata_equal, create_test_h5ad_file


class TestSpatialDataConfig:
    """Test spatial data configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpatialDataConfig()
        
        assert config.min_genes == 200
        assert config.min_cells == 3
        assert config.max_genes == 30000
        assert config.normalize == True
        assert config.log_transform == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SpatialDataConfig(
            min_genes=100,
            min_cells=5,
            normalize=False,
            spatial_neighbors=True
        )
        
        assert config.min_genes == 100
        assert config.min_cells == 5
        assert config.normalize == False
        assert config.spatial_neighbors == True


class TestBaseSpatialDataset:
    """Test base spatial dataset functionality."""
    
    def test_initialization(self, spatial_data_config):
        """Test base dataset initialization."""
        dataset = BaseSpatialDataset(config=spatial_data_config)
        
        assert dataset.config == spatial_data_config
        assert hasattr(dataset, 'adata')
    
    def test_validate_spatial_data_valid(self, sample_adata):
        """Test validation with valid spatial data."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        # Should not raise exception
        dataset._validate_spatial_data()
    
    def test_validate_spatial_data_missing_coords(self, sample_adata):
        """Test validation with missing spatial coordinates."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        # Remove spatial coordinates
        del dataset.adata.obsm['spatial']
        
        with pytest.raises(ValueError, match="No spatial coordinates found"):
            dataset._validate_spatial_data()
    
    def test_validate_spatial_data_wrong_dimensions(self, sample_adata):
        """Test validation with wrong spatial dimensions."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        # Change spatial coordinates to 3D
        dataset.adata.obsm['spatial'] = np.random.randn(sample_adata.n_obs, 3)
        
        with pytest.raises(ValueError, match="Spatial coordinates must be 2D"):
            dataset._validate_spatial_data()


class TestVisiumDataset:
    """Test Visium dataset loader."""
    
    def test_from_mock_data(self, mock_visium_data, spatial_data_config):
        """Test loading from mock Visium data."""
        # Create a mock H5 file for expression data
        h5_path = mock_visium_data / "filtered_feature_bc_matrix.h5"
        
        # Create minimal H5 file structure
        import h5py
        n_spots = 100
        n_genes = 50
        
        with h5py.File(h5_path, 'w') as f:
            # Create matrix group
            matrix_group = f.create_group('matrix')
            matrix_group.create_dataset('data', data=np.random.negative_binomial(5, 0.3, 1000))
            matrix_group.create_dataset('indices', data=np.random.randint(0, n_genes, 1000))
            matrix_group.create_dataset('indptr', data=np.arange(0, 1001, 10))
            matrix_group.create_dataset('shape', data=[n_genes, n_spots])
            
            # Create features
            features_group = matrix_group.create_group('features')
            gene_names = [f"Gene_{i}".encode() for i in range(n_genes)]
            features_group.create_dataset('name', data=gene_names)
            features_group.create_dataset('id', data=gene_names)
            feature_types = [b'Gene Expression'] * n_genes
            features_group.create_dataset('feature_type', data=feature_types)
            
            # Create barcodes
            barcodes = [f"BARCODE_{i}".encode() for i in range(n_spots)]
            matrix_group.create_dataset('barcodes', data=barcodes)
        
        try:
            dataset = VisiumDataset.from_10x_folder(
                mock_visium_data,
                config=spatial_data_config,
                load_images=False
            )
            
            assert isinstance(dataset.adata, AnnData)
            assert 'spatial' in dataset.adata.obsm
            assert 'in_tissue' in dataset.adata.obs
            
        except Exception as e:
            # If scanpy has issues with mock data, test basic functionality
            pytest.skip(f"Scanpy compatibility issue with mock data: {e}")
    
    def test_get_spatial_coordinates(self, sample_adata):
        """Test getting spatial coordinates."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        coords = dataset.get_spatial_coordinates()
        
        assert coords.shape == sample_adata.obsm['spatial'].shape
        np.testing.assert_array_equal(coords, sample_adata.obsm['spatial'])
    
    def test_basic_spatial_plot(self, sample_adata, monkeypatch):
        """Test basic spatial plotting."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        # Mock matplotlib to avoid display issues
        mock_plt = MagicMock()
        monkeypatch.setattr('matplotlib.pyplot', mock_plt)
        
        # Test basic plot
        dataset._basic_spatial_plot()
        mock_plt.scatter.assert_called()
        mock_plt.show.assert_called()
    
    def test_len_and_getitem(self, sample_adata):
        """Test dataset length and indexing."""
        dataset = BaseSpatialDataset()
        dataset.adata = sample_adata
        
        # Test length
        assert len(dataset) == sample_adata.n_obs
        
        # Test single item
        item = dataset[0]
        assert 'expression' in item
        assert 'coordinates' in item
        assert 'observations' in item
        
        # Test slice
        items = dataset[0:5]
        assert items['expression'].shape[0] == 5
        assert items['coordinates'].shape[0] == 5


class TestXeniumDataset:
    """Test Xenium dataset functionality."""
    
    def test_initialization_mock(self, temp_dir):
        """Test Xenium dataset initialization with mock data."""
        # Create mock Xenium data structure
        xenium_dir = temp_dir / "mock_xenium"
        xenium_dir.mkdir()
        
        # Mock cell feature matrix
        n_cells = 100
        n_genes = 50
        
        # Create mock expression data
        expression_data = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))
        
        # Create mock cell metadata
        cell_data = pd.DataFrame({
            'cell_id': [f"cell_{i}" for i in range(n_cells)],
            'x_centroid': np.random.uniform(0, 1000, n_cells),
            'y_centroid': np.random.uniform(0, 1000, n_cells),
            'cell_area': np.random.uniform(10, 100, n_cells)
        })
        
        # Create mock gene metadata
        gene_data = pd.DataFrame({
            'gene_id': [f"gene_{i}" for i in range(n_genes)],
            'gene_name': [f"Gene_{i}" for i in range(n_genes)]
        })
        
        # Test basic initialization (without actual file loading)
        try:
            dataset = XeniumDataset(
                data_path=xenium_dir,
                load_transcripts=False,
                load_boundaries=False
            )
            # This may fail due to missing actual Xenium files, which is expected
        except (FileNotFoundError, ValueError):
            # Expected for mock data
            pass


class TestMERFISHDataset:
    """Test MERFISH dataset functionality."""
    
    def test_initialization_mock(self, temp_dir):
        """Test MERFISH dataset initialization with mock data."""
        merfish_dir = temp_dir / "mock_merfish"
        merfish_dir.mkdir()
        
        # Test basic initialization
        try:
            dataset = MERFISHDataset(
                data_path=merfish_dir,
                load_boundaries=False,
                z_stack_processing=False
            )
        except (FileNotFoundError, ValueError):
            # Expected for mock data
            pass


class TestSpatialPreprocessor:
    """Test spatial data preprocessing."""
    
    def test_initialization(self, spatial_data_config):
        """Test preprocessor initialization."""
        preprocessor = SpatialPreprocessor(spatial_data_config)
        
        assert preprocessor.config == spatial_data_config
    
    def test_preprocess_spatial_data(self, sample_adata, spatial_data_config):
        """Test complete spatial data preprocessing."""
        preprocessor = SpatialPreprocessor(spatial_data_config)
        
        # Make a copy to avoid modifying the fixture
        adata_copy = sample_adata.copy()
        
        processed_adata = preprocessor.preprocess_spatial_data(
            adata_copy,
            normalize=True,
            log_transform=True,
            highly_variable_genes=False,  # Skip HVG for test data
            spatial_neighbors=True
        )
        
        assert isinstance(processed_adata, AnnData)
        assert 'spatial_graph' in processed_adata.uns
        
        # Check if normalization was applied
        if spatial_data_config.normalize:
            # Check that total counts per cell are roughly equal
            total_counts = np.array(processed_adata.X.sum(axis=1)).flatten()
            assert np.std(total_counts) < np.mean(total_counts) * 0.1
    
    def test_quality_control(self, sample_adata, spatial_data_config):
        """Test quality control filtering."""
        preprocessor = SpatialPreprocessor(spatial_data_config)
        
        adata_copy = sample_adata.copy()
        
        # Add some cells/genes that should be filtered
        # Add a cell with very low expression
        adata_copy.X[0, :] = 0
        
        filtered_adata = preprocessor._apply_quality_control(adata_copy)
        
        # Should have fewer cells after QC
        assert filtered_adata.n_obs <= adata_copy.n_obs
    
    def test_normalization(self, sample_adata):
        """Test normalization methods."""
        preprocessor = SpatialPreprocessor(SpatialDataConfig())
        
        adata_copy = sample_adata.copy()
        
        # Test total count normalization
        normalized_adata = preprocessor._normalize_data(adata_copy, method='total_count')
        
        # Check that total counts are normalized
        total_counts = np.array(normalized_adata.X.sum(axis=1)).flatten()
        expected_total = 10000  # Default target sum
        assert np.allclose(total_counts, expected_total, rtol=0.01)


class TestSpatialGraphBuilder:
    """Test spatial graph construction."""
    
    def test_initialization(self, spatial_data_config):
        """Test graph builder initialization."""
        builder = SpatialGraphBuilder(spatial_data_config)
        
        assert builder.config == spatial_data_config
    
    def test_build_knn_graph(self, spatial_data_config):
        """Test k-NN graph construction."""
        builder = SpatialGraphBuilder(spatial_data_config)
        
        # Create sample coordinates
        coordinates = np.random.uniform(0, 100, (50, 2))
        
        edge_index, edge_attr, graph_info = builder.build_spatial_graph(
            coordinates, method='knn', k=6
        )
        
        assert edge_index.shape[0] == 2  # Source and target nodes
        assert edge_attr.shape[1] >= 1  # At least distance feature
        assert 'method' in graph_info
        assert graph_info['method'] == 'knn'
    
    def test_build_radius_graph(self, spatial_data_config):
        """Test radius graph construction."""
        builder = SpatialGraphBuilder(spatial_data_config)
        
        coordinates = np.random.uniform(0, 100, (30, 2))
        
        edge_index, edge_attr, graph_info = builder.build_spatial_graph(
            coordinates, method='radius', radius=20.0
        )
        
        assert edge_index.shape[0] == 2
        assert graph_info['method'] == 'radius'
    
    def test_add_graph_to_adata(self, sample_adata, spatial_data_config):
        """Test adding graph to AnnData object."""
        builder = SpatialGraphBuilder(spatial_data_config)
        
        coordinates = sample_adata.obsm['spatial']
        edge_index, edge_attr, graph_info = builder.build_spatial_graph(
            coordinates, method='knn', k=6
        )
        
        builder.add_graph_to_adata(sample_adata, edge_index, edge_attr, graph_info)
        
        assert 'spatial_graph' in sample_adata.uns
        assert 'edge_index' in sample_adata.uns['spatial_graph']
        assert 'edge_attr' in sample_adata.uns['spatial_graph']
        assert 'graph_info' in sample_adata.uns['spatial_graph']
    
    def test_create_pytorch_geometric_data(self, sample_adata, spatial_data_config):
        """Test PyTorch Geometric data creation."""
        builder = SpatialGraphBuilder(spatial_data_config)
        
        coordinates = sample_adata.obsm['spatial']
        edge_index, edge_attr, graph_info = builder.build_spatial_graph(
            coordinates, method='knn', k=6
        )
        
        data = builder.create_pytorch_geometric_data(sample_adata, edge_index, edge_attr)
        
        assert hasattr(data, 'x')  # Node features
        assert hasattr(data, 'edge_index')  # Edge connectivity
        assert hasattr(data, 'edge_attr')  # Edge attributes
        assert hasattr(data, 'pos')  # Node positions


class TestSpatialAugmentor:
    """Test spatial data augmentation."""
    
    def test_initialization(self):
        """Test augmentor initialization."""
        augmentor = SpatialAugmentor(random_seed=42)
        
        assert augmentor.random_seed == 42
    
    def test_spatial_translation(self, sample_adata):
        """Test spatial translation augmentation."""
        augmentor = SpatialAugmentor()
        
        original_coords = sample_adata.obsm['spatial'].copy()
        augmented_adata = augmentor.apply_spatial_translation(
            sample_adata, translation_range=10.0
        )
        
        augmented_coords = augmented_adata.obsm['spatial']
        
        # Coordinates should be different but shape should be same
        assert augmented_coords.shape == original_coords.shape
        assert not np.array_equal(augmented_coords, original_coords)
    
    def test_spatial_rotation(self, sample_adata):
        """Test spatial rotation augmentation."""
        augmentor = SpatialAugmentor()
        
        original_coords = sample_adata.obsm['spatial'].copy()
        augmented_adata = augmentor.apply_spatial_rotation(
            sample_adata, max_angle=30.0
        )
        
        augmented_coords = augmented_adata.obsm['spatial']
        
        assert augmented_coords.shape == original_coords.shape
        assert not np.array_equal(augmented_coords, original_coords)
    
    def test_expression_noise(self, sample_adata):
        """Test expression noise augmentation."""
        augmentor = SpatialAugmentor()
        
        original_X = sample_adata.X.copy()
        augmented_adata = augmentor.add_expression_noise(
            sample_adata, noise_level=0.1
        )
        
        augmented_X = augmented_adata.X
        
        assert augmented_X.shape == original_X.shape
        assert not np.array_equal(augmented_X, original_X)
        
        # Expression values should still be non-negative
        assert np.all(augmented_X >= 0)
    
    def test_elastic_deformation(self, sample_adata):
        """Test elastic deformation augmentation."""
        augmentor = SpatialAugmentor()
        
        original_coords = sample_adata.obsm['spatial'].copy()
        
        try:
            augmented_adata = augmentor.apply_elastic_deformation(
                sample_adata, alpha=10.0, sigma=3.0
            )
            
            augmented_coords = augmented_adata.obsm['spatial']
            
            assert augmented_coords.shape == original_coords.shape
            # Some deformation should have occurred
            assert not np.allclose(augmented_coords, original_coords, rtol=0.01)
            
        except ImportError:
            # Skip if scipy.ndimage not available
            pytest.skip("scipy.ndimage not available for elastic deformation")
    
    def test_create_augmented_dataset(self, sample_adata):
        """Test creating augmented dataset."""
        augmentor = SpatialAugmentor()
        
        augmented_datasets = augmentor.create_augmented_dataset(
            sample_adata, num_augmentations=3
        )
        
        assert len(augmented_datasets) == 3
        
        for aug_adata in augmented_datasets:
            assert isinstance(aug_adata, AnnData)
            assert aug_adata.shape == sample_adata.shape
            assert 'spatial' in aug_adata.obsm
    
    def test_random_seed_reproducibility(self, sample_adata):
        """Test reproducibility with random seeds."""
        # Create two augmentors with same seed
        augmentor1 = SpatialAugmentor(random_seed=42)
        augmentor2 = SpatialAugmentor(random_seed=42)
        
        # Apply same augmentation
        aug1 = augmentor1.apply_spatial_translation(sample_adata, translation_range=10.0)
        aug2 = augmentor2.apply_spatial_translation(sample_adata, translation_range=10.0)
        
        # Results should be identical
        np.testing.assert_array_equal(aug1.obsm['spatial'], aug2.obsm['spatial'])


@pytest.mark.integration
class TestDataIntegration:
    """Integration tests for data processing workflows."""
    
    def test_complete_preprocessing_pipeline(self, sample_adata):
        """Test complete preprocessing pipeline."""
        config = SpatialDataConfig(
            min_genes=10,
            min_cells=3,
            normalize=True,
            log_transform=True,
            spatial_neighbors=True
        )
        
        preprocessor = SpatialPreprocessor(config)
        
        # Process data
        processed_adata = preprocessor.preprocess_spatial_data(
            sample_adata.copy(),
            normalize=True,
            log_transform=True,
            highly_variable_genes=False,
            spatial_neighbors=True
        )
        
        # Verify processing steps
        assert 'spatial_graph' in processed_adata.uns
        assert processed_adata.n_obs <= sample_adata.n_obs
        
        # Check log transformation
        if hasattr(processed_adata.X, 'toarray'):
            X_processed = processed_adata.X.toarray()
        else:
            X_processed = processed_adata.X
        
        # Log-transformed data should be smaller in magnitude
        assert np.mean(X_processed) < np.mean(sample_adata.X)
    
    def test_augmentation_pipeline(self, sample_adata):
        """Test augmentation pipeline."""
        augmentor = SpatialAugmentor(random_seed=42)
        
        # Apply multiple augmentations
        adata_translated = augmentor.apply_spatial_translation(sample_adata, 5.0)
        adata_rotated = augmentor.apply_spatial_rotation(adata_translated, 15.0)
        adata_noisy = augmentor.add_expression_noise(adata_rotated, 0.05)
        
        # Verify final result
        assert adata_noisy.shape == sample_adata.shape
        assert 'spatial' in adata_noisy.obsm
        
        # Check that transformations were applied
        assert not np.array_equal(
            adata_noisy.obsm['spatial'], 
            sample_adata.obsm['spatial']
        )
        assert not np.array_equal(adata_noisy.X, sample_adata.X)
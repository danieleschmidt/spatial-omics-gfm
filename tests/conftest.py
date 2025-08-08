"""
Pytest configuration and fixtures for Spatial-Omics GFM tests.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Tuple

from spatial_omics_gfm.models.graph_transformer import SpatialGraphTransformer, TransformerConfig
from spatial_omics_gfm.data.base import SpatialDataConfig


@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        'num_genes': 100,
        'num_cells': 500,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'spatial_dim': 2
    }


@pytest.fixture(scope="session")
def device():
    """Test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_adata(test_config):
    """Sample AnnData object for testing."""
    n_obs = test_config['num_cells']
    n_vars = test_config['num_genes']
    
    # Create expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create observations
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_obs),
        'sample_id': np.random.choice(['Sample1', 'Sample2'], n_obs),
        'batch': np.random.choice([0, 1], n_obs)
    })
    obs.index = [f"Cell_{i}" for i in range(n_obs)]
    
    # Create variables
    var = pd.DataFrame({
        'gene_name': [f"Gene_{i}" for i in range(n_vars)],
        'highly_variable': np.random.choice([True, False], n_vars)
    })
    var.index = [f"ENSG{i:08d}" for i in range(n_vars)]
    
    # Create spatial coordinates
    spatial_coords = np.random.uniform(0, 1000, size=(n_obs, 2)).astype(np.float32)
    
    # Create AnnData object
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm['spatial'] = spatial_coords
    
    # Add some processed data
    adata.layers['raw'] = X.copy()
    adata.obs['total_counts'] = np.array(X.sum(axis=1))
    adata.var['total_counts'] = np.array(X.sum(axis=0))
    
    return adata


@pytest.fixture
def sample_adata_large():
    """Large sample AnnData for performance testing."""
    n_obs = 5000
    n_vars = 2000
    
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC', 'TypeD'], n_obs),
        'sample_id': np.random.choice(['Sample1', 'Sample2', 'Sample3'], n_obs),
    })
    obs.index = [f"Cell_{i}" for i in range(n_obs)]
    
    var = pd.DataFrame({
        'gene_name': [f"Gene_{i}" for i in range(n_vars)]
    })
    var.index = [f"ENSG{i:08d}" for i in range(n_vars)]
    
    spatial_coords = np.random.uniform(0, 2000, size=(n_obs, 2)).astype(np.float32)
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm['spatial'] = spatial_coords
    
    return adata


@pytest.fixture
def transformer_config(test_config):
    """Sample transformer configuration."""
    return TransformerConfig(
        num_genes=test_config['num_genes'],
        hidden_dim=test_config['hidden_dim'],
        num_layers=test_config['num_layers'],
        num_heads=test_config['num_heads'],
        dropout=0.1,
        max_neighbors=6
    )


@pytest.fixture
def sample_model(transformer_config, device):
    """Sample transformer model for testing."""
    model = SpatialGraphTransformer(transformer_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def spatial_data_config():
    """Spatial data configuration for testing."""
    return SpatialDataConfig(
        min_genes=10,
        min_cells=10,
        max_genes=10000,
        normalize=True,
        log_transform=True,
        scale=False
    )


@pytest.fixture
def sample_spatial_graph():
    """Sample spatial graph data."""
    num_nodes = 100
    
    # Create random coordinates
    coords = np.random.uniform(0, 100, size=(num_nodes, 2))
    
    # Create k-NN graph
    from sklearn.neighbors import NearestNeighbors
    
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Build edge list
    edges = []
    edge_weights = []
    
    for i in range(num_nodes):
        for j in range(1, k+1):  # Skip self (index 0)
            if j < len(indices[i]):
                neighbor = indices[i][j]
                edges.append([i, neighbor])
                edge_weights.append(distances[i][j])
    
    edge_index = torch.tensor(edges).T.long()
    edge_attr = torch.tensor(edge_weights).float().unsqueeze(1)
    
    return {
        'coordinates': coords,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes
    }


@pytest.fixture
def sample_batch_data(sample_adata, device):
    """Sample batch data for model testing."""
    # Extract data
    if hasattr(sample_adata.X, 'toarray'):
        expression = torch.from_numpy(sample_adata.X.toarray()).float()
    else:
        expression = torch.from_numpy(sample_adata.X).float()
    
    spatial_coords = torch.from_numpy(sample_adata.obsm['spatial']).float()
    
    # Create simple graph
    num_nodes = expression.shape[0]
    edges = [[i, (i+1) % num_nodes] for i in range(num_nodes)]
    edge_index = torch.tensor(edges).T.long()
    edge_attr = torch.ones(len(edges), 1).float()
    
    # Move to device
    return {
        'gene_expression': expression.to(device),
        'spatial_coords': spatial_coords.to(device),
        'edge_index': edge_index.to(device),
        'edge_attr': edge_attr.to(device)
    }


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_visium_data(temp_dir):
    """Mock Visium data structure for testing."""
    visium_dir = temp_dir / "mock_visium"
    visium_dir.mkdir()
    
    # Create spatial directory
    spatial_dir = visium_dir / "spatial"
    spatial_dir.mkdir()
    
    # Create mock files
    n_spots = 1000
    
    # Tissue positions
    positions_data = {
        'barcode': [f"BARCODE_{i}" for i in range(n_spots)],
        'in_tissue': np.random.choice([0, 1], n_spots),
        'array_row': np.random.randint(0, 100, n_spots),
        'array_col': np.random.randint(0, 100, n_spots),
        'pxl_col_in_fullres': np.random.randint(0, 2000, n_spots),
        'pxl_row_in_fullres': np.random.randint(0, 2000, n_spots)
    }
    
    positions_df = pd.DataFrame(positions_data)
    positions_df.to_csv(spatial_dir / "tissue_positions_list.csv", index=False)
    
    # Scale factors
    scalefactors = {
        "tissue_hires_scalef": 0.17011238,
        "tissue_lowres_scalef": 0.051033,
        "fiducial_diameter_fullres": 144.5,
        "spot_diameter_fullres": 89.43
    }
    
    import json
    with open(spatial_dir / "scalefactors_json.json", 'w') as f:
        json.dump(scalefactors, f)
    
    # Create dummy images
    import numpy as np
    from PIL import Image
    
    # High-res image
    hires_img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    Image.fromarray(hires_img).save(spatial_dir / "tissue_hires_image.png")
    
    # Low-res image  
    lowres_img = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    Image.fromarray(lowres_img).save(spatial_dir / "tissue_lowres_image.png")
    
    return visium_dir


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing metrics."""
    n_samples = 200
    n_classes = 3
    
    # Generate predictions
    predictions = {
        'cell_types': np.random.randint(0, n_classes, n_samples),
        'cell_type_probabilities': np.random.dirichlet([1]*n_classes, n_samples),
        'interactions': np.random.beta(2, 5, n_samples),
        'pathway_scores': np.random.beta(3, 3, (n_samples, 10)),
        'uncertainties': np.random.beta(2, 8, n_samples)
    }
    
    # Generate ground truth
    ground_truth = {
        'cell_types': np.random.randint(0, n_classes, n_samples),
        'interactions': np.random.choice([0, 1], n_samples),
        'pathway_scores': np.random.beta(3, 3, (n_samples, 10))
    }
    
    # Generate coordinates
    coordinates = np.random.uniform(0, 100, (n_samples, 2))
    
    return predictions, ground_truth, coordinates


@pytest.fixture
def performance_test_data():
    """Data for performance testing."""
    sizes = [100, 500, 1000, 2000]
    datasets = {}
    
    for size in sizes:
        # Create dataset
        X = np.random.negative_binomial(5, 0.3, size=(size, 500)).astype(np.float32)
        coords = np.random.uniform(0, 1000, size=(size, 2)).astype(np.float32)
        
        obs = pd.DataFrame({
            'cell_type': np.random.choice(['A', 'B', 'C'], size)
        })
        obs.index = [f"Cell_{i}" for i in range(size)]
        
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(500)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm['spatial'] = coords
        
        datasets[size] = adata
    
    return datasets


# Parametrized fixtures for testing different configurations
@pytest.fixture(params=[True, False])
def enable_cuda(request):
    """Parametrized fixture for CUDA testing."""
    return request.param and torch.cuda.is_available()


@pytest.fixture(params=['knn', 'radius', 'delaunay'])
def graph_method(request):
    """Parametrized fixture for different graph construction methods."""
    return request.param


@pytest.fixture(params=[1, 4, 8])
def batch_sizes(request):
    """Parametrized fixture for different batch sizes."""
    return request.param


@pytest.fixture(params=['dynamic', 'static'])
def quantization_methods(request):
    """Parametrized fixture for quantization methods."""
    return request.param


# Utility functions for tests
def assert_tensor_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert tensors are equal within tolerance."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
        f"Tensors not equal: max diff = {torch.max(torch.abs(tensor1 - tensor2))}"


def assert_adata_equal(adata1, adata2):
    """Assert AnnData objects are equal."""
    assert adata1.shape == adata2.shape
    np.testing.assert_array_equal(adata1.X, adata2.X)
    pd.testing.assert_frame_equal(adata1.obs, adata2.obs)
    pd.testing.assert_frame_equal(adata1.var, adata2.var)
    
    for key in adata1.obsm.keys():
        if key in adata2.obsm:
            np.testing.assert_array_equal(adata1.obsm[key], adata2.obsm[key])


def create_test_h5ad_file(temp_dir: Path, name: str = "test.h5ad") -> Path:
    """Create a test H5AD file."""
    n_obs, n_vars = 100, 50
    
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm['spatial'] = np.random.uniform(0, 100, (n_obs, 2))
    
    file_path = temp_dir / name
    adata.write_h5ad(file_path)
    
    return file_path


# Skip markers for conditional tests
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

slow_test = pytest.mark.slow
integration_test = pytest.mark.integration
performance_test = pytest.mark.performance
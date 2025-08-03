"""
Base classes for spatial transcriptomics datasets.

This module provides the foundation classes that all platform-specific
datasets inherit from, ensuring consistent interfaces and functionality.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import anndata as ad
from dataclasses import dataclass
import warnings


@dataclass
class SpatialDataConfig:
    """Configuration for spatial data processing."""
    normalize: bool = True
    log_transform: bool = True
    filter_genes: bool = True
    filter_cells: bool = True
    min_cells_per_gene: int = 10
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 5000
    highly_variable_genes: int = 3000
    spatial_graph_k: int = 6
    spatial_graph_radius: Optional[float] = None
    edge_features: List[str] = None
    
    def __post_init__(self):
        if self.edge_features is None:
            self.edge_features = ["distance", "direction"]


class BaseSpatialDataset(Dataset, ABC):
    """
    Abstract base class for spatial transcriptomics datasets.
    
    This class defines the common interface and functionality that all
    platform-specific datasets must implement.
    """
    
    def __init__(
        self,
        config: Optional[SpatialDataConfig] = None,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None
    ):
        super().__init__()
        
        self.config = config or SpatialDataConfig()
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Data containers
        self._adata: Optional[ad.AnnData] = None
        self._spatial_graph: Optional[torch.Tensor] = None
        self._edge_features: Optional[torch.Tensor] = None
        self._processed_data: Optional[List[Data]] = None
        
        # Metadata
        self._num_genes: Optional[int] = None
        self._num_cells: Optional[int] = None
        self._gene_names: Optional[List[str]] = None
        self._cell_ids: Optional[List[str]] = None
    
    @abstractmethod
    def load_data(self, data_path: Union[str, Path], **kwargs) -> ad.AnnData:
        """
        Load platform-specific data.
        
        Args:
            data_path: Path to data files
            **kwargs: Platform-specific arguments
            
        Returns:
            AnnData object with loaded data
        """
        pass
    
    @abstractmethod
    def validate_data(self, adata: ad.AnnData) -> bool:
        """
        Validate that the data is suitable for processing.
        
        Args:
            adata: AnnData object to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def preprocess_data(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply standard preprocessing steps.
        
        Args:
            adata: Raw AnnData object
            
        Returns:
            Preprocessed AnnData object
        """
        from .preprocessing import SpatialPreprocessor
        
        preprocessor = SpatialPreprocessor(self.config)
        return preprocessor.process(adata)
    
    def build_spatial_graph(self, adata: ad.AnnData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build spatial graph from coordinates.
        
        Args:
            adata: AnnData object with spatial coordinates
            
        Returns:
            Tuple of (edge_index, edge_features)
        """
        from .graph_construction import SpatialGraphBuilder
        
        builder = SpatialGraphBuilder(self.config)
        return builder.build_graph(adata)
    
    def setup(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        Setup the dataset by loading and preprocessing data.
        
        Args:
            data_path: Path to data files
            **kwargs: Platform-specific arguments
        """
        # Load raw data
        self._adata = self.load_data(data_path, **kwargs)
        
        # Validate data
        if not self.validate_data(self._adata):
            raise ValueError("Data validation failed")
        
        # Preprocess data
        self._adata = self.preprocess_data(self._adata)
        
        # Build spatial graph
        self._spatial_graph, self._edge_features = self.build_spatial_graph(self._adata)
        
        # Store metadata
        self._num_genes = self._adata.n_vars
        self._num_cells = self._adata.n_obs
        self._gene_names = self._adata.var_names.tolist()
        self._cell_ids = self._adata.obs_names.tolist()
        
        # Create processed data
        self._create_pytorch_data()
    
    def _create_pytorch_data(self) -> None:
        """Create PyTorch Geometric Data objects."""
        if self._adata is None:
            raise RuntimeError("Data not loaded. Call setup() first.")
        
        # Get expression matrix
        if hasattr(self._adata, 'X') and hasattr(self._adata.X, 'toarray'):
            # Sparse matrix
            gene_expression = torch.FloatTensor(self._adata.X.toarray())
        else:
            # Dense matrix
            gene_expression = torch.FloatTensor(self._adata.X)
        
        # Get spatial coordinates
        if 'spatial' in self._adata.obsm:
            spatial_coords = torch.FloatTensor(self._adata.obsm['spatial'])
        else:
            raise ValueError("No spatial coordinates found in data")
        
        # Create Data object
        data = Data(
            x=gene_expression,
            pos=spatial_coords,
            edge_index=self._spatial_graph,
            edge_attr=self._edge_features
        )
        
        # Add metadata
        if 'cell_type' in self._adata.obs:
            data.cell_type = self._adata.obs['cell_type'].values
        
        if 'batch' in self._adata.obs:
            data.batch = torch.LongTensor(self._adata.obs['batch'].values)
        else:
            data.batch = torch.zeros(self._num_cells, dtype=torch.long)
        
        # Apply pre-transform if specified
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        self._processed_data = [data]
    
    def len(self) -> int:
        """Return the number of samples in the dataset."""
        if self._processed_data is None:
            return 0
        return len(self._processed_data)
    
    def get(self, idx: int) -> Data:
        """
        Get a data sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            PyTorch Geometric Data object
        """
        if self._processed_data is None:
            raise RuntimeError("Data not processed. Call setup() first.")
        
        if idx >= len(self._processed_data):
            raise IndexError(f"Index {idx} out of range")
        
        data = self._processed_data[idx].clone()
        
        # Apply transform if specified
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    @property
    def adata(self) -> Optional[ad.AnnData]:
        """Get the AnnData object."""
        return self._adata
    
    @property
    def num_genes(self) -> int:
        """Get the number of genes."""
        if self._num_genes is None:
            raise RuntimeError("Data not loaded")
        return self._num_genes
    
    @property
    def num_cells(self) -> int:
        """Get the number of cells."""
        if self._num_cells is None:
            raise RuntimeError("Data not loaded")
        return self._num_cells
    
    @property
    def gene_names(self) -> List[str]:
        """Get gene names."""
        if self._gene_names is None:
            raise RuntimeError("Data not loaded")
        return self._gene_names
    
    @property
    def cell_ids(self) -> List[str]:
        """Get cell IDs."""
        if self._cell_ids is None:
            raise RuntimeError("Data not loaded")
        return self._cell_ids
    
    @property
    def spatial_graph(self) -> torch.Tensor:
        """Get spatial graph edge indices."""
        if self._spatial_graph is None:
            raise RuntimeError("Spatial graph not built")
        return self._spatial_graph
    
    @property
    def edge_features(self) -> torch.Tensor:
        """Get edge features."""
        if self._edge_features is None:
            raise RuntimeError("Edge features not computed")
        return self._edge_features
    
    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Get a PyTorch DataLoader for the dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        from torch_geometric.loader import DataLoader as GeometricDataLoader
        
        return GeometricDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple['BaseSpatialDataset', 'BaseSpatialDataset', 'BaseSpatialDataset']:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # For now, return copies of the same dataset
        # In practice, you might implement cell-level or region-level splitting
        warnings.warn(
            "split_data is not fully implemented. "
            "Returning copies of the full dataset."
        )
        
        return self, self, self
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the dataset to disk.
        
        Args:
            path: Path to save the dataset
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save AnnData object
        if self._adata is not None:
            self._adata.write(path / "adata.h5ad")
        
        # Save processed data
        if self._processed_data is not None:
            torch.save(self._processed_data, path / "processed_data.pt")
        
        # Save configuration
        config_dict = {
            'normalize': self.config.normalize,
            'log_transform': self.config.log_transform,
            'filter_genes': self.config.filter_genes,
            'filter_cells': self.config.filter_cells,
            'min_cells_per_gene': self.config.min_cells_per_gene,
            'min_genes_per_cell': self.config.min_genes_per_cell,
            'max_genes_per_cell': self.config.max_genes_per_cell,
            'highly_variable_genes': self.config.highly_variable_genes,
            'spatial_graph_k': self.config.spatial_graph_k,
            'spatial_graph_radius': self.config.spatial_graph_radius,
            'edge_features': self.config.edge_features
        }
        
        import json
        with open(path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseSpatialDataset':
        """
        Load a saved dataset from disk.
        
        Args:
            path: Path to saved dataset
            
        Returns:
            Loaded dataset
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        # Load configuration
        config_path = path / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = SpatialDataConfig(**config_dict)
        else:
            config = SpatialDataConfig()
        
        # Create dataset instance
        dataset = cls(config=config)
        
        # Load AnnData object
        adata_path = path / "adata.h5ad"
        if adata_path.exists():
            dataset._adata = ad.read_h5ad(adata_path)
            
            # Update metadata
            dataset._num_genes = dataset._adata.n_vars
            dataset._num_cells = dataset._adata.n_obs
            dataset._gene_names = dataset._adata.var_names.tolist()
            dataset._cell_ids = dataset._adata.obs_names.tolist()
        
        # Load processed data
        processed_path = path / "processed_data.pt"
        if processed_path.exists():
            dataset._processed_data = torch.load(processed_path)
        
        return dataset
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        if self._adata is None:
            return f"{self.__class__.__name__}(not loaded)"
        
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.num_cells}, "
            f"n_genes={self.num_genes})"
        )
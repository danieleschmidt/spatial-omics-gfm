"""
Visium dataset loader for 10X Genomics Visium spatial transcriptomics data.
Handles standard Visium outputs including H5 files, spatial coordinates,
and tissue detection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import h5py
import json
from PIL import Image
import scanpy as sc
from anndata import AnnData
import torch
from torch_geometric.data import Data

from .base import BaseSpatialDataset, SpatialDataConfig
from .preprocessing import SpatialPreprocessor
from .graph_construction import SpatialGraphBuilder

logger = logging.getLogger(__name__)


class VisiumDataset(BaseSpatialDataset):
    """
    Dataset loader for 10X Genomics Visium spatial transcriptomics data.
    
    Handles loading from standard Visium outputs including:
    - Feature-barcode matrix (H5 or MEX format)
    - Spatial coordinates and tissue positions
    - High-resolution tissue images
    - Scalefactor information
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        load_images: bool = True,
        filter_tissue: bool = True,
        **kwargs
    ):
        """
        Initialize Visium dataset.
        
        Args:
            data_path: Path to Visium data directory or H5 file
            config: Dataset configuration
            load_images: Whether to load tissue images
            filter_tissue: Whether to filter spots to tissue areas only
            **kwargs: Additional arguments
        """
        super().__init__(config=config, **kwargs)
        
        self.data_path = Path(data_path)
        self.load_images = load_images
        self.filter_tissue = filter_tissue
        
        # Initialize components
        self.preprocessor = SpatialPreprocessor(config)
        self.graph_builder = SpatialGraphBuilder(config)
        
        # Load data
        self._load_data()
        
        # Validate loaded data
        self._validate_spatial_data()
        
        logger.info(f"Loaded Visium dataset: {self.adata.shape}")
    
    def load_data(self, data_path: Union[str, Path], **kwargs) -> AnnData:
        """
        Load platform-specific data.
        
        Args:
            data_path: Path to data files
            **kwargs: Platform-specific arguments
            
        Returns:
            AnnData object with loaded data
        """
        # Store original path and reload
        original_path = self.data_path
        self.data_path = Path(data_path)
        self._load_data()
        return self.adata
    
    def validate_data(self, adata: AnnData) -> bool:
        """
        Validate that the data is suitable for processing.
        
        Args:
            adata: AnnData object to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check basic structure
            if adata.n_obs == 0 or adata.n_vars == 0:
                logger.error("Empty dataset")
                return False
            
            # Check for spatial coordinates
            if 'spatial' not in adata.obsm:
                logger.error("No spatial coordinates found")
                return False
            
            # Check coordinate dimensions
            if adata.obsm['spatial'].shape[1] != 2:
                logger.error("Spatial coordinates must be 2D")
                return False
            
            # Check for valid expression data
            if hasattr(adata.X, 'nnz') and adata.X.nnz == 0:
                logger.error("No expression data found")
                return False
            
            # Check for reasonable expression values
            if hasattr(adata.X, 'max'):
                max_val = adata.X.max()
                if np.isinf(max_val) or np.isnan(max_val):
                    logger.error("Invalid expression values detected")
                    return False
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    @classmethod
    def from_10x_folder(
        cls,
        visium_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'VisiumDataset':
        """
        Load Visium data from standard 10X output folder.
        
        Expected folder structure:
        visium_path/
        ├── filtered_feature_bc_matrix.h5 (or matrix.mtx.gz + features.tsv.gz + barcodes.tsv.gz)
        ├── spatial/
        │   ├── tissue_positions_list.csv
        │   ├── scalefactors_json.json
        │   ├── tissue_hires_image.png
        │   └── tissue_lowres_image.png
        
        Args:
            visium_path: Path to Visium output directory
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            VisiumDataset instance
        """
        return cls(visium_path, config=config, **kwargs)
    
    @classmethod
    def from_h5_file(
        cls,
        h5_path: Union[str, Path],
        spatial_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'VisiumDataset':
        """
        Load Visium data from H5 file and separate spatial folder.
        
        Args:
            h5_path: Path to H5 feature-barcode matrix
            spatial_path: Path to spatial information folder
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            VisiumDataset instance
        """
        # Create temporary structure for loading
        temp_dict = {
            'h5_path': h5_path,
            'spatial_path': spatial_path
        }
        
        instance = cls.__new__(cls)
        instance.config = config or SpatialDataConfig()
        instance.load_images = kwargs.get('load_images', True)
        instance.filter_tissue = kwargs.get('filter_tissue', True)
        instance.data_dict = temp_dict
        
        instance.preprocessor = SpatialPreprocessor(instance.config)
        instance.graph_builder = SpatialGraphBuilder(instance.config)
        
        instance._load_from_paths()
        instance._validate_spatial_data()
        
        logger.info(f"Loaded Visium dataset: {instance.adata.shape}")
        return instance
    
    def _load_data(self) -> None:
        """Load Visium data from directory structure."""
        if self.data_path.is_file():
            # Single H5 file provided
            if self.data_path.suffix == '.h5':
                spatial_path = self.data_path.parent / 'spatial'
                self._load_from_h5_and_spatial(self.data_path, spatial_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        else:
            # Directory provided
            self._load_from_directory()
    
    def _load_from_paths(self) -> None:
        """Load from explicitly provided paths."""
        h5_path = self.data_dict['h5_path']
        spatial_path = self.data_dict['spatial_path']
        self._load_from_h5_and_spatial(Path(h5_path), Path(spatial_path))
    
    def _load_from_directory(self) -> None:
        """Load from standard Visium directory structure."""
        logger.info(f"Loading Visium data from directory: {self.data_path}")
        
        # Look for H5 file
        h5_candidates = list(self.data_path.glob("*.h5"))
        if not h5_candidates:
            # Try MEX format
            self._load_from_mex()
            return
        
        h5_path = h5_candidates[0]
        spatial_path = self.data_path / 'spatial'
        
        self._load_from_h5_and_spatial(h5_path, spatial_path)
    
    def _load_from_h5_and_spatial(self, h5_path: Path, spatial_path: Path) -> None:
        """Load data from H5 file and spatial directory."""
        logger.info(f"Loading expression data from: {h5_path}")
        
        # Load expression matrix
        try:
            self.adata = sc.read_10x_h5(h5_path)
            self.adata.var_names_unique()
        except Exception as e:
            logger.error(f"Failed to load H5 file: {e}")
            raise
        
        # Load spatial information
        self._load_spatial_info(spatial_path)
        
        # Filter to tissue spots if requested
        if self.filter_tissue:
            self._filter_tissue_spots()
        
        # Set up observations
        self.adata.obs_names = [f"Spot_{i}" for i in range(self.adata.n_obs)]
        
        # Store original data path
        self.adata.uns['visium_path'] = str(self.data_path)
    
    def _load_from_mex(self) -> None:
        """Load from MEX format (matrix.mtx, features.tsv, barcodes.tsv)."""
        logger.info("Loading from MEX format")
        
        # Look for MEX files
        matrix_file = None
        features_file = None
        barcodes_file = None
        
        for pattern in ["matrix.mtx*", "*.mtx*"]:
            candidates = list(self.data_path.glob(pattern))
            if candidates:
                matrix_file = candidates[0]
                break
        
        for pattern in ["features.tsv*", "genes.tsv*"]:
            candidates = list(self.data_path.glob(pattern))
            if candidates:
                features_file = candidates[0]
                break
                
        for pattern in ["barcodes.tsv*"]:
            candidates = list(self.data_path.glob(pattern))
            if candidates:
                barcodes_file = candidates[0]
                break
        
        if not all([matrix_file, features_file, barcodes_file]):
            raise FileNotFoundError("Could not find complete MEX format files")
        
        # Load using scanpy
        self.adata = sc.read_10x_mtx(
            self.data_path,
            var_names='gene_symbols',
            cache=True
        )
        self.adata.var_names_unique()
        
        # Load spatial information
        spatial_path = self.data_path / 'spatial'
        self._load_spatial_info(spatial_path)
        
        if self.filter_tissue:
            self._filter_tissue_spots()
    
    def _load_spatial_info(self, spatial_path: Path) -> None:
        """Load spatial coordinates and metadata."""
        logger.info(f"Loading spatial information from: {spatial_path}")
        
        if not spatial_path.exists():
            raise FileNotFoundError(f"Spatial directory not found: {spatial_path}")
        
        # Load tissue positions
        positions_file = spatial_path / 'tissue_positions_list.csv'
        if not positions_file.exists():
            # Try alternative name
            positions_file = spatial_path / 'tissue_positions.csv'
            
        if not positions_file.exists():
            raise FileNotFoundError(f"Tissue positions file not found in {spatial_path}")
        
        # Read positions - handle different formats
        try:
            positions_df = pd.read_csv(positions_file, header=0, index_col=0)
        except:
            positions_df = pd.read_csv(positions_file, header=None, index_col=0)
            positions_df.columns = ['in_tissue', 'array_row', 'array_col', 
                                   'pxl_col_in_fullres', 'pxl_row_in_fullres']
        
        # Ensure proper column names
        expected_cols = ['in_tissue', 'array_row', 'array_col', 
                        'pxl_col_in_fullres', 'pxl_row_in_fullres']
        if len(positions_df.columns) >= 5:
            positions_df.columns = expected_cols[:len(positions_df.columns)]
        
        # Match barcodes between expression and spatial data
        common_barcodes = positions_df.index.intersection(self.adata.obs.index)
        if len(common_barcodes) == 0:
            logger.warning("No matching barcodes found, attempting barcode matching")
            common_barcodes = self._match_barcodes(positions_df.index, self.adata.obs.index)
        
        # Filter both datasets to common barcodes
        self.adata = self.adata[common_barcodes].copy()
        positions_df = positions_df.loc[common_barcodes]
        
        # Add spatial information to adata
        self.adata.obs['in_tissue'] = positions_df['in_tissue'].astype(bool)
        self.adata.obs['array_row'] = positions_df['array_row'].astype(int)
        self.adata.obs['array_col'] = positions_df['array_col'].astype(int)
        
        # Store spatial coordinates (using pixel coordinates)
        spatial_coords = positions_df[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        self.adata.obsm['spatial'] = spatial_coords.astype(float)
        
        # Load scale factors
        self._load_scale_factors(spatial_path)
        
        # Load images if requested
        if self.load_images:
            self._load_images(spatial_path)
    
    def _load_scale_factors(self, spatial_path: Path) -> None:
        """Load spatial scale factors."""
        scalefactors_file = spatial_path / 'scalefactors_json.json'
        
        if scalefactors_file.exists():
            with open(scalefactors_file, 'r') as f:
                scalefactors = json.load(f)
            
            self.adata.uns['spatial'] = {
                'scalefactors': scalefactors
            }
            
            logger.info(f"Loaded scale factors: {scalefactors}")
        else:
            logger.warning("Scale factors file not found, using defaults")
            self.adata.uns['spatial'] = {
                'scalefactors': {
                    'tissue_hires_scalef': 1.0,
                    'tissue_lowres_scalef': 1.0,
                    'fiducial_diameter_fullres': 1.0,
                    'spot_diameter_fullres': 1.0
                }
            }
    
    def _load_images(self, spatial_path: Path) -> None:
        """Load tissue images."""
        logger.info("Loading tissue images")
        
        # Load high-resolution image
        hires_path = spatial_path / 'tissue_hires_image.png'
        lowres_path = spatial_path / 'tissue_lowres_image.png'
        
        images = {}
        
        if hires_path.exists():
            try:
                hires_img = np.array(Image.open(hires_path))
                images['hires'] = hires_img
                logger.info(f"Loaded high-res image: {hires_img.shape}")
            except Exception as e:
                logger.warning(f"Failed to load high-res image: {e}")
        
        if lowres_path.exists():
            try:
                lowres_img = np.array(Image.open(lowres_path))
                images['lowres'] = lowres_img  
                logger.info(f"Loaded low-res image: {lowres_img.shape}")
            except Exception as e:
                logger.warning(f"Failed to load low-res image: {e}")
        
        if images:
            if 'spatial' not in self.adata.uns:
                self.adata.uns['spatial'] = {}
            self.adata.uns['spatial']['images'] = images
    
    def _filter_tissue_spots(self) -> None:
        """Filter spots to only those in tissue."""
        if 'in_tissue' not in self.adata.obs:
            logger.warning("No tissue information available, skipping tissue filtering")
            return
        
        n_spots_before = self.adata.n_obs
        tissue_spots = self.adata.obs['in_tissue']
        self.adata = self.adata[tissue_spots].copy()
        n_spots_after = self.adata.n_obs
        
        logger.info(f"Filtered to tissue spots: {n_spots_before} -> {n_spots_after}")
    
    def _match_barcodes(self, spatial_barcodes: pd.Index, expression_barcodes: pd.Index) -> pd.Index:
        """Attempt to match barcodes between spatial and expression data."""
        logger.info("Attempting barcode matching")
        
        # Try with and without suffix
        spatial_clean = spatial_barcodes.str.replace(r'-\d+$', '', regex=True)
        expression_clean = expression_barcodes.str.replace(r'-\d+$', '', regex=True)
        
        # Find intersection of cleaned barcodes
        common_clean = spatial_clean.intersection(expression_clean)
        
        if len(common_clean) > 0:
            # Map back to original barcodes
            spatial_mapping = dict(zip(spatial_clean, spatial_barcodes))
            expression_mapping = dict(zip(expression_clean, expression_barcodes))
            
            common_spatial = [spatial_mapping[bc] for bc in common_clean]
            common_expression = [expression_mapping[bc] for bc in common_clean]
            
            # Update indices to match
            self.adata.obs.index = [expression_mapping.get(ec, ec) for ec in expression_clean]
            
            return pd.Index(common_spatial)
        
        return pd.Index([])
    
    def _validate_spatial_data(self) -> None:
        """Validate that spatial data is properly loaded."""
        if 'spatial' not in self.adata.obsm:
            raise ValueError("No spatial coordinates found")
            
        if self.adata.obsm['spatial'].shape[1] != 2:
            raise ValueError("Spatial coordinates must be 2D")
            
        # Check for valid coordinates
        coords = self.adata.obsm['spatial']
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            logger.warning("Invalid coordinates detected, cleaning data")
            valid_mask = ~(np.isnan(coords).any(axis=1) | np.isinf(coords).any(axis=1))
            self.adata = self.adata[valid_mask].copy()
    
    def get_spatial_coordinates(self) -> np.ndarray:
        """Get spatial coordinates."""
        return self.adata.obsm['spatial'].copy()
    
    def get_spatial_neighbors(
        self,
        method: str = 'knn',
        n_neighbors: int = 6,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial neighborhood graph.
        
        Args:
            method: Graph construction method
            n_neighbors: Number of neighbors
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        coordinates = self.get_spatial_coordinates()
        edge_index, edge_attr, graph_info = self.graph_builder.build_spatial_graph(
            coordinates, method=method, k=n_neighbors, **kwargs
        )
        
        # Store graph in adata
        self.graph_builder.add_graph_to_adata(
            self.adata, edge_index, edge_attr, graph_info
        )
        
        return edge_index, edge_attr
    
    def to_pytorch_geometric(self) -> Data:
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            PyTorch Geometric Data object
        """
        # Ensure spatial graph exists
        if 'spatial_graph' not in self.adata.uns:
            logger.info("Computing spatial graph for PyTorch Geometric conversion")
            self.get_spatial_neighbors()
        
        graph_data = self.adata.uns['spatial_graph']
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float32)
        
        return self.graph_builder.create_pytorch_geometric_data(
            self.adata, edge_index, edge_attr
        )
    
    def preprocess(
        self,
        normalize: bool = True,
        log_transform: bool = True,
        highly_variable_genes: bool = True,
        compute_neighbors: bool = True
    ) -> 'VisiumDataset':
        """
        Preprocess the dataset.
        
        Args:
            normalize: Whether to normalize expression
            log_transform: Whether to log transform
            highly_variable_genes: Whether to select HVGs  
            compute_neighbors: Whether to compute spatial neighbors
            
        Returns:
            Self for method chaining
        """
        self.adata = self.preprocessor.preprocess_spatial_data(
            self.adata,
            normalize=normalize,
            log_transform=log_transform,
            highly_variable_genes=highly_variable_genes,
            spatial_neighbors=compute_neighbors
        )
        
        return self
    
    def plot_spatial(self, color: Optional[str] = None, **kwargs) -> None:
        """Plot spatial data."""
        try:
            import squidpy as sq
            sq.pl.spatial_scatter(self.adata, color=color, **kwargs)
        except ImportError:
            logger.warning("Squidpy not available, using basic matplotlib plot")
            self._basic_spatial_plot(color, **kwargs)
    
    def _basic_spatial_plot(self, color: Optional[str] = None, **kwargs) -> None:
        """Basic spatial plot using matplotlib."""
        import matplotlib.pyplot as plt
        
        coords = self.get_spatial_coordinates()
        
        if color is None:
            plt.scatter(coords[:, 0], coords[:, 1], **kwargs)
        else:
            if color in self.adata.obs.columns:
                c = self.adata.obs[color]
            elif color in self.adata.var_names:
                c = self.adata.obs_vector(color)
            else:
                raise ValueError(f"Color '{color}' not found")
            
            plt.scatter(coords[:, 0], coords[:, 1], c=c, **kwargs)
            plt.colorbar()
        
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(f'Visium Spatial Plot{f" - {color}" if color else ""}')
        plt.show()
    
    def __len__(self) -> int:
        """Return number of spots."""
        return self.adata.n_obs
    
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Dict[str, Any]:
        """Get spot data by index."""
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))
        
        spot_data = {
            'expression': self.adata.X[idx].copy(),
            'coordinates': self.adata.obsm['spatial'][idx].copy(),
            'observations': self.adata.obs.iloc[idx].copy()
        }
        
        return spot_data
"""
Slide-seq dataset loader for Broad Institute Slide-seq spatial transcriptomics data.
Handles Slide-seq V1 and V2 data formats with high spatial resolution.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
from anndata import AnnData
import torch
from torch_geometric.data import Data

from .base import BaseSpatialDataset, SpatialDataConfig
from .preprocessing import SpatialPreprocessor
from .graph_construction import SpatialGraphBuilder

logger = logging.getLogger(__name__)


class SlideSeqDataset(BaseSpatialDataset):
    """
    Dataset loader for Broad Institute Slide-seq spatial transcriptomics data.
    
    Handles both Slide-seq V1 and V2 formats with high spatial resolution
    (typically 10-15 micron diameter beads).
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        coordinates_path: Optional[Union[str, Path]] = None,
        config: Optional[SpatialDataConfig] = None,
        version: str = 'auto',
        **kwargs
    ):
        """
        Initialize Slide-seq dataset.
        
        Args:
            data_path: Path to expression data (CSV, H5, or MTX)
            coordinates_path: Path to coordinates file (if separate)
            config: Dataset configuration
            version: Slide-seq version ('v1', 'v2', 'auto')
            **kwargs: Additional arguments
        """
        super().__init__(config=config, **kwargs)
        
        self.data_path = Path(data_path)
        self.coordinates_path = Path(coordinates_path) if coordinates_path else None
        self.version = version
        
        # Initialize components
        self.preprocessor = SpatialPreprocessor(config)
        self.graph_builder = SpatialGraphBuilder(config)
        
        # Load data
        self._load_data()
        
        # Validate loaded data
        self._validate_spatial_data()
        
        logger.info(f"Loaded Slide-seq dataset: {self.adata.shape}")
    
    @classmethod
    def from_csv_files(
        cls,
        expression_path: Union[str, Path],
        coordinates_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'SlideSeqDataset':
        """
        Load Slide-seq data from separate CSV files.
        
        Args:
            expression_path: Path to expression matrix CSV
            coordinates_path: Path to coordinates CSV
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            SlideSeqDataset instance
        """
        return cls(
            data_path=expression_path,
            coordinates_path=coordinates_path,
            config=config,
            **kwargs
        )
    
    @classmethod
    def from_combined_file(
        cls,
        data_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'SlideSeqDataset':
        """
        Load Slide-seq data from single file with embedded coordinates.
        
        Args:
            data_path: Path to combined data file
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            SlideSeqDataset instance
        """
        return cls(data_path=data_path, config=config, **kwargs)
    
    def _load_data(self) -> None:
        """Load Slide-seq data from files."""
        logger.info(f"Loading Slide-seq data from: {self.data_path}")
        
        # Determine file format and load accordingly
        if self.data_path.suffix == '.csv':
            self._load_from_csv()
        elif self.data_path.suffix == '.h5':
            self._load_from_h5()
        elif self.data_path.suffix in ['.mtx', '.txt']:
            self._load_from_matrix()
        elif self.data_path.is_dir():
            self._load_from_directory()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _load_from_csv(self) -> None:
        """Load from CSV format."""
        logger.info("Loading from CSV format")
        
        # Load expression data
        try:
            expr_df = pd.read_csv(self.data_path, index_col=0)
            logger.info(f"Loaded expression matrix: {expr_df.shape}")
        except Exception as e:
            logger.error(f"Failed to load expression CSV: {e}")
            raise
        
        # Check if coordinates are embedded in the same file
        if self.coordinates_path is None:
            # Look for coordinate columns in expression file
            coord_cols = self._detect_coordinate_columns(expr_df.columns)
            if coord_cols:
                logger.info(f"Found coordinate columns: {coord_cols}")
                coordinates = expr_df[coord_cols].values
                expression = expr_df.drop(columns=coord_cols)
            else:
                raise ValueError(
                    "No coordinate columns found and no separate coordinates file provided"
                )
        else:
            # Load coordinates from separate file
            coordinates = self._load_coordinates()
            expression = expr_df
        
        # Create AnnData object
        self.adata = self._create_anndata_from_dataframes(expression, coordinates)
    
    def _load_from_h5(self) -> None:
        """Load from H5 format."""
        logger.info("Loading from H5 format")
        
        try:
            self.adata = sc.read_h5ad(self.data_path)
        except:
            # Try reading as 10X H5 format
            self.adata = sc.read_10x_h5(self.data_path)
        
        # Load coordinates if separate file provided
        if self.coordinates_path:
            coordinates = self._load_coordinates()
            self.adata.obsm['spatial'] = coordinates
        elif 'spatial' not in self.adata.obsm:
            raise ValueError("No spatial coordinates found in H5 file")
    
    def _load_from_matrix(self) -> None:
        """Load from matrix format (MTX/TXT)."""
        logger.info("Loading from matrix format")
        
        if self.data_path.suffix == '.mtx':
            # Load sparse matrix
            expression_matrix = sparse.mmread(self.data_path).T.tocsr()
        else:
            # Load dense matrix
            expression_df = pd.read_csv(self.data_path, sep='\t', index_col=0)
            expression_matrix = expression_df.values
        
        # Create basic AnnData
        self.adata = AnnData(expression_matrix)
        
        # Load coordinates
        if self.coordinates_path:
            coordinates = self._load_coordinates()
            self.adata.obsm['spatial'] = coordinates
        else:
            raise ValueError("Coordinates file required for matrix format")
    
    def _load_from_directory(self) -> None:
        """Load from directory containing multiple files."""
        logger.info("Loading from directory")
        
        # Look for standard Slide-seq file patterns
        expr_files = list(self.data_path.glob("*expression*")) + \
                    list(self.data_path.glob("*count*")) + \
                    list(self.data_path.glob("*.csv"))
        
        coord_files = list(self.data_path.glob("*coordinate*")) + \
                     list(self.data_path.glob("*position*")) + \
                     list(self.data_path.glob("*location*"))
        
        if not expr_files:
            raise FileNotFoundError("No expression files found in directory")
        
        # Use the first found files
        expr_file = expr_files[0]
        coord_file = coord_files[0] if coord_files else None
        
        # Update paths and load
        self.data_path = expr_file
        self.coordinates_path = coord_file
        self._load_data()
    
    def _load_coordinates(self) -> np.ndarray:
        """Load spatial coordinates from file."""
        logger.info(f"Loading coordinates from: {self.coordinates_path}")
        
        try:
            if self.coordinates_path.suffix == '.csv':
                coords_df = pd.read_csv(self.coordinates_path)
            else:
                coords_df = pd.read_csv(self.coordinates_path, sep='\t')
            
            logger.info(f"Loaded coordinates: {coords_df.shape}")
            
            # Detect coordinate columns
            coord_cols = self._detect_coordinate_columns(coords_df.columns)
            if not coord_cols:
                # Assume first two numeric columns are coordinates
                numeric_cols = coords_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    coord_cols = numeric_cols[:2].tolist()
                else:
                    raise ValueError("Could not detect coordinate columns")
            
            coordinates = coords_df[coord_cols].values
            logger.info(f"Using coordinate columns: {coord_cols}")
            
            return coordinates.astype(float)
            
        except Exception as e:
            logger.error(f"Failed to load coordinates: {e}")
            raise
    
    def _detect_coordinate_columns(self, columns: pd.Index) -> List[str]:
        """Detect which columns contain spatial coordinates."""
        coord_patterns = [
            'x', 'y', 'X', 'Y',
            'xcoord', 'ycoord',
            'x_coord', 'y_coord',
            'pos_x', 'pos_y',
            'position_x', 'position_y',
            'coordinate_x', 'coordinate_y',
            'spatial_x', 'spatial_y'
        ]
        
        detected = []
        for pattern in coord_patterns:
            if pattern in columns:
                detected.append(pattern)
                if len(detected) == 2:
                    break
        
        # If exact matches not found, look for partial matches
        if len(detected) < 2:
            for col in columns:
                col_lower = col.lower()
                if any(p.lower() in col_lower for p in ['x', 'pos', 'coord']) and 'x' in col_lower:
                    detected.append(col)
                elif any(p.lower() in col_lower for p in ['y', 'pos', 'coord']) and 'y' in col_lower:
                    detected.append(col)
                
                if len(detected) == 2:
                    break
        
        return detected[:2]  # Return first two detected columns
    
    def _create_anndata_from_dataframes(
        self,
        expression: pd.DataFrame,
        coordinates: np.ndarray
    ) -> AnnData:
        """Create AnnData object from expression and coordinates."""
        logger.info("Creating AnnData object")
        
        # Ensure matching dimensions
        if len(expression) != len(coordinates):
            logger.warning(f"Dimension mismatch: expression {len(expression)}, coordinates {len(coordinates)}")
            # Take minimum length
            min_len = min(len(expression), len(coordinates))
            expression = expression.iloc[:min_len]
            coordinates = coordinates[:min_len]
        
        # Create AnnData
        adata = AnnData(
            X=expression.values,
            obs=pd.DataFrame(index=expression.index),
            var=pd.DataFrame(index=expression.columns)
        )
        
        # Add spatial coordinates
        adata.obsm['spatial'] = coordinates
        
        # Detect Slide-seq version from data characteristics
        if self.version == 'auto':
            self.version = self._detect_slideseq_version(adata)
        
        # Add metadata
        adata.uns['slideseq_version'] = self.version
        adata.uns['spatial_resolution'] = self._get_spatial_resolution()
        
        return adata
    
    def _detect_slideseq_version(self, adata: AnnData) -> str:
        """Detect Slide-seq version from data characteristics."""
        # V2 typically has higher bead density and different coordinate scales
        coordinates = adata.obsm['spatial']
        
        # Calculate bead density (beads per unit area)
        x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
        y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
        area = x_range * y_range
        density = len(coordinates) / area if area > 0 else 0
        
        # V2 characteristics (heuristic-based detection)
        if density > 0.01 and x_range > 1000:  # High density, large coordinate scale
            return 'v2'
        elif density < 0.001 or x_range < 100:  # Lower density, smaller scale
            return 'v1'
        else:
            return 'v2'  # Default to v2 for recent data
    
    def _get_spatial_resolution(self) -> float:
        """Get spatial resolution based on Slide-seq version."""
        if self.version == 'v1':
            return 10.0  # ~10 micron bead diameter
        else:  # v2
            return 10.0  # ~10 micron bead diameter (improved)
    
    def _validate_spatial_data(self) -> None:
        """Validate spatial data integrity."""
        if 'spatial' not in self.adata.obsm:
            raise ValueError("No spatial coordinates found")
        
        coords = self.adata.obsm['spatial']
        if coords.shape[1] != 2:
            raise ValueError("Spatial coordinates must be 2D")
        
        # Check for invalid coordinates
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            logger.warning("Invalid coordinates detected, cleaning data")
            valid_mask = ~(np.isnan(coords).any(axis=1) | np.isinf(coords).any(axis=1))
            self.adata = self.adata[valid_mask].copy()
        
        # Check coordinate range (detect potential scaling issues)
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        if x_range < 1 or y_range < 1:
            logger.warning("Coordinates appear to be in very small scale, consider scaling")
        elif x_range > 100000 or y_range > 100000:
            logger.warning("Coordinates appear to be in very large scale, consider scaling")
    
    def get_spatial_coordinates(self) -> np.ndarray:
        """Get spatial coordinates."""
        return self.adata.obsm['spatial'].copy()
    
    def get_spatial_neighbors(
        self,
        method: str = 'knn',
        n_neighbors: int = 8,  # Higher default for Slide-seq
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial neighborhood graph.
        
        Args:
            method: Graph construction method
            n_neighbors: Number of neighbors (higher default for dense Slide-seq)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        coordinates = self.get_spatial_coordinates()
        
        # Adjust default radius based on version
        if method == 'radius' and 'radius' not in kwargs:
            if self.version == 'v1':
                kwargs['radius'] = 15.0  # Slightly larger radius for v1
            else:
                kwargs['radius'] = 12.0  # Tighter radius for v2
        
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
    ) -> 'SlideSeqDataset':
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
    
    def filter_by_spatial_region(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        center: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None
    ) -> 'SlideSeqDataset':
        """
        Filter dataset to a specific spatial region.
        
        Args:
            x_range: Range of x coordinates (min, max)
            y_range: Range of y coordinates (min, max)
            center: Center coordinates for circular region
            radius: Radius for circular region
            
        Returns:
            Filtered dataset
        """
        coords = self.get_spatial_coordinates()
        mask = np.ones(len(coords), dtype=bool)
        
        if x_range:
            mask &= (coords[:, 0] >= x_range[0]) & (coords[:, 0] <= x_range[1])
        
        if y_range:
            mask &= (coords[:, 1] >= y_range[0]) & (coords[:, 1] <= y_range[1])
        
        if center and radius:
            distances = np.linalg.norm(coords - np.array(center), axis=1)
            mask &= distances <= radius
        
        self.adata = self.adata[mask].copy()
        logger.info(f"Filtered to spatial region: {self.adata.shape}")
        
        return self
    
    def plot_spatial(
        self,
        color: Optional[str] = None,
        size: float = 1.0,
        **kwargs
    ) -> None:
        """Plot spatial data."""
        import matplotlib.pyplot as plt
        
        coords = self.get_spatial_coordinates()
        
        # Adjust point size based on bead density
        if self.version == 'v2':
            size *= 0.5  # Smaller points for denser v2 data
        
        if color is None:
            plt.scatter(coords[:, 0], coords[:, 1], s=size, **kwargs)
        else:
            if color in self.adata.obs.columns:
                c = self.adata.obs[color]
            elif color in self.adata.var_names:
                c = self.adata.obs_vector(color)
            else:
                raise ValueError(f"Color '{color}' not found")
            
            plt.scatter(coords[:, 0], coords[:, 1], c=c, s=size, **kwargs)
            plt.colorbar()
        
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(f'Slide-seq {self.version.upper()} Spatial Plot{f" - {color}" if color else ""}')
        plt.axis('equal')
        plt.show()
    
    def get_bead_density(self) -> float:
        """Calculate bead density (beads per unit area)."""
        coords = self.get_spatial_coordinates()
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        area = x_range * y_range
        
        return len(coords) / area if area > 0 else 0
    
    def __len__(self) -> int:
        """Return number of beads."""
        return self.adata.n_obs
    
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Dict[str, Any]:
        """Get bead data by index."""
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))
        
        bead_data = {
            'expression': self.adata.X[idx].copy(),
            'coordinates': self.adata.obsm['spatial'][idx].copy(),
            'observations': self.adata.obs.iloc[idx].copy()
        }
        
        return bead_data
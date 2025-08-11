"""
Xenium dataset loader for 10X Genomics Xenium spatial transcriptomics data.
Handles subcellular resolution spatial transcriptomics with cell boundaries
and transcript-level information.
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


class XeniumDataset(BaseSpatialDataset):
    """
    Dataset loader for 10X Genomics Xenium spatial transcriptomics data.
    
    Handles loading from Xenium outputs including:
    - Cell-by-gene expression matrix
    - Subcellular transcript coordinates
    - Cell boundary polygons
    - High-resolution imaging data
    - Field of view (FOV) information
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        load_transcripts: bool = False,
        load_boundaries: bool = True,
        merge_fov: bool = True,
        subcellular_resolution: bool = False,
        **kwargs
    ):
        """
        Initialize Xenium dataset.
        
        Args:
            data_path: Path to Xenium output directory
            config: Dataset configuration
            load_transcripts: Whether to load transcript-level data
            load_boundaries: Whether to load cell boundaries
            merge_fov: Whether to merge multiple fields of view
            subcellular_resolution: Whether to use subcellular resolution
            **kwargs: Additional arguments
        """
        super().__init__(config=config, **kwargs)
        
        self.data_path = Path(data_path)
        self.load_transcripts = load_transcripts
        self.load_boundaries = load_boundaries
        self.merge_fov = merge_fov
        self.subcellular_resolution = subcellular_resolution
        
        # Initialize components
        self.preprocessor = SpatialPreprocessor(config)
        self.graph_builder = SpatialGraphBuilder(config)
        
        # Load data
        self._load_data()
        
        # Validate loaded data
        self._validate_spatial_data()
        
        logger.info(f"Loaded Xenium dataset: {self.adata.shape}")
    
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
        
        # Update other parameters if provided
        if 'load_transcripts' in kwargs:
            self.load_transcripts = kwargs['load_transcripts']
        if 'load_boundaries' in kwargs:
            self.load_boundaries = kwargs['load_boundaries']
        if 'merge_fov' in kwargs:
            self.merge_fov = kwargs['merge_fov']
        
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
            
            # Xenium specific validations
            # Check if we have cell metadata
            if 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns:
                # Verify coordinate consistency
                obs_coords = adata.obs[['x_centroid', 'y_centroid']].values
                obsm_coords = adata.obsm['spatial']
                if not np.allclose(obs_coords, obsm_coords, atol=1e-6):
                    logger.warning("Coordinate mismatch between obs and obsm")
            
            # Check for field of view information
            if 'fov' in adata.obs.columns:
                n_fovs = adata.obs['fov'].nunique()
                logger.info(f"Found {n_fovs} fields of view")
            
            logger.info("Xenium data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    @classmethod
    def from_xenium_folder(
        cls,
        xenium_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'XeniumDataset':
        """
        Load Xenium data from standard output folder.
        
        Expected folder structure:
        xenium_path/
        ├── cell_feature_matrix.h5
        ├── cells.csv.gz
        ├── cell_boundaries.csv.gz (optional)
        ├── transcripts.csv.gz (optional)
        ├── morphology.ome.tif (optional)
        └── experiment.xenium
        
        Args:
            xenium_path: Path to Xenium output directory
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            XeniumDataset instance
        """
        return cls(xenium_path, config=config, **kwargs)
    
    def _load_data(self) -> None:
        """Load Xenium data from directory structure."""
        logger.info(f"Loading Xenium data from directory: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Xenium directory not found: {self.data_path}")
        
        # Load expression matrix
        self._load_expression_matrix()
        
        # Load cell metadata
        self._load_cell_metadata()
        
        # Load cell boundaries if requested
        if self.load_boundaries:
            self._load_cell_boundaries()
        
        # Load transcripts if requested
        if self.load_transcripts:
            self._load_transcripts()
        
        # Load imaging data
        self._load_images()
        
        # Handle field of view merging
        if self.merge_fov:
            self._merge_fields_of_view()
    
    def _load_expression_matrix(self) -> None:
        """Load cell-by-gene expression matrix."""
        # Look for H5 file
        h5_candidates = [
            self.data_path / "cell_feature_matrix.h5",
            self.data_path / "cell_feature_matrix" / "matrix.h5",
        ]
        
        h5_path = None
        for candidate in h5_candidates:
            if candidate.exists():
                h5_path = candidate
                break
        
        if h5_path is None:
            # Try MEX format
            mex_path = self.data_path / "cell_feature_matrix"
            if mex_path.exists():
                self._load_from_mex(mex_path)
                return
            else:
                raise FileNotFoundError("No expression matrix found")
        
        logger.info(f"Loading expression matrix from: {h5_path}")
        
        try:
            self.adata = sc.read_10x_h5(h5_path)
            self.adata.var_names_unique()
            
            # Xenium typically has genes as features
            if 'Gene Expression' in self.adata.var.columns:
                gene_mask = self.adata.var['Gene Expression'] == 'Gene Expression'
                self.adata = self.adata[:, gene_mask].copy()
            
        except Exception as e:
            logger.error(f"Failed to load H5 file: {e}")
            raise
    
    def _load_from_mex(self, mex_path: Path) -> None:
        """Load from MEX format."""
        logger.info(f"Loading from MEX format: {mex_path}")
        
        self.adata = sc.read_10x_mtx(
            mex_path,
            var_names='gene_symbols',
            cache=True
        )
        self.adata.var_names_unique()
    
    def _load_cell_metadata(self) -> None:
        """Load cell metadata including coordinates."""
        cells_file = self.data_path / "cells.csv.gz"
        if not cells_file.exists():
            cells_file = self.data_path / "cells.csv"
            
        if not cells_file.exists():
            raise FileNotFoundError(f"Cells metadata file not found: {cells_file}")
        
        logger.info(f"Loading cell metadata from: {cells_file}")
        
        # Load cells metadata
        cells_df = pd.read_csv(cells_file)
        
        # Expected columns: cell_id, x_centroid, y_centroid, fov, cell_area, etc.
        required_cols = ['cell_id', 'x_centroid', 'y_centroid']
        missing_cols = [col for col in required_cols if col not in cells_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in cells.csv: {missing_cols}")
        
        # Set cell_id as index
        cells_df = cells_df.set_index('cell_id')
        
        # Match cell IDs between expression and metadata
        common_cells = cells_df.index.intersection(self.adata.obs.index)
        if not common_cells:
            logger.warning("No matching cell IDs found, attempting cell ID matching")
            common_cells = self._match_cell_ids(cells_df.index, self.adata.obs.index)
        
        # Filter both datasets to common cells
        self.adata = self.adata[common_cells].copy()
        cells_df = cells_df.loc[common_cells]
        
        # Add metadata to adata
        for col in cells_df.columns:
            self.adata.obs[col] = cells_df[col]
        
        # Store spatial coordinates
        spatial_coords = cells_df[['x_centroid', 'y_centroid']].values
        self.adata.obsm['spatial'] = spatial_coords.astype(float)
        
        # Store field of view information if available
        if 'fov' in cells_df.columns:
            self.adata.obs['fov'] = cells_df['fov'].astype('category')
    
    def _load_cell_boundaries(self) -> None:
        """Load cell boundary polygons."""
        boundaries_file = self.data_path / "cell_boundaries.csv.gz"
        if not boundaries_file.exists():
            boundaries_file = self.data_path / "cell_boundaries.csv"
            
        if not boundaries_file.exists():
            logger.warning("Cell boundaries file not found, skipping boundary loading")
            return
        
        logger.info(f"Loading cell boundaries from: {boundaries_file}")
        
        try:
            boundaries_df = pd.read_csv(boundaries_file)
            
            # Expected columns: cell_id, vertex_x, vertex_y
            if all(col in boundaries_df.columns for col in ['cell_id', 'vertex_x', 'vertex_y']):
                # Group by cell_id to create polygons
                cell_polygons = {}
                for cell_id, group in boundaries_df.groupby('cell_id'):
                    if cell_id in self.adata.obs.index:
                        vertices = group[['vertex_x', 'vertex_y']].values
                        cell_polygons[cell_id] = vertices
                
                # Store in uns
                self.adata.uns['cell_boundaries'] = cell_polygons
                logger.info(f"Loaded boundaries for {len(cell_polygons)} cells")
            
        except Exception as e:
            logger.warning(f"Failed to load cell boundaries: {e}")
    
    def _load_transcripts(self) -> None:
        """Load transcript-level data."""
        transcripts_file = self.data_path / "transcripts.csv.gz"
        if not transcripts_file.exists():
            transcripts_file = self.data_path / "transcripts.csv"
            
        if not transcripts_file.exists():
            logger.warning("Transcripts file not found, skipping transcript loading")
            return
        
        logger.info(f"Loading transcripts from: {transcripts_file}")
        
        try:
            # Load in chunks for large files
            chunk_size = 100000
            transcript_chunks = []
            
            for chunk in pd.read_csv(transcripts_file, chunksize=chunk_size):
                # Filter to cells in our dataset
                if 'cell_id' in chunk.columns:
                    chunk = chunk[chunk['cell_id'].isin(self.adata.obs.index)]
                transcript_chunks.append(chunk)
            
            if transcript_chunks:
                transcripts_df = pd.concat(transcript_chunks, ignore_index=True)
                self.adata.uns['transcripts'] = transcripts_df
                logger.info(f"Loaded {len(transcripts_df)} transcripts")
            
        except Exception as e:
            logger.warning(f"Failed to load transcripts: {e}")
    
    def _load_images(self) -> None:
        """Load morphology images."""
        # Look for standard Xenium image files
        image_candidates = [
            self.data_path / "morphology.ome.tif",
            self.data_path / "morphology_focus.ome.tif",
            self.data_path / "morphology_mip.ome.tif",
        ]
        
        images = {}
        
        for img_path in image_candidates:
            if img_path.exists():
                try:
                    # For OME-TIFF files, we need special handling
                    # For now, skip complex OME-TIFF loading
                    logger.info(f"Found image: {img_path} (OME-TIFF loading not implemented)")
                    images[img_path.stem] = str(img_path)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
        
        if images:
            if 'spatial' not in self.adata.uns:
                self.adata.uns['spatial'] = {}
            self.adata.uns['spatial']['images'] = images
    
    def _merge_fields_of_view(self) -> None:
        """Merge multiple fields of view into single coordinate system."""
        if 'fov' not in self.adata.obs:
            logger.info("No field of view information found, skipping FOV merging")
            return
        
        logger.info("Merging fields of view")
        
        # Get FOV-specific coordinate offsets (simplified approach)
        fov_offsets = {}
        for fov in self.adata.obs['fov'].cat.categories:
            fov_mask = self.adata.obs['fov'] == fov
            fov_coords = self.adata.obsm['spatial'][fov_mask]
            
            # Simple grid-based offset (assumes square FOVs)
            fov_idx = int(fov) if str(fov).isdigit() else hash(fov) % 100
            offset_x = (fov_idx % 10) * 2000  # Arbitrary spacing
            offset_y = (fov_idx // 10) * 2000
            
            fov_offsets[fov] = (offset_x, offset_y)
        
        # Apply offsets to coordinates
        new_coords = self.adata.obsm['spatial'].copy()
        for fov, (offset_x, offset_y) in fov_offsets.items():
            fov_mask = self.adata.obs['fov'] == fov
            new_coords[fov_mask, 0] += offset_x
            new_coords[fov_mask, 1] += offset_y
        
        self.adata.obsm['spatial'] = new_coords
        logger.info(f"Merged {len(fov_offsets)} fields of view")
    
    def _match_cell_ids(self, metadata_ids: pd.Index, expression_ids: pd.Index) -> pd.Index:
        """Attempt to match cell IDs between datasets."""
        logger.info("Attempting cell ID matching")
        
        # Try different matching strategies
        # Strategy 1: Direct string matching
        common_ids = metadata_ids.intersection(expression_ids)
        if len(common_ids) > 0:
            return common_ids
        
        # Strategy 2: Remove prefixes/suffixes
        metadata_clean = metadata_ids.str.replace(r'^cell_', '', regex=True)
        expression_clean = expression_ids.str.replace(r'^cell_', '', regex=True)
        
        common_clean = metadata_clean.intersection(expression_clean)
        if len(common_clean) > 0:
            # Map back to original IDs
            metadata_mapping = dict(zip(metadata_clean, metadata_ids))
            expression_mapping = dict(zip(expression_clean, expression_ids))
            
            common_original = [metadata_mapping[cid] for cid in common_clean]
            return pd.Index(common_original)
        
        logger.warning("No matching cell IDs found")
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
    
    def get_transcripts(self) -> Optional[pd.DataFrame]:
        """Get transcript-level data if available."""
        return self.adata.uns.get('transcripts', None)
    
    def get_cell_boundaries(self) -> Optional[Dict[str, np.ndarray]]:
        """Get cell boundary polygons if available."""
        return self.adata.uns.get('cell_boundaries', None)
    
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
    ) -> 'XeniumDataset':
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
        plt.title(f'Xenium Spatial Plot{f" - {color}" if color else ""}')
        plt.show()
    
    def plot_cell_boundaries(self, cell_ids: Optional[List[str]] = None, **kwargs) -> None:
        """Plot cell boundaries if available."""
        boundaries = self.get_cell_boundaries()
        if boundaries is None:
            logger.warning("No cell boundaries available")
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot cell boundaries
        patches = []
        cell_ids_to_plot = cell_ids or list(boundaries.keys())[:100]  # Limit for performance
        
        for cell_id in cell_ids_to_plot:
            if cell_id in boundaries:
                vertices = boundaries[cell_id]
                polygon = Polygon(vertices, closed=True)
                patches.append(polygon)
        
        if patches:
            patch_collection = PatchCollection(patches, alpha=0.3, edgecolors='black', linewidths=0.5)
            ax.add_collection(patch_collection)
            
            # Set axis limits
            all_vertices = np.vstack([boundaries[cid] for cid in cell_ids_to_plot if cid in boundaries])
            ax.set_xlim(all_vertices[:, 0].min(), all_vertices[:, 0].max())
            ax.set_ylim(all_vertices[:, 1].min(), all_vertices[:, 1].max())
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Cell Boundaries')
        plt.show()
    
    def __len__(self) -> int:
        """Return number of cells."""
        return self.adata.n_obs
    
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Dict[str, Any]:
        """Get cell data by index."""
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))
        
        cell_data = {
            'expression': self.adata.X[idx].copy(),
            'coordinates': self.adata.obsm['spatial'][idx].copy(),
            'observations': self.adata.obs.iloc[idx].copy()
        }
        
        return cell_data
"""
MERFISH dataset loader for Multiplexed Error-Robust Fluorescence In Situ Hybridization data.
Handles MERFISH-specific data formats including codebooks, blank correction,
and z-stack processing.
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


class MERFISHDataset(BaseSpatialDataset):
    """
    Dataset loader for MERFISH spatial transcriptomics data.
    
    Handles loading from MERFISH outputs including:
    - Gene expression matrix with barcode decoding
    - Spatial coordinates in 2D/3D
    - Codebook for barcode-to-gene mapping
    - Blank probe correction
    - Z-stack information
    - Imaging metadata
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        codebook_path: Optional[Union[str, Path]] = None,
        blank_correction: bool = True,
        z_stack_projection: str = "max",
        min_counts_per_cell: int = 10,
        min_genes_per_cell: int = 5,
        **kwargs
    ):
        """
        Initialize MERFISH dataset.
        
        Args:
            data_path: Path to MERFISH data directory or file
            config: Dataset configuration
            codebook_path: Path to codebook file (if separate)
            blank_correction: Whether to apply blank probe correction
            z_stack_projection: Method for z-stack projection ('max', 'mean', 'sum')
            min_counts_per_cell: Minimum counts per cell for filtering
            min_genes_per_cell: Minimum genes per cell for filtering
            **kwargs: Additional arguments
        """
        super().__init__(config=config, **kwargs)
        
        self.data_path = Path(data_path)
        self.codebook_path = Path(codebook_path) if codebook_path else None
        self.blank_correction = blank_correction
        self.z_stack_projection = z_stack_projection
        self.min_counts_per_cell = min_counts_per_cell
        self.min_genes_per_cell = min_genes_per_cell
        
        # Initialize components
        self.preprocessor = SpatialPreprocessor(config)
        self.graph_builder = SpatialGraphBuilder(config)
        
        # Storage for MERFISH-specific data
        self.codebook = None
        self.blank_probes = None
        self.decoded_spots = None
        
        # Load data
        self._load_data()
        
        # Validate loaded data
        self._validate_spatial_data()
        
        logger.info(f"Loaded MERFISH dataset: {self.adata.shape}")
    
    def load_data(self, data_path: Union[str, Path], **kwargs) -> AnnData:
        """
        Load platform-specific data.
        
        Args:
            data_path: Path to data files
            **kwargs: Platform-specific arguments
            
        Returns:
            AnnData object with loaded data
        """
        # Store original path and parameters
        original_path = self.data_path
        original_codebook_path = self.codebook_path
        
        self.data_path = Path(data_path)
        
        # Update parameters if provided
        if 'codebook_path' in kwargs:
            self.codebook_path = Path(kwargs['codebook_path']) if kwargs['codebook_path'] else None
        if 'blank_correction' in kwargs:
            self.blank_correction = kwargs['blank_correction']
        if 'z_stack_projection' in kwargs:
            self.z_stack_projection = kwargs['z_stack_projection']
        
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
            
            # Check coordinate dimensions (MERFISH can have 2D or 3D)
            if adata.obsm['spatial'].shape[1] not in [2, 3]:
                logger.error("Spatial coordinates must be 2D or 3D for MERFISH")
                return False
            
            # Check for valid expression data
            if hasattr(adata.X, 'nnz') and adata.X.nnz == 0:
                logger.error("No expression data found")
                return False
            
            # MERFISH specific validations
            # Check for reasonable cell counts per gene (MERFISH typically has discrete counts)
            if hasattr(adata.X, 'toarray'):
                sample_data = adata.X[:100].toarray() if adata.X.shape[0] > 100 else adata.X.toarray()
            else:
                sample_data = adata.X[:100] if adata.X.shape[0] > 100 else adata.X
            
            # Check if data looks like count data (mostly integers)
            if sample_data.size > 0:
                non_integer_fraction = np.mean(sample_data != np.round(sample_data))
                if non_integer_fraction > 0.1:
                    logger.warning("Data doesn't appear to be count-based, which is unusual for MERFISH")
            
            # Check for z-coordinate if available
            if 'z_centroid' in adata.obs.columns:
                logger.info("Found z-coordinate information")
            
            # Check if we have blank probe information
            blank_genes = [gene for gene in adata.var_names if 'blank' in gene.lower()]
            if blank_genes:
                logger.info(f"Found {len(blank_genes)} potential blank probe genes")
            
            logger.info("MERFISH data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    @classmethod
    def from_merfish_folder(
        cls,
        merfish_path: Union[str, Path],
        config: Optional[SpatialDataConfig] = None,
        **kwargs
    ) -> 'MERFISHDataset':
        """
        Load MERFISH data from standard output folder.
        
        Expected folder structure:
        merfish_path/
        ├── decoded_spots.csv (or similar)
        ├── codebook.csv
        ├── cell_by_gene_matrix.csv (optional)
        ├── cell_metadata.csv (optional)
        ├── images/ (optional)
        └── analysis_metadata.json (optional)
        
        Args:
            merfish_path: Path to MERFISH output directory
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            MERFISHDataset instance
        """
        return cls(merfish_path, config=config, **kwargs)
    
    def _load_data(self) -> None:
        """Load MERFISH data from directory or file."""
        logger.info(f"Loading MERFISH data from: {self.data_path}")
        
        if self.data_path.is_file():
            # Single file provided (assume it's the main data file)
            self._load_from_file()
        else:
            # Directory provided
            self._load_from_directory()
        
        # Load codebook
        self._load_codebook()
        
        # Apply blank correction if requested
        if self.blank_correction:
            self._apply_blank_correction()
        
        # Filter low-quality cells
        self._filter_cells()
    
    def _load_from_file(self) -> None:
        """Load from single file (CSV, H5, etc.)."""
        file_ext = self.data_path.suffix.lower()
        
        if file_ext == '.csv':
            self._load_from_csv()
        elif file_ext == '.h5':
            self._load_from_h5()
        elif file_ext in ['.xlsx', '.xls']:
            self._load_from_excel()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_from_directory(self) -> None:
        """Load from directory structure."""
        # Look for common MERFISH file patterns
        data_files = [
            "decoded_spots.csv",
            "cell_by_gene_matrix.csv",
            "gene_expression_matrix.csv",
            "spots.csv",
            "transcripts.csv"
        ]
        
        main_file = None
        for filename in data_files:
            candidate = self.data_path / filename
            if candidate.exists():
                main_file = candidate
                break
        
        if main_file is None:
            # Look for any CSV files
            csv_files = list(self.data_path.glob("*.csv"))
            if csv_files:
                main_file = csv_files[0]
                logger.warning(f"No standard MERFISH files found, using: {main_file}")
            else:
                raise FileNotFoundError("No suitable MERFISH data files found")
        
        logger.info(f"Loading main data from: {main_file}")
        self.data_path = main_file
        self._load_from_csv()
        
        # Load additional metadata if available
        self._load_additional_metadata()
    
    def _load_from_csv(self) -> None:
        """Load from CSV file."""
        logger.info(f"Loading MERFISH data from CSV: {self.data_path}")
        
        try:
            data_df = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
        
        # Check if this is decoded spots or cell-by-gene matrix
        if 'gene' in data_df.columns and 'x' in data_df.columns and 'y' in data_df.columns:
            # This is decoded spots format
            self._process_decoded_spots(data_df)
        elif data_df.index.name in ['cell_id', 'cell'] or 'cell_id' in data_df.columns:
            # This is cell-by-gene matrix
            self._process_cell_matrix(data_df)
        else:
            # Try to infer format
            self._infer_and_process_format(data_df)
    
    def _load_from_h5(self) -> None:
        """Load from H5 file."""
        logger.info(f"Loading MERFISH data from H5: {self.data_path}")
        
        try:
            # Try scanpy first
            self.adata = sc.read_h5ad(self.data_path)
            
            # Ensure spatial coordinates exist
            if 'spatial' not in self.adata.obsm:
                if 'X_spatial' in self.adata.obsm:
                    self.adata.obsm['spatial'] = self.adata.obsm['X_spatial']
                else:
                    raise ValueError("No spatial coordinates found in H5 file")
                    
        except Exception as e:
            logger.warning(f"Scanpy loading failed, trying manual H5 parsing: {e}")
            self._manual_h5_parsing()
    
    def _load_from_excel(self) -> None:
        """Load from Excel file."""
        logger.info(f"Loading MERFISH data from Excel: {self.data_path}")
        
        # Load all sheets to inspect structure
        excel_data = pd.read_excel(self.data_path, sheet_name=None)
        
        # Look for main data sheet
        main_sheet = None
        for sheet_name, sheet_data in excel_data.items():
            if 'gene' in sheet_data.columns or len(sheet_data.columns) > 10:
                main_sheet = sheet_data
                break
        
        if main_sheet is None:
            raise ValueError("Could not identify main data sheet in Excel file")
        
        # Process based on detected format
        if 'gene' in main_sheet.columns and 'x' in main_sheet.columns:
            self._process_decoded_spots(main_sheet)
        else:
            self._process_cell_matrix(main_sheet)
    
    def _process_decoded_spots(self, spots_df: pd.DataFrame) -> None:
        """Process decoded spots format into cell-by-gene matrix."""
        logger.info("Processing decoded spots format")
        
        self.decoded_spots = spots_df.copy()
        
        # Required columns
        required_cols = ['gene', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in spots_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for cell assignment
        if 'cell_id' not in spots_df.columns:
            logger.info("No cell assignments found, performing cell segmentation")
            self._assign_spots_to_cells(spots_df)
        
        # Create cell-by-gene matrix
        cell_gene_matrix = spots_df.groupby(['cell_id', 'gene']).size().unstack(fill_value=0)
        
        # Create AnnData object
        self.adata = AnnData(
            X=cell_gene_matrix.values,
            obs=pd.DataFrame(index=cell_gene_matrix.index),
            var=pd.DataFrame(index=cell_gene_matrix.columns)
        )
        
        # Add spatial coordinates
        cell_coords = spots_df.groupby('cell_id')[['x', 'y']].mean()
        self.adata.obsm['spatial'] = cell_coords.loc[self.adata.obs.index].values
        
        # Store additional spot information
        if 'z' in spots_df.columns:
            z_coords = spots_df.groupby('cell_id')['z'].mean()
            self.adata.obs['z_centroid'] = z_coords.loc[self.adata.obs.index]
        
        # Store spot counts per cell
        spot_counts = spots_df.groupby('cell_id').size()
        self.adata.obs['total_spots'] = spot_counts.loc[self.adata.obs.index]
    
    def _process_cell_matrix(self, matrix_df: pd.DataFrame) -> None:
        """Process cell-by-gene matrix format."""
        logger.info("Processing cell-by-gene matrix format")
        
        # Set appropriate index
        if 'cell_id' in matrix_df.columns:
            matrix_df = matrix_df.set_index('cell_id')
        
        # Separate coordinates from gene expression
        coord_cols = [col for col in matrix_df.columns if col.lower() in ['x', 'y', 'z', 'x_centroid', 'y_centroid']]
        gene_cols = [col for col in matrix_df.columns if col not in coord_cols]
        
        if not coord_cols:
            raise ValueError("No spatial coordinate columns found")
        
        # Create AnnData object
        expression_data = matrix_df[gene_cols]
        self.adata = AnnData(
            X=expression_data.values,
            obs=pd.DataFrame(index=matrix_df.index),
            var=pd.DataFrame(index=expression_data.columns)
        )
        
        # Add spatial coordinates
        coord_data = matrix_df[coord_cols]
        if 'x' in coord_data.columns and 'y' in coord_data.columns:
            self.adata.obsm['spatial'] = coord_data[['x', 'y']].values
        elif 'x_centroid' in coord_data.columns and 'y_centroid' in coord_data.columns:
            self.adata.obsm['spatial'] = coord_data[['x_centroid', 'y_centroid']].values
        else:
            # Use first two coordinate columns
            self.adata.obsm['spatial'] = coord_data.iloc[:, :2].values
        
        # Add z-coordinate if available
        if 'z' in coord_data.columns:
            self.adata.obs['z_centroid'] = coord_data['z']
    
    def _infer_and_process_format(self, data_df: pd.DataFrame) -> None:
        """Infer data format and process accordingly."""
        logger.info("Inferring MERFISH data format")
        
        # Check dimensions and column names to infer format
        n_rows, n_cols = data_df.shape
        
        if n_rows > n_cols * 10:  # Many more rows than columns - likely spots
            logger.info("Inferred format: decoded spots (many rows)")
            # Assume first few columns are coordinates and gene info
            if n_cols >= 3:
                data_df.columns = ['gene', 'x', 'y'] + [f'col_{i}' for i in range(3, n_cols)]
                self._process_decoded_spots(data_df)
            else:
                raise ValueError("Insufficient columns for decoded spots format")
        else:  # Fewer rows - likely cell-by-gene matrix
            logger.info("Inferred format: cell-by-gene matrix")
            # Assume rows are cells, columns are genes with some coordinate columns
            self._process_cell_matrix(data_df)
    
    def _assign_spots_to_cells(self, spots_df: pd.DataFrame) -> None:
        """Assign spots to cells using simple spatial clustering."""
        logger.info("Assigning spots to cells using spatial clustering")
        
        from sklearn.cluster import DBSCAN
        
        # Get coordinates
        coords = spots_df[['x', 'y']].values
        
        # Use DBSCAN for cell segmentation
        # Parameters might need tuning based on data
        clustering = DBSCAN(eps=5.0, min_samples=3).fit(coords)
        
        # Assign cell IDs (noise points get -1)
        spots_df['cell_id'] = clustering.labels_
        
        # Remove noise spots
        spots_df = spots_df[spots_df['cell_id'] >= 0].copy()
        
        # Rename cell IDs to be more meaningful
        spots_df['cell_id'] = spots_df['cell_id'].astype(str)
        spots_df['cell_id'] = 'cell_' + spots_df['cell_id']
        
        logger.info(f"Assigned spots to {spots_df['cell_id'].nunique()} cells")
    
    def _load_codebook(self) -> None:
        """Load MERFISH codebook for barcode-to-gene mapping."""
        # Look for codebook file
        codebook_candidates = []
        
        if self.codebook_path:
            codebook_candidates.append(self.codebook_path)
        
        if self.data_path.is_dir():
            codebook_candidates.extend([
                self.data_path / "codebook.csv",
                self.data_path / "codebook.xlsx",
                self.data_path / "gene_codebook.csv",
                self.data_path / "barcode_map.csv"
            ])
        else:
            # Look in same directory as data file
            data_dir = self.data_path.parent
            codebook_candidates.extend([
                data_dir / "codebook.csv",
                data_dir / "codebook.xlsx",
            ])
        
        codebook_file = None
        for candidate in codebook_candidates:
            if candidate.exists():
                codebook_file = candidate
                break
        
        if codebook_file is None:
            logger.warning("No codebook file found, skipping codebook loading")
            return
        
        logger.info(f"Loading codebook from: {codebook_file}")
        
        try:
            if codebook_file.suffix == '.xlsx':
                self.codebook = pd.read_excel(codebook_file)
            else:
                self.codebook = pd.read_csv(codebook_file)
            
            # Identify blank probes
            if 'gene' in self.codebook.columns:
                blank_mask = self.codebook['gene'].str.contains('blank', case=False, na=False)
                self.blank_probes = self.codebook[blank_mask]['gene'].tolist()
                logger.info(f"Identified {len(self.blank_probes)} blank probes")
            
        except Exception as e:
            logger.warning(f"Failed to load codebook: {e}")
    
    def _apply_blank_correction(self) -> None:
        """Apply blank probe correction to remove background signal."""
        if self.blank_probes is None or not self.blank_probes:
            logger.warning("No blank probes available for correction")
            return
        
        logger.info("Applying blank probe correction")
        
        # Find blank probes in our data
        available_blanks = [gene for gene in self.blank_probes if gene in self.adata.var_names]
        
        if not available_blanks:
            logger.warning("No blank probes found in expression data")
            return
        
        # Calculate blank probe statistics
        blank_data = self.adata[:, available_blanks].X
        blank_threshold = np.percentile(blank_data, 95)  # Use 95th percentile as threshold
        
        # Apply correction by subtracting blank threshold
        corrected_data = self.adata.X.copy()
        corrected_data = np.maximum(corrected_data - blank_threshold, 0)
        
        # Update data
        self.adata.X = corrected_data
        
        # Remove blank probes from final dataset
        non_blank_genes = [gene for gene in self.adata.var_names if gene not in self.blank_probes]
        self.adata = self.adata[:, non_blank_genes].copy()
        
        logger.info(f"Applied blank correction with threshold {blank_threshold:.2f}")
    
    def _filter_cells(self) -> None:
        """Filter low-quality cells."""
        logger.info("Filtering low-quality cells")
        
        # Calculate QC metrics
        self.adata.var['total_counts'] = np.array(self.adata.X.sum(axis=0)).flatten()
        self.adata.obs['total_counts'] = np.array(self.adata.X.sum(axis=1)).flatten()
        self.adata.obs['n_genes'] = (self.adata.X > 0).sum(axis=1)
        
        # Filter cells
        n_cells_before = self.adata.n_obs
        
        cell_filter = (
            (self.adata.obs['total_counts'] >= self.min_counts_per_cell) &
            (self.adata.obs['n_genes'] >= self.min_genes_per_cell)
        )
        
        self.adata = self.adata[cell_filter].copy()
        
        n_cells_after = self.adata.n_obs
        logger.info(f"Filtered cells: {n_cells_before} -> {n_cells_after}")
    
    def _load_additional_metadata(self) -> None:
        """Load additional metadata files if available."""
        if not self.data_path.parent.is_dir():
            return
        
        metadata_dir = self.data_path.parent
        
        # Look for cell metadata
        metadata_files = [
            "cell_metadata.csv",
            "metadata.csv",
            "analysis_metadata.json"
        ]
        
        for filename in metadata_files:
            filepath = metadata_dir / filename
            if filepath.exists():
                self._load_metadata_file(filepath)
    
    def _load_metadata_file(self, filepath: Path) -> None:
        """Load individual metadata file."""
        logger.info(f"Loading metadata from: {filepath}")
        
        try:
            if filepath.suffix == '.json':
                with open(filepath) as f:
                    metadata = json.load(f)
                # Store in uns
                self.adata.uns.update(metadata)
            elif filepath.suffix == '.csv':
                metadata_df = pd.read_csv(filepath, index_col=0)
                # Match with current cells
                common_cells = metadata_df.index.intersection(self.adata.obs.index)
                if len(common_cells) > 0:
                    for col in metadata_df.columns:
                        self.adata.obs[col] = metadata_df.loc[common_cells, col]
        except Exception as e:
            logger.warning(f"Failed to load metadata from {filepath}: {e}")
    
    def _manual_h5_parsing(self) -> None:
        """Manual parsing of H5 file structure."""
        logger.info("Attempting manual H5 parsing")
        
        with h5py.File(self.data_path, 'r') as f:
            # Explore H5 structure
            def explore_h5(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.info(f"Dataset: {name}, shape: {obj.shape}")
                else:
                    logger.info(f"Group: {name}")
            
            f.visititems(explore_h5)
            
            # Try to find expression and coordinate data
            # This would need to be customized based on specific H5 structure
            raise NotImplementedError("Manual H5 parsing needs customization for specific format")
    
    def _validate_spatial_data(self) -> None:
        """Validate that spatial data is properly loaded."""
        if 'spatial' not in self.adata.obsm:
            raise ValueError("No spatial coordinates found")
            
        if self.adata.obsm['spatial'].shape[1] not in [2, 3]:
            raise ValueError("Spatial coordinates must be 2D or 3D")
            
        # Check for valid coordinates
        coords = self.adata.obsm['spatial']
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            logger.warning("Invalid coordinates detected, cleaning data")
            valid_mask = ~(np.isnan(coords).any(axis=1) | np.isinf(coords).any(axis=1))
            self.adata = self.adata[valid_mask].copy()
    
    def get_spatial_coordinates(self) -> np.ndarray:
        """Get spatial coordinates."""
        return self.adata.obsm['spatial'].copy()
    
    def get_decoded_spots(self) -> Optional[pd.DataFrame]:
        """Get decoded spots data if available."""
        return self.decoded_spots
    
    def get_codebook(self) -> Optional[pd.DataFrame]:
        """Get codebook if available."""
        return self.codebook
    
    def get_blank_probes(self) -> Optional[List[str]]:
        """Get list of blank probe names."""
        return self.blank_probes
    
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
    ) -> 'MERFISHDataset':
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
        
        # Handle 3D coordinates by projecting to 2D
        if coords.shape[1] == 3:
            coords = coords[:, :2]  # Use X, Y coordinates
        
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
        plt.title(f'MERFISH Spatial Plot{f" - {color}" if color else ""}')
        plt.show()
    
    def plot_decoded_spots(self, genes: Optional[List[str]] = None, max_spots: int = 10000, **kwargs) -> None:
        """Plot decoded spots if available."""
        if self.decoded_spots is None:
            logger.warning("No decoded spots data available")
            return
        
        import matplotlib.pyplot as plt
        
        spots_to_plot = self.decoded_spots.copy()
        
        # Filter by genes if specified
        if genes:
            spots_to_plot = spots_to_plot[spots_to_plot['gene'].isin(genes)]
        
        # Subsample for performance
        if len(spots_to_plot) > max_spots:
            spots_to_plot = spots_to_plot.sample(max_spots)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by gene
        if 'gene' in spots_to_plot.columns:
            unique_genes = spots_to_plot['gene'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genes)))
            color_map = dict(zip(unique_genes, colors))
            
            for gene in unique_genes:
                gene_spots = spots_to_plot[spots_to_plot['gene'] == gene]
                ax.scatter(gene_spots['x'], gene_spots['y'], 
                          c=[color_map[gene]], label=gene, alpha=0.6, s=1)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(spots_to_plot['x'], spots_to_plot['y'], alpha=0.6, s=1)
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('MERFISH Decoded Spots')
        plt.tight_layout()
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
"""
Validation utilities for spatial transcriptomics data and model inputs.
Implements comprehensive data validation, quality control, and error checking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pathlib import Path

logger = logging.getLogger(__name__)


class SpatialDataValidator:
    """
    Comprehensive validator for spatial transcriptomics data.
    
    Validates:
    - Data formats and structures
    - Spatial coordinates
    - Gene expression matrices
    - Metadata consistency
    - Quality control metrics
    """
    
    def __init__(
        self,
        min_cells_per_gene: int = 3,
        min_genes_per_cell: int = 100,
        max_genes_per_cell: int = 10000,
        min_counts_per_cell: int = 500,
        max_mitochondrial_pct: float = 20.0,
        max_ribosomal_pct: float = 50.0
    ):
        """
        Initialize validator with quality control thresholds.
        
        Args:
            min_cells_per_gene: Minimum cells expressing each gene
            min_genes_per_cell: Minimum genes per cell
            max_genes_per_cell: Maximum genes per cell
            min_counts_per_cell: Minimum total counts per cell
            max_mitochondrial_pct: Maximum mitochondrial gene percentage
            max_ribosomal_pct: Maximum ribosomal gene percentage
        """
        self.min_cells_per_gene = min_cells_per_gene
        self.min_genes_per_cell = min_genes_per_cell
        self.max_genes_per_cell = max_genes_per_cell
        self.min_counts_per_cell = min_counts_per_cell
        self.max_mitochondrial_pct = max_mitochondrial_pct
        self.max_ribosomal_pct = max_ribosomal_pct
        
        logger.info("Initialized SpatialDataValidator")
    
    def validate_adata(self, adata: AnnData, fix_issues: bool = False) -> Dict[str, Any]:
        """
        Comprehensive validation of AnnData object.
        
        Args:
            adata: AnnData object to validate
            fix_issues: Whether to automatically fix issues when possible
            
        Returns:
            Validation report with issues and recommendations
        """
        logger.info("Validating AnnData object")
        
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'quality_metrics': {},
            'fixes_applied': []
        }
        
        try:
            # Basic structure validation
            self._validate_basic_structure(adata, validation_report)
            
            # Expression matrix validation
            self._validate_expression_matrix(adata, validation_report, fix_issues)
            
            # Spatial coordinates validation
            self._validate_spatial_coordinates(adata, validation_report, fix_issues)
            
            # Gene and cell filtering validation
            self._validate_filtering_criteria(adata, validation_report, fix_issues)
            
            # Quality control metrics
            self._compute_quality_metrics(adata, validation_report)
            
            # Metadata validation
            self._validate_metadata(adata, validation_report)
            
            # Final validation status
            validation_report['is_valid'] = not validation_report['errors']
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Validation failed: {e}")
        
        # Log summary
        self._log_validation_summary(validation_report)
        
        return validation_report
    
    def _validate_basic_structure(self, adata: AnnData, report: Dict[str, Any]) -> None:
        """Validate basic AnnData structure."""
        logger.debug("Validating basic structure")
        
        # Check if AnnData object is properly initialized
        if adata.X is None:
            report['errors'].append("Expression matrix (X) is None")
            return
        
        # Check dimensions
        if adata.n_obs == 0:
            report['errors'].append("No cells (observations) found")
        
        if adata.n_vars == 0:
            report['errors'].append("No genes (variables) found")
        
        # Check for empty expression matrix
        if hasattr(adata.X, 'nnz'):  # Sparse matrix
            if adata.X.nnz == 0:
                report['warnings'].append("Expression matrix contains no non-zero values")
        else:  # Dense matrix
            if np.all(adata.X == 0):
                report['warnings'].append("Expression matrix contains only zeros")
        
        # Check for proper indexing
        if adata.obs.index.duplicated().any():
            report['errors'].append("Duplicate cell (observation) indices found")
        
        if adata.var.index.duplicated().any():
            report['errors'].append("Duplicate gene (variable) indices found")
        
        logger.debug(f"Basic structure validation completed: {adata.shape}")
    
    def _validate_expression_matrix(self, adata: AnnData, report: Dict[str, Any], fix_issues: bool) -> None:
        """Validate gene expression matrix."""
        logger.debug("Validating expression matrix")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Check for invalid values
        if np.any(np.isnan(X)):
            if fix_issues:
                # Replace NaN with 0
                if hasattr(adata.X, 'toarray'):
                    adata.X.data[np.isnan(adata.X.data)] = 0
                else:
                    adata.X[np.isnan(adata.X)] = 0
                report['fixes_applied'].append("Replaced NaN values with 0")
            else:
                report['errors'].append("Expression matrix contains NaN values")
        
        if np.any(np.isinf(X)):
            if fix_issues:
                # Replace inf with maximum finite value
                max_finite = np.max(X[np.isfinite(X)])
                if hasattr(adata.X, 'toarray'):
                    adata.X.data[np.isinf(adata.X.data)] = max_finite
                else:
                    adata.X[np.isinf(adata.X)] = max_finite
                report['fixes_applied'].append("Replaced infinite values with maximum finite value")
            else:
                report['errors'].append("Expression matrix contains infinite values")
        
        # Check for negative values
        if np.any(X < 0):
            if fix_issues:
                # Set negative values to 0
                if hasattr(adata.X, 'toarray'):
                    adata.X.data[adata.X.data < 0] = 0
                else:
                    adata.X[adata.X < 0] = 0
                report['fixes_applied'].append("Set negative expression values to 0")
            else:
                report['warnings'].append("Expression matrix contains negative values")
        
        # Check data type
        if not np.issubdtype(X.dtype, np.floating):
            report['warnings'].append(f"Expression matrix has non-float dtype: {X.dtype}")
        
        # Check for extremely high values (potential outliers)
        max_value = np.max(X)
        if max_value > 1000:
            report['warnings'].append(f"Very high expression values detected (max: {max_value:.2f})")
        
        logger.debug("Expression matrix validation completed")
    
    def _validate_spatial_coordinates(self, adata: AnnData, report: Dict[str, Any], fix_issues: bool) -> None:
        """Validate spatial coordinates."""
        logger.debug("Validating spatial coordinates")
        
        if 'spatial' not in adata.obsm:
            report['errors'].append("No spatial coordinates found in obsm['spatial']")
            return
        
        coords = adata.obsm['spatial']
        
        # Check coordinate dimensions
        if coords.shape[1] not in [2, 3]:
            report['errors'].append(f"Spatial coordinates must be 2D or 3D, got {coords.shape[1]}D")
        
        # Check for invalid coordinates
        if np.any(np.isnan(coords)):
            if fix_issues:
                # Remove cells with invalid coordinates
                valid_mask = ~np.isnan(coords).any(axis=1)
                adata._inplace_subset_obs(valid_mask)
                report['fixes_applied'].append(f"Removed {np.sum(~valid_mask)} cells with invalid coordinates")
            else:
                report['errors'].append("Spatial coordinates contain NaN values")
        
        if np.any(np.isinf(coords)):
            if fix_issues:
                # Remove cells with infinite coordinates
                valid_mask = ~np.isinf(coords).any(axis=1)
                adata._inplace_subset_obs(valid_mask)
                report['fixes_applied'].append(f"Removed {np.sum(~valid_mask)} cells with infinite coordinates")
            else:
                report['errors'].append("Spatial coordinates contain infinite values")
        
        # Check coordinate range
        coord_ranges = np.ptp(coords, axis=0)
        if np.any(coord_ranges == 0):
            report['warnings'].append("Some spatial dimensions have zero range")
        
        # Check for duplicated coordinates
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < len(coords):
            duplicate_fraction = 1 - len(unique_coords) / len(coords)
            if duplicate_fraction > 0.1:  # More than 10% duplicates
                report['warnings'].append(f"High fraction of duplicate coordinates: {duplicate_fraction:.2%}")
            else:
                report['recommendations'].append(f"Small fraction of duplicate coordinates: {duplicate_fraction:.2%}")
        
        logger.debug("Spatial coordinates validation completed")
    
    def _validate_filtering_criteria(self, adata: AnnData, report: Dict[str, Any], fix_issues: bool) -> None:
        """Validate gene and cell filtering criteria."""
        logger.debug("Validating filtering criteria")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Gene filtering
        cells_per_gene = np.sum(X > 0, axis=0)
        genes_to_filter = cells_per_gene < self.min_cells_per_gene
        
        if np.any(genes_to_filter):
            n_genes_filter = np.sum(genes_to_filter)
            if fix_issues:
                adata._inplace_subset_var(~genes_to_filter)
                report['fixes_applied'].append(f"Filtered {n_genes_filter} genes with < {self.min_cells_per_gene} expressing cells")
            else:
                report['recommendations'].append(f"Consider filtering {n_genes_filter} genes with < {self.min_cells_per_gene} expressing cells")
        
        # Cell filtering
        genes_per_cell = np.sum(X > 0, axis=1)
        total_counts_per_cell = np.sum(X, axis=1)
        
        # Low gene count cells
        low_gene_cells = genes_per_cell < self.min_genes_per_cell
        if np.any(low_gene_cells):
            n_low_gene_cells = np.sum(low_gene_cells)
            if fix_issues:
                adata._inplace_subset_obs(~low_gene_cells)
                report['fixes_applied'].append(f"Filtered {n_low_gene_cells} cells with < {self.min_genes_per_cell} genes")
            else:
                report['recommendations'].append(f"Consider filtering {n_low_gene_cells} cells with < {self.min_genes_per_cell} genes")
        
        # High gene count cells (potential doublets)
        high_gene_cells = genes_per_cell > self.max_genes_per_cell
        if np.any(high_gene_cells):
            n_high_gene_cells = np.sum(high_gene_cells)
            report['warnings'].append(f"{n_high_gene_cells} cells have > {self.max_genes_per_cell} genes (potential doublets)")
        
        # Low count cells
        low_count_cells = total_counts_per_cell < self.min_counts_per_cell
        if np.any(low_count_cells):
            n_low_count_cells = np.sum(low_count_cells)
            if fix_issues:
                adata._inplace_subset_obs(~low_count_cells)
                report['fixes_applied'].append(f"Filtered {n_low_count_cells} cells with < {self.min_counts_per_cell} total counts")
            else:
                report['recommendations'].append(f"Consider filtering {n_low_count_cells} cells with < {self.min_counts_per_cell} total counts")
        
        logger.debug("Filtering criteria validation completed")
    
    def _compute_quality_metrics(self, adata: AnnData, report: Dict[str, Any]) -> None:
        """Compute quality control metrics."""
        logger.debug("Computing quality metrics")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        metrics = {}
        
        # Basic metrics
        metrics['n_cells'] = adata.n_obs
        metrics['n_genes'] = adata.n_vars
        metrics['total_counts'] = np.sum(X)
        metrics['mean_counts_per_cell'] = np.mean(np.sum(X, axis=1))
        metrics['median_counts_per_cell'] = np.median(np.sum(X, axis=1))
        metrics['mean_genes_per_cell'] = np.mean(np.sum(X > 0, axis=1))
        metrics['median_genes_per_cell'] = np.median(np.sum(X > 0, axis=1))
        
        # Gene expression statistics
        gene_means = np.mean(X, axis=0)
        metrics['mean_expression_per_gene'] = np.mean(gene_means)
        metrics['median_expression_per_gene'] = np.median(gene_means)
        metrics['highly_expressed_genes'] = np.sum(gene_means > np.percentile(gene_means, 95))
        
        # Sparsity
        if hasattr(adata.X, 'nnz'):
            sparsity = 1 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
        else:
            sparsity = np.mean(X == 0)
        metrics['sparsity'] = sparsity
        
        # Mitochondrial and ribosomal gene percentages
        mito_genes = self._identify_mitochondrial_genes(adata.var_names)
        ribo_genes = self._identify_ribosomal_genes(adata.var_names)
        
        if len(mito_genes) > 0:
            mito_counts = np.sum(X[:, mito_genes], axis=1)
            total_counts = np.sum(X, axis=1)
            mito_pct = (mito_counts / total_counts) * 100
            metrics['mean_mitochondrial_pct'] = np.mean(mito_pct)
            metrics['cells_high_mito'] = np.sum(mito_pct > self.max_mitochondrial_pct)
        
        if len(ribo_genes) > 0:
            ribo_counts = np.sum(X[:, ribo_genes], axis=1)
            total_counts = np.sum(X, axis=1)
            ribo_pct = (ribo_counts / total_counts) * 100
            metrics['mean_ribosomal_pct'] = np.mean(ribo_pct)
            metrics['cells_high_ribo'] = np.sum(ribo_pct > self.max_ribosomal_pct)
        
        # Spatial metrics if coordinates available
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
            spatial_range = np.ptp(coords, axis=0)
            metrics['spatial_range_x'] = spatial_range[0]
            metrics['spatial_range_y'] = spatial_range[1]
            if coords.shape[1] > 2:
                metrics['spatial_range_z'] = spatial_range[2]
            
            # Spatial density
            from scipy.spatial.distance import pdist
            if len(coords) > 1:
                distances = pdist(coords)
                metrics['mean_nearest_neighbor_distance'] = np.mean(np.sort(distances)[:len(coords)])
        
        report['quality_metrics'] = metrics
        logger.debug("Quality metrics computation completed")
    
    def _identify_mitochondrial_genes(self, gene_names: pd.Index) -> List[int]:
        """Identify mitochondrial genes."""
        mito_patterns = ['MT-', 'mt-', 'MT_', 'mt_']
        mito_indices = []
        
        for i, gene in enumerate(gene_names):
            if any(gene.startswith(pattern) for pattern in mito_patterns):
                mito_indices.append(i)
        
        return mito_indices
    
    def _identify_ribosomal_genes(self, gene_names: pd.Index) -> List[int]:
        """Identify ribosomal genes."""
        ribo_patterns = ['RPS', 'RPL', 'rps', 'rpl']
        ribo_indices = []
        
        for i, gene in enumerate(gene_names):
            if any(gene.startswith(pattern) for pattern in ribo_patterns):
                ribo_indices.append(i)
        
        return ribo_indices
    
    def _validate_metadata(self, adata: AnnData, report: Dict[str, Any]) -> None:
        """Validate metadata consistency."""
        logger.debug("Validating metadata")
        
        # Check for required metadata
        if adata.obs.empty:
            report['warnings'].append("No cell metadata (obs) found")
        
        if adata.var.empty:
            report['warnings'].append("No gene metadata (var) found")
        
        # Check for common metadata columns
        recommended_obs_columns = ['total_counts', 'n_genes', 'cell_type']
        recommended_var_columns = ['gene_ids', 'feature_types', 'highly_variable']
        
        missing_obs = [col for col in recommended_obs_columns if col not in adata.obs.columns]
        if missing_obs:
            report['recommendations'].append(f"Consider adding obs columns: {missing_obs}")
        
        missing_var = [col for col in recommended_var_columns if col not in adata.var.columns]
        if missing_var:
            report['recommendations'].append(f"Consider adding var columns: {missing_var}")
        
        # Check for categorical data types
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'object' and adata.obs[col].nunique() < 100:
                report['recommendations'].append(f"Consider converting obs['{col}'] to categorical")
        
        logger.debug("Metadata validation completed")
    
    def _log_validation_summary(self, report: Dict[str, Any]) -> None:
        """Log validation summary."""
        status = "PASSED" if report['is_valid'] else "FAILED"
        logger.info(f"Validation {status}")
        
        if report['errors']:
            logger.error(f"Errors found: {len(report['errors'])}")
            for error in report['errors']:
                logger.error(f"  - {error}")
        
        if report['warnings']:
            logger.warning(f"Warnings: {len(report['warnings'])}")
            for warning in report['warnings']:
                logger.warning(f"  - {warning}")
        
        if report['fixes_applied']:
            logger.info(f"Fixes applied: {len(report['fixes_applied'])}")
            for fix in report['fixes_applied']:
                logger.info(f"  - {fix}")


class ModelInputValidator:
    """
    Validator for model inputs and configurations.
    """
    
    def __init__(self):
        logger.info("Initialized ModelInputValidator")
    
    def validate_model_inputs(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Validate model input tensors.
        
        Args:
            gene_expression: Gene expression tensor
            spatial_coords: Spatial coordinates tensor
            edge_index: Edge index tensor
            edge_attr: Edge attribute tensor
            
        Returns:
            Validation report
        """
        logger.debug("Validating model inputs")
        
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate gene expression
        self._validate_gene_expression_tensor(gene_expression, report)
        
        # Validate spatial coordinates
        self._validate_spatial_coords_tensor(spatial_coords, report)
        
        # Validate edge tensors
        self._validate_edge_tensors(edge_index, edge_attr, report)
        
        # Check consistency between tensors
        self._validate_tensor_consistency(
            gene_expression, spatial_coords, edge_index, report
        )
        
        report['is_valid'] = not report['errors']
        
        return report
    
    def _validate_gene_expression_tensor(self, tensor: torch.Tensor, report: Dict[str, Any]) -> None:
        """Validate gene expression tensor."""
        if tensor.dim() != 2:
            report['errors'].append(f"Gene expression must be 2D, got {tensor.dim()}D")
        
        if torch.any(torch.isnan(tensor)):
            report['errors'].append("Gene expression contains NaN values")
        
        if torch.any(torch.isinf(tensor)):
            report['errors'].append("Gene expression contains infinite values")
        
        if torch.any(tensor < 0):
            report['warnings'].append("Gene expression contains negative values")
        
        if tensor.dtype not in [torch.float32, torch.float64]:
            report['warnings'].append(f"Gene expression has non-float dtype: {tensor.dtype}")
    
    def _validate_spatial_coords_tensor(self, tensor: torch.Tensor, report: Dict[str, Any]) -> None:
        """Validate spatial coordinates tensor."""
        if tensor.dim() != 2:
            report['errors'].append(f"Spatial coordinates must be 2D, got {tensor.dim()}D")
        
        if tensor.size(1) not in [2, 3]:
            report['errors'].append(f"Spatial coordinates must have 2 or 3 columns, got {tensor.size(1)}")
        
        if torch.any(torch.isnan(tensor)):
            report['errors'].append("Spatial coordinates contain NaN values")
        
        if torch.any(torch.isinf(tensor)):
            report['errors'].append("Spatial coordinates contain infinite values")
    
    def _validate_edge_tensors(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        report: Dict[str, Any]
    ) -> None:
        """Validate edge tensors."""
        if edge_index.dim() != 2:
            report['errors'].append(f"Edge index must be 2D, got {edge_index.dim()}D")
        
        if edge_index.size(0) != 2:
            report['errors'].append(f"Edge index must have 2 rows, got {edge_index.size(0)}")
        
        if edge_index.dtype != torch.long:
            report['errors'].append(f"Edge index must have long dtype, got {edge_index.dtype}")
        
        if torch.any(edge_index < 0):
            report['errors'].append("Edge index contains negative values")
        
        if edge_attr is not None:
            if edge_attr.dim() != 2:
                report['errors'].append(f"Edge attributes must be 2D, got {edge_attr.dim()}D")
            
            if edge_attr.size(0) != edge_index.size(1):
                report['errors'].append("Edge attributes and edge index have inconsistent sizes")
            
            if torch.any(torch.isnan(edge_attr)):
                report['errors'].append("Edge attributes contain NaN values")
            
            if torch.any(torch.isinf(edge_attr)):
                report['errors'].append("Edge attributes contain infinite values")
    
    def _validate_tensor_consistency(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        report: Dict[str, Any]
    ) -> None:
        """Validate consistency between tensors."""
        n_cells = gene_expression.size(0)
        
        if spatial_coords.size(0) != n_cells:
            report['errors'].append("Gene expression and spatial coordinates have different number of cells")
        
        if torch.max(edge_index) >= n_cells:
            report['errors'].append("Edge index contains node indices larger than number of cells")


def validate_file_format(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate file format and basic structure.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Validation report
    """
    file_path = Path(file_path)
    
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    # Check if file exists
    if not file_path.exists():
        report['is_valid'] = False
        report['errors'].append(f"File does not exist: {file_path}")
        return report
    
    # Get file info
    report['file_info'] = {
        'path': str(file_path),
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'suffix': file_path.suffix
    }
    
    # Validate based on file type
    if file_path.suffix == '.h5':
        _validate_h5_file(file_path, report)
    elif file_path.suffix == '.h5ad':
        _validate_h5ad_file(file_path, report)
    elif file_path.suffix in ['.csv', '.tsv']:
        _validate_csv_file(file_path, report)
    elif file_path.suffix == '.zarr':
        _validate_zarr_file(file_path, report)
    else:
        report['warnings'].append(f"Unknown file format: {file_path.suffix}")
    
    return report


def _validate_h5_file(file_path: Path, report: Dict[str, Any]) -> None:
    """Validate H5 file structure."""
    try:
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Check for common datasets
            if 'X' not in f and 'matrix' not in f:
                report['warnings'].append("No expression matrix found (expected 'X' or 'matrix')")
            
            # Check for spatial data
            has_spatial = False
            if 'obsm' in f and 'spatial' in f['obsm']:
                has_spatial = True
            elif 'spatial' in f:
                has_spatial = True
            
            if not has_spatial:
                report['warnings'].append("No spatial coordinates found")
            
            report['file_info']['datasets'] = list(f.keys())
            
    except Exception as e:
        report['errors'].append(f"Failed to read H5 file: {str(e)}")


def _validate_h5ad_file(file_path: Path, report: Dict[str, Any]) -> None:
    """Validate H5AD file structure."""
    try:
        import scanpy as sc
        
        # Try to read file info without loading full data
        adata = sc.read_h5ad(file_path, backed='r')
        
        report['file_info']['n_obs'] = adata.n_obs
        report['file_info']['n_vars'] = adata.n_vars
        
        if 'spatial' not in adata.obsm:
            report['warnings'].append("No spatial coordinates found in obsm['spatial']")
        
    except Exception as e:
        report['errors'].append(f"Failed to read H5AD file: {str(e)}")


def _validate_csv_file(file_path: Path, report: Dict[str, Any]) -> None:
    """Validate CSV file structure."""
    try:
        # Read first few rows to check structure
        df = pd.read_csv(file_path, nrows=5)
        
        report['file_info']['n_columns'] = len(df.columns)
        report['file_info']['columns'] = list(df.columns)
        
        # Check for spatial columns
        spatial_cols = [col for col in df.columns if col.lower() in ['x', 'y', 'spatial_x', 'spatial_y']]
        if len(spatial_cols) < 2:
            report['warnings'].append("No spatial coordinate columns found")
        
    except Exception as e:
        report['errors'].append(f"Failed to read CSV file: {str(e)}")


def _validate_zarr_file(file_path: Path, report: Dict[str, Any]) -> None:
    """Validate Zarr file structure."""
    try:
        import zarr
        
        store = zarr.open(file_path, mode='r')
        
        if 'X' not in store:
            report['warnings'].append("No expression matrix found (expected 'X')")
        
        if 'spatial' not in store:
            report['warnings'].append("No spatial coordinates found")
        
        report['file_info']['arrays'] = list(store.keys())
        
    except Exception as e:
        report['errors'].append(f"Failed to read Zarr file: {str(e)}")


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Validation report
    """
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Required fields
    required_fields = ['num_genes', 'hidden_dim']
    for field in required_fields:
        if field not in config:
            report['errors'].append(f"Missing required field: {field}")
    
    # Validate numeric ranges
    if 'hidden_dim' in config:
        if config['hidden_dim'] <= 0:
            report['errors'].append("hidden_dim must be positive")
        elif config['hidden_dim'] % 64 != 0:
            report['recommendations'].append("hidden_dim should be divisible by 64 for optimal performance")
    
    if 'num_layers' in config:
        if config['num_layers'] <= 0:
            report['errors'].append("num_layers must be positive")
        elif config['num_layers'] > 50:
            report['warnings'].append("Very deep model (>50 layers) may be difficult to train")
    
    if 'num_heads' in config:
        if 'hidden_dim' in config and config['hidden_dim'] % config['num_heads'] != 0:
            report['errors'].append("hidden_dim must be divisible by num_heads")
    
    if 'dropout' in config:
        if not (0 <= config['dropout'] <= 1):
            report['errors'].append("dropout must be between 0 and 1")
    
    report['is_valid'] = not report['errors']
    
    return report
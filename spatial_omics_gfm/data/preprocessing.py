"""
Data preprocessing utilities for spatial transcriptomics data.
Handles normalization, quality control, and feature selection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from anndata import AnnData

from .base import SpatialDataConfig

logger = logging.getLogger(__name__)


class SpatialPreprocessor:
    """
    Comprehensive preprocessing pipeline for spatial transcriptomics data.
    
    Handles quality control, normalization, feature selection, and data validation
    across different spatial platforms.
    """
    
    def __init__(self, config: Optional[SpatialDataConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or SpatialDataConfig()
        self._setup_scanpy()
    
    def _setup_scanpy(self) -> None:
        """Configure scanpy settings for optimal performance."""
        sc.settings.verbosity = 1  # Reduce verbosity
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        
    def preprocess_spatial_data(
        self,
        adata: AnnData,
        normalize: bool = True,
        log_transform: bool = True,
        highly_variable_genes: bool = True,
        spatial_neighbors: bool = True
    ) -> AnnData:
        """
        Complete preprocessing pipeline for spatial data.
        
        Args:
            adata: Annotated data object
            normalize: Whether to normalize gene expression
            log_transform: Whether to apply log transformation
            highly_variable_genes: Whether to select highly variable genes
            spatial_neighbors: Whether to compute spatial neighbors
            
        Returns:
            Processed AnnData object
        """
        logger.info("Starting spatial data preprocessing pipeline")
        
        # Quality control
        adata = self.quality_control(adata)
        
        # Gene filtering
        adata = self.filter_genes(adata)
        
        # Cell filtering  
        adata = self.filter_cells(adata)
        
        # Store raw data
        adata.raw = adata.copy()
        
        # Normalization
        if normalize:
            adata = self.normalize_expression(adata)
        
        # Log transformation
        if log_transform:
            adata = self.log_transform(adata)
            
        # Feature selection
        if highly_variable_genes:
            adata = self.select_highly_variable_genes(adata)
            
        # Spatial neighbors
        if spatial_neighbors and 'spatial' in adata.obsm:
            adata = self.compute_spatial_neighbors(adata)
            
        logger.info(f"Preprocessing complete. Final shape: {adata.shape}")
        return adata
    
    def quality_control(self, adata: AnnData) -> AnnData:
        """
        Compute quality control metrics.
        
        Args:
            adata: Input data
            
        Returns:
            Data with QC metrics added
        """
        logger.info("Computing quality control metrics")
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # Mitochondrial genes
        adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))  # Ribosomal
        
        # Per-cell metrics
        sc.pp.calculate_qc_metrics(
            adata, 
            percent_top=None, 
            log1p=False, 
            inplace=True,
            var_type='genes'
        )
        
        # Add custom metrics
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
        if sp.issparse(adata.X):
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['total_counts'] = adata.X.sum(axis=1)
            
        # Mitochondrial gene percentage
        if adata.var['mt'].sum() > 0:
            sc.pp.calculate_qc_metrics(
                adata, 
                qc_vars=['mt'], 
                percent_top=None, 
                log1p=False, 
                inplace=True
            )
        else:
            adata.obs['pct_counts_mt'] = 0
            
        return adata
    
    def filter_genes(
        self, 
        adata: AnnData, 
        min_cells: Optional[int] = None,
        min_counts: Optional[int] = None
    ) -> AnnData:
        """
        Filter genes based on expression criteria.
        
        Args:
            adata: Input data
            min_cells: Minimum number of cells expressing gene
            min_counts: Minimum total counts per gene
            
        Returns:
            Filtered data
        """
        min_cells = min_cells or self.config.quality_control.min_cells_per_gene
        min_counts = min_counts or self.config.quality_control.get('min_counts_per_gene', 0)
        
        logger.info(f"Filtering genes: min_cells={min_cells}, min_counts={min_counts}")
        
        n_genes_before = adata.n_vars
        
        # Filter by minimum cells
        if min_cells > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells)
            
        # Filter by minimum counts
        if min_counts > 0:
            gene_counts = np.array(adata.X.sum(axis=0)).flatten()
            gene_mask = gene_counts >= min_counts
            adata = adata[:, gene_mask].copy()
            
        n_genes_after = adata.n_vars
        logger.info(f"Filtered {n_genes_before - n_genes_after} genes. Remaining: {n_genes_after}")
        
        return adata
    
    def filter_cells(
        self,
        adata: AnnData,
        min_genes: Optional[int] = None,
        max_genes: Optional[int] = None,
        max_mt_percent: Optional[float] = None
    ) -> AnnData:
        """
        Filter cells based on quality criteria.
        
        Args:
            adata: Input data
            min_genes: Minimum genes per cell
            max_genes: Maximum genes per cell (filter doublets)
            max_mt_percent: Maximum mitochondrial percentage
            
        Returns:
            Filtered data
        """
        min_genes = min_genes or self.config.quality_control.min_genes_per_cell
        max_genes = max_genes or self.config.quality_control.max_genes_per_cell
        max_mt_percent = max_mt_percent or self.config.quality_control.mitochondrial_threshold
        
        logger.info(f"Filtering cells: min_genes={min_genes}, max_genes={max_genes}, max_mt={max_mt_percent}")
        
        n_cells_before = adata.n_obs
        
        # Filter by gene count
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
            
        if max_genes is not None:
            adata = adata[adata.obs['n_genes'] < max_genes].copy()
            
        # Filter by mitochondrial percentage
        if max_mt_percent is not None and 'pct_counts_mt' in adata.obs:
            adata = adata[adata.obs['pct_counts_mt'] < max_mt_percent * 100].copy()
            
        n_cells_after = adata.n_obs
        logger.info(f"Filtered {n_cells_before - n_cells_after} cells. Remaining: {n_cells_after}")
        
        return adata
    
    def normalize_expression(
        self,
        adata: AnnData,
        method: str = 'total_counts',
        target_sum: float = 1e4
    ) -> AnnData:
        """
        Normalize gene expression data.
        
        Args:
            adata: Input data
            method: Normalization method ('total_counts', 'scran', 'quantile')
            target_sum: Target sum for total counts normalization
            
        Returns:
            Normalized data
        """
        method = method or self.config.preprocessing.normalization
        
        logger.info(f"Normalizing expression with method: {method}")
        
        if method == 'total_counts':
            sc.pp.normalize_total(adata, target_sum=target_sum)
        elif method == 'scran':
            # Scran normalization (requires scran package)
            try:
                import scanpy.external as sce
                sce.pp.scran_normalize(adata)
            except ImportError:
                logger.warning("Scran not available, falling back to total counts")
                sc.pp.normalize_total(adata, target_sum=target_sum)
        elif method == 'quantile':
            self._quantile_normalize(adata)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return adata
    
    def _quantile_normalize(self, adata: AnnData) -> None:
        """Apply quantile normalization."""
        from scipy.stats import rankdata
        
        if sp.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X.copy()
            
        # Rank genes within each cell
        ranks = np.apply_along_axis(rankdata, 1, X)
        
        # Compute mean expression for each rank
        sorted_X = np.sort(X, axis=1)
        mean_sorted = np.mean(sorted_X, axis=0)
        
        # Map ranks to quantile values
        for i in range(X.shape[0]):
            X[i] = mean_sorted[ranks[i].astype(int) - 1]
            
        adata.X = X
    
    def log_transform(self, adata: AnnData, base: Optional[float] = None) -> AnnData:
        """
        Apply log transformation to expression data.
        
        Args:
            adata: Input data
            base: Log base (None for natural log)
            
        Returns:
            Log-transformed data
        """
        logger.info("Applying log transformation")
        
        if base is None:
            sc.pp.log1p(adata)
        else:
            adata.X = np.log1p(adata.X) / np.log(base)
            
        return adata
    
    def select_highly_variable_genes(
        self,
        adata: AnnData,
        n_top_genes: Optional[int] = None,
        min_mean: float = 0.0125,
        max_mean: float = 3,
        min_disp: float = 0.5
    ) -> AnnData:
        """
        Select highly variable genes for downstream analysis.
        
        Args:
            adata: Input data
            n_top_genes: Number of top genes to select
            min_mean: Minimum mean expression
            max_mean: Maximum mean expression  
            min_disp: Minimum dispersion
            
        Returns:
            Data with HVG selection
        """
        n_top_genes = n_top_genes or self.config.preprocessing.highly_variable_genes
        
        logger.info(f"Selecting highly variable genes: n_top={n_top_genes}")
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            flavor='seurat_v3' if adata.n_obs > 10000 else 'seurat'
        )
        
        # Store all genes info
        adata.var['highly_variable_rank'] = np.nan
        hvg_mask = adata.var['highly_variable']
        if hvg_mask.sum() > 0:
            adata.var.loc[hvg_mask, 'highly_variable_rank'] = np.arange(hvg_mask.sum())
            
        # Keep only highly variable genes
        adata = adata[:, adata.var['highly_variable']].copy()
        
        logger.info(f"Selected {adata.n_vars} highly variable genes")
        return adata
    
    def compute_spatial_neighbors(
        self,
        adata: AnnData,
        n_neighbors: int = 6,
        coord_key: str = 'spatial',
        method: str = 'knn'
    ) -> AnnData:
        """
        Compute spatial neighborhood graph.
        
        Args:
            adata: Input data with spatial coordinates
            n_neighbors: Number of neighbors
            coord_key: Key for spatial coordinates in obsm
            method: Method for neighbor computation ('knn', 'radius')
            
        Returns:
            Data with spatial neighbors
        """
        logger.info(f"Computing spatial neighbors: method={method}, n_neighbors={n_neighbors}")
        
        if coord_key not in adata.obsm:
            raise ValueError(f"Spatial coordinates '{coord_key}' not found in obsm")
            
        # Use squidpy if available for spatial neighbors
        try:
            import squidpy as sq
            sq.gr.spatial_neighbors(
                adata,
                coord_type='generic',
                n_neighs=n_neighbors,
                spatial_key=coord_key
            )
        except ImportError:
            # Fallback to scanpy
            logger.warning("Squidpy not available, using scanpy for spatial neighbors")
            sc.pp.neighbors(adata, use_rep=f'X_{coord_key}', n_neighbors=n_neighbors)
            
        return adata
    
    def scale_data(
        self,
        adata: AnnData,
        method: str = 'standard',
        max_value: Optional[float] = None
    ) -> AnnData:
        """
        Scale expression data.
        
        Args:
            adata: Input data
            method: Scaling method ('standard', 'robust', 'minmax')
            max_value: Maximum value for clipping
            
        Returns:
            Scaled data
        """
        logger.info(f"Scaling data with method: {method}")
        
        if method == 'standard':
            sc.pp.scale(adata, max_value=max_value)
        elif method == 'robust':
            if sp.issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = adata.X.copy()
                
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            if max_value is not None:
                X_scaled = np.clip(X_scaled, -max_value, max_value)
                
            adata.X = X_scaled
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        return adata
    
    def remove_batch_effects(
        self,
        adata: AnnData,
        batch_key: str = 'batch',
        method: str = 'combat'
    ) -> AnnData:
        """
        Remove batch effects from expression data.
        
        Args:
            adata: Input data
            batch_key: Key for batch information
            method: Method for batch correction
            
        Returns:
            Batch-corrected data
        """
        if batch_key not in adata.obs:
            logger.warning(f"Batch key '{batch_key}' not found, skipping batch correction")
            return adata
            
        logger.info(f"Removing batch effects with method: {method}")
        
        if method == 'combat':
            sc.pp.combat(adata, key=batch_key)
        else:
            raise ValueError(f"Unknown batch correction method: {method}")
            
        return adata
    
    def validate_data(self, adata: AnnData) -> Dict[str, Any]:
        """
        Validate processed data quality.
        
        Args:
            adata: Processed data
            
        Returns:
            Validation report
        """
        logger.info("Validating processed data")
        
        report = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'sparsity': 1 - (adata.X > 0).sum() / adata.X.size,
            'has_spatial': 'spatial' in adata.obsm,
            'has_neighbors': 'neighbors' in adata.obsp,
            'expression_range': (adata.X.min(), adata.X.max()),
            'total_counts_range': (
                adata.obs['total_counts'].min(),
                adata.obs['total_counts'].max()
            )
        }
        
        # Check for potential issues
        warnings_list = []
        
        if report['sparsity'] > 0.95:
            warnings_list.append("Very sparse data (>95% zeros)")
            
        if report['n_genes'] < 1000:
            warnings_list.append("Low gene count (<1000)")
            
        if not report['has_spatial']:
            warnings_list.append("No spatial coordinates found")
            
        report['warnings'] = warnings_list
        
        logger.info(f"Validation complete: {report['n_cells']} cells, {report['n_genes']} genes")
        if warnings_list:
            logger.warning(f"Data quality warnings: {', '.join(warnings_list)}")
            
        return report
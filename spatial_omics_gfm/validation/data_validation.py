"""
Data validation utilities for spatial transcriptomics data.

Provides comprehensive validation for expression matrices, coordinates,
gene names, and other spatial omics data structures.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """Comprehensive data validator for spatial transcriptomics."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, raise errors for warnings. If False, issue warnings.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
    
    def validate_expression_matrix(
        self, 
        expression_matrix: np.ndarray,
        min_genes_per_cell: int = 50,
        max_genes_per_cell: int = 10000,
        min_cells_per_gene: int = 3,
        max_mitochondrial_percent: float = 50.0,
        gene_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate expression matrix with comprehensive checks.
        
        Args:
            expression_matrix: Cell x Gene expression matrix
            min_genes_per_cell: Minimum genes per cell threshold
            max_genes_per_cell: Maximum genes per cell threshold  
            min_cells_per_gene: Minimum cells per gene threshold
            max_mitochondrial_percent: Maximum mitochondrial gene percentage
            gene_names: Optional gene names for additional validation
            
        Returns:
            Dictionary with validation results and metrics
            
        Raises:
            ValidationError: If critical validation fails
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Basic shape validation
            if not isinstance(expression_matrix, np.ndarray):
                raise ValidationError("Expression matrix must be numpy array")
            
            if len(expression_matrix.shape) != 2:
                raise ValidationError("Expression matrix must be 2D (cells x genes)")
            
            n_cells, n_genes = expression_matrix.shape
            results["metrics"]["n_cells"] = n_cells
            results["metrics"]["n_genes"] = n_genes
            
            # Check for negative values
            if np.any(expression_matrix < 0):
                error_msg = "Expression matrix contains negative values"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                results["warnings"].append(error_msg)
            
            # Check for NaN/inf values
            if np.any(~np.isfinite(expression_matrix)):
                raise ValidationError("Expression matrix contains NaN or infinite values")
            
            # Check data type
            if not np.issubdtype(expression_matrix.dtype, np.number):
                raise ValidationError("Expression matrix must contain numeric data")
            
            # Gene statistics
            genes_per_cell = np.sum(expression_matrix > 0, axis=1)
            cells_per_gene = np.sum(expression_matrix > 0, axis=0)
            
            results["metrics"]["mean_genes_per_cell"] = float(np.mean(genes_per_cell))
            results["metrics"]["mean_cells_per_gene"] = float(np.mean(cells_per_gene))
            
            # Quality control checks
            low_gene_cells = np.sum(genes_per_cell < min_genes_per_cell)
            high_gene_cells = np.sum(genes_per_cell > max_genes_per_cell)
            low_expression_genes = np.sum(cells_per_gene < min_cells_per_gene)
            
            results["metrics"]["low_gene_cells"] = int(low_gene_cells)
            results["metrics"]["high_gene_cells"] = int(high_gene_cells)
            results["metrics"]["low_expression_genes"] = int(low_expression_genes)
            
            # Issue warnings for quality issues
            if low_gene_cells > n_cells * 0.1:  # >10% of cells
                warning_msg = f"{low_gene_cells} cells have <{min_genes_per_cell} genes"
                results["warnings"].append(warning_msg)
            
            if low_expression_genes > n_genes * 0.3:  # >30% of genes
                warning_msg = f"{low_expression_genes} genes expressed in <{min_cells_per_gene} cells"
                results["warnings"].append(warning_msg)
            
            # Mitochondrial gene check (if gene names provided)
            if gene_names is not None:
                mito_genes = [i for i, gene in enumerate(gene_names) 
                             if gene.startswith(('MT-', 'mt-', 'MTRNR', 'MTATP'))]
                
                if mito_genes:
                    mito_expression = np.sum(expression_matrix[:, mito_genes], axis=1)
                    total_expression = np.sum(expression_matrix, axis=1)
                    mito_percent = (mito_expression / (total_expression + 1e-10)) * 100
                    
                    results["metrics"]["mean_mito_percent"] = float(np.mean(mito_percent))
                    high_mito_cells = np.sum(mito_percent > max_mitochondrial_percent)
                    
                    if high_mito_cells > 0:
                        warning_msg = f"{high_mito_cells} cells have >{max_mitochondrial_percent}% mitochondrial genes"
                        results["warnings"].append(warning_msg)
            
            # Expression distribution checks
            total_counts = np.sum(expression_matrix, axis=1)
            results["metrics"]["mean_total_counts"] = float(np.mean(total_counts))
            results["metrics"]["median_total_counts"] = float(np.median(total_counts))
            
            # Check for extremely low/high count cells
            low_count_threshold = np.percentile(total_counts, 5)
            high_count_threshold = np.percentile(total_counts, 95)
            
            if low_count_threshold < 100:
                results["warnings"].append("Some cells have very low total counts (<100)")
            
            # Memory usage estimation
            memory_gb = expression_matrix.nbytes / (1024**3)
            results["metrics"]["memory_usage_gb"] = memory_gb
            
            if memory_gb > 8:
                results["warnings"].append(f"Large memory usage: {memory_gb:.2f} GB")
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(str(e))
            
            if self.strict_mode:
                raise ValidationError(f"Expression matrix validation failed: {e}")
        
        return results
    
    def validate_coordinates(
        self,
        coordinates: np.ndarray,
        n_cells: Optional[int] = None,
        expected_dims: int = 2
    ) -> Dict[str, Any]:
        """
        Validate spatial coordinates.
        
        Args:
            coordinates: Spatial coordinates array
            n_cells: Expected number of cells (optional)
            expected_dims: Expected spatial dimensions (2 or 3)
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Basic validation
            if not isinstance(coordinates, np.ndarray):
                raise ValidationError("Coordinates must be numpy array")
            
            if len(coordinates.shape) != 2:
                raise ValidationError("Coordinates must be 2D array (cells x dimensions)")
            
            n_coord_cells, n_dims = coordinates.shape
            results["metrics"]["n_cells"] = n_coord_cells
            results["metrics"]["n_dimensions"] = n_dims
            
            # Check dimensions
            if n_dims != expected_dims:
                error_msg = f"Expected {expected_dims}D coordinates, got {n_dims}D"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                results["warnings"].append(error_msg)
            
            # Check cell count consistency
            if n_cells is not None and n_coord_cells != n_cells:
                raise ValidationError(f"Coordinate count ({n_coord_cells}) doesn't match expected ({n_cells})")
            
            # Check for NaN/inf values
            if np.any(~np.isfinite(coordinates)):
                raise ValidationError("Coordinates contain NaN or infinite values")
            
            # Check data type
            if not np.issubdtype(coordinates.dtype, np.number):
                raise ValidationError("Coordinates must be numeric")
            
            # Spatial statistics
            for dim in range(n_dims):
                dim_name = ['x', 'y', 'z'][dim] if dim < 3 else f'dim_{dim}'
                coord_values = coordinates[:, dim]
                
                results["metrics"][f"{dim_name}_range"] = [float(np.min(coord_values)), float(np.max(coord_values))]
                results["metrics"][f"{dim_name}_mean"] = float(np.mean(coord_values))
                results["metrics"][f"{dim_name}_std"] = float(np.std(coord_values))
            
            # Check for duplicate coordinates
            unique_coords = np.unique(coordinates, axis=0)
            if len(unique_coords) < len(coordinates):
                n_duplicates = len(coordinates) - len(unique_coords)
                warning_msg = f"Found {n_duplicates} duplicate coordinate positions"
                results["warnings"].append(warning_msg)
            
            # Check coordinate distribution
            for dim in range(n_dims):
                coord_values = coordinates[:, dim]
                if np.std(coord_values) < 1e-6:
                    dim_name = ['x', 'y', 'z'][dim] if dim < 3 else f'dim_{dim}'
                    results["warnings"].append(f"Very low variance in {dim_name} coordinates")
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(str(e))
            
            if self.strict_mode:
                raise ValidationError(f"Coordinate validation failed: {e}")
        
        return results
    
    def validate_gene_names(
        self,
        gene_names: List[str],
        n_genes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate gene names.
        
        Args:
            gene_names: List of gene names
            n_genes: Expected number of genes
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Basic validation
            if not isinstance(gene_names, (list, np.ndarray)):
                raise ValidationError("Gene names must be list or array")
            
            gene_names = list(gene_names)  # Convert to list for consistency
            results["metrics"]["n_genes"] = len(gene_names)
            
            # Check count consistency
            if n_genes is not None and len(gene_names) != n_genes:
                raise ValidationError(f"Gene name count ({len(gene_names)}) doesn't match expected ({n_genes})")
            
            # Check for empty or None names
            empty_names = sum(1 for name in gene_names if not name or name.strip() == "")
            if empty_names > 0:
                error_msg = f"Found {empty_names} empty gene names"
                results["errors"].append(error_msg)
            
            # Check for duplicates
            unique_names = set(gene_names)
            if len(unique_names) < len(gene_names):
                n_duplicates = len(gene_names) - len(unique_names)
                results["warnings"].append(f"Found {n_duplicates} duplicate gene names")
            
            # Check naming conventions
            non_standard = 0
            for name in gene_names:
                if isinstance(name, str):
                    # Check for unusual characters
                    if not name.replace('-', '').replace('_', '').replace('.', '').isalnum():
                        non_standard += 1
            
            if non_standard > 0:
                results["warnings"].append(f"Found {non_standard} genes with non-standard names")
            
            # Gene type analysis
            mito_genes = sum(1 for name in gene_names if isinstance(name, str) and name.startswith(('MT-', 'mt-', 'MTRNR', 'MTATP')))
            ribosomal_genes = sum(1 for name in gene_names if isinstance(name, str) and (name.startswith(('RPS', 'RPL')) or 'ribosom' in name.lower()))
            
            results["metrics"]["mitochondrial_genes"] = mito_genes
            results["metrics"]["ribosomal_genes"] = ribosomal_genes
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(str(e))
            
            if self.strict_mode:
                raise ValidationError(f"Gene name validation failed: {e}")
        
        return results
    
    def validate_spatial_data(
        self,
        expression_matrix: np.ndarray,
        coordinates: np.ndarray,
        gene_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of spatial transcriptomics data.
        
        Args:
            expression_matrix: Cell x Gene expression matrix
            coordinates: Cell x Spatial_dims coordinate matrix
            gene_names: Optional gene names
            **kwargs: Additional validation parameters
            
        Returns:
            Combined validation results
        """
        logger.info("Starting comprehensive spatial data validation")
        
        combined_results = {
            "overall_valid": True,
            "expression_validation": {},
            "coordinate_validation": {},
            "gene_name_validation": {},
            "compatibility_checks": {},
            "summary": {}
        }
        
        try:
            n_cells, n_genes = expression_matrix.shape
            
            # Validate expression matrix
            expr_results = self.validate_expression_matrix(
                expression_matrix, 
                gene_names=gene_names,
                **{k: v for k, v in kwargs.items() if k.startswith(('min_genes', 'max_genes', 'min_cells', 'max_mito'))}
            )
            combined_results["expression_validation"] = expr_results
            
            # Validate coordinates
            coord_results = self.validate_coordinates(
                coordinates,
                n_cells=n_cells,
                **{k: v for k, v in kwargs.items() if k.startswith('expected_dims')}
            )
            combined_results["coordinate_validation"] = coord_results
            
            # Validate gene names if provided
            if gene_names is not None:
                gene_results = self.validate_gene_names(gene_names, n_genes=n_genes)
                combined_results["gene_name_validation"] = gene_results
            
            # Check compatibility between data components
            compatibility_results = {
                "valid": True,
                "warnings": [],
                "errors": []
            }
            
            # Cell count consistency
            coord_cells = coordinates.shape[0] if coordinates is not None else 0
            if coord_cells != n_cells:
                compatibility_results["errors"].append(
                    f"Cell count mismatch: expression({n_cells}) vs coordinates({coord_cells})"
                )
                compatibility_results["valid"] = False
            
            # Gene count consistency
            if gene_names is not None and len(gene_names) != n_genes:
                compatibility_results["errors"].append(
                    f"Gene count mismatch: expression({n_genes}) vs names({len(gene_names)})"
                )
                compatibility_results["valid"] = False
            
            combined_results["compatibility_checks"] = compatibility_results
            
            # Overall validation status
            all_valid = (
                expr_results.get("valid", False) and
                coord_results.get("valid", False) and
                compatibility_results.get("valid", False) and
                (gene_names is None or combined_results["gene_name_validation"].get("valid", False))
            )
            
            combined_results["overall_valid"] = all_valid
            
            # Create summary
            total_warnings = (
                len(expr_results.get("warnings", [])) +
                len(coord_results.get("warnings", [])) +
                len(compatibility_results.get("warnings", [])) +
                len(combined_results["gene_name_validation"].get("warnings", []))
            )
            
            total_errors = (
                len(expr_results.get("errors", [])) +
                len(coord_results.get("errors", [])) +
                len(compatibility_results.get("errors", [])) +
                len(combined_results["gene_name_validation"].get("errors", []))
            )
            
            combined_results["summary"] = {
                "total_warnings": total_warnings,
                "total_errors": total_errors,
                "data_shape": (n_cells, n_genes),
                "coordinate_dims": coordinates.shape[1] if coordinates is not None else 0,
                "has_gene_names": gene_names is not None,
                "validation_passed": all_valid
            }
            
            logger.info(f"Validation complete: {total_errors} errors, {total_warnings} warnings")
            
        except Exception as e:
            combined_results["overall_valid"] = False
            combined_results["summary"] = {"validation_error": str(e)}
            
            if self.strict_mode:
                raise ValidationError(f"Spatial data validation failed: {e}")
        
        return combined_results


# Convenience functions
def validate_expression_matrix(expression_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Convenience function for expression matrix validation."""
    validator = DataValidator()
    return validator.validate_expression_matrix(expression_matrix, **kwargs)


def validate_coordinates(coordinates: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Convenience function for coordinate validation."""
    validator = DataValidator()
    return validator.validate_coordinates(coordinates, **kwargs)


def validate_gene_names(gene_names: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function for gene name validation."""
    validator = DataValidator()
    return validator.validate_gene_names(gene_names, **kwargs)
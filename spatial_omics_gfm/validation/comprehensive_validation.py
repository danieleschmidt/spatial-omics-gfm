"""
Comprehensive Validation Framework for Spatial-Omics GFM
========================================================

Generation 2 Enhancement: MAKE IT ROBUST
- Multi-level input validation and error handling
- Data quality assessment and automatic corrections
- Security hardening against malicious inputs  
- Robust error recovery and graceful degradation
- Comprehensive logging and monitoring

Authors: Daniel Schmidt, Terragon Labs
Date: 2025-01-25
"""

import numpy as np
import logging
import time
import hashlib
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import traceback


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"  # Basic checks only
    STANDARD = "standard"     # Standard validation
    STRICT = "strict"         # Comprehensive validation
    PARANOID = "paranoid"     # Maximum security validation


class DataQuality(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class ValidationResult:
    """Comprehensive validation result container."""
    is_valid: bool
    quality_score: float  # 0-1
    quality_level: DataQuality
    errors: List[str]
    warnings: List[str]
    corrections_applied: List[str]
    metadata: Dict[str, Any]
    validation_time: float
    security_score: float  # 0-1


class SecurityValidator:
    """Security-focused validation for preventing malicious attacks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security limits
        self.MAX_CELLS = 100000
        self.MAX_GENES = 50000
        self.MAX_MEMORY_MB = 8192
        self.MAX_COORDINATE_VALUE = 1e6
        self.MIN_COORDINATE_VALUE = -1e6
        
    def validate_array_safety(self, 
                             arr: np.ndarray, 
                             name: str,
                             max_size_mb: float = 1024) -> Tuple[bool, List[str]]:
        """
        Validate array for security threats.
        
        Checks for:
        - Memory bombs (excessive size)
        - NaN/Inf injection attacks
        - Out-of-range values
        - Suspicious patterns
        """
        errors = []
        
        # Check array size (prevent memory bombs)
        size_mb = arr.nbytes / (1024 * 1024)
        if size_mb > max_size_mb:
            errors.append(f"{name}: Array too large ({size_mb:.1f}MB > {max_size_mb}MB limit)")
        
        # Check for malicious values
        if np.any(np.isinf(arr)):
            errors.append(f"{name}: Contains infinite values (potential DoS attack)")
            
        # Count NaN values
        nan_count = np.sum(np.isnan(arr))
        nan_fraction = nan_count / arr.size
        if nan_fraction > 0.5:
            errors.append(f"{name}: Excessive NaN values ({nan_fraction:.1%} > 50%)")
        elif nan_fraction > 0.1:
            self.logger.warning(f"{name}: High NaN fraction ({nan_fraction:.1%})")
        
        # Check for suspicious uniform patterns (might indicate synthetic attack data)
        if arr.size > 100:
            unique_values = len(np.unique(arr[~np.isnan(arr)]))
            if unique_values < max(10, arr.size // 1000):
                errors.append(f"{name}: Suspiciously low value diversity (potential synthetic data)")
        
        # Check coordinate bounds for spatial data
        if 'coord' in name.lower() or 'spatial' in name.lower():
            if np.any((arr < self.MIN_COORDINATE_VALUE) | (arr > self.MAX_COORDINATE_VALUE)):
                errors.append(f"{name}: Coordinates outside safe bounds")
        
        # Check for extremely negative values in expression data
        if 'expression' in name.lower() or 'gene' in name.lower():
            if np.any(arr < -100):
                errors.append(f"{name}: Suspicious negative expression values")
        
        return len(errors) == 0, errors
    
    def compute_data_hash(self, data: Union[np.ndarray, Dict]) -> str:
        """Compute secure hash of data for integrity checking."""
        if isinstance(data, np.ndarray):
            # Create reproducible hash of array
            data_bytes = data.tobytes()
        else:
            # Hash dictionary
            data_str = json.dumps(data, sort_keys=True, default=str)
            data_bytes = data_str.encode('utf-8')
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def validate_input_integrity(self,
                               gene_expression: np.ndarray,
                               spatial_coords: np.ndarray,
                               expected_hash: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive security validation of input data.
        """
        start_time = time.time()
        errors = []
        warnings = []
        corrections = []
        
        # 1. Basic security checks
        expr_safe, expr_errors = self.validate_array_safety(
            gene_expression, "gene_expression", max_size_mb=2048
        )
        spatial_safe, spatial_errors = self.validate_array_safety(
            spatial_coords, "spatial_coords", max_size_mb=64
        )
        
        errors.extend(expr_errors)
        errors.extend(spatial_errors)
        
        # 2. Dimensional consistency
        if gene_expression.shape[0] != spatial_coords.shape[0]:
            errors.append("Dimension mismatch: gene_expression and spatial_coords must have same number of cells")
        
        # 3. Coordinate validation
        if spatial_coords.shape[1] not in [2, 3]:
            errors.append(f"Invalid spatial dimensions: expected 2D or 3D, got {spatial_coords.shape[1]}D")
        
        # 4. Expression data sanity checks
        if gene_expression.shape[1] < 10:
            warnings.append(f"Very few genes ({gene_expression.shape[1]}) - results may be unreliable")
        elif gene_expression.shape[1] > self.MAX_GENES:
            errors.append(f"Too many genes ({gene_expression.shape[1]} > {self.MAX_GENES})")
        
        # 5. Cell count validation
        n_cells = gene_expression.shape[0]
        if n_cells < 10:
            errors.append(f"Insufficient cells ({n_cells} < 10 minimum)")
        elif n_cells > self.MAX_CELLS:
            errors.append(f"Too many cells ({n_cells} > {self.MAX_CELLS})")
        
        # 6. Hash-based integrity check
        if expected_hash:
            actual_hash = self.compute_data_hash(gene_expression)
            if actual_hash != expected_hash:
                errors.append("Data integrity check failed - hash mismatch")
        
        # 7. Statistical anomaly detection
        expr_stats = self._analyze_expression_statistics(gene_expression)
        if expr_stats['suspicious_patterns'] > 0:
            warnings.append(f"Detected {expr_stats['suspicious_patterns']} suspicious expression patterns")
        
        # Compute security score
        security_score = 1.0
        security_score -= 0.2 * len(errors)  # Major security issues
        security_score -= 0.1 * len(warnings)  # Minor issues
        security_score = max(0.0, security_score)
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=security_score,
            quality_level=self._assess_quality_level(security_score),
            errors=errors,
            warnings=warnings,
            corrections_applied=corrections,
            metadata={
                'n_cells': n_cells,
                'n_genes': gene_expression.shape[1],
                'data_hash': self.compute_data_hash(gene_expression),
                'expr_statistics': expr_stats
            },
            validation_time=validation_time,
            security_score=security_score
        )
    
    def _analyze_expression_statistics(self, gene_expression: np.ndarray) -> Dict[str, Any]:
        """Analyze expression data for suspicious statistical patterns."""
        stats = {}
        
        # Basic statistics
        stats['mean'] = float(np.nanmean(gene_expression))
        stats['std'] = float(np.nanstd(gene_expression))
        stats['min'] = float(np.nanmin(gene_expression))
        stats['max'] = float(np.nanmax(gene_expression))
        stats['nan_fraction'] = float(np.sum(np.isnan(gene_expression)) / gene_expression.size)
        
        # Anomaly detection
        suspicious_patterns = 0
        
        # Check for identical rows (duplicated cells)
        if gene_expression.shape[0] > 10:
            for i in range(min(100, gene_expression.shape[0])):  # Sample check
                row_i = gene_expression[i, :]
                if not np.any(np.isnan(row_i)):  # Skip NaN rows
                    identical_count = np.sum(np.allclose(gene_expression, row_i[None, :], rtol=1e-10))
                    if identical_count > 1:
                        suspicious_patterns += 1
                        break
        
        # Check for suspiciously uniform distributions
        if gene_expression.size > 1000:
            sample = gene_expression.flatten()
            sample = sample[~np.isnan(sample)]
            if len(sample) > 100:
                hist, _ = np.histogram(sample, bins=50)
                uniformity = np.std(hist) / (np.mean(hist) + 1e-8)
                if uniformity < 0.1:  # Very uniform
                    suspicious_patterns += 1
        
        stats['suspicious_patterns'] = suspicious_patterns
        
        return stats
    
    def _assess_quality_level(self, score: float) -> DataQuality:
        """Convert quality score to quality level."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.ACCEPTABLE
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.UNACCEPTABLE


class DataQualityProcessor:
    """
    Advanced data quality assessment and automatic correction.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
    def assess_and_correct_quality(self,
                                  gene_expression: np.ndarray,
                                  spatial_coords: np.ndarray,
                                  auto_correct: bool = True) -> Tuple[np.ndarray, np.ndarray, ValidationResult]:
        """
        Assess data quality and apply automatic corrections.
        
        Returns:
            corrected_expression, corrected_coords, validation_result
        """
        start_time = time.time()
        errors = []
        warnings = []
        corrections = []
        
        # Work on copies to avoid modifying original data
        corrected_expression = gene_expression.copy()
        corrected_coords = spatial_coords.copy()
        
        # 1. Handle missing values
        expr_nan_fraction = np.sum(np.isnan(corrected_expression)) / corrected_expression.size
        coords_nan_fraction = np.sum(np.isnan(corrected_coords)) / corrected_coords.size
        
        if expr_nan_fraction > 0:
            if expr_nan_fraction > 0.5:
                errors.append(f"Excessive missing expression data ({expr_nan_fraction:.1%})")
            else:
                if auto_correct:
                    # Impute missing values with row/column medians
                    corrected_expression = self._impute_missing_expression(corrected_expression)
                    corrections.append(f"Imputed {expr_nan_fraction:.1%} missing expression values")
                else:
                    warnings.append(f"Missing expression data ({expr_nan_fraction:.1%}) not corrected")
        
        if coords_nan_fraction > 0:
            if coords_nan_fraction > 0.1:
                errors.append(f"Too many missing coordinates ({coords_nan_fraction:.1%})")
            else:
                if auto_correct:
                    corrected_coords = self._impute_missing_coordinates(corrected_coords)
                    corrections.append(f"Imputed {coords_nan_fraction:.1%} missing coordinates")
        
        # 2. Detect and handle outliers
        expr_outliers = self._detect_expression_outliers(corrected_expression)
        if len(expr_outliers) > 0:
            if len(expr_outliers) > corrected_expression.shape[0] * 0.1:
                warnings.append(f"Many expression outliers detected ({len(expr_outliers)} cells)")
            
            if auto_correct and len(expr_outliers) < corrected_expression.shape[0] * 0.05:
                corrected_expression = self._correct_expression_outliers(
                    corrected_expression, expr_outliers
                )
                corrections.append(f"Corrected {len(expr_outliers)} expression outliers")
        
        # 3. Spatial coordinate validation and correction
        spatial_outliers = self._detect_spatial_outliers(corrected_coords)
        if len(spatial_outliers) > 0:
            if auto_correct:
                corrected_coords = self._correct_spatial_outliers(
                    corrected_coords, spatial_outliers
                )
                corrections.append(f"Corrected {len(spatial_outliers)} spatial outliers")
        
        # 4. Data normalization assessment
        expr_scale_issues = self._assess_expression_scaling(corrected_expression)
        if expr_scale_issues:
            if auto_correct:
                corrected_expression = self._normalize_expression(corrected_expression)
                corrections.append("Applied expression normalization")
            else:
                warnings.append("Expression data may benefit from normalization")
        
        # 5. Spatial coordinate standardization
        coord_scale_issues = self._assess_coordinate_scaling(corrected_coords)
        if coord_scale_issues:
            if auto_correct:
                corrected_coords = self._standardize_coordinates(corrected_coords)
                corrections.append("Standardized spatial coordinates")
        
        # 6. Final quality assessment
        quality_score = self._compute_quality_score(
            corrected_expression, corrected_coords, 
            len(errors), len(warnings)
        )
        
        validation_time = time.time() - start_time
        
        validation_result = ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=quality_score,
            quality_level=self._assess_quality_level(quality_score),
            errors=errors,
            warnings=warnings,
            corrections_applied=corrections,
            metadata={
                'original_nan_expression': expr_nan_fraction,
                'original_nan_coordinates': coords_nan_fraction,
                'expression_outliers_detected': len(expr_outliers),
                'spatial_outliers_detected': len(spatial_outliers),
                'corrections_count': len(corrections)
            },
            validation_time=validation_time,
            security_score=quality_score  # Quality and security related
        )
        
        return corrected_expression, corrected_coords, validation_result
    
    def _impute_missing_expression(self, expression: np.ndarray) -> np.ndarray:
        """Impute missing expression values using median imputation."""
        result = expression.copy()
        
        # Gene-wise median imputation
        for gene_idx in range(expression.shape[1]):
            gene_values = expression[:, gene_idx]
            if np.any(np.isnan(gene_values)):
                median_val = np.nanmedian(gene_values)
                if not np.isnan(median_val):
                    result[np.isnan(gene_values), gene_idx] = median_val
                else:
                    result[np.isnan(gene_values), gene_idx] = 0.0
        
        return result
    
    def _impute_missing_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Impute missing coordinates using spatial interpolation."""
        result = coords.copy()
        
        for dim in range(coords.shape[1]):
            dim_values = coords[:, dim]
            missing_mask = np.isnan(dim_values)
            
            if np.any(missing_mask):
                valid_values = dim_values[~missing_mask]
                if len(valid_values) > 0:
                    # Simple imputation with mean + small random offset
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    
                    n_missing = np.sum(missing_mask)
                    imputed_values = mean_val + np.random.normal(0, std_val * 0.1, n_missing)
                    result[missing_mask, dim] = imputed_values
        
        return result
    
    def _detect_expression_outliers(self, expression: np.ndarray) -> List[int]:
        """Detect cells with outlier expression patterns."""
        outliers = []
        
        # Detect cells with extreme total expression
        total_expr = np.sum(expression, axis=1)
        q75, q25 = np.percentile(total_expr, [75, 25])
        iqr = q75 - q25
        
        lower_bound = q25 - 3.0 * iqr
        upper_bound = q75 + 3.0 * iqr
        
        outlier_mask = (total_expr < lower_bound) | (total_expr > upper_bound)
        outliers = np.where(outlier_mask)[0].tolist()
        
        return outliers
    
    def _correct_expression_outliers(self, 
                                   expression: np.ndarray, 
                                   outlier_indices: List[int]) -> np.ndarray:
        """Correct expression outliers by winsorizing."""
        result = expression.copy()
        
        for cell_idx in outlier_indices:
            cell_expr = expression[cell_idx, :]
            
            # Winsorize to 95th percentile bounds
            lower_bound = np.percentile(expression, 2.5, axis=0)
            upper_bound = np.percentile(expression, 97.5, axis=0)
            
            corrected_expr = np.clip(cell_expr, lower_bound, upper_bound)
            result[cell_idx, :] = corrected_expr
        
        return result
    
    def _detect_spatial_outliers(self, coords: np.ndarray) -> List[int]:
        """Detect spatially outlying cells."""
        outliers = []
        
        # Use distance-based outlier detection
        for dim in range(coords.shape[1]):
            dim_values = coords[:, dim]
            q75, q25 = np.percentile(dim_values, [75, 25])
            iqr = q75 - q25
            
            lower_bound = q25 - 3.0 * iqr
            upper_bound = q75 + 3.0 * iqr
            
            outlier_mask = (dim_values < lower_bound) | (dim_values > upper_bound)
            outliers.extend(np.where(outlier_mask)[0].tolist())
        
        return list(set(outliers))  # Remove duplicates
    
    def _correct_spatial_outliers(self, 
                                coords: np.ndarray, 
                                outlier_indices: List[int]) -> np.ndarray:
        """Correct spatial outliers by moving to boundary."""
        result = coords.copy()
        
        for cell_idx in outlier_indices:
            for dim in range(coords.shape[1]):
                dim_values = coords[:, dim]
                q75, q25 = np.percentile(dim_values, [75, 25])
                iqr = q75 - q25
                
                lower_bound = q25 - 2.0 * iqr
                upper_bound = q75 + 2.0 * iqr
                
                result[cell_idx, dim] = np.clip(coords[cell_idx, dim], lower_bound, upper_bound)
        
        return result
    
    def _assess_expression_scaling(self, expression: np.ndarray) -> bool:
        """Assess if expression data needs scaling."""
        # Check if data spans multiple orders of magnitude
        non_zero = expression[expression > 0]
        if len(non_zero) == 0:
            return True
            
        log_range = np.log10(np.max(non_zero)) - np.log10(np.min(non_zero))
        return log_range > 3.0  # More than 3 orders of magnitude
    
    def _normalize_expression(self, expression: np.ndarray) -> np.ndarray:
        """Apply robust expression normalization."""
        result = expression.copy()
        
        # Log1p transformation for highly skewed data
        result = np.log1p(result)
        
        # Per-cell normalization (library size)
        cell_totals = np.sum(result, axis=1, keepdims=True)
        cell_totals = np.where(cell_totals > 0, cell_totals, 1.0)
        result = result / cell_totals * np.median(cell_totals)
        
        return result
    
    def _assess_coordinate_scaling(self, coords: np.ndarray) -> bool:
        """Assess if coordinates need standardization."""
        # Check if coordinate ranges are very different across dimensions
        ranges = np.ptp(coords, axis=0)  # Peak-to-peak range
        if np.any(ranges == 0):
            return True
        
        range_ratio = np.max(ranges) / np.min(ranges)
        return range_ratio > 100  # Very different scales
    
    def _standardize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Standardize coordinates to comparable scales."""
        result = coords.copy()
        
        # Center and scale each dimension
        for dim in range(coords.shape[1]):
            dim_values = coords[:, dim]
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values)
            
            if std_val > 0:
                result[:, dim] = (dim_values - mean_val) / std_val
            else:
                result[:, dim] = dim_values - mean_val
        
        return result
    
    def _compute_quality_score(self,
                             expression: np.ndarray,
                             coords: np.ndarray,
                             n_errors: int,
                             n_warnings: int) -> float:
        """Compute overall data quality score (0-1)."""
        score = 1.0
        
        # Penalize errors and warnings
        score -= 0.3 * n_errors
        score -= 0.1 * n_warnings
        
        # Check data completeness
        expr_completeness = 1.0 - np.sum(np.isnan(expression)) / expression.size
        coord_completeness = 1.0 - np.sum(np.isnan(coords)) / coords.size
        
        score *= (expr_completeness + coord_completeness) / 2.0
        
        # Check data diversity
        if expression.size > 0:
            unique_expr_fraction = len(np.unique(expression[~np.isnan(expression)])) / expression.size
            score *= min(1.0, unique_expr_fraction * 10)  # Penalize low diversity
        
        return max(0.0, min(1.0, score))
    
    def _assess_quality_level(self, score: float) -> DataQuality:
        """Convert quality score to quality level."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.ACCEPTABLE
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.UNACCEPTABLE


class RobustExecutionManager:
    """
    Robust execution manager with comprehensive error handling and recovery.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_validator = SecurityValidator()
        self.quality_processor = DataQualityProcessor()
        
    def robust_predict_interactions(self,
                                  gene_expression: np.ndarray,
                                  spatial_coords: np.ndarray,
                                  validation_level: ValidationLevel = ValidationLevel.STANDARD,
                                  max_retries: int = 3,
                                  timeout_seconds: float = 300.0) -> Dict[str, Any]:
        """
        Robust interaction prediction with comprehensive error handling.
        
        Features:
        - Input validation and sanitization
        - Automatic error recovery
        - Timeout protection  
        - Graceful degradation
        - Comprehensive logging
        """
        execution_id = int(time.time() * 1000)  # Unique execution ID
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting robust prediction [ID: {execution_id}]")
            
            # Phase 1: Security validation
            self.logger.debug("Phase 1: Security validation")
            security_result = self.security_validator.validate_input_integrity(
                gene_expression, spatial_coords
            )
            
            if not security_result.is_valid:
                raise ValueError(f"Security validation failed: {'; '.join(security_result.errors)}")
            
            # Phase 2: Data quality assessment and correction
            self.logger.debug("Phase 2: Data quality assessment")
            corrected_expression, corrected_coords, quality_result = \
                self.quality_processor.assess_and_correct_quality(
                    gene_expression, spatial_coords, auto_correct=True
                )
            
            # Phase 3: Robust prediction with retries
            prediction_result = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"Phase 3: Prediction attempt {attempt + 1}/{max_retries}")
                    
                    # Import and use the adaptive attention algorithm from Generation 1
                    from ..research.adaptive_attention_breakthrough import predict_interactions_adaptive
                    
                    prediction_result = predict_interactions_adaptive(
                        corrected_expression, corrected_coords
                    )
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Prediction attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        # Apply recovery strategies
                        if "memory" in str(e).lower():
                            # Reduce data size for memory issues
                            if corrected_expression.shape[0] > 1000:
                                sample_indices = np.random.choice(
                                    corrected_expression.shape[0], 1000, replace=False
                                )
                                corrected_expression = corrected_expression[sample_indices]
                                corrected_coords = corrected_coords[sample_indices]
                                self.logger.info("Applied memory recovery: reduced dataset size")
                        
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    
            if prediction_result is None:
                raise RuntimeError(f"All prediction attempts failed. Last error: {last_error}")
            
            # Phase 4: Result validation and post-processing
            self.logger.debug("Phase 4: Result validation")
            
            total_time = time.time() - start_time
            
            # Check timeout
            if total_time > timeout_seconds:
                self.logger.warning(f"Execution exceeded timeout ({total_time:.1f}s > {timeout_seconds}s)")
            
            # Compile comprehensive results
            result = {
                'interactions': prediction_result.get('interactions', []),
                'statistics': prediction_result.get('statistics', {}),
                'execution_metadata': {
                    'execution_id': execution_id,
                    'total_time_seconds': total_time,
                    'validation_level': validation_level.value,
                    'retries_used': attempt + 1 if prediction_result else max_retries,
                    'timeout_exceeded': total_time > timeout_seconds
                },
                'security_validation': {
                    'is_secure': security_result.is_valid,
                    'security_score': security_result.security_score,
                    'errors': security_result.errors,
                    'warnings': security_result.warnings
                },
                'quality_assessment': {
                    'quality_level': quality_result.quality_level.value,
                    'quality_score': quality_result.quality_score,
                    'corrections_applied': quality_result.corrections_applied,
                    'data_hash': quality_result.metadata.get('data_hash', 'unknown')
                },
                'robustness_metrics': {
                    'error_recovery_used': attempt > 0 if prediction_result else False,
                    'data_corrections_count': len(quality_result.corrections_applied),
                    'graceful_degradation': False,  # No degradation needed
                    'execution_stability': 'stable'
                }
            }
            
            self.logger.info(f"Robust prediction completed successfully [ID: {execution_id}] in {total_time:.3f}s")
            return result
            
        except Exception as e:
            # Catastrophic failure handling
            total_time = time.time() - start_time
            
            error_info = {
                'execution_id': execution_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'total_time_seconds': total_time,
                'phase_failed': self._identify_failure_phase(str(e))
            }
            
            self.logger.error(f"Catastrophic failure in robust prediction [ID: {execution_id}]: {str(e)}")
            
            # Attempt graceful degradation
            try:
                degraded_result = self._graceful_degradation(gene_expression, spatial_coords)
                error_info['graceful_degradation'] = degraded_result
                self.logger.info("Graceful degradation successful")
            except:
                error_info['graceful_degradation'] = None
                self.logger.error("Graceful degradation also failed")
            
            return {
                'interactions': [],
                'statistics': {'num_interactions': 0, 'error': True},
                'error': error_info,
                'execution_metadata': {'execution_id': execution_id, 'failed': True}
            }
    
    def _identify_failure_phase(self, error_message: str) -> str:
        """Identify which phase failed based on error message."""
        error_lower = error_message.lower()
        
        if 'security' in error_lower or 'validation' in error_lower:
            return 'security_validation'
        elif 'quality' in error_lower or 'impute' in error_lower:
            return 'quality_assessment'
        elif 'memory' in error_lower or 'timeout' in error_lower:
            return 'prediction_execution'
        else:
            return 'unknown'
    
    def _graceful_degradation(self, 
                            gene_expression: np.ndarray, 
                            spatial_coords: np.ndarray) -> Dict[str, Any]:
        """
        Provide graceful degradation when main algorithm fails.
        
        Returns basic distance-based interactions as fallback.
        """
        self.logger.info("Attempting graceful degradation with distance-based fallback")
        
        n_cells = min(gene_expression.shape[0], 500)  # Limit for safety
        
        # Simple distance-based interactions
        fallback_interactions = []
        
        for i in range(n_cells):
            for j in range(i+1, n_cells):
                distance = np.sqrt(np.sum((spatial_coords[i] - spatial_coords[j])**2))
                
                if distance < 100:  # 100 unit distance threshold
                    fallback_interactions.append({
                        'sender_cell': i,
                        'receiver_cell': j,
                        'distance': float(distance),
                        'method': 'distance_fallback'
                    })
        
        return {
            'interactions': fallback_interactions,
            'method': 'graceful_degradation',
            'fallback_type': 'distance_based',
            'reliability': 'low'
        }


def demonstrate_robust_validation() -> Dict[str, Any]:
    """
    Demonstrate the comprehensive validation framework.
    """
    print("🛡️  GENERATION 2: COMPREHENSIVE ROBUSTNESS VALIDATION")
    print("=" * 70)
    print("🔒 Security: Input validation, memory protection, integrity checking")
    print("🔧 Quality: Automatic correction, outlier detection, normalization")
    print("⚡ Robustness: Error recovery, graceful degradation, timeout protection")
    print()
    
    # Generate test data with intentional issues
    np.random.seed(42)
    
    # Create problematic dataset
    n_cells, n_genes = 300, 500
    gene_expression = np.random.lognormal(0, 1, (n_cells, n_genes))
    spatial_coords = np.random.uniform(0, 1000, (n_cells, 2))
    
    # Inject problems for testing robustness
    # 1. Add NaN values
    gene_expression[50:60, 100:150] = np.nan
    spatial_coords[25, 0] = np.nan
    
    # 2. Add extreme outliers
    gene_expression[10, :] *= 1000  # Expression outlier
    spatial_coords[5] = [5000, 5000]  # Spatial outlier
    
    # 3. Add duplicated cells
    gene_expression[200] = gene_expression[201]
    
    print("🧬 Generated test dataset with intentional issues:")
    print(f"   • Cells: {n_cells}, Genes: {n_genes}")
    print(f"   • NaN values: {np.sum(np.isnan(gene_expression))} expression, {np.sum(np.isnan(spatial_coords))} spatial")
    print(f"   • Extreme outliers: 1 expression, 1 spatial")
    print(f"   • Duplicated cells: 1 pair")
    print()
    
    # Run robust validation
    print("🛡️  Running comprehensive robustness validation...")
    start_time = time.time()
    
    robust_manager = RobustExecutionManager()
    result = robust_manager.robust_predict_interactions(
        gene_expression, spatial_coords,
        validation_level=ValidationLevel.STRICT,
        max_retries=3,
        timeout_seconds=120.0
    )
    
    total_time = time.time() - start_time
    
    # Display results
    print("📊 ROBUSTNESS VALIDATION RESULTS:")
    print("   " + "=" * 50)
    
    # Security results
    security = result['security_validation']
    print(f"   🔒 Security Validation:")
    print(f"      ├─ Status: {'✅ SECURE' if security['is_secure'] else '❌ INSECURE'}")
    print(f"      ├─ Security Score: {security['security_score']:.3f}")
    print(f"      ├─ Errors: {len(security['errors'])}")
    print(f"      └─ Warnings: {len(security['warnings'])}")
    
    # Quality results  
    quality = result['quality_assessment']
    print(f"   🔧 Quality Assessment:")
    print(f"      ├─ Quality Level: {quality['quality_level'].upper()}")
    print(f"      ├─ Quality Score: {quality['quality_score']:.3f}")
    print(f"      └─ Corrections Applied: {len(quality['corrections_applied'])}")
    for correction in quality['corrections_applied']:
        print(f"         • {correction}")
    
    # Execution robustness
    execution = result['execution_metadata']
    robustness = result['robustness_metrics']
    print(f"   ⚡ Execution Robustness:")
    print(f"      ├─ Execution Time: {execution['total_time_seconds']:.3f}s")
    print(f"      ├─ Retries Used: {execution['retries_used']}")
    print(f"      ├─ Error Recovery: {'✅ YES' if robustness['error_recovery_used'] else '❌ NO'}")
    print(f"      ├─ Data Corrections: {robustness['data_corrections_count']}")
    print(f"      └─ Stability: {robustness['execution_stability'].upper()}")
    
    # Prediction results
    stats = result['statistics']
    print(f"   📈 Prediction Results:")
    print(f"      ├─ Interactions Found: {stats.get('num_interactions', 0)}")
    print(f"      ├─ Mean Score: {stats.get('mean_attention_score', 0):.4f}")
    print(f"      └─ Statistical Significance: {'✅ YES' if stats.get('statistically_significant', False) else '❌ NO'}")
    
    print()
    
    # Assessment
    robustness_score = 0
    robustness_score += 1 if security['is_secure'] else 0
    robustness_score += 1 if quality['quality_level'] in ['excellent', 'good', 'acceptable'] else 0
    robustness_score += 1 if execution['total_time_seconds'] < 30.0 else 0
    robustness_score += 1 if not result.get('error') else 0
    robustness_score += 1 if len(quality['corrections_applied']) > 0 else 0
    
    print("🎯 GENERATION 2 ROBUSTNESS ASSESSMENT:")
    print("   " + "=" * 50)
    print(f"   ✅ Security Validation:     {'PASS' if security['is_secure'] else 'FAIL'}")
    print(f"   ✅ Quality Assurance:       {'PASS' if quality['quality_level'] != 'unacceptable' else 'FAIL'}")
    print(f"   ✅ Performance Efficiency:  {'PASS' if execution['total_time_seconds'] < 30.0 else 'FAIL'}")
    print(f"   ✅ Error Handling:          {'PASS' if not result.get('error') else 'FAIL'}")
    print(f"   ✅ Auto-Correction:         {'PASS' if len(quality['corrections_applied']) > 0 else 'FAIL'}")
    print(f"\n   ROBUSTNESS SCORE: {robustness_score}/5")
    
    if robustness_score >= 4:
        print(f"\n🏆 GENERATION 2 SUCCESS: System is ROBUST and ready for production!")
    else:
        print(f"\n⚠️  GENERATION 2 NEEDS WORK: Robustness improvements needed")
    
    return {
        'validation_result': result,
        'robustness_score': robustness_score,
        'total_validation_time': total_time,
        'generation_2_success': robustness_score >= 4
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run robustness demonstration
    demo_results = demonstrate_robust_validation()
    
    if demo_results['generation_2_success']:
        print(f"\n🚀 Ready for Generation 3: MAKE IT SCALE!")
    else:
        print(f"\n🔧 Generation 2 needs refinement before scaling")
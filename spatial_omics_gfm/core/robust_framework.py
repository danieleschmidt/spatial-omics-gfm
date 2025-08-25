"""
Robust Framework - Generation 2 Implementation

Adds comprehensive error handling, validation, security, logging, and monitoring.
Builds upon Generation 1 with production-ready robustness features.
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityLevel(Enum):
    """Security strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    MAXIMUM = "maximum"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0


@dataclass
class SecurityCheckResult:
    """Result of security checks."""
    is_secure: bool
    threats_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_level: str = "low"


class RobustLogger:
    """Enhanced logging with multiple output streams and security features."""
    
    def __init__(
        self, 
        name: str = "spatial_omics_gfm",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_security_logging: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.enable_security_logging = enable_security_logging
        self.security_events = []
    
    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log info message."""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log error message."""
        self.logger.error(message, extra=extra or {})
    
    def critical(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=extra or {})
    
    def security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security-related events."""
        if self.enable_security_logging:
            event = {
                "timestamp": time.time(),
                "type": event_type,
                "details": details
            }
            self.security_events.append(event)
            self.logger.warning(f"SECURITY EVENT - {event_type}: {details}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events."""
        event_types = {}
        for event in self.security_events:
            event_type = event["type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.security_events),
            "event_types": event_types,
            "recent_events": self.security_events[-10:]  # Last 10 events
        }


class DataValidator:
    """Comprehensive data validation with multiple strictness levels."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = RobustLogger("data_validator")
    
    def validate_expression_matrix(
        self, 
        expression_matrix: List[List[float]],
        min_cells: int = 10,
        max_cells: int = 100000,
        min_genes: int = 50,
        max_genes: int = 50000
    ) -> ValidationResult:
        """Validate expression matrix data."""
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic structure validation
            if not expression_matrix:
                result.is_valid = False
                result.errors.append("Expression matrix is empty")
                return result
            
            n_cells = len(expression_matrix)
            n_genes = len(expression_matrix[0]) if expression_matrix else 0
            
            # Check dimensions
            if n_cells < min_cells:
                result.is_valid = False
                result.errors.append(f"Too few cells: {n_cells} < {min_cells}")
            
            if n_cells > max_cells:
                if self.validation_level == ValidationLevel.STRICT:
                    result.is_valid = False
                    result.errors.append(f"Too many cells: {n_cells} > {max_cells}")
                else:
                    result.warnings.append(f"Large number of cells: {n_cells}")
            
            if n_genes < min_genes:
                result.is_valid = False
                result.errors.append(f"Too few genes: {n_genes} < {min_genes}")
            
            if n_genes > max_genes:
                if self.validation_level == ValidationLevel.STRICT:
                    result.is_valid = False
                    result.errors.append(f"Too many genes: {n_genes} > {max_genes}")
                else:
                    result.warnings.append(f"Large number of genes: {n_genes}")
            
            # Check matrix consistency
            inconsistent_rows = []
            for i, row in enumerate(expression_matrix):
                if len(row) != n_genes:
                    inconsistent_rows.append(i)
            
            if inconsistent_rows:
                result.is_valid = False
                result.errors.append(
                    f"Inconsistent matrix rows: {len(inconsistent_rows)} rows with wrong length"
                )
            
            # Check for invalid values
            invalid_values = []
            negative_values = 0
            inf_values = 0
            nan_values = 0
            
            for i, row in enumerate(expression_matrix[:100]):  # Sample first 100 rows
                for j, value in enumerate(row):
                    try:
                        if value < 0:
                            negative_values += 1
                        elif value == float('inf') or value == float('-inf'):
                            inf_values += 1
                        elif value != value:  # NaN check
                            nan_values += 1
                    except (TypeError, ValueError):
                        invalid_values.append(f"Invalid value at [{i}, {j}]: {value}")
            
            if invalid_values:
                result.is_valid = False
                result.errors.extend(invalid_values[:5])  # Show first 5
                if len(invalid_values) > 5:
                    result.errors.append(f"... and {len(invalid_values) - 5} more invalid values")
            
            if negative_values > 0:
                if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    result.is_valid = False
                    result.errors.append(f"Found {negative_values} negative expression values")
                else:
                    result.warnings.append(f"Found {negative_values} negative expression values")
            
            if inf_values > 0:
                result.is_valid = False
                result.errors.append(f"Found {inf_values} infinite values")
            
            if nan_values > 0:
                result.is_valid = False
                result.errors.append(f"Found {nan_values} NaN values")
            
            # Statistical validation for paranoid mode
            if self.validation_level == ValidationLevel.PARANOID and result.is_valid:
                # Check for suspiciously uniform data
                sample_variances = []
                for row in expression_matrix[:min(50, n_cells)]:
                    if len(row) > 1:
                        mean_val = sum(row) / len(row)
                        variance = sum((x - mean_val) ** 2 for x in row) / len(row)
                        sample_variances.append(variance)
                
                if sample_variances:
                    mean_variance = sum(sample_variances) / len(sample_variances)
                    if mean_variance < 0.01:
                        result.warnings.append("Data appears suspiciously uniform (low variance)")
            
            # Store metadata
            result.metadata = {
                "n_cells": n_cells,
                "n_genes": n_genes,
                "negative_values": negative_values,
                "inf_values": inf_values,
                "nan_values": nan_values,
                "validation_level": self.validation_level.value
            }
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            self.logger.error(f"Expression matrix validation failed: {e}")
        
        finally:
            result.validation_time = time.time() - start_time
        
        return result
    
    def validate_coordinates(
        self,
        coordinates: List[List[float]],
        n_cells: int,
        spatial_dim: int = 2
    ) -> ValidationResult:
        """Validate spatial coordinates."""
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Check basic structure
            if not coordinates:
                result.is_valid = False
                result.errors.append("Coordinates are empty")
                return result
            
            if len(coordinates) != n_cells:
                result.is_valid = False
                result.errors.append(
                    f"Coordinate count mismatch: {len(coordinates)} != {n_cells}"
                )
            
            # Check coordinate dimensions
            invalid_coords = []
            for i, coord in enumerate(coordinates):
                if len(coord) != spatial_dim:
                    invalid_coords.append(i)
                
                # Check for invalid coordinate values
                for j, value in enumerate(coord):
                    try:
                        if value != value or value == float('inf') or value == float('-inf'):
                            result.is_valid = False
                            result.errors.append(f"Invalid coordinate at [{i}, {j}]: {value}")
                    except (TypeError, ValueError):
                        result.is_valid = False
                        result.errors.append(f"Non-numeric coordinate at [{i}, {j}]: {value}")
            
            if invalid_coords:
                result.is_valid = False
                result.errors.append(
                    f"{len(invalid_coords)} coordinates have wrong dimensions"
                )
            
            # Check for spatial reasonableness
            if result.is_valid and coordinates:
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                
                if x_range <= 0 or y_range <= 0:
                    result.warnings.append("Coordinates appear to be in a line or single point")
                
                # Check for duplicate coordinates
                coord_set = set(tuple(coord) for coord in coordinates)
                duplicates = len(coordinates) - len(coord_set)
                if duplicates > 0:
                    if self.validation_level == ValidationLevel.STRICT:
                        result.is_valid = False
                        result.errors.append(f"Found {duplicates} duplicate coordinates")
                    else:
                        result.warnings.append(f"Found {duplicates} duplicate coordinates")
                
                result.metadata = {
                    "spatial_dim": spatial_dim,
                    "x_range": x_range,
                    "y_range": y_range,
                    "duplicate_coordinates": duplicates
                }
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Coordinate validation error: {str(e)}")
            self.logger.error(f"Coordinate validation failed: {e}")
        
        finally:
            result.validation_time = time.time() - start_time
        
        return result
    
    def validate_gene_names(self, gene_names: List[str], n_genes: int) -> ValidationResult:
        """Validate gene names."""
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            if not gene_names:
                result.warnings.append("No gene names provided")
                return result
            
            if len(gene_names) != n_genes:
                result.is_valid = False
                result.errors.append(
                    f"Gene name count mismatch: {len(gene_names)} != {n_genes}"
                )
            
            # Check for empty or invalid gene names
            invalid_names = []
            for i, name in enumerate(gene_names):
                if not isinstance(name, str):
                    invalid_names.append(f"Position {i}: {type(name).__name__}")
                elif not name.strip():
                    invalid_names.append(f"Position {i}: empty name")
                elif len(name) > 100:  # Suspiciously long gene name
                    result.warnings.append(f"Very long gene name at position {i}: {name[:20]}...")
            
            if invalid_names:
                result.is_valid = False
                result.errors.extend(invalid_names[:5])
                if len(invalid_names) > 5:
                    result.errors.append(f"... and {len(invalid_names) - 5} more invalid names")
            
            # Check for duplicates
            unique_names = set(gene_names)
            duplicates = len(gene_names) - len(unique_names)
            if duplicates > 0:
                if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    result.is_valid = False
                    result.errors.append(f"Found {duplicates} duplicate gene names")
                else:
                    result.warnings.append(f"Found {duplicates} duplicate gene names")
            
            result.metadata = {
                "unique_names": len(unique_names),
                "duplicate_names": duplicates
            }
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Gene name validation error: {str(e)}")
            self.logger.error(f"Gene name validation failed: {e}")
        
        finally:
            result.validation_time = time.time() - start_time
        
        return result


class SecurityGuard:
    """Security validation and threat detection."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MINIMAL):
        self.security_level = security_level
        self.logger = RobustLogger("security_guard")
        
        # Suspicious patterns
        self.suspicious_patterns = [
            "__import__", "eval", "exec", "compile",
            "subprocess", "os.system", "os.popen",
            "<script", "javascript:", "data:text/html",
            "../", "..\\" 
        ]
        
        # File extension whitelist
        self.allowed_extensions = {
            ".json", ".csv", ".tsv", ".txt", ".h5", ".h5ad",
            ".xlsx", ".pkl", ".pickle", ".npz", ".zarr"
        }
    
    def sanitize_string_input(self, input_string: str) -> str:
        """Sanitize string input to remove potential threats."""
        if not isinstance(input_string, str):
            return str(input_string)
        
        # Remove null bytes
        sanitized = input_string.replace('\x00', '')
        
        # Limit length
        max_length = 10000 if self.security_level == SecurityLevel.MINIMAL else 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            self.logger.security_event(
                "INPUT_TRUNCATED", 
                {"original_length": len(input_string), "truncated_length": max_length}
            )
        
        return sanitized
    
    def check_file_path_security(self, file_path: str) -> SecurityCheckResult:
        """Check if file path is secure."""
        result = SecurityCheckResult(is_secure=True)
        
        try:
            # Convert to Path object for safer handling
            path = Path(file_path)
            
            # Check for path traversal attempts
            if ".." in str(path):
                result.is_secure = False
                result.threats_detected.append("Path traversal attempt detected")
                result.risk_level = "high"
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_extensions:
                if self.security_level == SecurityLevel.MAXIMUM:
                    result.is_secure = False
                    result.threats_detected.append(f"Disallowed file extension: {path.suffix}")
                    result.risk_level = "medium"
            
            # Check for absolute paths in paranoid mode
            if self.security_level == SecurityLevel.MAXIMUM:
                if path.is_absolute():
                    result.recommendations.append(
                        "Consider using relative paths for better security"
                    )
            
            # Check path length
            if len(str(path)) > 500:
                result.threats_detected.append("Suspiciously long file path")
                result.risk_level = "low"
        
        except Exception as e:
            result.is_secure = False
            result.threats_detected.append(f"Path validation error: {e}")
            result.risk_level = "high"
        
        return result
    
    def scan_data_for_threats(self, data: Any) -> SecurityCheckResult:
        """Scan data structure for security threats."""
        result = SecurityCheckResult(is_secure=True)
        threats_found = []
        
        def scan_recursive(obj: Any, path: str = "root") -> None:
            if isinstance(obj, str):
                # Check string for suspicious patterns
                lower_obj = obj.lower()
                for pattern in self.suspicious_patterns:
                    if pattern in lower_obj:
                        threats_found.append(f"Suspicious pattern '{pattern}' at {path}")
                
                # Check for very long strings
                if len(obj) > 50000:
                    threats_found.append(f"Very long string ({len(obj)} chars) at {path}")
            
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    scan_recursive(value, f"{path}.{key}")
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if i > 10000:  # Prevent scanning extremely large lists
                        break
                    scan_recursive(item, f"{path}[{i}]")
        
        try:
            scan_recursive(data)
            
            if threats_found:
                result.is_secure = False
                result.threats_detected = threats_found
                result.risk_level = "medium" if len(threats_found) < 5 else "high"
                
                # Log security event
                self.logger.security_event(
                    "THREAT_SCAN_FAILED",
                    {"threats_count": len(threats_found), "threats": threats_found[:5]}
                )
        
        except Exception as e:
            result.is_secure = False
            result.threats_detected.append(f"Threat scanning error: {e}")
            result.risk_level = "high"
        
        return result


class RobustSpatialData:
    """Robust spatial data with comprehensive validation and error handling."""
    
    def __init__(
        self,
        expression_matrix: List[List[float]],
        coordinates: List[List[float]],
        gene_names: Optional[List[str]] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        security_level: SecurityLevel = SecurityLevel.MINIMAL,
        enable_logging: bool = True
    ):
        self.logger = RobustLogger("robust_spatial_data") if enable_logging else None
        self.validator = DataValidator(validation_level)
        self.security = SecurityGuard(security_level)
        
        # Store validation and security results
        self.validation_results = {}
        self.security_results = {}
        
        # Initialize with validation
        self._initialize_with_validation(
            expression_matrix, coordinates, gene_names
        )
    
    def _initialize_with_validation(
        self,
        expression_matrix: List[List[float]],
        coordinates: List[List[float]],
        gene_names: Optional[List[str]]
    ) -> None:
        """Initialize data with comprehensive validation."""
        
        if self.logger:
            self.logger.info("Initializing robust spatial data with validation")
        
        # Security scan first
        security_result = self.security.scan_data_for_threats({
            "expression_matrix": expression_matrix,
            "coordinates": coordinates,
            "gene_names": gene_names
        })
        self.security_results["initialization"] = security_result
        
        if not security_result.is_secure:
            raise ValueError(f"Security threats detected: {security_result.threats_detected}")
        
        # Validate expression matrix
        expr_result = self.validator.validate_expression_matrix(expression_matrix)
        self.validation_results["expression_matrix"] = expr_result
        
        if not expr_result.is_valid:
            error_msg = "Expression matrix validation failed: " + "; ".join(expr_result.errors)
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log warnings
        if expr_result.warnings and self.logger:
            for warning in expr_result.warnings:
                self.logger.warning(f"Expression matrix: {warning}")
        
        # Store basic properties
        self.n_cells = len(expression_matrix)
        self.n_genes = len(expression_matrix[0]) if expression_matrix else 0
        
        # Validate coordinates
        coord_result = self.validator.validate_coordinates(coordinates, self.n_cells)
        self.validation_results["coordinates"] = coord_result
        
        if not coord_result.is_valid:
            error_msg = "Coordinates validation failed: " + "; ".join(coord_result.errors)
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate gene names
        if gene_names is None:
            gene_names = [f"Gene_{i:04d}" for i in range(self.n_genes)]
        
        gene_result = self.validator.validate_gene_names(gene_names, self.n_genes)
        self.validation_results["gene_names"] = gene_result
        
        if not gene_result.is_valid:
            error_msg = "Gene names validation failed: " + "; ".join(gene_result.errors)
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store validated data
        self.expression_matrix = expression_matrix
        self.coordinates = coordinates
        self.gene_names = gene_names
        
        if self.logger:
            self.logger.info(
                f"Successfully initialized robust spatial data: "
                f"{self.n_cells} cells, {self.n_genes} genes"
            )
    
    def safe_normalize_expression(self, method: str = "log1p_cpm") -> bool:
        """Safely normalize expression data with error handling."""
        
        if self.logger:
            self.logger.info(f"Starting safe normalization with method: {method}")
        
        try:
            if method == "log1p_cpm":
                # Counts per million normalization + log1p
                for i in range(self.n_cells):
                    cell_total = sum(self.expression_matrix[i])
                    if cell_total == 0:
                        if self.logger:
                            self.logger.warning(f"Cell {i} has zero total expression")
                        continue
                    
                    # Normalize to CPM
                    cpm_factor = 1000000 / cell_total
                    for j in range(self.n_genes):
                        normalized_value = self.expression_matrix[i][j] * cpm_factor
                        self.expression_matrix[i][j] = self._safe_log1p(normalized_value)
                    
            elif method == "z_score":
                # Z-score normalization per gene
                for j in range(self.n_genes):
                    gene_values = [self.expression_matrix[i][j] for i in range(self.n_cells)]
                    mean_val = sum(gene_values) / len(gene_values)
                    variance = sum((x - mean_val) ** 2 for x in gene_values) / len(gene_values)
                    
                    if variance > 0:
                        std_dev = variance ** 0.5
                        for i in range(self.n_cells):
                            self.expression_matrix[i][j] = (
                                self.expression_matrix[i][j] - mean_val
                            ) / std_dev
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Validate normalized data
            validation_result = self.validator.validate_expression_matrix(
                self.expression_matrix
            )
            
            if not validation_result.is_valid:
                if self.logger:
                    self.logger.error(
                        f"Normalization produced invalid data: {validation_result.errors}"
                    )
                return False
            
            if self.logger:
                self.logger.info("Normalization completed successfully")
            
            return True
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Normalization failed: {e}")
            return False
    
    def _safe_log1p(self, value: float) -> float:
        """Safely compute log1p with bounds checking."""
        try:
            if value < 0:
                return 0.0  # Handle negative values gracefully
            elif value > 1e10:  # Very large values
                return 25.0  # Reasonable upper bound for log values
            else:
                import math
                return math.log1p(value)
        except (ValueError, OverflowError):
            return 0.0
    
    def safe_find_neighbors(
        self, 
        k: int = 6, 
        max_distance: Optional[float] = None
    ) -> Optional[Dict[int, List[int]]]:
        """Safely find spatial neighbors with error handling."""
        
        if self.logger:
            self.logger.info(f"Finding spatial neighbors (k={k}, max_distance={max_distance})")
        
        try:
            # Validate parameters
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            
            if k >= self.n_cells:
                if self.logger:
                    self.logger.warning(f"k ({k}) >= n_cells ({self.n_cells}), using k={self.n_cells-1}")
                k = max(1, self.n_cells - 1)
            
            neighbors = {}
            
            for i in range(self.n_cells):
                try:
                    distances = []
                    for j in range(self.n_cells):
                        if i != j:
                            dist = self._safe_distance(self.coordinates[i], self.coordinates[j])
                            if max_distance is None or dist <= max_distance:
                                distances.append((dist, j))
                    
                    # Sort and take k nearest
                    distances.sort(key=lambda x: x[0])
                    neighbors[i] = [idx for _, idx in distances[:k]]
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to compute neighbors for cell {i}: {e}")
                    neighbors[i] = []
            
            if self.logger:
                avg_neighbors = sum(len(neighs) for neighs in neighbors.values()) / len(neighbors)
                self.logger.info(f"Computed neighbors successfully. Average neighbors per cell: {avg_neighbors:.2f}")
            
            return neighbors
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Neighbor computation failed: {e}")
            return None
    
    def _safe_distance(self, coord1: List[float], coord2: List[float]) -> float:
        """Safely compute Euclidean distance."""
        try:
            import math
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
        except (ValueError, OverflowError, TypeError):
            return float('inf')  # Return infinite distance for invalid coordinates
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {
            "validation_results": self.validation_results,
            "security_results": self.security_results,
            "data_properties": {
                "n_cells": self.n_cells,
                "n_genes": self.n_genes,
                "has_gene_names": len(self.gene_names) > 0
            }
        }
    
    def export_validation_report(self, file_path: str) -> bool:
        """Export validation report to file."""
        try:
            # Security check on file path
            security_result = self.security.check_file_path_security(file_path)
            if not security_result.is_secure:
                if self.logger:
                    self.logger.error(f"Insecure file path: {security_result.threats_detected}")
                return False
            
            validation_summary = self.get_validation_summary()
            with open(file_path, 'w') as f:
                json.dump(validation_summary, f, indent=2, default=str)
            
            if self.logger:
                self.logger.info(f"Validation report exported to {file_path}")
            
            return True
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to export validation report: {e}")
            return False


def run_robust_analysis_demo() -> Dict[str, Any]:
    """Demonstrate robust analysis capabilities - Generation 2."""
    print("=== Robust Spatial-Omics Analysis (Generation 2) ===")
    
    # Initialize robust logger
    logger = RobustLogger("robust_demo", log_level="INFO")
    logger.info("Starting robust analysis demonstration")
    
    try:
        # Import the enhanced basic example functions  
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from enhanced_basic_example import (
            create_enhanced_demo_data,
            EnhancedCellTypePredictor,
            EnhancedInteractionPredictor
        )
        
        # Create demo data with the pure Python implementation
        logger.info("Creating enhanced demo data")
        pure_python_data = create_enhanced_demo_data(n_cells=500, n_genes=120)
        
        # Convert to robust spatial data with validation
        logger.info("Converting to robust spatial data with comprehensive validation")
        robust_data = RobustSpatialData(
            expression_matrix=pure_python_data.expression_matrix,
            coordinates=pure_python_data.coordinates,
            gene_names=pure_python_data.gene_names,
            validation_level=ValidationLevel.STRICT,
            security_level=SecurityLevel.MINIMAL
        )
        
        logger.info("Validation completed successfully")
        
        # Safe normalization with error handling
        logger.info("Performing safe normalization")
        normalization_success = robust_data.safe_normalize_expression(method="log1p_cpm")
        
        if not normalization_success:
            logger.error("Normalization failed, using raw data")
        else:
            logger.info("Normalization completed successfully")
        
        # Safe neighbor finding
        logger.info("Computing spatial neighbors with error handling")
        neighbors = robust_data.safe_find_neighbors(k=8, max_distance=150.0)
        
        if neighbors is None:
            logger.error("Neighbor computation failed")
            neighbors = {}
        else:
            logger.info(f"Computed neighbors for {len(neighbors)} cells")
        
        # Robust cell type prediction with error handling
        logger.info("Running robust cell type prediction")
        try:
            cell_type_predictor = EnhancedCellTypePredictor()
            
            # Create compatible data object for the predictor
            class CompatibleData:
                def __init__(self, robust_data):
                    self.expression_matrix = robust_data.expression_matrix
                    self.coordinates = robust_data.coordinates
                    self.gene_names = robust_data.gene_names
                    self.n_cells = robust_data.n_cells
                    self.n_genes = robust_data.n_genes
                
                def calculate_distance(self, coord1, coord2):
                    import math
                    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                
                def find_spatial_neighbors(self, k=6):
                    neighbors = {}
                    for i in range(self.n_cells):
                        distances = []
                        for j in range(self.n_cells):
                            if i != j:
                                dist = self.calculate_distance(self.coordinates[i], self.coordinates[j])
                                distances.append((dist, j))
                        distances.sort(key=lambda x: x[0])
                        neighbors[i] = [idx for _, idx in distances[:k]]
                    return neighbors
            
            compatible_data = CompatibleData(robust_data)
            cell_type_predictions = cell_type_predictor.predict_cell_types(compatible_data)
            cell_type_assignments = cell_type_predictor.assign_best_cell_types(cell_type_predictions)
            
            logger.info(f"Successfully predicted cell types for {len(cell_type_assignments)} cells")
            
        except Exception as e:
            logger.error(f"Cell type prediction failed: {e}")
            cell_type_predictions = {}
            cell_type_assignments = []
        
        # Robust interaction prediction
        logger.info("Running robust interaction prediction")
        try:
            interaction_predictor = EnhancedInteractionPredictor()
            interactions = interaction_predictor.predict_interactions(
                compatible_data, max_distance=120.0, min_score=0.02
            )
            
            logger.info(f"Predicted {len(interactions)} interactions")
            
            # Pathway analysis
            if interactions:
                pathway_enrichment = interaction_predictor.analyze_pathway_enrichment(interactions)
                logger.info(f"Analyzed {len(pathway_enrichment)} pathways")
            else:
                pathway_enrichment = {}
            
        except Exception as e:
            logger.error(f"Interaction prediction failed: {e}")
            interactions = []
            pathway_enrichment = {}
        
        # Generate comprehensive results
        results = {
            "generation": "2_robust",
            "analysis_metadata": {
                "n_cells": robust_data.n_cells,
                "n_genes": robust_data.n_genes,
                "validation_level": "strict",
                "security_level": "standard",
                "normalization_success": normalization_success,
                "features_analyzed": [
                    "comprehensive_validation",
                    "security_scanning",
                    "safe_normalization",
                    "robust_neighbor_finding",
                    "error_handling",
                    "logging_monitoring"
                ]
            },
            "validation_summary": robust_data.get_validation_summary(),
            "cell_type_analysis": {
                "predictions": cell_type_predictions,
                "assignments": cell_type_assignments[:100],  # Limit size
                "success": len(cell_type_assignments) > 0
            },
            "interaction_analysis": {
                "interactions": interactions[:100],  # Limit size
                "total_interactions": len(interactions),
                "pathway_enrichment": pathway_enrichment,
                "success": len(interactions) > 0
            },
            "robustness_features": {
                "validation_errors": sum(
                    len(result.errors) for result in robust_data.validation_results.values()
                ),
                "validation_warnings": sum(
                    len(result.warnings) for result in robust_data.validation_results.values()
                ),
                "security_threats": sum(
                    len(result.threats_detected) for result in robust_data.security_results.values()
                ),
                "neighbor_computation_success": neighbors is not None,
                "normalization_success": normalization_success
            }
        }
        
        # Export validation report
        report_path = "/root/repo/robust_validation_report.json"
        export_success = robust_data.export_validation_report(report_path)
        
        logger.info(f"Validation report export: {'success' if export_success else 'failed'}")
        
        # Save results with proper serialization
        def make_serializable(obj):
            """Convert objects to JSON-serializable format."""
            if hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        serializable_results = make_serializable(results)
        with open("/root/repo/robust_generation2_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\u2705 Robust Generation 2 analysis complete!")
        print(f"   - Processed {robust_data.n_cells} cells and {robust_data.n_genes} genes")
        print(f"   - Validation errors: {results['robustness_features']['validation_errors']}")
        print(f"   - Validation warnings: {results['robustness_features']['validation_warnings']}")
        print(f"   - Security threats: {results['robustness_features']['security_threats']}")
        print(f"   - Normalization: {'✅' if normalization_success else '❌'}")
        print(f"   - Cell type prediction: {'✅' if len(cell_type_assignments) > 0 else '❌'}")
        print(f"   - Interaction prediction: {'✅' if len(interactions) > 0 else '❌'}")
        
        logger.info("Robust analysis completed successfully")
        return results
    
    except Exception as e:
        logger.critical(f"Robust analysis failed with critical error: {e}")
        return {
            "generation": "2_robust",
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    run_robust_analysis_demo()
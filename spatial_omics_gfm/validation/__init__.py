"""Validation and error handling for spatial-omics-gfm."""

from .data_validation import (
    DataValidator,
    ValidationError,
    validate_expression_matrix,
    validate_coordinates,
    validate_gene_names
)

from .input_validation import (
    InputValidator,
    sanitize_file_path,
    validate_parameters,
    check_memory_requirements
)

from .security import (
    SecurityValidator,
    check_file_safety,
    sanitize_user_input,
    validate_permissions
)

__all__ = [
    "DataValidator",
    "ValidationError", 
    "validate_expression_matrix",
    "validate_coordinates",
    "validate_gene_names",
    "InputValidator",
    "sanitize_file_path",
    "validate_parameters",
    "check_memory_requirements",
    "SecurityValidator",
    "check_file_safety",
    "sanitize_user_input",
    "validate_permissions"
]
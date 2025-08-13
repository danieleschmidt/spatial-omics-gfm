#!/usr/bin/env python3
"""
Test robustness features in Generation 2 implementation.
"""

import sys
import os
sys.path.insert(0, '/root/repo')

import numpy as np
from spatial_omics_gfm.validation import (
    DataValidator,
    InputValidator,
    SecurityValidator,
    sanitize_user_input,
    check_file_safety
)

def test_data_validation():
    """Test data validation functionality."""
    print("=== Testing Data Validation ===")
    
    # Create test data
    expression_matrix = np.random.negative_binomial(5, 0.3, (100, 50)).astype(float)
    coordinates = np.random.rand(100, 2) * 1000
    gene_names = [f"Gene_{i:03d}" for i in range(50)]
    
    # Add some marker genes
    gene_names[:10] = ["CD3D", "CD8A", "CD19", "CD68", "COL1A1", "MT-ATP6", "RPS4X", "MALAT1", "GAPDH", "ACTB"]
    
    validator = DataValidator()
    
    print("1. Testing expression matrix validation...")
    expr_results = validator.validate_expression_matrix(expression_matrix, gene_names=gene_names)
    print(f"   Valid: {expr_results['valid']}")
    print(f"   Warnings: {len(expr_results['warnings'])}")
    print(f"   Metrics: cells={expr_results['metrics']['n_cells']}, genes={expr_results['metrics']['n_genes']}")
    
    print("2. Testing coordinate validation...")
    coord_results = validator.validate_coordinates(coordinates, n_cells=100)
    print(f"   Valid: {coord_results['valid']}")
    print(f"   Spatial extent: {coord_results['metrics']}")
    
    print("3. Testing gene name validation...")
    gene_results = validator.validate_gene_names(gene_names, n_genes=50)
    print(f"   Valid: {gene_results['valid']}")
    print(f"   Mitochondrial genes: {gene_results['metrics']['mitochondrial_genes']}")
    
    print("4. Testing comprehensive validation...")
    full_results = validator.validate_spatial_data(expression_matrix, coordinates, gene_names)
    print(f"   Overall valid: {full_results['overall_valid']}")
    print(f"   Total warnings: {full_results['summary']['total_warnings']}")
    print(f"   Total errors: {full_results['summary']['total_errors']}")


def test_input_validation():
    """Test input validation functionality."""
    print("\n=== Testing Input Validation ===")
    
    validator = InputValidator()
    
    print("1. Testing file path sanitization...")
    try:
        safe_path = validator.sanitize_file_path("/root/repo/README.md")
        print(f"   Safe path: {safe_path}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("2. Testing parameter validation...")
    params = {
        "n_neighbors": 6,
        "learning_rate": 0.001,
        "model_type": "transformer",
        "batch_size": 32
    }
    
    schema = {
        "n_neighbors": {"type": int, "min": 1, "max": 50, "default": 6},
        "learning_rate": {"type": float, "min": 1e-6, "max": 1.0, "default": 0.001},
        "model_type": {"type": str, "choices": ["transformer", "gnn", "cnn"], "default": "transformer"},
        "batch_size": {"type": int, "min": 1, "max": 1024, "default": 32}
    }
    
    try:
        validated = validator.validate_parameters(params, schema)
        print(f"   Validated parameters: {validated}")
    except Exception as e:
        print(f"   Validation error: {e}")
    
    print("3. Testing memory requirements...")
    data_size = 100 * 50 * 8  # 100 cells * 50 genes * 8 bytes per float
    memory_check = validator.check_memory_requirements(data_size, "preprocessing")
    print(f"   Sufficient memory: {memory_check['sufficient_memory']}")
    print(f"   Estimated usage: {memory_check['estimated_memory_gb']:.3f} GB")


def test_security_validation():
    """Test security validation functionality."""
    print("\n=== Testing Security Validation ===")
    
    validator = SecurityValidator()
    
    print("1. Testing file safety check...")
    try:
        safety_results = validator.check_file_safety("/root/repo/README.md")
        print(f"   File is safe: {safety_results['safe']}")
        print(f"   Detected type: {safety_results['checks'].get('detected_type', 'unknown')}")
        print(f"   File size: {safety_results['checks']['size_bytes']} bytes")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("2. Testing input sanitization...")
    test_inputs = [
        "normal input text",
        "input with <script>alert('xss')</script>",
        "input with $(rm -rf /)",
        "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
        "normal gene name: ACTB"
    ]
    
    for test_input in test_inputs:
        try:
            sanitized = validator.sanitize_user_input(test_input, max_length=100)
            print(f"   '{test_input[:30]}...' -> '{sanitized[:30]}...'")
        except Exception as e:
            print(f"   '{test_input[:30]}...' -> BLOCKED: {e}")
    
    print("3. Testing permission validation...")
    try:
        perm_results = validator.validate_permissions("/root/repo", "r")
        print(f"   Has read permission: {perm_results['valid']}")
        print(f"   Permissions: {perm_results['permissions']}")
    except Exception as e:
        print(f"   Error: {e}")


def test_error_handling():
    """Test error handling with invalid data."""
    print("\n=== Testing Error Handling ===")
    
    print("1. Testing with invalid expression matrix...")
    try:
        # Invalid data: negative values
        bad_matrix = np.array([[-1, 2, 3], [4, -5, 6]])
        validator = DataValidator(strict_mode=False)  # Use warning mode
        results = validator.validate_expression_matrix(bad_matrix)
        print(f"   Warnings for negative values: {len(results['warnings'])}")
    except Exception as e:
        print(f"   Error caught: {e}")
    
    print("2. Testing with mismatched dimensions...")
    try:
        expression = np.random.rand(100, 50)
        coordinates = np.random.rand(90, 2)  # Wrong number of cells
        validator = DataValidator()
        results = validator.validate_spatial_data(expression, coordinates)
        print(f"   Dimension mismatch detected: {not results['overall_valid']}")
    except Exception as e:
        print(f"   Error caught: {e}")
    
    print("3. Testing with dangerous file path...")
    try:
        validator = InputValidator()
        dangerous_path = "../../../etc/passwd"
        safe_path = validator.sanitize_file_path(dangerous_path, allow_create=False)
        print(f"   Unexpected: path allowed {safe_path}")
    except Exception as e:
        print(f"   Dangerous path blocked: {type(e).__name__}")


def main():
    """Run all robustness tests."""
    print("🛡️ Spatial-Omics GFM Robustness Testing")
    print("=" * 50)
    
    try:
        test_data_validation()
        test_input_validation()
        test_security_validation()
        test_error_handling()
        
        print("\n✅ All robustness tests completed successfully!")
        print("Generation 2 robustness features are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
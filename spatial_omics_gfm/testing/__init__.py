"""
Lightweight Testing Framework for Spatial-Omics GFM

Provides testing capabilities that work without heavy ML dependencies.
"""

from .lightweight_test_framework import (
    LightweightTestRunner,
    TestResult,
    run_comprehensive_tests
)

__all__ = [
    'LightweightTestRunner',
    'TestResult', 
    'run_comprehensive_tests'
]
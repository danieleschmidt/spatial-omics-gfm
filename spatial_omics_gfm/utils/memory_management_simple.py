"""
Memory management utilities with automatic fallback to simple version.
This module detects available dependencies and imports the appropriate implementation.
"""

# Check for heavy dependencies
try:
    import torch
    import anndata
    import h5py
    HEAVY_DEPS_AVAILABLE = True
except ImportError:
    HEAVY_DEPS_AVAILABLE = False

if HEAVY_DEPS_AVAILABLE:
    # Import from full implementation when dependencies are available
    try:
        from .memory_management import MemoryMonitor, memory_managed_operation
    except Exception:
        # Fallback if full implementation fails for any reason
        from .simple_memory_management import MemoryMonitor, memory_managed_operation
else:
    # Use simple implementation when dependencies are missing
    from .simple_memory_management import MemoryMonitor, memory_managed_operation

# Export the same interface regardless of which implementation is used
__all__ = ['MemoryMonitor', 'memory_managed_operation']
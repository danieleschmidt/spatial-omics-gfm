"""
Test suite for Spatial-Omics GFM.
Comprehensive tests for all components with 85%+ coverage target.
"""

import pytest
import warnings
import numpy as np
import torch
from pathlib import Path

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_FIXTURES_DIR.mkdir(exist_ok=True)

# Global test configuration
torch.manual_seed(42)
np.random.seed(42)

# Set device for testing
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
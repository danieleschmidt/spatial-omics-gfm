# API Documentation

## Core Classes

### SimpleSpatialData
```python
from spatial_omics_gfm.core import SimpleSpatialData

class SimpleSpatialData:
    """Simple spatial transcriptomics data container"""
    
    def __init__(self, expression_matrix, coordinates, gene_names=None, cell_ids=None):
        """Initialize spatial data
        
        Args:
            expression_matrix: Gene expression matrix (cells x genes)
            coordinates: Spatial coordinates (cells x 2)
            gene_names: List of gene names
            cell_ids: List of cell identifiers
        """
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics of the data"""
    
    def find_spatial_neighbors(self, k: int = 6) -> list:
        """Find spatial neighbors for each cell"""
```

### Data Loaders
```python
from spatial_omics_gfm.data import VisiumDataset, XeniumDataset

# Load Visium data
dataset = VisiumDataset.from_10x_folder("path/to/visium/")

# Load Xenium data  
dataset = XeniumDataset("path/to/xenium/")
```

### Model Classes
```python
from spatial_omics_gfm.models import SpatialGraphTransformer

model = SpatialGraphTransformer(
    num_genes=3000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8
)
```

## REST API Endpoints

### Data Upload
```
POST /api/v1/data/upload
Content-Type: multipart/form-data

Parameters:
- file: Spatial data file (h5, csv)
- platform: Platform type (visium, xenium, slideseq)
```

### Analysis
```
POST /api/v1/analysis/cell-typing
Content-Type: application/json

{
  "data_id": "string",
  "model": "spatial-gfm-base",
  "parameters": {
    "confidence_threshold": 0.8
  }
}
```

### Results
```
GET /api/v1/results/{job_id}

Response:
{
  "status": "completed",
  "results": {
    "cell_types": [...],
    "interactions": [...],
    "pathways": [...]
  }
}
```

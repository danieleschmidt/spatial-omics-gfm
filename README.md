# spatial-omics-gfm

**Graph Foundation Model for Spatial Transcriptomics**

Processes spatially-resolved gene expression data by representing tissue sections as spatial k-NN graphs and learning joint representations of gene expression and tissue topology.

---

## What it does

Modern spatial transcriptomics platforms (10x Visium, Slide-seq, MERFISH, Xenium) capture thousands of genes at hundreds to millions of spatial locations across a tissue slice. Each measurement spot has:

- **(x, y) coordinates** in the tissue
- A **gene expression vector** (UMI counts per gene)

This library treats the tissue section as a **spatial graph** — spots are nodes, spatial proximity defines edges — and learns spot embeddings that integrate both transcriptional content and tissue architecture.

---

## Core classes

| Class | Description |
|-------|-------------|
| `SpatialSpot` | Single spot: (x, y) + gene expression vector. Handles log-normalization. |
| `SpatialGraph` | Builds a directed k-NN graph from spot coordinates. Nodes = spots, edges = spatial proximity, node features = normalized expression, edge features = (distance, angle). |
| `SpatialGraphTransformer` | 2–3 layer graph transformer. Encodes expression with a linear encoder, coordinates with sinusoidal Fourier features, fuses them, then propagates with spatial attention (edge features as attention bias). Pure PyTorch — no torch_geometric needed. |
| `CellCommunicationInferer` | Scores ligand-receptor co-expression across spatially adjacent spot pairs. Also provides expression correlation analysis and spatial domain detection via clustering. |
| `SyntheticSpatialDataset` | Generates synthetic Visium-like data: grid of spots with spatially varying expression patterns mimicking distinct tissue regions. |

---

## Quick start

```python
from spatial_omics_gfm import (
    SyntheticSpatialDataset,
    SpatialGraph,
    SpatialGraphTransformer,
    CellCommunicationInferer,
)

# 1. Generate synthetic data
dataset = SyntheticSpatialDataset(n_spots=100, n_genes=500, n_regions=5)

# 2. Build spatial graph
graph = SpatialGraph(dataset.spots, k=6)
# → SpatialGraph(nodes=100, edges=600, n_genes=500, k=6)

# 3. Train graph transformer
model = SpatialGraphTransformer.from_graph(graph, hidden_dim=128, num_layers=3, n_classes=5)
embeddings, logits = model(
    graph.node_features, graph.coords, graph.edge_index, graph.edge_attr
)

# 4. Detect spatial domains
domain_labels = CellCommunicationInferer.detect_spatial_domains(
    embeddings, graph.coords, n_domains=5
)

# 5. Infer cell-cell communication
lr_pairs = CellCommunicationInferer.make_synthetic_lr_pairs(n_genes=500)
infer = CellCommunicationInferer(lr_pairs)
results = infer.infer(graph.node_features, graph.edge_index, graph.edge_attr)
for r in results:
    print(r, r.top_interactions(n=3))
```

Run the full demo:

```bash
python3 examples/demo_spatial_gfm.py
```

---

## Architecture

### SpatialGraphTransformer

```
gene expression (n_genes)
        │
   LinearEncoder → LayerNorm → GELU         (expression embedding)
        │
        ├── SpatialPositionEncoding          (Fourier features → MLP)
        │
   FusionGate (concat → linear)
        │
   ┌─ GraphAttnLayer × num_layers ─┐
   │  Q/K/V projections             │
   │  + edge bias (dist, angle)     │
   │  + segment softmax             │
   │  + FFN + LayerNorm             │
   └────────────────────────────────┘
        │
   per-node embeddings (hidden_dim)
        │
   [optional] classification head (n_classes)
```

### CellCommunicationInferer

For each directed edge (sender → receiver) and each L-R pair:

```
score = max(ligand_expr(sender) − θ, 0)
      × max(receptor_expr(receiver) − θ, 0)
      × exp(−distance / distance_scale)
```

where θ is a minimum expression threshold.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU or CUDA)
- NumPy
- scikit-learn (for KMeans in domain detection)

No `torch_geometric` or other graph libraries needed.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

37 tests covering all modules.

---

## Project structure

```
spatial_omics_gfm/
    __init__.py          — public API
    spot.py              — SpatialSpot
    graph.py             — SpatialGraph
    model.py             — SpatialGraphTransformer, SpatialPositionEncoding
    communication.py     — CellCommunicationInferer, LRPair, CommunicationResult
    dataset.py           — SyntheticSpatialDataset
tests/
    test_spot.py
    test_graph.py
    test_model.py
    test_communication.py
    test_dataset.py
examples/
    demo_spatial_gfm.py  — end-to-end demo
```

---

## License

MIT

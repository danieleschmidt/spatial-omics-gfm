"""
Spatial-Omics GFM Demo
======================
1. Generate a 100-spot synthetic Visium-like dataset
2. Build a spatial k-NN graph
3. Train SpatialGraphTransformer for spatial domain classification
4. Run CellCommunicationInferer to detect L-R signalling
5. Print summary statistics

Run with:
    ~/anaconda3/bin/python3 examples/demo_spatial_gfm.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from spatial_omics_gfm import (
    SyntheticSpatialDataset,
    SpatialGraph,
    SpatialGraphTransformer,
    CellCommunicationInferer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Generating synthetic spatial transcriptomics dataset")
print("=" * 60)

dataset = SyntheticSpatialDataset(n_spots=100, n_genes=500, n_regions=5, seed=42)
print(dataset)
print(f"   Unique cell types: {dataset.unique_cell_types()}")
print(f"   Sample spot: {dataset[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build spatial graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Building spatial k-NN graph (k=6)")
print("=" * 60)

graph = SpatialGraph(dataset.spots, k=6)
print(graph)
print(f"   Avg degree: {graph.num_edges / graph.num_nodes:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train SpatialGraphTransformer
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Training SpatialGraphTransformer (3 layers, 4 heads)")
print("=" * 60)

# Encode cell types as integer labels
cell_types = dataset.unique_cell_types()
ct2idx = {ct: i for i, ct in enumerate(cell_types)}
labels = torch.tensor(
    [ct2idx[s.cell_type] for s in dataset.spots], dtype=torch.long
)

model = SpatialGraphTransformer.from_graph(
    graph,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    n_classes=len(cell_types),
).to(DEVICE)
print(model)

node_feat = graph.node_features.to(DEVICE)
coords = graph.coords.to(DEVICE)
edge_index = graph.edge_index.to(DEVICE)
edge_attr = graph.edge_attr.to(DEVICE)
labels_d = labels.to(DEVICE)

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

losses = []
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    _, logits = model(node_feat, coords, edge_index, edge_attr)
    loss = F.cross_entropy(logits, labels_d)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            _, logits_e = model(node_feat, coords, edge_index, edge_attr)
        preds = logits_e.argmax(dim=-1)
        acc = (preds == labels_d).float().mean().item()
        print(f"   Epoch {epoch:3d} | loss={loss.item():.4f} | acc={acc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spatial domain identification from embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Spatial domain detection via clustering on embeddings")
print("=" * 60)

model.eval()
with torch.no_grad():
    embeddings, _ = model(node_feat, coords, edge_index, edge_attr)

domain_labels = CellCommunicationInferer.detect_spatial_domains(
    embeddings.cpu(), graph.coords, n_domains=5
)

from collections import Counter
domain_counts = Counter(domain_labels.tolist())
print("   Spots per domain:", dict(sorted(domain_counts.items())))

# Compare with true cell-type labels
true_labels = labels.numpy()
# Simple agreement (majority label per cluster)
for d in sorted(domain_counts):
    mask = domain_labels == d
    majority = Counter(true_labels[mask].tolist()).most_common(1)[0]
    print(f"   Domain {d}: {domain_counts[d]} spots  — "
          f"majority cell type '{cell_types[majority[0]]}' ({majority[1]}/{domain_counts[d]})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cell-cell communication inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Inferring cell-cell communication (L-R co-expression)")
print("=" * 60)

lr_pairs = CellCommunicationInferer.make_synthetic_lr_pairs(n_genes=500, n_pairs=5)
infer = CellCommunicationInferer(lr_pairs, distance_scale=2.0)
results = infer.infer(graph.node_features, graph.edge_index, graph.edge_attr)

for res in results:
    top = res.top_interactions(n=3)
    print(f"\n   {res}")
    for sender, receiver, score in top:
        print(f"     spot_{sender:04d} → spot_{receiver:04d}  score={score:.4f}")

# Expression correlation across edges
corr = CellCommunicationInferer.pairwise_correlation(
    graph.node_features, graph.edge_index, top_genes=100
)
print(f"\n   Edge expression correlation: "
      f"mean={corr.mean():.3f} ± {corr.std():.3f}, "
      f"max={corr.max():.3f}")

print("\n" + "=" * 60)
print("Demo complete ✓")
print("=" * 60)

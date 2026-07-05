# Graph Representations

A graph representation treats a slide as nodes plus edges:

```math
\mathcal{X}_i=(V_i,E_i,H_i).
```

Nodes may be:

```text
patches
cells
superpixels
regions
tissue components
prototypes
```

Edges encode relationships:

```text
spatial adjacency
k-nearest neighbors
cell-cell contact
region containment
morphological similarity
learned topology
```

## Files

- `01_slide_as_graph.md`: graph object, node features, edge structure.
- `02_message_passing.md`: GNN maps and permutation equivariance.
- `03_graph_construction.md`: kNN, radius, cell graphs, heterogeneous graphs.
- `04_readout_and_surviving_statistics.md`: graph-level representations.
- `05_failure_modes.md`: oversmoothing, oversquashing, topology errors.

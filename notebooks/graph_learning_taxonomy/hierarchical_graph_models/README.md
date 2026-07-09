# Hierarchical Graph Models

This folder derives hierarchical graph models in computational pathology.

The anchor papers for this portion are:

```text
Pati et al.
HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for
Histopathological Image Classification
arXiv 2020
https://arxiv.org/abs/2007.00584
```

```text
Pati et al.
Hierarchical graph representations in digital pathology
Medical Image Analysis 2022
https://doi.org/10.1016/j.media.2021.102264
https://arxiv.org/abs/2102.11057
```

The important move is not merely deeper message passing. HACT changes the
mathematical object being processed:

```text
single graph over patches
    becomes
cell graph + tissue graph + hard cell-to-tissue assignment
```

So the slide or region is no longer represented by one untyped graph. It is a
two-level entity graph with an explicit hierarchy.

## Notation

The HACT papers use assignment notation such as `S_cell_to_tissue` or
`A_cell_to_tissue`. These notes write the hard assignment matrix as:

```math
B_i^{\mathrm{cell}\to\mathrm{tissue}}
```

to avoid overloading:

```text
A:
    adjacency matrices

S:
    supervision in the C/R/G/S decomposition
```

Class count is written as:

```math
K_{\mathrm{cls}}
```

so it does not collide with the context operator `C`.

## Files

- `01_hact_slide_object.md`: the HACT object and its relation to WSI graph
  learning.
- `02_cell_graph_and_tissue_graph_construction.md`: cell graph, tissue graph,
  topology, and paper-specific construction details.
- `03_cell_to_tissue_assignment_operator.md`: hard assignment, hierarchical
  pooling, and what information survives.
- `04_hact_net_context_and_readout.md`: GIN/PNA HACT-Net forward maps,
  jumping knowledge, readout, and supervision.
- `05_hact_failure_modes.md`: segmentation, hierarchy, pooling, topology, and
  field-of-view failures.

## C/R/G/S Placement

```text
G:
    typed hierarchical entity graph
    G_HACT = {G_cell, G_tissue, B_cell_to_tissue}

C:
    cell-level GNN
    hard cell-to-tissue pooling
    tissue-level GNN

R:
    tissue-node readout after hierarchical context injection

S:
    tissue-region classification label through cross-entropy
```

## Why It Matters

Patch-GCN asks:

```text
How do neighboring patches exchange information?
```

HACT asks:

```text
Which biological entities exist at different scales,
and how does one scale condition the next?
```

This is a different inductive bias. Patch-GCN keeps one entity type. HACT makes
the hierarchy part of the input object.

# Heterogeneous Graph Models

This folder derives heterogeneous graph models for WSI learning.

The anchor paper for this portion is:

```text
Chan et al.
Histopathology Whole Slide Image Analysis With Heterogeneous Graph
Representation Learning
CVPR 2023
https://arxiv.org/abs/2307.04189
```

The important move is:

```text
patch graph
    plus
node type from nucleus pseudo-labels
    plus
continuous edge attributes
```

This is different from Patch-GCN and HACT.

```text
Patch-GCN:
    homogeneous patch graph with coordinate kNN support

HACT:
    hierarchical cell graph + tissue graph + assignment matrix

Chan et al.:
    feature-kNN patch graph with HoVer-Net pseudo-label node types and
    Pearson edge attributes
```

## Files

- `01_heterogeneous_wsi_graph_object.md`: heterogeneous WSI graph object.
- `02_node_and_edge_type_construction.md`: patch nodes, HoVer-Net node types,
  KimiaNet features, feature-space kNN, and Pearson edge attributes.
- `03_heat_context_operator.md`: Heterogeneous Edge Attribute Transformer
  message passing.
- `04_pseudo_label_pooling.md`: semantic-consistent pooling by pseudo-label
  node type.
- `05_causal_localization_and_supervision.md`: classification losses and
  leave-one-node causal localization.
- `06_failure_modes.md`: pseudo-label noise, feature kNN support, edge
  attribute brittleness, PL pooling, and causality caveats.
- `07_full_forward_map.md`: full construction, HEAT, PL pooling, head, and
  localization pipeline.

## C/R/G/S Placement

```text
\mathcal{G}:
    heterogeneous patch graph with node types and edge attributes

\mathcal{C}:
    HEAT message passing with node-type-specific projections and edge
    attribute modulation

\mathcal{R}:
    pseudo-label graph pooling over fixed node-type clusters

\mathcal{S}:
    slide-level classification, staging, or typing label through cross-entropy
```

## Source-Specific Caveat

The node is still a patch. The node type is derived from nuclei inside that
patch by a pretrained HoVer-Net model and majority voting.

So this method is not:

```text
cell graph over individual nuclei
```

It is:

```text
patch graph whose patch nodes carry nucleus-type pseudo-labels
```

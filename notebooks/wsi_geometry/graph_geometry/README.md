# Graph Geometry

Graph geometry represents spatial or relational structure with edges:

```math
G_i
=
(V_i,E_i,A_i),
\qquad
V_i=\{1,\ldots,n_i\}.
```

Nodes may be patches, cells, regions, or tissue components. Edges may come from
coordinates, morphology similarity, radius thresholds, kNN, tissue hierarchy, or
learned relations.

## Files

- `01_edges_as_geometry.md`: adjacency as a spatial assumption.
- `02_message_passing_and_hop_neighborhoods.md`: graph context and receptive fields.
- `03_laplacian_smoothing_and_oversquashing.md`: what repeated message passing does.
- `04_failure_modes.md`: wrong edges, disconnected components, and density artifacts.

## C/R/G/S Placement

```text
G:
    graph adjacency, edge weights, and edge features

C:
    message passing or graph attention

R:
    graph-level pooling over contextualized nodes

S:
    slide labels, survival targets, or graph regularizers
```

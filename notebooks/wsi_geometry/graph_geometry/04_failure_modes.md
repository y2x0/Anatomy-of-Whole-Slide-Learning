# Failure Modes

Graph geometry fails when edges do not match the tissue relations needed by the
task.

## Wrong Adjacency

If the true context relation is $E^\star$ but the model uses $E$, then message
passing computes:

```math
h_j^{(L)}
=
F_\theta(\{h_k:d_E(j,k)\le L\})
```

instead of:

```math
h_j^{\star(L)}
=
F_\theta^\star(\{h_k:d_{E^\star}(j,k)\le L\}).
```

The model can be expressive and still reason over the wrong neighborhood.

## kNN Density Artifacts

In a kNN graph, every node has roughly fixed out-degree:

```math
|\mathcal{N}_k(j)|=k.
```

This ignores physical density. A sparse tissue area and dense tissue area may
receive the same number of neighbors but at very different physical radii:

```math
\max_{k'\in\mathcal{N}_k(j)}\|c_j-c_{k'}\|_2.
```

The graph normalizes topology while distorting physical scale.

## Radius Graph Degree Artifacts

Radius graphs preserve scale:

```math
\|c_j-c_k\|_2\le r.
```

But degree varies:

```math
d_j
=
\sum_k A_{jk}.
```

High-density regions may dominate aggregation unless normalization is handled
carefully.

## Disconnected Tissue Components

If the graph has connected components:

```math
V
=
\bigcup_{m=1}^{M}V_m,
\qquad
E(V_m,V_{m'})=\varnothing
```

then message passing cannot move information across components. The readout
must combine component-level evidence without contextual interaction.

## Feature-Similarity Edges Can Ignore Space

Morphology similarity graphs connect visually similar patches:

```math
\mathrm{sim}(h_j,h_k)\ge\tau.
```

This is useful for phenotype grouping, but it can connect distant regions and
erase local tissue architecture.

## Dense Summary

Graph failures are geometry failures:

```text
wrong neighbors
wrong physical scale
degree artifacts
disconnected components
oversmoothing
oversquashing
feature-space shortcuts
```

The graph should be evaluated as part of the model, not as harmless
preprocessing.

# BioX-CPath Stain-Aware Node Attention

BioX-CPath constructs multistain WSI graphs and derives stain-aware readouts
from graph attention and top-k node selection.

## 1. Graph Object

The input combines feature-space edges and region-adjacency edges:

```math
G^{\mathrm{FRA}}
=\left(V,E^{\mathrm{FS}}\cup E^{\mathrm{RA}},X,S\right),
```

where `S_v` is the stain type of node `v`. The two edge families have different
semantics even when they are processed by the same GNN.

## 2. Node Scores and Selection

At a SAAP block, a GNN produces node scores

```math
a=\mathrm{GNN}(X,A_{\mathrm{FRA}})
\in\mathbb R^{N}.
```

The top fraction is selected:

```math
V'=\mathrm{TopK}(a),
\qquad
X'=X[V'].
```

Selected features are scaled by normalized scores before stain-wise pooling.
This is a graph-dependent top-k readout, not a post-hoc attribution applied
after classification.

## 3. Stain Attention

Let `a'_v` be normalized selected-node scores and `V'_s` the selected nodes from
stain `s`. The stain attention is

```math
\alpha_s
=\sum_{v\in V'_s}a'_v.
```

It estimates how much selected node mass belongs to a stain. It is not a signed
class contribution unless the final head is explicitly decomposed by stain.

## 4. Spatial Rendering

BioX-CPath propagates node scores back to WSI coordinates and overlays them.
Because the score was produced after graph context and top-k selection, the
rendered region represents selected graph-node importance under that operator,
not isolated pixel evidence.


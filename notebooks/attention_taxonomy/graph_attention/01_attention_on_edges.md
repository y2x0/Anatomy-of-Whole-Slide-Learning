# Attention On Edges

Graph attention assigns weights only to graph neighbors.

Let:

```math
G
=
(V,E),
\qquad
h_u\in\mathbb{R}^{d}.
```

For node `u`, neighbors are:

```math
\mathcal{N}(u)
=
\{v:(u,v)\in E\}.
```

A graph attention layer computes:

```math
e_{uv}
=
\psi_\theta(W h_u,W h_v),
\qquad
v\in\mathcal{N}(u).
```

Weights are normalized only over the neighborhood:

```math
a_{uv}
=
\frac{\exp(e_{uv})}
{\sum_{r\in\mathcal{N}(u)}\exp(e_{ur})}.
```

The update is:

```math
h_u'
=
\sigma
\left(
\sum_{v\in\mathcal{N}(u)}
a_{uv}W h_v
\right).
```

## Attention Matrix

The graph attention matrix satisfies:

```math
a_{uv}=0
\quad
\text{if }(u,v)\notin E.
```

It is a row-stochastic weighted adjacency:

```math
A_\theta(G,H)\mathbf{1}
=
\mathbf{1}.
```

This statement assumes each node has at least one allowed source, usually by
including self-loops:

```math
|\mathcal{N}(u)|
>
0.
```

Thus graph attention is masked self-attention where the mask is the adjacency.

## WSI Graph Interpretation

For whole-slide graphs:

```text
nodes:
    patches, superpatches, cells, glands, regions

edges:
    spatial kNN, radius graph, Delaunay graph, learned similarity graph

attention:
    edge-specific learned importance
```

The graph defines who can communicate. Attention defines how strongly messages
move along available edges.

## Dense Summary

Graph attention is:

```math
\boxed{
h_u'
=
\sum_{v}
\frac{M_{uv}\exp(e_{uv})}
{\sum_r M_{ur}\exp(e_{ur})}
W h_v
}
```

with:

```math
M_{uv}
=
\mathbf{1}\{(u,v)\in E\}.
```

It learns weights on an existing topology unless the graph itself is also
learned.

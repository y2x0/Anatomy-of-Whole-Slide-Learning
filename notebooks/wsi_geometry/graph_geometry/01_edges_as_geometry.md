# Edges As Geometry

A graph over WSI units is:

```math
G_i
=
(V_i,E_i),
\qquad
V_i=\{1,\ldots,n_i\}.
```

Edges define which units are allowed to exchange information before readout.

## Coordinate kNN

Given coordinates $c_{ij}$:

```math
\mathcal{N}_k(j)
=
\mathrm{kNN}_k(c_{ij};\{c_{i\ell}\}_{\ell=1}^{n_i}).
```

Edges are:

```math
(j,k)\in E_i
\quad\Longleftrightarrow\quad
k\in\mathcal{N}_k(j).
```

Patch-GCN-style WSI graph models fit this geometry: WSI patches become points,
and local graph context is built from their spatial arrangement.

## Radius Graph

A radius graph uses physical distance:

```math
(j,k)\in E_i
\quad\Longleftrightarrow\quad
\|c_{ij}-c_{ik}\|_2\le r.
```

This preserves scale better than fixed kNN, but the node degree depends on
tissue density:

```math
d_j
=
|\{k:\|c_j-c_k\|_2\le r\}|.
```

## Similarity Graph

A morphology graph uses feature similarity:

```math
(j,k)\in E_i
\quad\Longleftrightarrow\quad
\mathrm{sim}(h_{ij},h_{ik})\ge\tau.
```

This is not physical adjacency. It is a relation graph in feature space.

## Weighted Edge Features

Edges may carry attributes:

```math
e_{jk}
=
\left[
c_j-c_k,
\|c_j-c_k\|_2,
\mathrm{sim}(h_j,h_k)
\right].
```

A message function can use them:

```math
m_j
=
\sum_{k\in\mathcal{N}(j)}
\psi_\theta(h_j,h_k,e_{jk}).
```

## C/R/G/S Placement

```text
G:
    adjacency E_i and edge features e_jk

C:
    graph message passing or graph attention

R:
    graph-level pooling after contextualization

S:
    task loss shapes whether the graph context becomes predictive
```

## Dense Summary

An edge is an inductive-bias statement:

```text
node k is relevant context for node j
```

The graph is not a neutral preprocessing step. It is the mathematical definition
of local tissue context.

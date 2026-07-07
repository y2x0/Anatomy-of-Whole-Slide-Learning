# Tree, DAG, And Scale Index

Hierarchies differ by whether parent assignment is hard, soft, single-parent, or
multi-parent.

## Tree Or Forest

A hard tree has one parent per node:

```math
\pi^{(\ell)}(v)
=
u.
```

The cross-level edge set is:

```math
B^{(\ell)}
=
\{(v,\pi^{(\ell)}(v)):v\in V^{(\ell)}\}.
```

This is appropriate when containment is unambiguous.

## Soft DAG

If a fine unit can contribute to multiple coarse units, use assignment weights:

```math
w_{vu}^{(\ell)}
\ge
0,
\qquad
\sum_{u\in V^{(\ell+1)}}w_{vu}^{(\ell)}
=
1.
```

Then:

```math
h_u^{(\ell+1)}
=
\sum_{v\in V^{(\ell)}}
w_{vu}^{(\ell)}
\phi_\theta(h_v^{(\ell)}).
```

This creates a directed acyclic graph rather than a tree.

## Scale Index

Each level has a field of view:

```math
\Delta_\ell>0,
\qquad
\Delta_0<\Delta_1<\cdots<\Delta_L.
```

A node is not only a feature. It is a feature at scale:

```math
(h_v^{(\ell)},c_v^{(\ell)},\Delta_\ell).
```

The same coordinate at different scales means different tissue support:

```math
B(c,\Delta_\ell)
\ne
B(c,\Delta_{\ell'}).
```

## Cross-Level Consistency

If a coarse node represents a region, its coordinate can be defined as:

```math
c_u^{(\ell+1)}
=
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}
c_v^{(\ell)}.
```

Its support is:

```math
\Omega_u^{(\ell+1)}
=
\bigcup_{v\in\mathrm{Ch}(u)}
\Omega_v^{(\ell)}.
```

Geometry is inconsistent if coarse supports do not match child supports.

## C/R/G/S Placement

```text
G:
    tree, DAG, assignment weights, and scale index

C:
    cross-level message passing or scale-aware attention

R:
    child-to-parent pooling or multiscale fusion

S:
    task and pretraining losses shape which scales matter
```

## Dense Summary

Hierarchy geometry is not one thing:

```text
hard tree:
    exact containment

soft DAG:
    uncertain or overlapping containment

scale-indexed hierarchy:
    explicit field of view at every level
```

A method should state which one it assumes.

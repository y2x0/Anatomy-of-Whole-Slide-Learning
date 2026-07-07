# Parent Maps And Multiscale Units

A hierarchy has node sets at multiple levels:

```math
V_i^{(0)},V_i^{(1)},\ldots,V_i^{(L)}.
```

Fine units live at low levels:

```text
cells or patches
```

Coarse units live at high levels:

```text
regions, tissue compartments, slide
```

## Parent Map

A hard hierarchy uses parent maps:

```math
\pi_i^{(\ell)}
:
V_i^{(\ell)}
\to
V_i^{(\ell+1)}.
```

The children of a coarse node $u\in V_i^{(\ell+1)}$ are:

```math
\mathrm{Ch}(u)
=
\{v\in V_i^{(\ell)}:\pi_i^{(\ell)}(v)=u\}.
```

Each fine unit has exactly one parent at the next level:

```math
\forall v\in V_i^{(\ell)}
\quad
\exists!\,u\in V_i^{(\ell+1)}
\quad
\text{such that}
\quad
\pi_i^{(\ell)}(v)=u.
```

## Local Aggregation

Region states are built from child states:

```math
h_u^{(\ell+1)}
=
\mathcal{R}_{\ell}
\left(
\{h_v^{(\ell)}:v\in\mathrm{Ch}(u)\}
\right).
```

For a local mean:

```math
h_u^{(\ell+1)}
=
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}
\phi_\theta(h_v^{(\ell)}).
```

For local attention:

```math
h_u^{(\ell+1)}
=
\sum_{v\in\mathrm{Ch}(u)}
a_{v\mid u}\phi_\theta(h_v^{(\ell)}).
```

## HIPT And HACT Geometry

HIPT-style methods use multiscale image regions: local patch or region tokens
compose coarser tokens, which compose slide-level representations.

HACT-style methods use a cell-to-tissue hierarchy: cell graphs and tissue graphs
are connected by containment or assignment edges.

Both are hierarchy geometry, but they differ in the units:

```text
HIPT:
    image-scale hierarchy

HACT:
    biological object hierarchy
```

## C/R/G/S Placement

```text
G:
    parent maps pi and child sets Ch

C:
    local context inside children or cross-level messages

R:
    repeated child-to-parent aggregation

S:
    slide label or self-supervised scale objective
```

## Dense Summary

Hierarchy geometry says:

```math
\text{fine units compose coarse units}.
```

The parent map $\pi$ is the core object. If $\pi$ is wrong, the hierarchy routes
information through the wrong biological units.

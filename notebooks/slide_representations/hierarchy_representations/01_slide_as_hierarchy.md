# Slide As Hierarchy

A whole slide has natural nested structure:

```text
cell
patch
region
tissue compartment
whole slide
```

A flat set representation collapses this immediately:

```math
\mathcal{X}_i=\{h_{ij}\}_{j=1}^{n_i}.
```

A hierarchy representation keeps parent-child structure:

```math
\mathcal{X}_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1},
\{H_i^{(\ell)}\}_{\ell=0}^{L}
\right).
```

Here:

```text
V_i^{(0)}:
    fine units, such as cells or small patches

V_i^{(L)}:
    coarse units, often the slide itself

pi_i^{(ell)}:
    parent map from level ell to level ell + 1
```

The parent map is:

```math
\pi_i^{(\ell)}:V_i^{(\ell)}\to V_i^{(\ell+1)}.
```

The children of a coarse unit $u\in V_i^{(\ell+1)}$ are:

```math
\operatorname{Ch}(u)
=
\{v\in V_i^{(\ell)}:\pi_i^{(\ell)}(v)=u\}.
```

## Tree Versus DAG

If every unit has exactly one parent, the hierarchy is a tree or forest:

```math
|\{\pi_i^{(\ell)}(v)\}|=1.
```

If a fine unit may contribute to multiple coarse regions, the object is a
directed acyclic graph:

```math
w_{vu}^{(\ell)}
\ge 0,
\qquad
\sum_{u\in V_i^{(\ell+1)}}w_{vu}^{(\ell)}=1.
```

Then parent aggregation is soft:

```math
h_u^{(\ell+1)}
=
\sum_{v\in V_i^{(\ell)}}w_{vu}^{(\ell)}
\phi_\theta(h_v^{(\ell)}).
```

Tree hierarchies impose hard containment. Soft hierarchies represent uncertain
boundaries.

## Scale As A Mathematical Index

Each level has a physical scale:

```math
\Delta_\ell>0.
```

For image-pyramid methods:

```math
\Delta_0 < \Delta_1 < \cdots < \Delta_L.
```

A patch-level encoder sees local morphology:

```math
h_v^{(0)}=E_0(x_v^{(0)}).
```

A region-level encoder sees a collection of children:

```math
h_u^{(1)}
=
E_1(\{h_v^{(0)}:v\in\operatorname{Ch}(u)\}).
```

A slide-level encoder sees regions:

```math
z_i
=
E_L(\{h_u^{(L-1)}:u\in V_i^{(L-1)}\}).
```

## Hierarchy Versus Graph

A graph representation says:

```text
which units are neighbors
```

A hierarchy representation says:

```text
which units compose larger units
```

Both can coexist. A hierarchical graph has within-level edges:

```math
E_i^{(\ell)}\subseteq V_i^{(\ell)}\times V_i^{(\ell)}
```

and cross-level edges:

```math
B_i^{(\ell)}
=
\{(v,\pi_i^{(\ell)}(v)):v\in V_i^{(\ell)}\}.
```

HACT-style representations are of this form.

## Dense Summary

```math
\boxed{
\mathcal{X}_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{E_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1},
\{H_i^{(\ell)}\}_{\ell=0}^{L}
\right)
}
```

A slide-as-hierarchy model represents tissue as nested units. The core inductive
bias is compositionality:

```text
fine morphology composes region state, and region state composes slide state
```

# Geometry As Equivalence Relation

A geometry choice defines which slides a model is allowed to distinguish.

Let:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i}.
```

A geometry-aware model is:

```math
f_\theta(X_i;G_i).
```

The model induces an equivalence relation:

```math
X_i
\sim_\theta
X_{i'}
\quad\Longleftrightarrow\quad
f_\theta(X_i;G_i)
=
f_\theta(X_{i'};G_{i'}).
```

The geometry determines which transformations are invisible.

## No Geometry

Projection:

```math
\Pi_{\varnothing}(X_i)
=
\{h_{ij}\}_{j=1}^{n_i}.
```

Equivalence:

```math
X_i
\sim_{\varnothing}
X_{i'}
\quad\Longleftrightarrow\quad
\{h_{ij}\}_{j=1}^{n_i}
=
\{h_{i'j}\}_{j=1}^{n_{i'}}.
```

All layouts with the same feature multiset collapse to one object.

## Coordinate Geometry

If absolute coordinates are used:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i}
```

is distinguishable from:

```math
X_i(a)
=
\{(h_{ij},c_{ij}+a)\}_{j=1}^{n_i}
```

unless the model is designed to remove translation.

If only distances are used, then rigid transformations become invisible:

```math
\|Rc_j+a-(Rc_k+a)\|_2
=
\|c_j-c_k\|_2.
```

## Grid Geometry

Grid geometry makes integer lattice displacement meaningful:

```math
q_j-q_k
\in
\mathbb{Z}^{2}.
```

Two slides are similar if their features align under the assumed lattice and
mask. A shifted tile origin can change the discrete representation:

```math
I(x)
\to
\{E(I|_{B(q,p)})\}_{q}
```

even when the underlying continuous tissue is nearly unchanged.

## Graph Geometry

Graph geometry preserves adjacency:

```math
G_i
=
(H_i,A_i).
```

Two graph slides are naturally compared up to permutation:

```math
(H,A)
\sim
(PH,PAP^\top).
```

Graph neural networks should be equivariant before readout:

```math
\mathcal{C}(PH,PAP^\top)
=
P\mathcal{C}(H,A).
```

After graph pooling, the slide output should be invariant:

```math
\mathcal{R}(P\widetilde H)
=
\mathcal{R}(\widetilde H).
```

## Hierarchy Geometry

Hierarchy preserves parent-child relations:

```math
\pi^{(\ell)}:V^{(\ell)}\to V^{(\ell+1)}.
```

Two hierarchies are equivalent only if feature states and parent maps align up
to relabeling within levels.

The key object is not just the node set. It is:

```math
\left(
\{H^{(\ell)}\}_{\ell=0}^{L},
\{\pi^{(\ell)}\}_{\ell=0}^{L-1}
\right).
```

## Learned Topology

Learned topology induces:

```math
A_\theta
=
A_\theta(H,C).
```

The equivalence relation becomes model-dependent:

```math
X_i
\sim_{\theta}
X_{i'}
\quad\Longleftrightarrow\quad
f_\theta(X_i,A_\theta(X_i))
=
f_\theta(X_{i'},A_\theta(X_{i'})).
```

This is expressive, but harder to interpret because the geometry is no longer
fixed before learning.

## Dense Summary

Geometry is an invariance contract:

```text
no geometry:
    same feature multiset means same slide

coordinates:
    location can matter

grid:
    lattice and mask can matter

graph:
    adjacency can matter

hierarchy:
    containment and scale can matter

learned topology:
    task-learned relations can matter
```

A model is wrong when its equivalence relation disagrees with the biology of the
task.

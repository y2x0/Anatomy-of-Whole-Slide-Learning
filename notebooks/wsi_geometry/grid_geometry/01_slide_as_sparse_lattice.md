# Slide As Sparse Lattice

If patches are extracted by a regular tiling process, each patch has an integer
grid index:

```math
q_{ij}
=
(r_{ij},s_{ij})
\in
\mathbb{Z}^{2}.
```

The physical coordinate is:

```math
c_{ij}
=
p\Delta q_{ij},
```

where
```math
p
```
 is tile width in pixels and
```math
\Delta
```
is microns per pixel.

## Dense Grid

A dense grid representation is a tensor:

```math
H_i
\in
\mathbb{R}^{R_i\times S_i\times d}.
```

Each site has:

```math
H_i[r,s]
=
h_i(r,s).
```

This representation supports ordinary convolution and windowed attention.

## Sparse Tissue Grid

WSIs are usually sparse after tissue masking:

```math
\Lambda_i
=
\{(r,s):\text{tile at }(r,s)\text{ contains tissue}\}.
```

The slide is:

```math
X_i
=
\{(h_i(q),q):q\in\Lambda_i\}.
```

The tissue mask is part of geometry:

```math
M_i(q)
=
\mathbf{1}\{q\in\Lambda_i\}.
```

## Grid Neighborhood

A radius-
```math
m
```
grid neighborhood is:

```math
\mathcal{N}_{m}(q)
=
\{q'\in\Lambda_i:\|q-q'\|_{\infty}\le m\}.
```

The grid gives a canonical local neighborhood, but the tissue mask can create
holes:

```math
\mathcal{N}_{m}(q)
\ne
\{q':\|q-q'\|_{\infty}\le m\}.
```

## Relation To Graph Geometry

A grid induces a graph:

```math
(q,q')\in E_i
\quad\Longleftrightarrow\quad
q'\in\mathcal{N}_{1}(q).
```

Thus grid geometry is a special graph geometry with structured coordinates and
regular local edges.

## C/R/G/S Placement

```text
G:
    sparse lattice Lambda_i and mask M_i

C:
    local grid operator, window attention, convolution, or scan

R:
    site pooling, window pooling, or grid-level readout

S:
    task loss and augmentations define which grid symmetries are expected
```

## Dense Summary

Grid geometry says:

```math
\text{nearby integer sites should interact similarly across the slide}.
```

It is powerful when tissue structure aligns with the lattice and brittle when
tissue is sparse, irregular, rotated, or scale-shifted.

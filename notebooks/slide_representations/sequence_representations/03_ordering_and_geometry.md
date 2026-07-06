# Ordering And Geometry

A WSI is spatially two-dimensional, but a sequence is one-dimensional. The order
map:

```math
\sigma:\text{patches}\to\{1,\ldots,n\}
```

compresses spatial layout into a traversal.

## Raster Order

Raster order sorts by row then column:

```math
\sigma(j)<\sigma(k)
\quad
\text{if}
\quad
y_j<y_k
\quad
\text{or}
\quad
(y_j=y_k,\ x_j<x_k).
```

It is simple but creates discontinuities at row boundaries.

Two consecutive sequence tokens can be far apart on the slide:

```math
\|c_{\sigma(t+1)}-c_{\sigma(t)}\|
\gg
\text{patch size}.
```

## Space-Filling Curves

Hilbert or Morton order tries to preserve locality:

```math
\|c_j-c_k\|\ \text{small}
\Rightarrow
|\sigma(j)-\sigma(k)|\ \text{often small}.
```

But no one-dimensional order perfectly preserves two-dimensional neighborhoods.
For growing two-dimensional grids, any one-dimensional traversal must create
spatial neighbors whose sequence indices are far apart. The issue is not that
Hilbert or Morton orders are bad; it is that a total order cannot faithfully
preserve all 2D adjacency relations.

A useful diagnostic is locality distortion:

```math
D(\sigma)
=
\frac{1}{|E_{\text{spatial}}|}
\sum_{(u,v)\in E_{\text{spatial}}}
|\sigma(u)-\sigma(v)|.
```

Here $E_{\text{spatial}}$ can be a grid, kNN, or radius-neighborhood
graph built from coordinates. Low distortion means spatial neighbors usually
stay near each other in sequence space.

## Learned Reordering

A model may learn or choose an order based on features:

```math
\sigma_i=\operatorname{argsort}(r_\theta(h_{ij},c_{ij})).
```

This can bring relevant patches closer in sequence space, but it introduces a
new learned preprocessing step.

If $\sigma_i$ depends on the slide features, the sequence object itself becomes
model-dependent.

In that case the forward map is better written:

```math
\sigma_i=f_\theta(H_i,C_i),
\qquad
\mathcal{X}_i
=
(h_{i\sigma_i(1)},\ldots,h_{i\sigma_i(n_i)}).
```

The representation and the model are no longer separable.

## Positional Features Versus Ordering

One can use sequence order and coordinates together:

```math
\widetilde{h}_{ij}
=
h_{i\sigma(j)}
+
\phi(c_{i\sigma(j)})
+
p_j.
```

Here:

```text
phi(c):
    physical position

p_j:
    sequence index
```

Both may matter.

## MambaMIL-Style Motivation

State-space MIL methods rely on sequence processing for long WSI bags. Reordering
is not cosmetic: it defines which patch dependencies are local in the SSM scan.

If prognostic regions are far apart in the chosen order, the state must preserve
information across long gaps.

## Dense Summary

```math
\text{WSI geometry}
\xrightarrow{\sigma}
\text{sequence locality}.
```

The ordering rule determines whether sequence-local computation corresponds to
spatially meaningful tissue locality.

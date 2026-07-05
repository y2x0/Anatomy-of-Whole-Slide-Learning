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

## Learned Reordering

A model may learn or choose an order based on features:

```math
\sigma_i=\operatorname{argsort}(r_\theta(h_{ij},c_{ij})).
```

This can bring relevant patches closer in sequence space, but it introduces a
new learned preprocessing step.

If $\sigma_i$ depends on the slide features, the sequence object itself becomes
model-dependent.

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

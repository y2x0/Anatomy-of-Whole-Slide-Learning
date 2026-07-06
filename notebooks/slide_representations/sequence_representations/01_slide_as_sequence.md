# Slide As Sequence

A sequence representation assigns an order to patch embeddings:

```math
\mathcal{X}_i
=
(h_{i1},h_{i2},\ldots,h_{in_i}).
```

Unlike a set, this object is not permutation invariant.

The order is a map:

```math
\sigma_i:\{1,\ldots,n_i\}\to\{1,\ldots,n_i\}.
```

The sequence is:

```math
(h_{i\sigma_i(1)},h_{i\sigma_i(2)},\ldots,h_{i\sigma_i(n_i)}).
```

## Order As Structure

If patch coordinates are:

```math
c_{ij}=(x_{ij},y_{ij}),
```

then order can be derived from coordinates:

```text
raster order
column-major order
Hilbert curve
nearest-neighbor traversal
region-by-region traversal
```

The order becomes a one-dimensional encoding of two-dimensional tissue layout.

## Sequence Function

A sequence model computes:

```math
u_{ij}
=
\mathcal{C}_{\operatorname{seq}}
(h_{i1},\ldots,h_{in_i})_j.
```

Readout:

```math
z_i=\mathcal{R}(u_{i1},\ldots,u_{in_i}).
```

Prediction:

```math
\widehat{y}_i=\mathcal{H}(z_i).
```

## Positional Encoding

A transformer sequence uses:

```math
\widetilde{h}_{ij}=h_{ij}+p_j.
```

Here $p_j$ is sequence position, not necessarily physical coordinate.

If using spatial coordinates:

```math
\widetilde{h}_{ij}=h_{ij}+\phi(c_{ij}).
```

The distinction matters:

```text
sequence index:
    where the patch appears in the chosen order

coordinate:
    where the patch lies on the slide
```

## Why Sequence Models Are Attractive

For very large bags, sequence models can be computationally efficient.

State-space models can process:

```math
O(n_i)
```

tokens instead of transformer-style:

```math
O(n_i^2).
```

This makes them appealing for WSI-scale patch sequences.

## Dense Summary

```math
\begin{aligned}
\mathcal{X}_i&=(h_{i\sigma(1)},\ldots,h_{i\sigma(n_i)}),\\
u_{ij}&=\mathcal{C}_{\operatorname{seq}}(\mathcal{X}_i)_j,\\
z_i&=\mathcal{R}(u_{i1},\ldots,u_{in_i}).
\end{aligned}
```

A slide-as-sequence model is only as meaningful as the ordering rule that
creates the sequence.

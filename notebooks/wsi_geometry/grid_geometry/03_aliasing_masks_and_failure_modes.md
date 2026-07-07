# Aliasing, Masks, And Failure Modes

Grid geometry fails when the lattice is mistaken for the tissue.

## Tissue Mask Holes

A full grid has sites:

```math
\{1,\ldots,R\}\times\{1,\ldots,S\}.
```

The observed tissue grid is:

```math
\Lambda_i
\subset
\{1,\ldots,R\}\times\{1,\ldots,S\}.
```

If missing sites are treated as zeros:

```math
H[q]
=
0
\quad
q\notin\Lambda_i,
```

then the model may confuse missing tissue with a real visual feature.

## Aliasing From Tile Size

Tile extraction discretizes a continuous tissue field:

```math
I_i:\Omega_i\to\mathbb{R}^{3}.
```

The patch feature is a local functional:

```math
h_i(q)
=
E_\phi(I_i|_{B(q,p)}).
```

If the relevant biological structure is smaller than or badly aligned with the
tile size $p$, the grid representation aliases it:

```math
I_i
\to
\{h_i(q)\}_{q\in\Lambda_i}
```

loses sub-tile arrangement.

## Window Boundary Artifacts

For window attention:

```math
\mathcal{N}(q)
=
\mathcal{W}(q).
```

Two adjacent patches may be unable to interact if they fall in different
windows:

```math
\|q-q'\|_2=1,
\qquad
\mathcal{W}(q)\ne\mathcal{W}(q').
```

The boundary is algorithmic, not biological.

## Scan-Order Distortion

A sequence order $\sigma$ induces path distance:

```math
d_{\sigma}(q,q')
=
|\sigma^{-1}(q)-\sigma^{-1}(q')|.
```

This can disagree with physical distance:

```math
d_{\sigma}(q,q')
\gg
\|q-q'\|_2.
```

or:

```math
d_{\sigma}(q,q')
\ll
\|q-q'\|_2.
```

State-space or recurrent models inherit this distortion unless the ordering is
carefully designed.

## Dense Summary

Grid geometry assumes:

```text
lattice neighbors approximate tissue neighbors
```

The major failure modes are:

```text
missing tissue treated as signal
tile-size aliasing
window boundary artifacts
scan-order distortion
scale mismatch
```

The stress test is to perturb tiling, mask, window origin, and scan order. A
stable model should not be overly sensitive to arbitrary grid choices.

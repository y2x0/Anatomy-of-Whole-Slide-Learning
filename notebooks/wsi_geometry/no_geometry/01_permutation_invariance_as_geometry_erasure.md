# Permutation Invariance As Geometry Erasure

No-geometry WSI learning starts from the finite set:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i}
\subset
\mathbb{R}^{d}.
```

The coordinates exist in preprocessing, but the model does not consume them.
The slide is represented only by the feature multiset.

## Invariance

Let $P$ be a permutation matrix acting on the order of patch embeddings. A
no-geometry model satisfies:

```math
f(PH_i)
=
f(H_i).
```

For a Deep Sets style model:

```math
f(H_i)
=
\rho_\theta
\left(
\sum_{j=1}^{n_i}\phi_\theta(h_{ij})
\right),
```

permutation invariance follows because:

```math
\sum_{j=1}^{n_i}\phi_\theta(h_{i\sigma(j)})
=
\sum_{j=1}^{n_i}\phi_\theta(h_{ij}).
```

For attention MIL:

```math
a_{ij}
=
\frac{\exp(s_\theta(h_{ij}))}
{\sum_{\ell=1}^{n_i}\exp(s_\theta(h_{i\ell}))},
\qquad
z_i
=
\sum_{j=1}^{n_i}a_{ij}v_\theta(h_{ij}),
```

the readout is also invariant to patch reordering because the softmax is taken
over the set of scores, not over physical location.

## What Is Erased

If the original slide object is:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i},
```

then no-geometry MIL applies the projection:

```math
\Pi_{\varnothing}(X_i)
=
\{h_{ij}\}_{j=1}^{n_i}.
```

All slides in the fiber:

```math
\Pi_{\varnothing}^{-1}(H_i)
=
\left\{
\{(h_{ij},c'_{ij})\}_{j=1}^{n_i}
:
\{h_{ij}\}_{j=1}^{n_i}\ \text{fixed}
\right\}
```

become indistinguishable.

This is the exact mathematical meaning of ignoring geometry.

## C/R/G/S Placement

```text
G:
    empty

C:
    patchwise, set-equivariant, or complete attention without physical adjacency

R:
    permutation-invariant pooling

S:
    only task supervision decides which morphology statistics matter
```

No-geometry does not mean no inductive bias. It means the inductive bias is:

```text
the label depends on morphology content, not spatial arrangement
```

## Relation To Papers

ABMIL, CLAM, DSMIL, and many MIL baselines can be run in this regime when patch
coordinates are not used by the model. They may still learn strong classifiers
because patch morphology can be highly predictive. The geometry claim is only
that spatial arrangement cannot affect prediction except through features
already encoded in $h_{ij}$.

## Dense Summary

No-geometry models replace:

```math
\{(h_j,c_j)\}_{j=1}^{n}
```

with:

```math
\{h_j\}_{j=1}^{n}.
```

That projection is a modeling decision. It preserves morphology distribution
and erases layout.

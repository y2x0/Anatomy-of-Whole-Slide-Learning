# Sequence Representation Failure Modes

Sequence models introduce order, but WSI order is usually constructed rather
than inherent.

## 1. Arbitrary Order Artifact

If the order is arbitrary:

```math
(h_1,\ldots,h_n)
```

then a sequence model may learn dependencies that are artifacts of enumeration.

A set model would satisfy:

```math
f(H)=f(PH).
```

A sequence model generally does not.

## 2. Spatial Discontinuities

Raster order can make distant patches adjacent at row boundaries.

The sequence model sees:

```math
h_{\sigma(t)}
\to
h_{\sigma(t+1)}
```

as local, even if:

```math
\|c_{\sigma(t)}-c_{\sigma(t+1)}\|
```

is large.

## 3. Long-Range Dependence Loss

If two relevant regions are far apart in the sequence:

```math
|\sigma(j)-\sigma(k)|\gg1,
```

the model must preserve information across many steps.

State-space models are designed for long sequences, but finite state dimension
still creates compression.

## 4. Direction Bias

A one-way scan treats earlier and later patches differently:

```math
s_{j+1}=f(s_j,h_j).
```

This can be inappropriate for unordered tissue fields.

Bidirectional or multi-scan approaches reduce this issue.

## 5. Shortcut Ordering

If ordering correlates with tissue amount, scanner tiling, or slide boundaries,
the model can learn shortcuts tied to sequence position.

Example:

```text
tumor always appears in central crop positions after preprocessing
background dominates early or late sequence positions
```

## Diagnostic Questions

1. What defines the order?
2. Does local sequence adjacency imply spatial adjacency?
3. Is the model robust to order perturbation?
4. Are coordinates included separately?
5. Is the scan bidirectional or multi-directional?

## Dense Summary

Sequence representations trade permutation invariance for order-aware modeling.

That is useful only when the order encodes meaningful slide structure.

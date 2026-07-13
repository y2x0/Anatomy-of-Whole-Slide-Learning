# Set Transformer Pooling Failure Modes

Set Transformer pooling is expressive, but the expressivity creates its own
failure modes.

## 1. Quadratic Cost

Full set attention costs:

```math
O(n_i^2).
```

For WSI bags with many patches, this can be prohibitive. ISAB reduces cost using
inducing points, but then information must pass through a bottleneck of size
```math
M
```
.

## 2. Inducing-Point Bottleneck

With
```math
M
```
inducing points:

```math
A_i
=
\mathrm{MHA}(I,H_i,H_i).
```

If
```math
M
```
 is too small, rare morphologies may never be represented in
```math
A_i
```
.

## 3. Seed Collapse

Multiple PMA seeds can learn redundant queries:

```math
z_{i1}\approx z_{i2}\approx\cdots\approx z_{iK}.
```

Then multi-seed pooling behaves like a single attention summary.

## 4. No Geometry Unless Supplied

Set attention over patches without coordinates treats the slide as a set:

```math
f(H_i)=f(PH_i).
```

It can model pairwise feature interactions but not physical adjacency unless
coordinates or positional encodings are included.

## 5. Over-Smoothing By Attention

Repeated attention blocks can make states similar:

```math
\widetilde h_{ij}
\approx
\widetilde h_{ik}.
```

If rare patch identities are washed into global context, pooling cannot recover
them.

## Dense Summary

Set Transformer pooling fails when:

```text
the set is too large,
the inducing bottleneck is too narrow,
seed queries are redundant,
or geometry is needed but absent.
```

Its strength is interaction-aware invariant readout. Its cost is computation and
the risk of compressing rare evidence before pooling.

# Failure Modes

Hierarchy geometry fails when scale or containment is wrong.

## Premature Compression

If local pooling maps:

```math
\{h_v:v\in\mathrm{Ch}(u)\}
\to
h_u,
```

then all fine-level configurations with the same
```math
h_u
```
become equivalent:

```math
\mathcal{R}_{\mathrm{local}}(A)
=
\mathcal{R}_{\mathrm{local}}(B)
\quad
\Longrightarrow
\quad
A\sim B
```

for the global model.

If rare diagnostic patches are erased locally, global readout cannot recover
them.

## Bad Region Boundaries

A parent map:

```math
\pi:V^{(0)}\to V^{(1)}
```

may split a coherent tissue structure across parents or merge unrelated
structures into one parent.

Then regional states estimate the wrong object:

```math
h_u^{(1)}
\approx
\text{summary of algorithmic region, not biological region}.
```

## Scale Leakage

A coarse token may encode information that should belong to a different scale,
for example scanner artifacts or tissue mask size:

```math
h_u^{(\ell)}
=
\phi(\text{biology},\text{scale artifact}).
```

The model can then learn scale-specific shortcuts.

## Unequal Child Counts

If region pooling is an unweighted mean over regions:

```math
z
=
\frac{1}{M}\sum_{u=1}^{M}h_u,
```

then a small region and a large region have equal global weight. At patch level:

```math
w_v
=
\frac{1}{M|\mathrm{Ch}(\pi(v))|}.
```

Patches inside small regions receive larger effective weight.

## Dense Summary

Hierarchy failures are usually one of:

```text
wrong parent map
wrong scale
premature local pooling
unequal effective weights
boundary fragmentation
scale-specific shortcuts
```

Hierarchy is powerful when tissue really is compositional. It is harmful when
the hierarchy is mostly an artifact of tiling, segmentation, or memory limits.

# Failure Modes

Coordinate geometry fails when location features encode the wrong invariance.

## Absolute Coordinate Shortcut

If labels correlate with slide position because of preprocessing, tissue
placement, or scanning artifacts, a coordinate-aware model can learn:

```math
P(Y\mid C)
```

instead of:

```math
P(Y\mid H,C)
```

This is dangerous when:

```math
P_{\mathrm{train}}(Y\mid C)
\ne
P_{\mathrm{test}}(Y\mid C).
```

The model may appear spatially intelligent while learning a cohort artifact.

## Scale Mismatch

A radius graph built in pixel units:

```math
(j,k)\in E
\quad\Longleftrightarrow\quad
\|c_j^{\mathrm{px}}-c_k^{\mathrm{px}}\|_2\le r
```

does not preserve physical neighborhoods if microns-per-pixel varies across
slides:

```math
c^{\mu m}
=
r_i c^{\mathrm{px}}.
```

The same numerical radius becomes different tissue scales.

## Boundary Effects

For patches near tissue boundaries, local neighborhoods are truncated:

```math
|\mathcal{N}(j)|
\ll
|\mathcal{N}(k)|
```

even if $j$ and $k$ are biologically similar. This can bias local density,
attention, and graph aggregation.

## Rotation And Orientation

If a model uses raw coordinates:

```math
f(\{(h_j,c_j)\}_j)
```

then a rotated slide:

```math
\{(h_j,Rc_j)\}_j
```

may receive a different prediction. That is correct only if orientation is a
meaningful part of the task.

## Coordinate-Morphology Confounding

Coordinates can proxy for morphology when tissue preparation is structured:

```math
C
\to
H
\to
Y.
```

or proxy for nonbiological artifacts:

```math
C
\to
A
\to
Y.
```

The same coordinate dependence can be causal, biological, or spurious.

## Dense Summary

Coordinate geometry should be stress-tested by transformations:

```text
translation
rotation
rescaling
tissue-mask perturbation
scanner-resolution change
origin shift
```

If the prediction changes under a transformation that should be irrelevant, the
geometry is too strong. If it fails to change under a transformation that
changes the biology, the geometry is too weak.

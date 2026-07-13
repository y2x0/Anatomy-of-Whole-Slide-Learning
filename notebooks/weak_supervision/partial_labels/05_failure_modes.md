# Failure Modes

Partial labels fail when the observed subset is not representative or when
constraints are treated as stronger than they are.

## Missing Not At Random

If pathologists annotate the most obvious regions:

```math
P(M=1\mid Z,h)
```

depends on difficulty, salience, or disease severity.

Then:

```math
\widehat R_{\mathrm{mask}}
```

estimates performance on annotated regions, not all tissue.

## Annotation Geometry Bias

If annotations are concentrated in a spatial region:

```math
P(M=1\mid c)
\ne
P(M=1),
```

then the model may learn location-dependent supervision artifacts.

## Region Label Overinterpretation

A positive region label:

```math
Y_r=1
```

may only imply:

```math
\max_{j\in B_r}Z_j=1.
```

It does not imply:

```math
Z_j=1
\quad
\forall j\in B_r.
```

Treating all region children as positive creates label noise.

## Negative Space Ambiguity

Unannotated tissue is often assumed negative:

```math
M_j=0
\Rightarrow
Z_j=0.
```

This is usually false. It turns sparse annotation into systematic false
negatives.

## Ranking Calibration Failure

Ranking constraints identify order:

```math
r_a>r_b.
```

They do not identify calibrated probability:

```math
P(Y=1\mid a).
```

Models trained only on ranking may perform poorly when thresholded as
classifiers.

## Dense Summary

Partial labels require modeling the annotation process:

```math
S^{\mathrm{obs}}
=
(M,M\odot U).
```

The mask
```math
M
```
is part of the supervision signal. Ignoring it can turn helpful
partial information into biased training.

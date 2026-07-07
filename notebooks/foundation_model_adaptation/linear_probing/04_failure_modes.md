# Failure Modes

## Nonlinear Readout Requirement

If the task is:

```math
Y
=
\mathbf{1}\{z_1z_2>0\},
```

then no single hyperplane separates the classes. A failed linear probe does not
prove the frozen representation is useless; it proves the chosen readout class
is insufficient.

## Shortcut Direction

A probe may find a nuisance direction:

```math
w^\top z
\approx
\text{scanner/site/stain}
```

if that nuisance correlates with $Y$ in training. The representation can be
powerful and the probe can still learn the wrong axis.

## Small Sample Instability

When $D\gg N$, many probe directions can fit training labels:

```math
\{w:\mathrm{CE}(w)\approx 0\}
```

may be large. Regularization controls this:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda\|w\|_2^2.
```

## False Negative Interpretation

If a linear probe fails:

```math
\min_{w,b}
\mathcal{L}(w,b)
\text{ is high},
```

the conclusion should be:

```text
the task is not linearly readable from this frozen representation
```

not:

```text
the foundation model has no useful task information
```

## Dense Summary

A linear probe is a diagnostic instrument. Its failure identifies a mismatch
between frozen features and linear readout, not necessarily a failure of the
entire foundation model.

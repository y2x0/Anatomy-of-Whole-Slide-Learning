# Counts, Residuals, And Transport

Prototype prevalence records how much of each morphology appears:

```math
p_{im}
=
\frac{1}{n_i}\sum_j q_{ijm}.
```

This is only the first prototype statistic. More information can survive if the
readout also stores residuals or uses prototype geometry.

## Count Versus Prevalence

Prevalence is normalized:

```math
p_{im}
=
\frac{1}{n_i}\sum_j q_{ijm}.
```

Count-like burden is unnormalized:

```math
b_{im}
=
\sum_j q_{ijm}.
```

These answer different questions:

```text
prevalence:
    what fraction of tissue resembles prototype m?

burden:
    how much tissue resembles prototype m?
```

For WSI, this distinction matters because $n_i$ may reflect tissue area,
sampling density, or preprocessing.

## Residual Statistics

Within each prototype, define a residual:

```math
r_{ijm}
=
h_{ij}-c_m.
```

A prototype-specific residual mean is:

```math
\bar r_{im}
=
\frac{
\sum_j q_{ijm}(h_{ij}-c_m)
}{
\sum_j q_{ijm}+\epsilon
}.
```

The slide representation can concatenate:

```math
z_i
=
[p_{i1},\bar r_{i1},\ldots,p_{iM},\bar r_{iM}].
```

Now the statistic preserves both:

```text
how much prototype mass exists
how that mass deviates from the prototype center
```

## Covariance Within Prototype

A second-order residual statistic is:

```math
\Sigma_{im}
=
\frac{
\sum_j q_{ijm}(h_{ij}-c_m)(h_{ij}-c_m)^\top
}{
\sum_j q_{ijm}+\epsilon
}.
```

This is richer but high-dimensional. Most practical systems compress it,
diagonalize it, or drop it.

## Prototype Geometry

A prevalence vector treats prototypes as categories. But prototypes live in
embedding space, so distances between them matter:

```math
C_{mn}
=
\|c_m-c_n\|^2.
```

Transport compares two slides by the cost of moving prototype mass:

```math
W_C(p_i,p_k)
=
\min_{T\ge 0}
\sum_{m,n}T_{mn}C_{mn}
```

subject to:

```math
\sum_n T_{mn}=p_{im},
\qquad
\sum_m T_{mn}=p_{kn}.
```

This says that confusing two nearby prototypes is less severe than confusing two
distant prototypes.

## Dense Summary

Prototype pooling can preserve a ladder of statistics:

```text
level 1:
    prevalence p_im

level 2:
    count or burden b_im

level 3:
    residual mean r_im

level 4:
    covariance or Fisher-like deviation

level 5:
    geometry-aware comparison by transport
```

The more statistics survive, the less the slide collapses. The cost is higher
dimension, noisier estimates, and less direct interpretability.


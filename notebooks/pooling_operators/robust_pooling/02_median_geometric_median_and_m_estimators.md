# Median, Geometric Median, And M-Estimators

Median-style pooling estimates robust location instead of an average.

## Coordinatewise Median

For each coordinate $r$:

```math
z_{ir}
=
\mathrm{median}_{j}\ u_{ijr}.
```

This has high resistance to coordinatewise outliers, but the resulting vector:

```math
z_i
=
(z_{i1},\ldots,z_{id})
```

may not correspond to any actual patch.

## Geometric Median

The geometric median solves:

```math
z_i
=
\arg\min_{z\in\mathbb{R}^{d}}
\sum_{j=1}^{n_i}\|u_{ij}-z\|_2.
```

Mean pooling solves:

```math
z_i^{\mathrm{mean}}
=
\arg\min_z
\sum_j\|u_{ij}-z\|_2^2.
```

The median uses linear distance and is less sensitive to far outliers.

## M-Estimator Pooling

An M-estimator solves:

```math
z_i
=
\arg\min_z
\sum_j \rho(\|u_{ij}-z\|).
```

For squared loss:

```math
\rho(t)=t^2,
```

we recover the mean. For Huber-style loss, large residuals grow linearly rather
than quadratically.

The estimating equation is:

```math
\sum_j
\psi(\|u_{ij}-z_i\|)
\frac{u_{ij}-z_i}{\|u_{ij}-z_i\|}
=
0,
```

where:

```math
\psi(t)
=
\rho'(t).
```

## Dense Summary

Robust location pooling replaces:

```math
\text{average all points}
```

with:

```math
\text{find a location with bounded outlier influence}.
```

It is useful for artifact resistance and dangerous for sparse-positive tasks.


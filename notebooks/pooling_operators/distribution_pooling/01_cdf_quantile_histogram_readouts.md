# CDF, Quantile, And Histogram Readouts

Distribution pooling starts from a scalar or vector feature map:

```math
r_{ij}
=
\phi(u_{ij}).
```

For scalar $r_{ij}$, the empirical CDF is:

```math
F_i(t)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\mathbf{1}\{r_{ij}\le t\}.
```

## Quantile Readout

The $\alpha$ quantile is:

```math
Q_i(\alpha)
=
\inf\{t:F_i(t)\ge\alpha\}.
```

A finite quantile vector is:

```math
z_i
=
(Q_i(\alpha_1),\ldots,Q_i(\alpha_M)).
```

This preserves distribution shape along the chosen scalar feature.

## Histogram Readout

For bins $B_1,\ldots,B_M$:

```math
p_{im}
=
\mu_i(B_m)
=
\frac{1}{n_i}
\sum_j\mathbf{1}\{u_{ij}\in B_m\}.
```

Soft histograms use assignments:

```math
p_{im}
=
\frac{1}{n_i}
\sum_j q_m(u_{ij}),
\qquad
\sum_m q_m(u)=1.
```

## CDF Versus Moment

Moments preserve expectations:

```math
\mathbb{E}_{\mu_i}[\phi(u)].
```

CDF/quantile summaries preserve order statistics:

```math
Q_i(\alpha).
```

They are more sensitive to tails than a mean and less brittle than a maximum.

## C/R/G/S Placement

```text
G:
    bin geometry or scalar ordering induced by phi

C:
    feature map phi or bin assignment q_m

R:
    CDF, quantile, or histogram statistic

S:
    task loss if phi or bins are learned
```

## Dense Summary

CDF and quantile pooling ask:

```text
what does the whole score distribution look like?
```

This is useful when phenotype depends on tails, spread, or prevalence rather
than only mean or max.


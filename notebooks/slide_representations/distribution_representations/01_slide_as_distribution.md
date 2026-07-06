# Slide As Distribution

The set view says:

```math
H_i=\{h_{ij}\}_{j=1}^{n_i}.
```

The distribution view says:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{h_{ij}}.
```

The slide is the empirical distribution of patch embeddings.

This is stronger than saying "mean pooling." Mean pooling is only one statistic
of $\mu_i$:

```math
m_i
=
\int h\,d\mu_i(h).
```

Distribution representation asks for a richer statistic:

```math
z_i=T(\mu_i).
```

## Distributional Equivalence

Two slides are distributionally equivalent under statistic $T$ if:

```math
T(\mu_i)=T(\mu_k).
```

For mean pooling:

```math
\int h\,d\mu_i(h)
=
\int h\,d\mu_k(h)
```

does not imply:

```math
\mu_i=\mu_k.
```

Many different morphology distributions can share the same first moment.

## Moment Representation

A moment statistic uses functions $\phi_m$:

```math
z_{im}
=
\int \phi_m(h)\,d\mu_i(h)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\phi_m(h_{ij}).
```

If:

```math
\phi_m(h)=h_m,
```

we get first moments. If:

```math
\phi_{ab}(h)=h_a h_b,
```

we get second moments:

```math
M_i
=
\int hh^\top\,d\mu_i(h).
```

The covariance is:

```math
\Sigma_i
=
\int (h-m_i)(h-m_i)^\top\,d\mu_i(h).
```

Higher moments encode more shape but grow quickly in dimension and estimation
variance.

## Histogram Representation

Let $B_1,\ldots,B_M$ partition embedding space.

The histogram statistic is:

```math
p_{im}
=
\mu_i(B_m)
=
\frac{1}{n_i}
\sum_j\mathbf{1}\{h_{ij}\in B_m\}.
```

Soft histograms replace hard bins with assignment functions:

```math
p_{im}
=
\int q_m(h)\,d\mu_i(h),
\qquad
\sum_{m=1}^{M}q_m(h)=1.
```

This turns a slide into a morphology prevalence vector.

## Distribution Versus Prototype

A prototype representation chooses learned or estimated centers
$c_1,\ldots,c_M$ and summarizes the slide by assignments to those centers:

```math
q_m(h)
=
\frac{\exp(-\|h-c_m\|^2/\tau)}
{\sum_{r=1}^{M}\exp(-\|h-c_r\|^2/\tau)}.
```

Then:

```math
p_{im}
=
\frac{1}{n_i}
\sum_j q_m(h_{ij}).
```

The slide becomes:

```math
z_i=(p_{i1},\ldots,p_{iM}).
```

This is still permutation invariant, but it preserves more of the distribution
than a single mean.

## Dense Summary

```math
\boxed{
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}},
\qquad
z_i=T(\mu_i)
}
```

A distribution representation makes the slide a measure over morphology. The
central design problem is choosing $T$: the finite statistic that survives.

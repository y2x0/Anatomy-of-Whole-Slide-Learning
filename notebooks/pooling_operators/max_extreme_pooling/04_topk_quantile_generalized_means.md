# Top-K, Quantile, And Generalized Means

Hard max is only one extreme statistic. Many pooling operators interpolate
between mean and max.

Let instance scores be:

```math
s_{ij}
=
g_\theta(u_{ij}).
```

## Top-K Pooling

Let $I_K(i)$ be the indices of the $K$ largest scores:

```math
I_K(i)
=
\mathrm{TopK}_{j}\ s_{ij}.
```

Top-k average pooling is:

```math
z_i^{(K)}
=
\frac{1}{K}
\sum_{j\in I_K(i)}s_{ij}.
```

Special cases:

```math
K=1
\quad\Rightarrow\quad
z_i^{(K)}=\max_j s_{ij},
```

```math
K=n_i
\quad\Rightarrow\quad
z_i^{(K)}=\frac{1}{n_i}\sum_j s_{ij}.
```

Top-k pooling assumes the label depends on a small but not necessarily single
set of high-evidence patches.

## Quantile Pooling

For a quantile level $\alpha\in(0,1)$, define:

```math
Q_i(\alpha)
=
\inf\{t:F_i(t)\ge \alpha\},
```

where the empirical CDF of scores is:

```math
F_i(t)
=
\frac{1}{n_i}
\sum_j \mathbf{1}\{s_{ij}\le t\}.
```

High quantiles such as $Q_i(0.95)$ preserve near-extreme evidence while being
less sensitive to a single outlier than max pooling.

## Generalized Mean

For positive transformed scores $r_{ij}>0$, the generalized mean is:

```math
M_p(r_i)
=
\left(
\frac{1}{n_i}
\sum_j r_{ij}^{p}
\right)^{1/p}.
```

Limits:

```math
\lim_{p\to 1}M_p
=
\text{arithmetic mean},
```

```math
\lim_{p\to\infty}M_p
=
\max_j r_{ij}.
```

Increasing $p$ makes the readout more extreme-focused.

## C/R/G/S Placement

```text
G:
    ignored unless scores include geometric context

C:
    instance score map g_theta

R:
    top-k, quantile, or generalized-mean statistic

S:
    slide labels choosing how sparse evidence should be
```

## Dense Summary

These operators form a spectrum:

```text
mean:
    all instances survive equally

top-k / quantile:
    upper tail survives

max:
    only strongest instance survives
```

The design question is how many high-evidence patches should define the bag.


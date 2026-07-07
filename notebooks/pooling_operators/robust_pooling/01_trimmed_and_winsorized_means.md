# Trimmed And Winsorized Means

Robust pooling starts by limiting the effect of extreme instance scores.

Let:

```math
s_{ij}
=
g_\theta(u_{ij}).
```

Sort the scores:

```math
s_{i(1)}
\le
s_{i(2)}
\le
\cdots
\le
s_{i(n_i)}.
```

## Trimmed Mean

For trim fraction $\alpha$, let:

```math
k
=
\lfloor \alpha n_i\rfloor.
```

The trimmed mean is:

```math
T_{\alpha}(s_i)
=
\frac{1}{n_i-2k}
\sum_{r=k+1}^{n_i-k}s_{i(r)}.
```

This discards both low and high extremes.

For one-sided upper trimming:

```math
T_{\alpha}^{\mathrm{upper}}(s_i)
=
\frac{1}{n_i-k}
\sum_{r=1}^{n_i-k}s_{i(r)}.
```

This limits high-score artifacts.

## Winsorized Mean

Winsorization clips extremes instead of removing them:

```math
\widetilde s_{ij}
=
\min
\left(
\max(s_{ij},s_{i(k+1)}),
s_{i(n_i-k)}
\right).
```

Then:

```math
W_\alpha(s_i)
=
\frac{1}{n_i}\sum_j\widetilde s_{ij}.
```

## Vector Version

For vector states, robust pooling can apply coordinatewise:

```math
z_{ir}
=
T_\alpha(\{u_{ijr}\}_{j=1}^{n_i}).
```

Coordinatewise trimming is simple but can create synthetic vectors that do not
correspond to any real patch.

## C/R/G/S Placement

```text
G:
    none by default

C:
    score map g_theta or coordinate projection

R:
    trimmed or winsorized empirical mean

S:
    robustness assumption that extremes may be artifacts
```

## Dense Summary

Trimmed pooling changes the surviving statistic from:

```math
\frac{1}{n_i}\sum_j s_{ij}
```

to:

```math
\frac{1}{|\mathcal{I}_i|}
\sum_{j\in\mathcal{I}_i}s_{ij},
```

where $\mathcal{I}_i$ excludes extreme ranks. This protects against outliers but
can remove rare true positives.


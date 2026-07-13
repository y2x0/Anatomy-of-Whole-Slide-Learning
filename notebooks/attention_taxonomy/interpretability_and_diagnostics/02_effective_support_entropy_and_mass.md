# Effective Support, Entropy, And Mass

Attention diagnostics should measure the distribution, not just display a heat
map.

For a readout vector:

```math
a
=
(a_1,\ldots,a_n),
\qquad
\sum_ja_j=1.
```

## Entropy

Entropy is:

```math
H(a)
=
-
\sum_j a_j\log a_j.
```

Normalized entropy is:

```math
\bar H(a)
=
\frac{H(a)}{\log n}.
```

This lies between zero and one.

## Effective Number

The exponential effective support is:

```math
N_{\mathrm{eff}}^{(1)}
=
\exp(H(a)).
```

The quadratic effective support is:

```math
N_{\mathrm{eff}}^{(2)}
=
\frac{1}{\sum_j a_j^2}.
```

Both equal `n` under uniform attention and `1` under hard attention.

For any probability vector:

```math
1
\le
N_{\mathrm{eff}}^{(2)}
\le
N_{\mathrm{eff}}^{(1)}
\le
n.
```

The normalized entropy `Hbar` is useful when bags have different patch counts,
because raw entropy has a moving maximum `log(n)`. Effective support still has
to be reported with `n`; a value of 100 means something different in a bag of
1,000 patches than in a bag of 100,000.

## Top-k Mass

Let:

```math
a_{(1)}
\ge
a_{(2)}
\ge
\cdots
\ge
a_{(n)}.
```

Top-k mass is:

```math
M_k(a)
=
\sum_{r=1}^{k}a_{(r)}.
```

This answers:

```text
how much of the slide statistic comes from the k most attended patches?
```

## Spatial Spread

If patch coordinates are:

```math
c_j\in\mathbb{R}^{2},
```

the attention center is:

```math
\bar c
=
\sum_j a_jc_j.
```

The attention spatial variance is:

```math
\Sigma_a
=
\sum_j a_j(c_j-\bar c)(c_j-\bar c)^\top.
```

This distinguishes concentrated local attention from diffuse slide-wide
attention.

The covariance trace has a direct interpretation:

```math
\mathrm{tr}(\Sigma_a)
=
\sum_j
a_j
\|c_j-\bar c\|_2^2.
```

It is measured in squared coordinate units. Spatial spread is therefore not
comparable across pixel, micron, and normalized-slide coordinates until the
coordinate system is fixed or the statistic is normalized.

## Dense Summary

A useful attention report includes:

```text
entropy
effective support
top-k mass
spatial spread
overlap with tissue compartments
stability across augmentations
```

The question is not "where is the heat map bright?" but "what measure did the
model actually use?"

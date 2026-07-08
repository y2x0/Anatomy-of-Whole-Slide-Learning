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


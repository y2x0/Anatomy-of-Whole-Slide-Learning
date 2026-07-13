# Softmax Geometry, Temperature, And Entropy

Temperature controls how score differences become probability mass.

Given scores:

```math
s
=
(s_1,\ldots,s_n),
```

define:

```math
a_j^{(\beta)}
=
\frac{\exp(\beta s_j)}
{\sum_{\ell=1}^{n}\exp(\beta s_\ell)}.
```

Here `beta` is inverse temperature.

## Limits

As temperature becomes infinite, equivalently as inverse temperature vanishes:

```math
\beta\to 0
\quad\Longrightarrow\quad
a_j^{(\beta)}
\to
\frac{1}{n}.
```

As temperature tends to zero, equivalently as inverse temperature diverges:

```math
\beta\to\infty
\quad\Longrightarrow\quad
a_j^{(\beta)}
\to
\frac{\mathbf{1}\{j\in M\}}{|M|},
```

where:

```math
M
=
\arg\max_j s_j.
```

Softmax therefore interpolates between mean pooling and max-like selection.

## Entropy

Attention entropy is:

```math
H(a)
=
-
\sum_{j=1}^{n}a_j\log a_j.
```

Uniform attention gives:

```math
H(a)
=
\log n.
```

One-hot hard selection gives:

```math
H(a)
=
0.
```

The effective support size is:

```math
N_{\mathrm{eff}}(a)
=
\exp(H(a)),
```

or, using the inverse participation ratio:

```math
N_2(a)
=
\frac{1}{\sum_j a_j^2}.
```

These two diagnostics measure how many patches are effectively used.

The entropy changes monotonically with inverse temperature. Let:

```math
Z(\beta)
=
\sum_j\exp(\beta s_j),
\qquad
H(\beta)
=
\log Z(\beta)
-
\beta\,\mathbb{E}_{a^{(\beta)}}[s].
```

Then:

```math
\frac{\mathrm{d}H}{\mathrm{d}\beta}
=
-\beta\,\mathrm{Var}_{a^{(\beta)}}(s)
\le 0.
```

Thus increasing `beta` cannot increase attention entropy. It may have almost no
effect when all scores are equal, and it can have a sharp effect when several
scores are close to the maximum.

## Gradient Geometry

For softmax:

```math
\frac{\partial a_j}{\partial s_k}
=
a_j
(\mathbf{1}\{j=k\}-a_k).
```

With inverse temperature:

```math
\frac{\partial a_j^{(\beta)}}{\partial s_k}
=
\beta a_j^{(\beta)}
(\mathbf{1}\{j=k\}-a_k^{(\beta)}).
```

Low temperature increases local sensitivity near uncertain decisions but can
also saturate when one score dominates. For the readout:

```math
m(\beta)
=
\sum_j a_j^{(\beta)}v_j,
```

the temperature derivative is:

```math
\frac{\mathrm{d}m}{\mathrm{d}\beta}
=
\sum_j
a_j^{(\beta)}
\left(
s_j
-
\mathbb{E}_{a^{(\beta)}}[s]
\right)v_j.
```

The readout therefore moves toward values associated with above-average scores;
temperature changes the statistic itself, not merely the appearance of its
weight map.

## Denominator Coupling

Adding a new patch with score `s_new` changes every old weight:

```math
a_j'
=
\frac{\exp(s_j)}
{\sum_{\ell=1}^{n}\exp(s_\ell)+\exp(s_{\mathrm{new}})}
=
a_j
\frac{Z}{Z+\exp(s_{\mathrm{new}})}.
```

This is the denominator effect. Even irrelevant patches perturb the measure
unless their scores are very low.

## Dense Summary

Softmax attention has dense support:

```math
a_j>0
\quad
\text{for all finite }s_j.
```

Its practical behavior is governed by:

```text
score scale
temperature
entropy
bag size
denominator competition
```

The same attention architecture can behave like mean pooling, top-instance
pooling, or something in between depending on this geometry.

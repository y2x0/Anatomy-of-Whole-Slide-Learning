# Noisy-Or Bag Probability

Classic positive-instance MIL assumes:

```math
y_i
=
\max_j y_{ij},
\qquad
y_{ij}\in\{0,1\}.
```

Noisy-or replaces latent binary instance labels with probabilities:

```math
p_{ij}
=
P(y_{ij}=1\mid u_{ij})
=
\mathrm{sigmoid}(g_\theta(u_{ij})).
```

Assuming conditional independence of instance failures:

```math
P(y_i=0\mid H_i)
=
\prod_{j=1}^{n_i}(1-p_{ij}).
```

Therefore:

```math
P(y_i=1\mid H_i)
=
1-\prod_{j=1}^{n_i}(1-p_{ij}).
```

## Complement Survival View

Noisy-or is easier to understand through the negative event:

```text
bag is negative
    iff every instance fails to trigger positivity
```

Mathematically:

```math
y_i=0
\quad
\Leftrightarrow
\quad
y_{i1}=0,\ldots,y_{in_i}=0.
```

If each instance independently fails with probability
```math
1-p_{ij}
```
, multiply the
failure probabilities.

## Bag Logit

Let:

```math
r_i
=
1-\prod_j(1-p_{ij}).
```

Binary cross-entropy is:

```math
\mathcal{L}_i
=
-y_i\log r_i
-(1-y_i)\log(1-r_i).
```

The readout itself is the probability
```math
r_i
```
, not a vector embedding.

## Small-Probability Approximation

If all
```math
p_{ij}
```
are small:

```math
\prod_j(1-p_{ij})
\approx
1-\sum_j p_{ij}.
```

Then:

```math
r_i
\approx
\sum_j p_{ij}.
```

So noisy-or behaves like additive evidence in the rare-event, low-probability
regime.

## C/R/G/S Placement

```text
G:
    absent unless u_ij already contains geometry

C:
    instance probability model p_ij

R:
    1 - product_j (1 - p_ij)

S:
    bag-level binary label
```

## Dense Summary

Noisy-or pooling is:

```math
\boxed{
\mathcal{R}(H_i)
=
1-\prod_j(1-p_{ij})
}
```

It is the probabilistic readout for an at-least-one-positive bag assumption.

# Log-Space Gradients And Limits

Noisy-or can underflow when a slide has many patches. It is safer to compute the
negative probability in log space.

Let:

```math
q_i
=
P(y_i=0\mid H_i)
=
\prod_j(1-p_{ij}).
```

Then:

```math
\log q_i
=
\sum_j \log(1-p_{ij}).
```

The positive probability is:

```math
r_i
=
1-q_i
=
1-\exp\left(\sum_j\log(1-p_{ij})\right).
```

## Gradient With Respect To Instance Probability

The derivative of $r_i$ is:

```math
\frac{\partial r_i}{\partial p_{ik}}
=
\prod_{j\ne k}(1-p_{ij}).
```

Equivalently:

```math
\frac{\partial r_i}{\partial p_{ik}}
=
\frac{1-r_i}{1-p_{ik}}.
```

This shows the coupling: an instance receives gradient depending on the failure
probability of all other instances.

## Positive Bag Loss Gradient

For a positive bag:

```math
\mathcal{L}_i^{+}
=
-\log r_i.
```

Then:

```math
\frac{\partial \mathcal{L}_i^{+}}{\partial p_{ik}}
=
-
\frac{1}{r_i}
\prod_{j\ne k}(1-p_{ij}).
```

When $r_i$ is already close to $1$, gradients shrink. Once the model finds one
high-probability patch, the bag is nearly explained.

## Negative Bag Loss Gradient

For a negative bag:

```math
\mathcal{L}_i^{-}
=
-\log(1-r_i)
=
-\sum_j\log(1-p_{ij}).
```

Therefore:

```math
\frac{\partial \mathcal{L}_i^{-}}{\partial p_{ik}}
=
\frac{1}{1-p_{ik}}.
```

Every instance in a negative bag is pushed downward.

## Max Limit

Let:

```math
p_{\max}
=
\max_j p_{ij}.
```

Noisy-or satisfies:

```math
r_i
\ge
p_{\max}.
```

If one probability dominates and all others are near zero:

```math
r_i
\approx
p_{\max}.
```

## Dense Summary

Noisy-or gradients encode competition through the complement:

```math
\frac{\partial r_i}{\partial p_{ik}}
=
\prod_{j\ne k}(1-p_{ij}).
```

The operator accumulates weak probabilities, but it saturates once the bag is
almost certainly positive.


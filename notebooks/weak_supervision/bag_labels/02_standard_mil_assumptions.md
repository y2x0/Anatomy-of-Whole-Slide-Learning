# Standard MIL Assumptions

MIL assumptions specify how latent instance truth produces a bag label.

## Witness Assumption

The witness assumption says a positive bag contains at least one positive
instance:

```math
Y_i
=
\max_j Z_{ij}.
```

The word "witness" means that one instance can witness bag positivity.

The ideal witness index is:

```math
j_i^\star
\in
\{j:Z_{ij}=1\}.
```

But $j_i^\star$ is hidden. Max pooling estimates it by:

```math
\widehat j_i
=
\arg\max_j s_\theta(h_{ij}).
```

This is a latent witness estimator.

## Collective Assumption

The collective assumption says the bag label depends on many instances:

```math
Y_i
=
F
\left(
\frac{1}{n_i}\sum_{j=1}^{n_i}\phi(Z_{ij},h_{ij})
\right).
```

Mean pooling, additive evidence, prototype histograms, and distribution pooling
fit this more naturally than hard max pooling.

## Standard MIL Versus Collective MIL

Under witness MIL:

```math
\{1,0,0,\ldots,0\}
\quad
\text{and}
\quad
\{1,1,1,\ldots,1\}
```

give the same positive label.

Under collective MIL, these may differ because burden matters:

```math
\frac{1}{n_i}\sum_j Z_{ij}
```

is different.

## Multi-Class MIL

For $C$ classes:

```math
Z_{ij}
\in
\{0,1,\ldots,C\}.
```

A slide label $Y_i=c$ may mean:

```math
\exists j:Z_{ij}=c
```

or:

```math
c
=
\arg\max_{r}
\sum_j \mathbf{1}\{Z_{ij}=r\}.
```

Those are different bag maps. Multi-class MIL is often ambiguous unless the
paper states the latent class assumption.

## Multi-Label MIL

For multi-label classification:

```math
Y_{ic}
=
\max_j Z_{ijc},
\qquad
c=1,\ldots,C.
```

Each class has its own latent witness set.

CLAM-style class-specific attention is compatible with this idea because each
class has a separate readout:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

## C/R/G/S Placement

```text
G:
    optional spatial prior on which witnesses are plausible

C:
    instance scorer or contextualizer

R:
    max, noisy-or, attention, mean, additive, or distributional bag map

S:
    bag label constraining Gamma(Z), not Z itself
```

## Dense Summary

The key question is:

```text
Does one instance make the bag positive, or does the bag label summarize a
population of instances?
```

That choice determines whether max-like, noisy-or, additive, attention, or
distribution pooling is mathematically aligned with the supervision.

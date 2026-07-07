# Ranking And Order Constraints

Some partial annotations do not say:

```text
this patch has label 1
```

They say:

```text
this patch, region, or slide should rank above that one
```

This is common when experts can compare severity more reliably than assign
calibrated labels.

## Pairwise Ranking Constraint

Let $r_\theta(a)$ be a score for item $a$. A pairwise annotation:

```math
a
\succ
b
```

means:

```math
r_\theta(a)
>
r_\theta(b).
```

A logistic ranking loss is:

```math
\mathcal{L}_{\mathrm{rank}}
=
\log
\left(
1+
\exp
\left(
-
(r_\theta(a)-r_\theta(b))
\right)
\right).
```

## Region Ranking

For regions:

```math
r_{ir}
=
\mathcal{R}_{\mathrm{reg}}
(\{h_{ij}:j\in B_{ir}\}).
```

If region $r$ is more suspicious than $s$:

```math
B_{ir}
\succ
B_{is},
```

then:

```math
u^\top r_{ir}
>
u^\top r_{is}.
```

## Slide Ranking

Survival and grade annotations can induce slide ranking:

```math
i
\succ
i'
\quad
\Longrightarrow
\quad
\eta_i
>
\eta_{i'}.
```

This resembles survival ranking losses but can also be used for coarse ordinal
pathology labels.

## Partial Order Likelihood

A set of comparisons defines a partial order:

```math
\mathcal{P}
=
\{(a,b):a\succ b\}.
```

The Bradley-Terry probability is:

```math
P(a\succ b)
=
\frac{\exp(r_a)}
{\exp(r_a)+\exp(r_b)}.
```

The negative log likelihood is:

```math
\mathcal{L}_{\mathrm{BT}}
=
-
\sum_{(a,b)\in\mathcal{P}}
\log
\frac{\exp(r_a)}
{\exp(r_a)+\exp(r_b)}.
```

## C/R/G/S Placement

```text
G:
    rankings may be local to regions or global across slides

C:
    score function learns ordered evidence

R:
    creates region or slide scores

S:
    partial order constraints, not absolute labels
```

## Dense Summary

Ranking supervision identifies score differences, not calibrated probabilities:

```math
r_a-r_b
```

is constrained, but the absolute scale of $r$ is not. Ranking labels are useful
when relative severity is reliable and absolute class boundaries are noisy.

# Sparse Positive MIL

Max pooling is tied to a particular weak-supervision assumption:

```text
a bag is positive if at least one instance is positive
```

This is the sparse-positive MIL regime.

## Instance Labels And Bag Labels

Let latent instance labels be:

```math
y_{ij}\in\{0,1\}.
```

The bag label is:

```math
y_i
=
\max_j y_{ij}.
```

Equivalently:

```math
y_i=1
\quad
\Longleftrightarrow
\quad
\exists j:\ y_{ij}=1.
```

This is a logical OR over instances.

## Score-Based Relaxation

If a model predicts instance scores:

```math
s_{ij}
=
g_\theta(u_{ij}),
```

then the bag score is:

```math
r_i
=
\max_j s_{ij}.
```

The predicted probability can be:

```math
\widehat p_i
=
\sigma(r_i),
```

where $\sigma$ is a sigmoid function.

This means one high-scoring patch can drive the whole slide prediction.

## Noisy-Or Relaxation

If:

```math
p_{ij}
=
p(y_{ij}=1\mid u_{ij}),
```

and instance positives are conditionally independent, then:

```math
p(y_i=0\mid H_i)
=
\prod_j(1-p_{ij}).
```

Therefore:

```math
p(y_i=1\mid H_i)
=
1-\prod_j(1-p_{ij}).
```

This is noisy-or pooling. It is a probabilistic relaxation of the OR rule.

For small probabilities:

```math
1-\prod_j(1-p_{ij})
\approx
\sum_j p_{ij}.
```

So noisy-or behaves like additive evidence when all instance probabilities are
small, but saturates near one when enough evidence accumulates.

## Max Versus Noisy-Or

Max pooling:

```math
r_i=\max_j s_{ij}
```

keeps only the strongest instance.

Noisy-or:

```math
p_i
=
1-\prod_j(1-p_{ij})
```

combines all positive probabilities.

If one instance dominates:

```math
p_{ij^\star}\gg p_{ij}
\quad
\text{for}
\quad
j\ne j^\star,
```

then noisy-or and max-like pooling behave similarly.

If many weak positives accumulate, noisy-or can detect evidence that max may
underrepresent.

## Identifiability Problem

The observed loss is bag-level:

```math
\ell(\widehat p_i,y_i).
```

Patch labels are unobserved. Even if $y_i=1$, the data only imply:

```math
\exists j:\ y_{ij}=1.
```

They do not identify which $j$ is positive.

Many instance-label assignments are compatible with the same bag label:

```math
(1,0,0,0),
\quad
(0,1,0,0),
\quad
(1,1,0,0)
```

all imply:

```math
y_i=1.
```

Thus max-style MIL can localize only under extra assumptions from architecture,
regularization, priors, or data.

## Failure Modes

Wrong winner:

```text
early high-scoring nuisance patch receives most gradient
```

Diffuse phenotype:

```text
many moderate patches matter, but no single patch dominates
```

Multiple evidence regions:

```text
max ignores whether evidence appears once or repeatedly
```

Noisy positive patch:

```text
one artifact can determine the bag score
```

## Diagnostic Questions

1. Can one patch truly determine the label?
2. Is disease evidence focal or diffuse?
3. Should repeated evidence increase confidence?
4. Are instance labels identifiable from bag labels?
5. Does training become dominated by early false winners?

## Dense Summary

```math
\boxed{
y_i
=
\max_j y_{ij}
}
```

Sparse-positive MIL is a logical assumption about the label, not just a pooling
choice. Max pooling is appropriate when that logical assumption matches the
biology.

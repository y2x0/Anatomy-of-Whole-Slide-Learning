# InfoNCE And Mutual Information Bounds

InfoNCE is a classification loss over positives and sampled negatives.

For query
```math
q
```
, positive key
```math
k^+
```
, and negative keys
```math
k_1^-,\ldots,k_M^-
```
:

```math
\mathcal{L}_{\mathrm{NCE}}
=
-
\log
\frac{
\exp(\mathrm{sim}(q,k^+)/\tau)
}{
\exp(\mathrm{sim}(q,k^+)/\tau)
+
\sum_{m=1}^{M}
\exp(\mathrm{sim}(q,k_m^-)/\tau)
}.
```

Here
```math
\tau
```
is temperature.

## Softmax Classification View

The loss is cross entropy for identifying the positive key among candidates:

```math
P(k^+\mid q,\mathcal{K})
=
\frac{
\exp(\mathrm{sim}(q,k^+)/\tau)
}{
\sum_{k\in\mathcal{K}}
\exp(\mathrm{sim}(q,k)/\tau)
}.
```

Thus contrastive labels are not class labels. They are index labels within a
candidate set.

## Mutual Information Bound

Under ideal sampling assumptions, the expected InfoNCE objective lower bounds
mutual information:

```math
I(Q;K)
\ge
\log(M+1)
-
\mathbb{E}[\mathcal{L}_{\mathrm{NCE}}].
```

This statement is not a property of an arbitrary minibatch. It relies on:

```math
(q,k^+)
\sim
P_{QK},
```

```math
k_m^-
\sim
P_K
\quad
\text{iid and independent of }q,
```

and on a critic rich enough to approximate a density ratio:

```math
\exp(f(q,k))
\propto
\frac{p(k\mid q)}{p(k)}.
```

In WSI, this can fail because negatives may share disease class, tissue type,
organ, stain, or patient-level factors with the query.

If negatives are sampled from a biased batch distribution
```math
P_B
```
rather than
the population marginal
```math
P_K
```
, the contrastive objective estimates a different
classification problem:

```math
k_m^-
\sim
P_B
\ne
P_K.
```

## Gradient Shape

Let:

```math
s_k
=
\mathrm{sim}(q,k)/\tau.
```

The softmax weight is:

```math
\pi_k
=
\frac{\exp(s_k)}
{\sum_{\ell}\exp(s_\ell)}.
```

The gradient pulls the query toward the positive:

```math
\nabla_q\mathcal{L}
\supset
-
\nabla_q s_{k^+},
```

and pushes it away from all candidates proportional to
```math
\pi_k
```
:

```math
\nabla_q\mathcal{L}
=
\sum_k
\pi_k\nabla_q s_k
-
\nabla_q s_{k^+}.
```

Hard negatives with high
```math
\pi_k
```
dominate repulsion.

## Temperature

Small
```math
\tau
```
sharpens the denominator:

```math
\pi_k
\to
\mathbf{1}\{k=\arg\max_\ell s_\ell\}.
```

Large
```math
\tau
```
spreads gradients across more negatives.

## C/R/G/S Placement

```text
G:
    can determine which negatives are truly negative or false negative

C:
    encoder learns invariances from positive construction

R:
    object embedding is the representation being contrasted

S:
    positive index among sampled candidates
```

## Dense Summary

InfoNCE is mathematically clean only after specifying the sampling distribution:

```math
k^+\sim P(k\mid q),
\qquad
k^-\sim P(k).
```

In pathology, these distributions are design choices, not facts.

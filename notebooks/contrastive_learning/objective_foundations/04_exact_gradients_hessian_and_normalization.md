# Exact Gradients, Hessian, And Normalization

The InfoNCE gradient is not a generic pull-push metaphor. It is the difference
between a Gibbs-weighted candidate mean and the target key, followed by the
Jacobian of representation normalization.

Primary sources:

- Chen et al. "A Simple Framework for Contrastive Learning of Visual
  Representations." ICML 2020. https://arxiv.org/abs/2002.05709
- Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
  https://arxiv.org/abs/2004.11362

## Directional Loss

Let query `q` and candidate keys `k_1,...,k_N` be fixed-dimensional column
vectors. Let `y` be the positive index. Define:

```math
\ell_j
=
\frac{q^{\top}k_j}{\tau}.
```

The candidate softmax is:

```math
\pi_j
=
\frac{\exp(\ell_j)}
{\sum_{r=1}^{N}\exp(\ell_r)}.
```

The directional loss is:

```math
\mathcal{L}
=
-\ell_y
+
\log
\sum_{j=1}^{N}
\exp(\ell_j).
```

## Logit Gradient

For every candidate:

```math
\frac{\partial\mathcal{L}}
{\partial\ell_j}
=
\pi_j
-
\mathbb{1}\{j=y\}.
```

The positive coefficient is:

```math
\pi_y-1
\le
0.
```

Each negative coefficient is:

```math
\pi_j
\ge
0,
\qquad
j\ne y.
```

## Query Gradient

Because:

```math
\nabla_q\ell_j
=
\frac{k_j}{\tau},
```

the query gradient is:

```math
\nabla_q\mathcal{L}
=
\frac{1}{\tau}
\left(
\sum_{j=1}^{N}
\pi_jk_j
-
k_y
\right).
```

Define the Gibbs candidate mean:

```math
\mu_{\pi}
=
\sum_{j=1}^{N}
\pi_jk_j.
```

Then:

```math
\nabla_q\mathcal{L}
=
\frac{
\mu_{\pi}-k_y
}{\tau}.
```

Gradient descent moves the query toward the positive and away from the current
softmax-weighted candidate center.

## Key Gradients

For a candidate key:

```math
\nabla_{k_j}\mathcal{L}
=
\frac{
\pi_j-\mathbb{1}\{j=y\}
}{\tau}
q.
```

The positive key is attracted toward the query. A negative is repelled in
proportion to its current softmax probability.

If the same embedding appears as a key for many anchors, its total gradient is
the sum over all those roles. Per-anchor reasoning alone does not describe the
full symmetric minibatch update.

## Hessian With Respect To The Query

Holding keys fixed:

```math
\nabla_q^2\mathcal{L}
=
\frac{1}{\tau^2}
\left(
\sum_{j=1}^{N}
\pi_jk_jk_j^{\top}
-
\mu_{\pi}\mu_{\pi}^{\top}
\right).
```

Equivalently:

```math
\nabla_q^2\mathcal{L}
=
\frac{1}{\tau^2}
\mathrm{Cov}_{j\sim\pi}
[k_j].
```

The Hessian is positive semidefinite. For fixed keys, directional InfoNCE is
convex in the unconstrained query vector.

Curvature is large in directions where high-probability keys vary and zero in
directions absent from the candidate covariance.

## Hard-Negative Weighting

For negative `j`:

```math
\lVert
\nabla_{k_j}\mathcal{L}
\rVert_2
=
\frac{\pi_j}{\tau}
\lVert q\rVert_2.
```

A negative with high similarity receives exponentially larger weight through:

```math
\pi_j
\propto
\exp(q^{\top}k_j/\tau).
```

This is implicit hard-negative weighting. It is also why a high-similarity
false negative can be especially destructive.

## Normalization Jacobian

Let a raw vector be:

```math
w
\in
\mathbb{R}^{d},
```

and its normalized form:

```math
u
=
\frac{w}{\lVert w\rVert_2}.
```

The Jacobian is:

```math
J_{\mathrm{norm}}(w)
=
\frac{1}{\lVert w\rVert_2}
\left(
I_d
-
uu^{\top}
\right).
```

For a loss defined on `u`:

```math
\nabla_w\mathcal{L}
=
\frac{1}{\lVert w\rVert_2}
\left(
I_d-uu^{\top}
\right)
\nabla_u\mathcal{L}.
```

The projector:

```math
I_d-uu^{\top}
```

removes the radial component. Therefore:

```math
u^{\top}
\nabla_w\mathcal{L}
=
0.
```

The immediate gradient changes direction on the sphere rather than norm along
the current ray.

## Norm-Dependent Learning Rate

Although normalized logits ignore raw norm, the gradient carries:

```math
\lVert w\rVert_2^{-1}.
```

Two samples with identical normalized directions but different raw norms can
receive different gradient magnitudes:

```math
w_1
=
c_1u,
\qquad
w_2
=
c_2u,
```

```math
\frac{
\lVert\nabla_{w_1}\mathcal{L}\rVert_2
}{
\lVert\nabla_{w_2}\mathcal{L}\rVert_2
}
=
\frac{c_2}{c_1}.
```

Forward scale invariance does not imply optimization-scale invariance.

## Temperature And Curvature

The query gradient scales as:

```math
\tau^{-1},
```

while the fixed-key Hessian scales as:

```math
\tau^{-2}.
```

Lower temperature simultaneously sharpens candidate probabilities and
increases local curvature. It can emphasize useful hard comparisons or amplify
optimization instability and mislabeled pairs.

## C/R/G/S Placement

```text
\mathcal{G}:
    candidate vectors and their Gibbs covariance

\mathcal{C}:
    encoder Jacobians receiving the embedding gradient

\mathcal{R}:
    projection and normalization Jacobian

\mathcal{S}:
    one-hot positive index determining pi - y
```

## Dense Summary

The exact local mechanism is:

```math
\text{gradient}
=
\frac{
\text{softmax candidate mean}
-
\text{positive key}
}{\tau},
```

followed by tangent projection when embeddings are normalized.

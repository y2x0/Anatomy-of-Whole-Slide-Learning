# Temperature And Hyperspherical Geometry

SimCLR applies normalized temperature-scaled cross-entropy in projection space.
Normalization and temperature jointly determine the geometry and gradient
competition of the objective.

Primary source:

- Chen et al. "A Simple Framework for Contrastive Learning of Visual
  Representations." ICML 2020. https://arxiv.org/abs/2002.05709

## Normalized Projection

Let an encoder and projector produce:

```math
h_i
=
f_{\theta}(v_i),
\qquad
z_i
=
g_{\phi}(h_i).
```

Normalize the projected vector:

```math
u_i
=
\frac{z_i}{\lVert z_i\rVert_2}
\in
\mathbb{S}^{d-1}.
```

The cosine score is:

```math
s_{ij}
=
u_i u_j^{\top}
\in
[-1,1].
```

The contrastive logit is:

```math
\ell_{ij}
=
\frac{s_{ij}}{\tau},
\qquad
\tau>0.
```

## Angular And Euclidean Equivalence

For unit vectors:

```math
\lVert u_i-u_j\rVert_2^2
=
2-2u_i u_j^{\top}.
```

Therefore:

```math
s_{ij}
=
1
-
\frac{1}{2}
\lVert u_i-u_j\rVert_2^2.
```

Maximizing cosine similarity is equivalent to minimizing squared Euclidean
distance on the unit sphere. This equivalence does not hold for unnormalized
vectors because vector norm then changes the dot product independently of
angle.

## NT-Xent As Attraction Plus Log-Partition

Let `p(i)` be the positive index for anchor `i`, and let `A(i)` contain all
candidate indices except the anchor. The directional loss is:

```math
\mathcal{L}_i
=
-
\frac{s_{i,p(i)}}{\tau}
+
\log
\sum_{a\in A(i)}
\exp
\left(
\frac{s_{ia}}{\tau}
\right).
```

The first term attracts the positive. The log-partition compares that
attraction against every candidate, including the positive.

Substituting spherical distances cancels the common constant:

```math
\mathcal{L}_i
=
\frac{
\lVert u_i-u_{p(i)}\rVert_2^2
}{2\tau}
+
\log
\sum_{a\in A(i)}
\exp
\left(
-
\frac{
\lVert u_i-u_a\rVert_2^2
}{2\tau}
\right).
```

Thus temperature is the scale of a Gibbs distribution over spherical squared
distances.

## Candidate Gibbs Measure

Define:

```math
\pi_{ia}(\tau)
=
\frac{
\exp(s_{ia}/\tau)
}{
\sum_{r\in A(i)}
\exp(s_{ir}/\tau)
}.
```

This distribution determines which candidates dominate the log-partition and
the gradient.

Its entropy is:

```math
\mathcal{H}_i(\tau)
=
-
\sum_{a\in A(i)}
\pi_{ia}(\tau)
\log
\pi_{ia}(\tau).
```

Small temperature tends to lower entropy; large temperature tends to increase
it toward the logarithm of candidate-set size.

## Low-Temperature Limit

The log-sum-exp limit is:

```math
\lim_{\tau\downarrow 0}
\tau
\log
\sum_{a\in A(i)}
\exp(s_{ia}/\tau)
=
\max_{a\in A(i)}s_{ia}.
```

Therefore:

```math
\lim_{\tau\downarrow 0}
\tau\mathcal{L}_i
=
\max_{a\in A(i)}s_{ia}
-
s_{i,p(i)}.
```

At low temperature, the loss approaches a hardest-candidate gap. If a negative
has larger similarity than the positive, that negative controls the leading
term.

If several candidates tie for maximum similarity, the unscaled loss retains a
logarithmic multiplicity term even when the scaled limit is the same.

## High-Temperature Expansion

Let:

```math
M_i
=
|A(i)|,
```

```math
\overline s_i
=
\frac{1}{M_i}
\sum_{a\in A(i)}s_{ia}.
```

For large temperature:

```math
\mathcal{L}_i
=
\log M_i
+
\frac{
\overline s_i-s_{i,p(i)}
}{\tau}
+
\frac{
\mathrm{Var}_{a\sim\mathrm{Unif}(A(i))}
[s_{ia}]
}{2\tau^2}
+
\mathcal{O}(\tau^{-3}).
```

The loss approaches `log M_i`, while representation-dependent gradients shrink
with inverse temperature.

## Temperature Derivative

Holding similarities fixed:

```math
\frac{\partial\mathcal{L}_i}
{\partial\tau}
=
\frac{
s_{i,p(i)}
-
\sum_{a\in A(i)}
\pi_{ia}s_{ia}
}{\tau^2}.
```

Temperature therefore compares positive similarity with the Gibbs-weighted
candidate similarity. It is not only a numerical softmax parameter.

## Scale Invariance And Its Gradient Cost

For any positive scalar `c`:

```math
\frac{cz_i}{\lVert cz_i\rVert_2}
=
u_i.
```

Forward logits are invariant to pre-normalization scale. Gradients are not,
because the normalization Jacobian contains:

```math
\frac{1}{\lVert z_i\rVert_2}.
```

Large raw projection norms reduce gradient magnitude through normalization even
though they do not change the forward similarity.

## Projection Geometry Versus Retained Geometry

SimCLR optimizes:

```math
u_i
=
\mathrm{Normalize}
\left(
g_{\phi}(h_i)
\right),
```

but retains:

```math
h_i
=
f_{\theta}(v_i)
```

for downstream evaluation. The projector can absorb invariances that would be
harmful if imposed directly on every coordinate of `h_i`.

Consequently:

```math
u_i^{\top}u_j
```

is not automatically the correct downstream similarity between retained
representations.

## C/R/G/S Placement

```text
\mathcal{G}:
    unit-sphere angular geometry and candidate support A(i)

\mathcal{C}:
    base encoder f_theta

\mathcal{R}:
    projection g_phi followed by L2 normalization

\mathcal{S}:
    positive index and temperature-scaled candidate cross-entropy
```

## Dense Summary

Temperature controls which angular comparisons matter:

```text
small tau:
    hardest-candidate geometry

large tau:
    diffuse candidate averaging with weak gradients
```

Normalization removes radial scoring freedom and moves optimization onto the
tangent geometry of a sphere.

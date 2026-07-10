# RetCCL Subqueues And Weighted InfoNCE

RetCCL partitions a memory bank by online k-means and attenuates negatives from
the cluster most similar to the current key. Its weight scales the negative
logit, not the exponential term outside the logit.

Primary anchors:

- Wang et al. "RetCCL: Clustering-guided Contrastive Learning for Whole-slide
  Image Retrieval." Medical Image Analysis 2023.
  https://pubmed.ncbi.nlm.nih.gov/36270093/
- Official RetCCL implementation and pretrained model:
  https://github.com/Xiyue-Wang/RetCCL

## Three Augmented Inputs

RetCCL constructs three augmented images from one patch:

```math
x_p,
\qquad
x_k,
\qquad
x_q.
```

The paper's feature-extraction diagram uses two encoder roles:

```math
h_p
=
h(x_p),
\qquad
f_k
=
f(x_k),
\qquad
h_q
=
h(x_q).
```

Two projection heads separate instance-level and group-level objectives. For
the `p` and `q` paths:

```math
g_{p1}
=
g_1(h_p),
\qquad
g_{p2}
=
g_2(h_p),
```

```math
g_{q1}
=
g_1(h_q),
\qquad
g_{q2}
=
g_2(h_q).
```

The key path uses the second head:

```math
g_k
=
g_2(f_k).
```

Thus:

```math
\left(
g_{p2},g_k
\right),
\qquad
\left(
g_{q2},g_k
\right)
```

are the two instance-level positive pairs, while `g_{p1}` and
`g_{q1}` feed group discrimination.

## Memory Partition

Let the negative memory bank contain:

```math
\mathcal{M}
=
\left\{
g_{k_1^{-}},
\ldots,
g_{k_L^{-}}
\right\}.
```

K-means partitions it into `Q` subqueues:

```math
\mathcal{M}
=
\bigsqcup_{r=1}^{Q}
\mathcal{Q}_r,
```

with centroids:

```math
\mu_1,
\ldots,
\mu_Q.
```

For the current key `g_k`, select the most similar centroid:

```math
r_{\max}
=
\arg\max_{r\in\left\{
1,\ldots,Q
\right\}}
\mathrm{sim}
\left(
g_k,\mu_r
\right).
```

Define:

```math
\mathcal{Q}_{\max}
=
\mathcal{Q}_{r_{\max}}.
```

Negatives in this subqueue are treated as false-negative-like because they lie
in the cluster closest to the current key.

## Exact Piecewise Weight

RetCCL defines:

```math
\phi
\left(
g_{k_i^{-}}
\right)
=
\begin{cases}
w,
&
g_{k_i^{-}}
\in
\mathcal{Q}_{\max},
\\
1,
&
\text{otherwise},
\end{cases}
```

where:

```math
w
\in
\left[
0,1
\right).
```

The weight is cluster-membership based. It is not a continuous posterior
probability that a negative is false.

## Exact Weighted InfoNCE

The paper's weighted instance loss is:

```math
\mathcal{L}_{\mathrm{W\mbox{-}InfoNCE}}
=
-
\frac{1}{2}
\log
\frac{
\exp
\left(
g_{p2}^{\top}g_k/\tau
\right)
}{
\exp
\left(
g_{p2}^{\top}g_k/\tau
\right)
+
\sum_{i=1}^{L}
\exp
\left(
\phi
\left(
g_{k_i^{-}}
\right)
g_{p2}^{\top}g_{k_i^{-}}/\tau
\right)
}
```

```math
\phantom{
\mathcal{L}_{\mathrm{W\mbox{-}InfoNCE}}
}
-
\frac{1}{2}
\log
\frac{
\exp
\left(
g_{q2}^{\top}g_k/\tau
\right)
}{
\exp
\left(
g_{q2}^{\top}g_k/\tau
\right)
+
\sum_{i=1}^{L}
\exp
\left(
\phi
\left(
g_{k_i^{-}}
\right)
g_{q2}^{\top}g_{k_i^{-}}/\tau
\right)
}.
```

The two query-side augmentations share the positive key and negative memory.

## Weighted Logit

For one anchor `a`, positive `k`, and negative `n_i`, define:

```math
s_{+}
=
a^{\top}k,
\qquad
s_i
=
a^{\top}n_i,
\qquad
\phi_i
=
\phi(n_i).
```

The negative logit is:

```math
\ell_i^{-}
=
\frac{
\phi_i s_i
}{
\tau
}.
```

This differs from multiplying negative exponential mass:

```math
\phi_i
\exp
\left(
s_i/\tau
\right).
```

The RetCCL formula uses:

```math
\exp
\left(
\phi_i s_i/\tau
\right).
```

## Exact Similarity Gradient

For one directional loss:

```math
\mathcal{L}_a
=
-
\log
\frac{
\exp
\left(
s_{+}/\tau
\right)
}{
\exp
\left(
s_{+}/\tau
\right)
+
\sum_i
\exp
\left(
\phi_i s_i/\tau
\right)
},
```

define the negative softmax mass:

```math
\pi_i
=
\frac{
\exp
\left(
\phi_i s_i/\tau
\right)
}{
\exp
\left(
s_{+}/\tau
\right)
+
\sum_r
\exp
\left(
\phi_r s_r/\tau
\right)
}.
```

Holding the cluster weight fixed:

```math
\frac{
\partial\mathcal{L}_a
}{
\partial s_i
}
=
\frac{
\phi_i
}{
\tau
}
\pi_i.
```

The direct repulsive force through the similarity is multiplied by
`\phi_i`.

## Anchor Gradient

Treating candidates and weights as fixed:

```math
\nabla_a
\mathcal{L}_a
=
\frac{1}{\tau}
\left[
\sum_i
\pi_i\phi_i n_i
-
\left(
1-\pi_{+}
\right)
k
\right],
```

where:

```math
\pi_{+}
=
\frac{
\exp
\left(
s_{+}/\tau
\right)
}{
\exp
\left(
s_{+}/\tau
\right)
+
\sum_r
\exp
\left(
\phi_r s_r/\tau
\right)
}.
```

Same-subqueue negatives contribute scaled vectors `wn_i` to the
repulsive barycenter.

## Sign-Dependent Denominator Effect

For:

```math
0
\le
w
<
1,
```

and a positive negative-pair similarity:

```math
s_i
>
0,
```

weighting reduces its exponential mass:

```math
\exp
\left(
ws_i/\tau
\right)
<
\exp
\left(
s_i/\tau
\right).
```

For a negative similarity:

```math
s_i
<
0,
```

weighting moves the logit toward zero:

```math
ws_i
>
s_i,
```

so:

```math
\exp
\left(
ws_i/\tau
\right)
>
\exp
\left(
s_i/\tau
\right).
```

The loss value can receive more denominator mass from an already-dissimilar
same-cluster negative even though its derivative with respect to similarity is
scaled by `w`. "Downweighting" is monotone in gradient strength but not
in exponential mass for negative dot products.

## Zero Weight Does Not Delete A Candidate

If:

```math
w
=
0,
```

then a same-subqueue term becomes:

```math
\exp
\left(
0
\cdot
s_i/\tau
\right)
=
1.
```

Its similarity gradient is:

```math
\frac{
\partial\mathcal{L}_a
}{
\partial s_i
}
=
0,
```

but its denominator contribution remains one. Therefore `w=0` blocks
repulsion through that similarity while leaving a constant normalization
penalty. It is not equivalent to removing the negative.

## Hard Cluster Boundary

The weight switches at the nearest-centroid rule:

```math
\phi_i
=
w
\quad
\Longleftrightarrow
\quad
n_i
\in
\mathcal{Q}_{r_{\max}(g_k)}.
```

When two key-centroid similarities tie:

```math
\mathrm{sim}
\left(
g_k,\mu_a
\right)
=
\mathrm{sim}
\left(
g_k,\mu_b
\right),
```

the identity of `\mathcal{Q}_{\max}` can switch. Many negative weights
can change simultaneously.

## Statistical Interpretation Boundary

Classical InfoNCE assumes candidates from a proposal distribution and uses
unweighted logits. RetCCL modifies the critic with a data-dependent cluster
weight:

```math
s_{\mathrm{RetCCL}}
\left(
a,n
\right)
=
\phi
\left(
n;
g_k,\mathcal{M}
\right)
a^{\top}n.
```

The standard unrestricted density-ratio optimum does not directly describe
this restricted, cluster-conditioned score family.

## C/R/G/S Placement

```text
\mathcal{G}:
    momentum-style positive pair, memory negatives, online k-means subqueues,
    and the key-nearest subqueue relation

\mathcal{C}:
    encoder branches plus memory partition and piecewise weight assignment

\mathcal{R}:
    weighted negative logits in the g_2 projection space

\mathcal{S}:
    two-direction average of weighted InfoNCE terms sharing one key
```

## Surviving Statistic

The weighted branch preserves instance discrimination while reducing the
similarity gradient from negatives assigned to the current key's nearest
memory cluster. The repair is binary at the cluster level and does not convert
those negatives into positives.

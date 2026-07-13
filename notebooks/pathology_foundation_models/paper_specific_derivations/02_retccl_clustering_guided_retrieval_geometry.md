# RetCCL: Clustering-Guided Retrieval Geometry

References: [RetCCL paper](https://pubmed.ncbi.nlm.nih.gov/36270093/),
[official implementation](https://github.com/Xiyue-Wang/RetCCL).

RetCCL has two distinct parts: clustering-guided contrastive learning (CCL)
for the patch encoder and a nonparametric WSI retrieval system. Its CCL does
not simply turn a cluster into a positive set. It attenuates a cluster of
negative logits and adds a separate cross-branch centroid objective.

## 1. Three Views And Two Projection Spaces

For one patch, RetCCL constructs three views:

```math
x_p,
\qquad
x_k,
\qquad
x_q.
```

The query views use encoder `h`, while the key view uses momentum encoder `f`:

```math
h_p=h(x_p),
\qquad
f_k=f(x_k),
\qquad
h_q=h(x_q).
```

Two heads define separate geometries:

```math
g_{p1}=g_1(h_p),
\qquad
g_{p2}=g_2(h_p),
```

```math
g_{q1}=g_1(h_q),
\qquad
g_{q2}=g_2(h_q),
\qquad
g_k=g_2(f_k).
```

The second head feeds instance CCL; the first head feeds group CCL.

## 2. Subqueue-Based Negative Geometry

Let the memory bank be partitioned by online k-means:

```math
\mathcal M
=
\bigsqcup_{r=1}^{Q}\mathcal Q_r,
\qquad
\mu_r
=
\frac{1}{|\mathcal Q_r|}
\sum_{v\in\mathcal Q_r}v.
```

For current key `g_k`, select its most similar centroid:

```math
r^{\star}
=
\arg\max_{r\in\{1,\ldots,Q\}}
\mathrm{sim}(g_k,\mu_r).
```

The paper's piecewise negative weight is:

```math
\phi(v)
=
\begin{cases}
w,&v\in\mathcal Q_{r^{\star}},\\
1,&v\notin\mathcal Q_{r^{\star}},
\end{cases}
\qquad
0\le w<1.
```

The nearest subqueue is treated as false-negative-like. The weight is not a
probability of semantic equivalence and is not a differentiable confidence.

## 3. Weighted Instance InfoNCE

For anchor `a`, positive key `k`, and memory negatives `v_l`, RetCCL uses:

```math
\ell_{\mathrm{W\mbox{-}NCE}}(a,k)
=
-\log
\frac{
\exp(a^{\top}k/\tau)
}{
\exp(a^{\top}k/\tau)
+
\sum_{l=1}^{L}
\exp
\left(
\phi(v_l)a^{\top}v_l/\tau
\right)
}.
```

The symmetric instance objective is:

```math
\mathcal L_{\mathrm{W\mbox{-}InfoNCE}}
=
\frac{1}{2}
\ell_{\mathrm{W\mbox{-}NCE}}(g_{p2},g_k)
+
\frac{1}{2}
\ell_{\mathrm{W\mbox{-}NCE}}(g_{q2},g_k).
```

The weight multiplies the negative logit before exponentiation. It is not the
same as multiplying the negative exponential by `w`:

```math
\exp
\left(
w a^{\top}v/\tau
\right)
\neq
w\exp
\left(
a^{\top}v/\tau
\right).
```

For `s=a^\top v`, the direct derivative is:

```math
\frac{\partial\ell_{\mathrm{W\mbox{-}NCE}}}{\partial s}
=
\frac{\phi(v)}{\tau}\pi_v,
```

where `pi_v` is that negative's denominator probability. Same-subqueue
negatives therefore receive weaker similarity gradients, but their denominator
mass is not simply deleted. At `w=0`, the term becomes the constant `1` in the
denominator.

## 4. Cross-Branch Group InfoNCE

The first projection head is clustered separately on the two query branches:

```math
\mathcal S^p
=
\left\{S_1^p,\ldots,S_S^p\right\},
\qquad
\mathcal S^q
=
\left\{S_1^q,\ldots,S_S^q\right\}.
```

For `g_{p1}`, select the nearest opposite-branch centroid:

```math
r_p^{\star}
=
\arg\max_r
g_{p1}^{\top}S_r^q,
\qquad
S^{q+}=S_{r_p^{\star}}^q.
```

The remaining `S-1` centroids are negatives. The `p` to `q` direction is:

```math
\ell_{p\to q}
=
-\log
\frac{
\exp(g_{p1}^{\top}S^{q+}/\tau)
}{
\sum_{r=1}^{S}
\exp(g_{p1}^{\top}S_r^q/\tau)
}.
```

The reverse direction uses `g_{q1}` and `\mathcal S^p`, giving:

```math
\mathcal L_{\mathrm{G\mbox{-}InfoNCE}}
=
\frac{1}{2}
\ell_{p\to q}
+
\frac{1}{2}
\ell_{q\to p}.
```

Conditional on the selected centroid, the anchor gradient is:

```math
\nabla_{g_{p1}}\ell_{p\to q}
=
\frac{1}{\tau}
\left[
\sum_{r=1}^{S}
\pi_r S_r^q
-
S_{r_p^{\star}}^q
\right].
```

The argmax makes the target self-generated. A small margin between the first
and second centroid scores can switch the target discontinuously.

## 5. Combined CCL Objective

The paper combines the two geometries:

```math
\mathcal L_{\mathrm{CCL}}
=
\mathcal L_{\mathrm{W\mbox{-}InfoNCE}}
+
\lambda
\mathcal L_{\mathrm{G\mbox{-}InfoNCE}}.
```

The encoder receives gradients from both heads, but the two heads can organize
different projected spaces. The cluster assignments, subqueue choice, and
cross-branch centroid target are generated supervision, not ground-truth
pathology labels.

## 6. WSI Mosaic Construction

For slide `i`, the pretrained CCL encoder maps foreground patches to:

```math
z_{ij}=f_{\widehat\theta}(p_{ij}),
\qquad
r_{ij}\in\mathbb R^2.
```

RetCCL first applies feature k-means with `K_1=9` clusters and then performs a
spatial clustering step within each feature cluster using the reported ratio
`R=0.2`. The mosaic is:

```math
\mathcal M_i
=
\bigcup_{k=1}^{K_1}
\mathcal M_{i,k},
```

where `M_{i,k}` contains the selected spatial representatives. The slide is
therefore represented by a selected set of patch features and metadata:

```math
\mathcal M_i
\subset
\mathbb R^d
\times
\mathcal I
\times
\mathcal Y.
```

It is not reduced to one first moment before retrieval.

## 7. Query Bags And Ranking

Each query patch retrieves a bag of database patches:

```math
\mathbb B_i
=
\left\{
b_i^1,\ldots,b_i^t
\right\}.
```

For diagnosis `m`, the bag distribution uses similarity and database-label
weights:

```math
p_{i,m}
=
\frac{
\sum_{j=1}^{t}
\mathbf 1[y_i^j=m]
w_{y_i^j}(d_i^j+1)/2
}{
\sum_{j=1}^{t}
w_{y_i^j}(d_i^j+1)/2
}.
```

Entropy ranks uncertainty:

```math
H_i
=
-\sum_m p_{i,m}\log p_{i,m}.
```

This is a diagnosis-aware retrieval readout. The CCL encoder is self-supervised,
but the full retrieval system uses archive diagnosis metadata.

## 8. C/R/G/S Placement

```text
C:
    momentum key encoder, subqueue assignment, cross-branch centroid comparison,
    feature-spatial mosaic construction, and query-bag ranking

R:
    weighted instance InfoNCE, centroid group InfoNCE, then diagnosis-aware
    patch retrieval and source-slide ranking

G:
    angular patch geometry, k-means subqueues, 2D spatial representatives, and
    archive-neighbor geometry

S:
    augmentation pairs, false-negative-like subqueue weights, generated centroid
    targets, and diagnosis metadata during retrieval
```

## 9. Failure Matrix

| Failure | Mathematical source | Consequence |
|---|---|---|
| cluster shortcut | nearest-centroid subqueue | stain or organ can define the downweighted negatives |
| weight misread | `exp(phi s / tau)` versus `phi exp(s / tau)` | wrong gradient and denominator interpretation |
| self-labeling loop | cross-branch argmax centroid target | early cluster mistakes reinforce themselves |
| mosaic omission | feature and spatial selection | rare diagnostic patches may never enter the archive |
| archive dependence | diagnosis-aware bag ranking | retrieval behavior changes with cohort composition |
| layout loss | final cosine search ignores selected-patch adjacency | similar feature sets with different arrangements can collide |

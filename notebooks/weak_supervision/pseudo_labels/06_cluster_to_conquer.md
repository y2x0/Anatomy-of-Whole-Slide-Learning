# Cluster-To-Conquer

Cluster-to-Conquer is a weakly supervised WSI method that uses clusters to
choose informative patch subsets, but its supervision is still slide-level. The
important distinction is:

```math
S_i^{\mathrm{obs}}
=
Y_i,
```

while patch clusters and patch targets are generated:

```math
\kappa_i
=
\Psi_{\mathrm{kmeans}}(H_i),
\qquad
\widehat Z_{ij}
=
Y_i.
```

Reference: [Cluster-to-Conquer paper](https://proceedings.mlr.press/v143/sharma21a.html).

## Local Clustering Per Slide

For each WSI, patch embeddings are clustered locally:

```math
\kappa_i(j)
\in
\{1,\ldots,K_i\}.
```

The cluster
```math
k
```
is:

```math
\mathcal{C}_{ik}
=
\{j:\kappa_i(j)=k\}.
```

This is not a global pathology vocabulary shared across all slides. It is a
within-slide partition used to sample diverse patches.

## Cluster-Based Sampling

Rather than training on every patch, the method samples from clusters:

```math
\mathcal{S}_i
\subset
\{1,\ldots,n_i\},
```

with sampling constrained by:

```math
\mathcal{S}_i
\cap
\mathcal{C}_{ik}
\ne
\varnothing
```

for selected clusters. The mathematical purpose is to reduce redundant tissue
sampling and expose the model to multiple local modes of the slide distribution.

## Slide-Level MIL Loss

The sampled patches are encoded and aggregated:

```math
z_i
=
\mathcal{R}_\theta(\{h_{ij}:j\in\mathcal{S}_i\}),
```

```math
\widehat p_i
=
\mathcal{H}_\theta(z_i).
```

The slide-level cross-entropy is:

```math
\mathcal{L}_{\mathrm{slide}}
=
\sum_i
\mathrm{CE}(\widehat p_i,Y_i).
```

## Weak Patch-Level Cross-Entropy

Cluster-to-Conquer also applies a weak patch-level loss by copying the slide
label to selected patches:

```math
\widehat Z_{ij}
=
Y_i,
\qquad
j\in\mathcal{S}_i.
```

If
```math
p_{ij}
```
is the patch prediction, the weak patch loss is:

```math
\mathcal{L}_{\mathrm{patch}}
=
\sum_i
\sum_{j\in\mathcal{S}_i}
\mathrm{CE}(p_{ij},Y_i).
```

This is intentionally noisy. In a positive slide, many sampled patches may not
be positive:

```math
Y_i=1,
\qquad
Z_{ij}=0,
\qquad
\widehat Z_{ij}=1.
```

The cluster sampler is supposed to make the noisy copied labels less damaging
by increasing coverage of relevant tissue modes.

## Attention Regularization Within Clusters

The method also regularizes attention so patches from the same local cluster do
not collapse to an arbitrary single instance. Let attention weights inside
cluster
```math
k
```
be normalized:

```math
\bar a_{ij}^{(k)}
=
\frac{a_{ij}}
{\sum_{\ell\in\mathcal{C}_{ik}}a_{i\ell}},
\qquad
j\in\mathcal{C}_{ik}.
```

The uniform distribution over that cluster is:

```math
u_{ij}^{(k)}
=
\frac{1}{|\mathcal{C}_{ik}|}.
```

A KL-style regularizer has the form:

```math
\mathcal{L}_{\mathrm{KL}}
=
\sum_i
\sum_{k=1}^{K_i}
\mathrm{KL}
\left(
u_i^{(k)}
\;\|\;
\bar a_i^{(k)}
\right).
```

The orientation of the KL is less important than the inductive bias: within a
local cluster, attention should not become a degenerate one-patch selector too
early.

## Full Objective

The combined objective can be summarized as:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{slide}}
+
\lambda_{\mathrm{patch}}
\mathcal{L}_{\mathrm{patch}}
+
\lambda_{\mathrm{KL}}
\mathcal{L}_{\mathrm{KL}}.
```

Each term uses a different generated signal:

```text
slide CE:
    observed slide label

patch CE:
    copied slide label on sampled patches

KL:
    local cluster membership
```

## Failure Mode

The key failure is cluster-label mismatch. A cluster can be morphologically
coherent without being diagnostically relevant:

```math
h_{ij},h_{i\ell}\in\mathcal{C}_{ik}
\not\Rightarrow
Z_{ij}=Z_{i\ell}.
```

Even if the cluster is coherent, copying
```math
Y_i
```
to every sampled patch still
assumes:

```math
P(Z_{ij}=Y_i\mid j\in\mathcal{S}_i)
```

is high enough to be useful. That is not guaranteed by k-means.

## C/R/G/S Placement

```text
G:
    local k-means clusters define a within-slide sampling geometry

C:
    weak patch CE shapes patch embeddings directly

R:
    attention readout is regularized by cluster membership

S:
    observed slide label plus generated patch targets and cluster relations
```

## Dense Summary

Cluster-to-Conquer turns:

```math
Y_i
```

into three training signals:

```math
Y_i,
\qquad
\widehat Z_{ij}=Y_i,
\qquad
\kappa_i(j).
```

Its mathematical bet is that local cluster structure makes copied slide labels
less destructive by improving sampling diversity and attention stability.

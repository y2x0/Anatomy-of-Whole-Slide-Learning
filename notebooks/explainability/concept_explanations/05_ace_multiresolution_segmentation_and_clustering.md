# ACE Multiresolution Segmentation and Clustering

ACE automates concept proposal before applying TCAV.

## 1. Segment Generation

For each class image `x`, apply SLIC at several resolutions:

```math
\mathcal S(x)
=\bigcup_{r\in\mathcal R}
\mathrm{SLIC}_r(x).
```

Each segment is converted to a model-compatible image patch and embedded at
layer `l`:

```math
z_s=\phi_l(\mathrm{Resize}(s)).
```

Multi-resolution segmentation expands the candidate family from textures to
larger object parts. It also creates overlapping and duplicate candidates.

## 2. Clustering

ACE groups segment embeddings using Euclidean geometry. For `K` clusters,

```math
\min_{\mu_1,\ldots,\mu_K}
\sum_s\min_{1\le k\le K}
\|z_s-\mu_k\|_2^2.
```

The provisional concept assignment is

```math
c(s)=\arg\min_k\|z_s-\mu_k\|_2^2.
```

Outliers far from a cluster center are removed to increase within-cluster
coherence. Small clusters are filtered because they may reflect rare noise
rather than recurrent class structure.

## 3. Discovery Is Representation-Conditional

ACE discovers repeated regions under three imposed geometries:

```math
\text{SLIC pixel geometry}
\to
\text{network activation geometry}
\to
\text{Euclidean K-means geometry}.
```

A pathology pattern split across noncontiguous regions may not survive SLIC. A
feature absent from layer `l` cannot be recovered by clustering. A nonconvex
activation manifold can be fragmented by K-means.

## 4. Hyperparameter Nonidentifiability

Changing resolution set, cluster count, minimum cluster size, or outlier
threshold changes the concept inventory:

```math
\mathcal C
=\mathcal A
(\mathcal D,l,\mathcal R,K,t_{\mathrm{out}},t_{\mathrm{size}}).
```

Automatic discovery removes the need to name concepts in advance; it does not
remove analyst choices. Stability across this parameter family is part of the
explanation evidence.


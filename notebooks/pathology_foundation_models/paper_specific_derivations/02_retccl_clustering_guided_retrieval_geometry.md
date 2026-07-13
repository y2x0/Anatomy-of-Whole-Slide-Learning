# RetCCL: Clustering-Guided Retrieval Geometry

## 1. Instance Embeddings and Clusters

Let patch embeddings be normalized as `u_i`. A clustering stage assigns each
embedding a cluster label `c_i` or soft assignment `q_i`.

```math
q_i\in\Delta^{K-1},
\qquad
c_i=\arg\max_k q_{ik}.
```

RetCCL uses clustering to define more informative positives for contrastive
learning and targets retrieval-quality pathology neighborhoods.

## 2. Cluster-Guided Positive Set

For anchor `i`, let `P(i)` contain samples selected from its cluster or related
cluster structure. A multi-positive objective is

```math
\ell_i
=-\frac1{|P(i)|}\sum_{p\in P(i)}
\log
\frac{\exp(u_i^{\top}u_p/\tau)}
{\sum_{a\ne i}\exp(u_i^{\top}u_a/\tau)}.
```

The learned geometry is jointly shaped by instance augmentation and cluster
membership.

## 3. Retrieval Readout

Given query `u`, the retrieved references are

```math
\mathcal N_K(u)
=\underset{A:|A|=K}{\mathrm{arg\,min}}
\sum_{r\in A}(1-u^{\top}u_r).
```

RetCCL's interpretive object is the neighborhood, not a classifier logit.

## 4. Circularity Risk

If clustering errors are used to define positives, the contrastive objective can
make those errors geometrically stable. Retrieval coherence must be evaluated
with independent pathology labels and across institutions.

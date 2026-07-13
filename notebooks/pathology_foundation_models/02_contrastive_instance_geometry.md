# Contrastive Instance Geometry

## 1. Positive Pair Objective

For two augmented views of an instance, let

```math
u_i=g(f_\phi(v_i(x))),
\qquad
v_i=g(f_\phi(v_i'(x))).
```

With normalized embeddings and temperature `tau`, an InfoNCE term is

```math
\ell_i
=-\log
\frac{\exp(\mathrm{sim}(u_i,v_i)/\tau)}
{\sum_{j=1}^{B}\exp(\mathrm{sim}(u_i,v_j)/\tau)}.
```

CTransPath and related pathology contrastive models instantiate this kind of
view-invariance pressure at patch scale.

## 2. RetCCL's Cluster Guidance

Retrieval-oriented contrastive learning can use cluster assignments to define
additional positives. Let `q_i` be a cluster assignment and `P(i)` its positive
index set:

```math
\ell_i^{\mathrm{multi}}
=-\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{\exp(\mathrm{sim}(u_i,u_p)/\tau)}
{\sum_{a\ne i}\exp(\mathrm{sim}(u_i,u_a)/\tau)}.
```

The representation retains cluster-consistent similarity, which can aid
retrieval but can also reinforce clustering errors.

## 3. Surviving Statistic

The objective shapes pairwise angles or distances. It does not directly ensure
that individual coordinates are interpretable. Downstream linear probing tests
class separation in the learned geometry; nearest-neighbor retrieval tests a
different functional.

## 4. Pathology Failure Modes

```text
positive augmentation changes morphology rather than nuisance;
negative patches share the same tissue state;
cluster labels encode scanner or cohort;
rare pathology is treated as a negative;
patch-level invariance destroys spatial information.
```


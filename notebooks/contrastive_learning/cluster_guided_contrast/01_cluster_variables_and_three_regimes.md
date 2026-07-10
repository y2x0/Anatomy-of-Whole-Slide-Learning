# Cluster Variables And Three Regimes

Cluster-guided learning introduces a latent assignment between representation
and supervision. The assignment can be global or minibatch-local, hard or
soft, and either a target itself or side information for another contrastive
loss.

Primary anchors:

- DeepCluster: https://arxiv.org/abs/1807.05520
- SwAV: https://arxiv.org/abs/2006.09882
- RetCCL: https://pubmed.ncbi.nlm.nih.gov/36270093/

## Representation Collection

Let:

```math
x_n
\sim
p_X,
\qquad
z_n
=
f_{\theta}(x_n)
\in
\mathbb{R}^{d}.
```

Stack the features:

```math
Z
=
\begin{bmatrix}
z_1^{\top}\\
\vdots\\
z_N^{\top}
\end{bmatrix}
\in
\mathbb{R}^{N\times d}.
```

Let:

```math
C
=
\begin{bmatrix}
c_1&
\cdots&
c_K
\end{bmatrix}
\in
\mathbb{R}^{d\times K}
```

contain cluster centroids or trainable prototypes.

## Hard Assignment

A hard assignment is:

```math
y_n
\in
\left\{
e_1,\ldots,e_K
\right\},
```

where `e_k` is the `k`-th standard basis vector. Equivalently:

```math
y_n
\in
\left\{
0,1
\right\}^{K},
\qquad
y_n^{\top}\mathbf{1}_K
=
1.
```

The selected centroid is:

```math
Cy_n.
```

DeepCluster uses this regime.

## Soft Assignment

A soft code is:

```math
q_n
\in
\Delta^{K-1},
```

where:

```math
\Delta^{K-1}
=
\left\{
q
\in
\mathbb{R}_{+}^{K}
:
q^{\top}\mathbf{1}_K
=
1
\right\}.
```

Its prototype barycenter is:

```math
Cq_n.
```

SwAV computes soft codes by an entropy-regularized balanced assignment
problem.

## Cluster Side Information

A cluster variable can alter a contrastive loss without becoming its
categorical target. Let:

```math
\kappa
\left(
z,z^{-}
\right)
\in
\left\{
1,\ldots,K
\right\}
```

describe the relation of an anchor or key to a memory cluster. A weight:

```math
\phi
\left(
z^{-};
\kappa
\right)
```

can modify a negative logit:

```math
\frac{
z^{\top}z^{-}
}{
\tau
}
\longrightarrow
\frac{
\phi
\left(
z^{-};
\kappa
\right)
z^{\top}z^{-}
}{
\tau
}.
```

RetCCL uses clusters this way in its weighted InfoNCE branch and uses
cross-branch centroids as targets in a second branch.

## Cluster Labels Are Permutation Invariant

Let `P` be any permutation matrix:

```math
P
\in
\left\{
0,1
\right\}^{K\times K}.
```

Transform:

```math
\widetilde C
=
CP^{\top},
\qquad
\widetilde y_n
=
Py_n.
```

Then:

```math
\widetilde C
\widetilde y_n
=
CP^{\top}Py_n
=
Cy_n.
```

The partition is identified only up to relabeling. A statement about "cluster
3" has no cross-run meaning without alignment.

## Partition Versus Biological Class

Let `U_n` be latent morphology and `D_n` acquisition nuisance.
The learned partition can be:

```math
C_n
=
\kappa
\left(
U_n,D_n
\right).
```

A pure tissue partition would satisfy:

```math
C_n
\perp
D_n
\mid
U_n.
```

No unsupervised clustering objective guarantees this conditional
independence.

## Three Optimization Regimes

### Global Hard Alternation

```math
Z_t
\xrightarrow{
\mathrm{k\mbox{-}means}
}
Y_t
\xrightarrow{
\mathrm{cross\mbox{-}entropy}
}
\theta_{t+1}.
```

The assignment uses the full feature collection and remains fixed during the
subsequent discriminative update.

### Online Soft Balanced Assignment

```math
Z_{\mathcal{B},t}
\xrightarrow{
\mathrm{Sinkhorn}
}
Q_{\mathcal{B},t}
\xrightarrow{
\mathrm{swapped\ prediction}
}
\theta_{t+1},C_{t+1}.
```

Codes are computed from the current minibatch and prototypes.

### Cluster-Guided Contrast

```math
Z_t
\xrightarrow{
\mathrm{k\mbox{-}means}
}
\left(
\text{subqueues},
\text{centroids}
\right)
\xrightarrow{
\mathrm{weighted/group\ InfoNCE}
}
\theta_{t+1}.
```

Clusters modify candidate importance and define cross-instance group targets.

## Positive Relation Graphs

For DeepCluster:

```math
(i,j)
\in
E_{+}
\quad
\Longleftrightarrow
\quad
y_i
=
y_j
```

only indirectly through shared pseudo-label prediction.

For SwAV, two views are connected through swapped codes:

```math
z_i^{(1)}
\longrightarrow
q_i^{(2)},
\qquad
z_i^{(2)}
\longrightarrow
q_i^{(1)}.
```

For RetCCL, a group edge can connect an instance to the nearest centroid in the
other branch:

```math
z_i^{(p)}
\longrightarrow
c_{\star}^{(q)}.
```

These are not the same graph.

## C/R/G/S Template

```text
\mathcal{G}:
    the support on which assignments are computed and any balance constraints

\mathcal{C}:
    encoder plus global clustering, online prototypes, or subqueue construction

\mathcal{R}:
    hard label, soft code, centroid relation, or cluster-weighted logit

\mathcal{S}:
    pseudo-label prediction, swapped-code prediction, or cluster-guided InfoNCE
```

## Surviving Object

Cluster-guided methods preserve a partition or prototype coordinate system in
the learned feature geometry. What that partition means depends on the metric,
balance rule, temporal update, and data mixture that created it.

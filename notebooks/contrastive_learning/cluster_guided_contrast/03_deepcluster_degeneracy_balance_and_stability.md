# DeepCluster Degeneracy, Balance, And Stability

Jointly learning features and their own pseudo-labels admits trivial solutions.
DeepCluster uses algorithmic repairs rather than a single explicit
anti-collapse regularizer.

Primary anchor:

- DeepCluster: https://arxiv.org/abs/1807.05520

## Collapsed Feature Solution

If:

```math
f_{\theta}(x_n)
=
0,
\qquad
\forall n,
```

then the one-cluster distortion is:

```math
\sum_{n=1}^{N}
\left\lVert
f_{\theta}(x_n)-0
\right\rVert_2^2
=
0.
```

Without constraints forcing multiple occupied groups, feature collapse is a
global optimum of a joint representation-plus-k-means distortion objective.

DeepCluster does not optimize that joint objective end to end, but its
alternating loop is still exposed to empty clusters and heavily imbalanced
assignments.

## Empty-Cluster Repair

When cluster `k` is empty:

```math
n_k
=
0,
```

its centroid mean is undefined. The paper uses a scalable k-means repair:

1. sample a nonempty cluster;
2. copy its centroid with a small random perturbation;
3. split the source cluster's assigned points across the two centroids.

Schematically:

```math
c_{\mathrm{empty}}
\leftarrow
c_r
+
\varepsilon,
```

followed by local reassignment.

This prevents an empty output index. It does not prove that every cluster
corresponds to a stable semantic mode.

## Trivial Parametrization Under Imbalance

Let:

```math
n_k
=
\sum_n y_{nk}.
```

If most samples occupy a few clusters, empirical cross-entropy:

```math
\frac{1}{N}
\sum_{n=1}^{N}
\ell_n
```

is dominated by those assignments. A classifier can reduce loss by predicting
the majority pseudo-label for many inputs, producing weak features for small
clusters.

## Uniform-Over-Cluster Sampling

DeepCluster samples pseudo-labels uniformly. The induced sampling rule is:

```math
K
\sim
\mathrm{Uniform}
\left\{
1,\ldots,K
\right\},
```

```math
N
\sim
\mathrm{Uniform}
\left\{
n
:
y_{nk}=1
\right\}
\mid
K=k.
```

The expected risk becomes:

```math
\mathcal{J}_{\mathrm{balanced}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{1}{n_k}
\sum_{
n:
y_{nk}=1
}
\ell_n.
```

Equivalently, each observation receives weight proportional to:

```math
w_n
\propto
\frac{1}{
n_{c(n)}
}.
```

This is the inverse-cluster-size equivalence stated in the paper.

## Prior Shift

The empirical pseudo-label prior is:

```math
\widehat\pi_k
=
\frac{n_k}{N}.
```

Uniform cluster sampling replaces it with:

```math
\widetilde\pi_k
=
\frac{1}{K}.
```

The training risk therefore targets a balanced partition prior, not the
observed data prevalence.

For pathology, common tissue can occupy most patches:

```math
\widehat\pi_{\mathrm{stroma}}
\gg
\widehat\pi_{\mathrm{rare}}.
```

Balancing can protect rare morphology from being ignored, but it can also
amplify tiny clusters caused by artifacts, pen marks, blur, or scanner noise.

## Weight Variance

Under uniform observation sampling, the inverse-size weight has second moment:

```math
\mathbb{E}
\left[
w_N^2
\right]
\propto
\frac{1}{N}
\sum_{k=1}^{K}
n_k
\frac{1}{n_k^2}
=
\frac{1}{N}
\sum_{k=1}^{K}
\frac{1}{n_k}.
```

Very small clusters increase gradient-weight variance. Balance changes both
bias and optimization noise.

## Assignment Boundary

For centroids `c_a` and `c_b`, define the squared-distance margin:

```math
\Delta_{a,b}(z)
=
\left\lVert
z-c_b
\right\rVert_2^2
-
\left\lVert
z-c_a
\right\rVert_2^2.
```

Point `z` prefers cluster `a` over `b` when:

```math
\Delta_{a,b}(z)
>
0.
```

Expanding:

```math
\Delta_{a,b}(z)
=
2z^{\top}
\left(
c_a-c_b
\right)
+
\lVert c_b\rVert_2^2
-
\lVert c_a\rVert_2^2.
```

Under feature perturbation `z\mapsto z+\delta`:

```math
\Delta_{a,b}(z+\delta)
-
\Delta_{a,b}(z)
=
2\delta^{\top}
\left(
c_a-c_b
\right).
```

Hence:

```math
\left|
\Delta_{a,b}(z+\delta)
-
\Delta_{a,b}(z)
\right|
\le
2
\lVert\delta\rVert_2
\lVert c_a-c_b\rVert_2.
```

Assignment against `b` is guaranteed stable when:

```math
\Delta_{a,b}(z)
>
2
\lVert\delta\rVert_2
\lVert c_a-c_b\rVert_2.
```

The smallest pairwise margin controls local pseudo-label stability.

## Centroid Drift

Across epochs:

```math
c_{k,t}
\longrightarrow
c_{\pi_t(k),t+1},
```

where `\pi_t` is an unknown label alignment permutation. Direct cluster
index comparison is meaningless without matching centroids or assignments.

DeepCluster uses normalized mutual information between successive assignments:

```math
\mathrm{NMI}
\left(
Y_t,Y_{t+1}
\right)
=
\frac{
I
\left(
Y_t;Y_{t+1}
\right)
}{
\sqrt{
H(Y_t)H(Y_{t+1})
}
}.
```

This statistic is invariant to cluster-label permutation.

The paper observes increasing NMI over training but saturation below one,
meaning a nontrivial fraction of observations continues to change assignment.

## Hard-Target Discontinuity

The prediction target is:

```math
y_n
=
e_{
\arg\min_k
\lVert z_n-c_k\rVert_2^2
}.
```

An arbitrarily small perturbation at a Voronoi boundary can change:

```math
y_n
\quad
\text{from}
\quad
e_a
\quad
\text{to}
\quad
e_b.
```

Cross-entropy then changes the target class discontinuously even though the
feature changed continuously.

## Pathology Counterexample

Assume a large normal-tissue mode and two small modes:

```math
\pi
=
\left(
0.96,
0.03,
0.01
\right),
```

where the last mode is a scanner artifact rather than rare disease.

Uniform cluster sampling changes the effective mode masses to:

```math
\widetilde\pi
=
\left(
\frac{1}{3},
\frac{1}{3},
\frac{1}{3}
\right).
```

The artifact receives thirty-three times its empirical mass, while normal
tissue receives roughly one third of its balanced risk share. Balance is
anti-collapse supervision, not a guarantee of biological importance.

## Failure Matrix

| Failure | Repair in DeepCluster | Residual risk |
|---|---|---|
| empty cluster | split a nonempty cluster with perturbed centroid | artificial unstable groups |
| dominant pseudo-label | uniform-over-cluster sampling | prior distortion and noisy rare groups |
| feature collapse | alternating discrimination plus occupied clusters | no global theorem for biological non-collapse |
| assignment switching | repeated re-clustering | hard discontinuous targets |
| cluster-label permutation | permutation-invariant comparison such as NMI | no stable semantic cluster names |
| metric shortcut | PCA, whitening, normalization | transformed geometry can still reflect nuisance |

## C/R/G/S Placement

```text
\mathcal{G}:
    hard Voronoi cells plus an imposed uniform cluster sampler

\mathcal{C}:
    k-means reassignment and discriminative feature update

\mathcal{R}:
    one-hot pseudo-label with inverse-size effective weighting

\mathcal{S}:
    predict current cluster under data augmentation
```

## Failure Principle

DeepCluster prevents numerical cluster death and representation domination by
changing assignment and sampling dynamics. Those repairs define a balanced
latent task; they do not reveal the natural prevalence or semantics of tissue
states.

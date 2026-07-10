# DeepCluster Alternating Map

DeepCluster alternates global feature clustering with discriminative prediction
of the resulting assignments. The centroids generate pseudo-labels but are not
used by the classifier during the prediction phase.

Primary anchor:

- Caron et al. "Deep Clustering for Unsupervised Learning of Visual Features."
  ECCV 2018. https://arxiv.org/abs/1807.05520

## Supervised Reference Objective

For labeled observations:

```math
\left\{
\left(
x_n,y_n
\right)
\right\}_{n=1}^{N},
```

let the encoder and linear classifier be:

```math
z_n
=
f_{\theta}(x_n),
```

```math
\ell_n
=
W^{\top}z_n
\in
\mathbb{R}^{K}.
```

The multinomial logistic objective is:

```math
\mathcal{J}_{\mathrm{CE}}
\left(
\theta,W;Y
\right)
=
\frac{1}{N}
\sum_{n=1}^{N}
\ell_{\mathrm{CE}}
\left(
W^{\top}f_{\theta}(x_n),
y_n
\right).
```

DeepCluster replaces observed labels by assignments generated from the
features.

## K-Means Assignment Step

Let:

```math
C
\in
\mathbb{R}^{d\times K}
```

be the centroid matrix and:

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

The paper solves:

```math
\min_{
C,
\left\{
y_n
\right\}_{n=1}^{N}
}
\frac{1}{N}
\sum_{n=1}^{N}
\left\lVert
f_{\theta}(x_n)
-
Cy_n
\right\rVert_2^2.
```

For fixed centroids:

```math
y_n^{\star}
=
e_{
\arg\min\limits_{k}
\left\lVert
f_{\theta}(x_n)
-
c_k
\right\rVert_2^2
}.
```

For fixed nonempty assignments:

```math
c_k^{\star}
=
\frac{
1
}{
n_k
}
\sum_{
n:
y_{nk}=1
}
f_{\theta}(x_n),
```

where:

```math
n_k
=
\sum_{n=1}^{N}
y_{nk}.
```

## Pseudo-Label Prediction Step

After k-means, the assignments:

```math
Y_t
=
\left\{
y_{n,t}^{\star}
\right\}_{n=1}^{N}
```

are treated as fixed labels. The network update approximately solves:

```math
\left(
\theta_{t+1},W_{t+1}
\right)
\approx
\arg\min_{\theta,W}
\mathcal{J}_{\mathrm{CE}}
\left(
\theta,W;Y_t
\right).
```

The centroid matrix `C_t` is not used in this discriminative loss. Its
role ends after assignment generation.

## Full Iteration

At iteration `t`:

```math
Z_t
=
\left\{
f_{\theta_t}(x_n)
\right\}_{n=1}^{N},
```

```math
\left(
C_t,Y_t
\right)
\in
\arg\min_{C,Y}
\mathcal{J}_{\mathrm{kmeans}}
\left(
C,Y;Z_t
\right),
```

```math
\left(
\theta_{t+1},W_{t+1}
\right)
\approx
\arg\min_{\theta,W}
\mathcal{J}_{\mathrm{CE}}
\left(
\theta,W;Y_t
\right).
```

The next representation collection changes:

```math
Z_{t+1}
\ne
Z_t,
```

so assignments are recomputed.

## Not Exact Coordinate Descent On One Printed Objective

The clustering step minimizes feature distortion:

```math
\mathcal{J}_{\mathrm{kmeans}}
=
\frac{1}{N}
\sum_n
\left\lVert
f_{\theta_t}(x_n)
-
Cy_n
\right\rVert_2^2.
```

The network step minimizes classification loss:

```math
\mathcal{J}_{\mathrm{CE}}
=
\frac{1}{N}
\sum_n
\ell_{\mathrm{CE}}
\left(
W^{\top}f_{\theta}(x_n),
y_n
\right).
```

The second step does not minimize the first objective with respect to
`\theta`, and the first step does not minimize the second with respect to
`Y`. Therefore the printed algorithm resembles block alternation but is
not exact coordinate descent on either displayed scalar objective.

It is also not standard expectation-maximization:

```math
\text{k-means assignment}
\ne
\text{posterior expectation under a stated probabilistic model},
```

and the discriminative update is not a complete-data likelihood maximization.

## Metric Used By The Assignment Step

The paper applies to extracted features:

```text
1. PCA reduction to 256 dimensions
2. whitening
3. L2 normalization
4. k-means
```

Let the combined linear preprocessing be `A`. Before normalization:

```math
\widetilde z_n
=
A
\left(
z_n-\overline z
\right).
```

After unit normalization:

```math
u_n
=
\frac{
\widetilde z_n
}{
\lVert
\widetilde z_n
\rVert_2
}.
```

K-means therefore minimizes distances in the transformed geometry:

```math
\left\lVert
u_n-\mu_k
\right\rVert_2^2,
```

not raw encoder-space Euclidean distance.

For normalized vectors:

```math
\left\lVert
u_n-\mu_k
\right\rVert_2^2
=
\lVert u_n\rVert_2^2
+
\lVert\mu_k\rVert_2^2
-
2u_n^{\top}\mu_k.
```

If centroids were also normalized, nearest-centroid assignment would be
equivalent to maximum cosine similarity. Standard k-means centroids need not
have unit norm, so centroid norm remains part of the decision.

## Assignment View Versus Training View

The paper clusters features from central crops, then trains the convnet using
random horizontal flips and random crops of varying sizes and aspect ratios.

Let:

```math
y_n
=
\kappa
\left(
f_{\theta}
\left(
x_n^{\mathrm{center}}
\right)
\right).
```

The training objective asks augmented views to predict that assignment:

```math
\ell_{\mathrm{CE}}
\left(
W^{\top}
f_{\theta}
\left(
T(x_n)
\right),
y_n
\right).
```

This indirectly enforces augmentation invariance at the cluster-label level.

## Computational Cost

Each reassignment cycle requires:

```math
\Theta
\left(
N
\cdot
\mathrm{encoder\ forward\ cost}
\right)
```

to extract all features.

Lloyd-style k-means with `I` iterations costs approximately:

```math
\Theta
\left(
INKd
\right)
```

for dense distance evaluation.

The paper reports that clustering includes the full forward pass and consumes
a substantial fraction of training time. Assignments were updated every epoch
in the ImageNet setup.

## C/R/G/S Placement

```text
\mathcal{G}:
    a full-dataset Euclidean Voronoi partition after PCA, whitening, and
    normalization

\mathcal{C}:
    global k-means followed by an encoder and K-way classifier

\mathcal{R}:
    hard one-hot pseudo-label; centroids are discarded before prediction

\mathcal{S}:
    cross-entropy prediction of the current pseudo-label under augmented views
```

## Surviving Statistic

DeepCluster preserves predictability of a repeatedly recomputed hard partition.
It does not directly preserve centroid distances during the network update.

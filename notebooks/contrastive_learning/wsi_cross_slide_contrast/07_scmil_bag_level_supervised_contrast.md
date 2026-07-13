# SC-MIL Bag-Level Supervised Contrast

Primary anchor:

- Juyal et al. "SC-MIL: Supervised Contrastive Multiple Instance Learning for
  Imbalanced Classification in Digital Pathology." WACV 2024.
  https://openaccess.thecvf.com/content/WACV2024/html/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.html

## Bag Embedding Before Contrast

```math
b_i
=
b
\left(
\mathcal{X}_i
\right)
\in
\mathbb{R}^{d},
```

```math
z_i
=
\frac{g(b_i)}{\left\|g(b_i)\right\|_2}.
```

SC-MIL applies supervised contrast after attention aggregation, avoiding
direct propagation of slide labels to patches. With this normalization,
`z_i^{\top}z_j` is cosine similarity. Omitting the normalization changes the
critic to a norm-sensitive dot product and is not the forward map used in the
paper's pseudocode.

## Exact Loss

```math
\mathcal{P}_i^{+}
=
\left\{
j\ne i:Y_j=Y_i
\right\},
```

```math
\mathcal{B}_i
=
\left\{
k:k\ne i
\right\}.
```

```math
\mathcal{L}_{\mathrm{SCL}}
=
\sum_i
-\frac{1}{|\mathcal{P}_i^{+}|}
\sum_{j\in\mathcal{P}_i^{+}}
\log
\frac{
\exp
\left(
z_i^{\top}z_j/\tau
\right)
}{
\sum_{k\in\mathcal{B}_i}
\exp
\left(
z_i^{\top}z_k/\tau
\right)
}.
```

## Readout Bottleneck

```math
b(\mathcal{X}_a)
=
b(\mathcal{X}_b)
```

implies that contrast cannot distinguish the two patch distributions. Every
inductive bias of the MIL aggregator enters before cross-slide geometry.

## Bag Gradient

For candidate softmax probabilities `q_ik`:

```math
\nabla_{z_i}\mathcal{L}_i
=
\frac{1}{\tau}
\left[
\sum_{k\in\mathcal{B}_i}
q_{ik}z_k
-
\frac{1}{|\mathcal{P}_i^{+}|}
\sum_{j\in\mathcal{P}_i^{+}}
z_j
\right].
```

Patch gradients pass through the readout:

```math
\nabla_{h_{ir}}
\mathcal{L}_i
=
\left(
\frac{\partial b_i}
{\partial h_{ir}}
\right)^{\top}
\left(
\frac{\partial z_i}
{\partial b_i}
\right)^{\top}
\nabla_{z_i}
\mathcal{L}_i.
```

Low-attention patches can receive negligible cross-slide supervision.

## Empty Positives

If:

```math
|\mathcal{P}_i^{+}|=0,
```

the anchor loss is undefined. Batch sampling must provide class partners or
omit unsupported anchors.

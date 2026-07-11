# SC-MIL Bag-Level Supervised Contrast

Primary anchor:

- Juyal et al. "SC-MIL: Supervised Contrastive Multiple Instance Learning for
  Imbalanced Classification in Digital Pathology." WACV 2024.
  https://openaccess.thecvf.com/content/WACV2024/html/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.html

## Bag Embedding Before Contrast

For one input bag, an attention MIL model produces:

```math
b_i
=
b
\left(
\mathcal{X}_i
\right)
\in
\mathbb{R}^{d}.
```

A nonlinear projection head gives:

```math
z_i
=
g(b_i).
```

SC-MIL applies supervised contrast after aggregation, avoiding direct
propagation of slide labels to patches.

## Exact Loss

Define same-class positives:

```math
\mathcal{P}_i^{+}
=
\left\{
j\ne i:
Y_j=Y_i
\right\},
```

and all other bags in the batch:

```math
\mathcal{B}_i
=
\left\{
k:k\ne i
\right\}.
```

The paper's bag-level supervised contrastive loss is:

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

If two bags satisfy:

```math
b(\mathcal{X}_a)
=
b(\mathcal{X}_b),
```

the contrastive objective cannot distinguish their patch distributions. Every
inductive bias of the MIL aggregator enters before contrast.

## Bag-Level Gradient

Let candidate softmax probabilities be `q_ik`. Then:

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

The gradient reaches patches only through the bag readout Jacobian:

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

Low-attention patches may receive negligible cross-slide supervision.

## Empty-Positive Batch

If:

```math
|\mathcal{P}_i^{+}|=0,
```

the anchor loss is undefined. Batch sampling must place at least two examples
of each represented class or omit unsupported anchors.

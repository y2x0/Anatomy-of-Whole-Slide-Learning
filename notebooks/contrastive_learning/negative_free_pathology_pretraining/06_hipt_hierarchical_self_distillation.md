# HIPT Hierarchical Self-Distillation

Primary anchor:

- Chen et al. "Scaling Vision Transformers to Gigapixel Images via
  Hierarchical Self-Supervised Learning." CVPR 2022.
  https://arxiv.org/abs/2206.02647

## Three Spatial Scales

HIPT treats a WSI as a hierarchy:

```math
16\times16
\longrightarrow
256\times256
\longrightarrow
4096\times4096
\longrightarrow
\mathrm{WSI}.
```

A 256-pixel patch contains 256 non-overlapping 16-pixel tokens. A 4096-pixel
region contains 256 non-overlapping 256-pixel patches.

## First Encoder

For patch `x_{k,j}^{256}` in region `k`:

```math
z_{k,j}^{256}
=
\mathrm{ViT}_{256\text{-}16}
\left(
x_{k,j}^{256}
\right)_{\mathrm{CLS}}
\in
\mathbb{R}^{384}.
```

This encoder is pretrained with DINO on 256-pixel pathology patches.

## Second Encoder

For 4096-pixel region `k`, stack 256 patch-level class tokens:

```math
Z_k^{256}
=
\left[
z_{k,1}^{256};
\ldots;
z_{k,256}^{256}
\right].
```

The region representation is:

```math
z_k^{4096}
=
\mathrm{ViT}_{4096\text{-}256}
\left(
Z_k^{256}
\right)_{\mathrm{CLS}}
\in
\mathbb{R}^{192}.
```

During second-stage pretraining, the first encoder is fixed and serves as the
tokenization layer.

## Slide-Level Composition

For `M` tissue regions:

```math
Z^{4096}
=
\left[
z_1^{4096};
\ldots;
z_M^{4096}
\right],
```

```math
z^{\mathrm{WSI}}
=
\mathrm{ViT}_{\mathrm{WSI}\text{-}4096}
\left(
Z^{4096}
\right)_{\mathrm{CLS}}.
```

The number of tokens contracts by roughly a factor of 256 at each aggregation
stage.

## Frozen-Tokenizer Information Bound

The second stage sees only:

```math
z_{k,j}^{256}
=
f_{256}
\left(
x_{k,j}^{256}
\right).
```

If:

```math
f_{256}(x_a)
=
f_{256}(x_b),
```

no region encoder can recover the distinction between `x_a` and `x_b`.
Hierarchical context cannot restore details removed by the lower-scale class
token.

## Stagewise Versus End-To-End Gradients

With frozen first stage:

```math
\frac{\partial\mathcal{L}_{4096}}
{\partial\theta_{256}}
=
0.
```

The region objective cannot reshape cell-level features for a region-level
task. This stabilizes training and makes gigapixel computation feasible, but
creates a hard scale boundary.

## Hierarchical Positive Relation

DINO views at the first stage preserve patch identity under crops. At the
second stage they preserve region identity under transformations of patch-token
grids. These are different semantic claims:

```math
\text{same patch}
\ne
\text{same tissue region}.
```

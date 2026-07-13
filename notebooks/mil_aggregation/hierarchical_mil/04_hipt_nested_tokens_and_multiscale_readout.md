# HIPT Nested Tokens And Multiscale Readout

Source:

```text
Chen et al., Scaling Vision Transformers to Gigapixel Images via Hierarchical
Self-Supervised Learning, CVPR 2022.
https://arxiv.org/abs/2206.02647
```

## 1. HIPT's slide object

HIPT does not flatten a WSI directly into one sequence of patch embeddings. It
models the slide as a nested collection of fixed-scale token sequences:

```math
\mathcal X_i
=
\left\{
\left\{
\left\{
x_{i r p c}
\right\}_{c=1}^{256}
\right\}_{p=1}^{256}
\right\}_{r=1}^{R_i}.
```

Here c indexes cell-scale tokens inside a 256 by 256 image, p indexes
256 by 256 images inside a 4096 by 4096 region, and r indexes regions in the
slide-level tissue sequence. The exact number R_i varies with the tissue area.

The nesting is a computational partition. It is not a claim that every
4096 by 4096 window is a biological region with a ground-truth boundary.

## 2. Three aggregation stages

The first stage applies a ViT with 16 by 16 input patches to each
256 by 256 image. Writing the [CLS] output as h_{irp}, the cell-scale
aggregation is

```math
h_{irp}
=
\mathrm{ViT}_{16}^{(1)}
\left(
x_{irpc}
\right)_{c=1}^{256}
\in\mathbb R^{d_1}.
```

The second stage treats the 256 [CLS] outputs from a 4096 by 4096 region as a
sequence. Its output is a region token:

```math
h_{ir}
=
\mathrm{ViT}_{256}^{(2)}
\left(
h_{irp}
\right)_{p=1}^{256}
\in\mathbb R^{d_2}.
```

The third stage processes the variable-length sequence of region tokens:

```math
z_i
=
\mathrm{ViT}_{4096}^{(3)}
\left(
h_{ir}
\right)_{r=1}^{R_i}
\in\mathbb R^{d_3}.
```

In the released hierarchy, the [CLS] token from one stage becomes an input
token for the next stage. The sequence length therefore contracts by a factor
of 256 at each aggregation boundary rather than requiring attention over all
cell-scale tokens at once.

## 3. Transformer context at each scale

For a token matrix X with d-dimensional channels, one self-attention head is

```math
\mathrm{Attn}(X)
=
\mathrm{softmax}
\left(
\frac{(XW_Q)(XW_K)^{\mathsf T}}{\sqrt{d_k}}
\right)
XW_V.
```

A transformer block applies residual attention and a pointwise feed-forward
map:

```math
X^{+}
=
X+\mathrm{Attn}(X),
\qquad
X^{\mathrm{out}}
=
X^{+}
+
\mathrm{FFN}(X^{+}).
```

Within a stage, the attention support is the token sequence for one parent
window, except at the slide stage where the support is the sequence of region
tokens. The hierarchy changes the support on which pairwise interactions occur.

If stage ell has n_ell tokens per parent and d_ell channels, its dense
self-attention has pairwise score cost

```math
O(n_\ell^2d_\ell).
```

HIPT keeps n_ell approximately fixed at 256 for the first two aggregation
stages, then pays the variable slide-level region cost R_i^2d_3.

## 4. Hierarchical DINO pretraining

HIPT pretrains the lower aggregation stages with a DINO-style teacher-student
objective. For a teacher view v_t and student view v_s, let

```math
p_t
=
\mathrm{softmax}
\left(
\frac{g_{\xi}(h_t)-c}{\tau_t}
\right),
\qquad
p_s
=
\mathrm{softmax}
\left(
\frac{g_{\theta}(h_s)}{\tau_s}
\right).
```

The cross-entropy over teacher output coordinates is

```math
\mathcal L_{\mathrm{DINO}}
=
-\sum_{k=1}^{K}
p_{t,k}\log p_{s,k}.
```

The teacher parameters are updated by an exponential moving average:

```math
\xi
\leftarrow
\mu\xi+(1-\mu)\theta.
```

The first pretraining stage learns cell-scale aggregation from views of
256 by 256 patches. With the first stage fixed, its [CLS] tokens are used as
the embedding inputs for the second DINO stage on 4096 by 4096 regions.

This distinction matters: the two-stage hierarchical pretraining objective does
not directly apply a DINO loss to the complete WSI. It pretrains local and
regional aggregation layers, after which the slide-level layer is used for
downstream representation learning.

## 5. Downstream WSI prediction

A downstream slide representation can be the output z_i of the third stage,
or a task head applied to it:

```math
\widehat y_i
=
\mathcal H_{\omega}(z_i).
```

For a survival head with scalar risk score eta_i,

```math
\eta_i
=
w_{\omega}^{\mathsf T}z_i+b_{\omega}.
```

The hierarchy therefore places contextual processing before the task-specific
readout. A comparison against ABMIL using the same frozen cell encoder is not
only a comparison of readouts: HIPT also pretrains aggregation layers that
change the representation presented to the head.

## 6. What survives

At each stage, the [CLS] output is a learned summary of a parent window. The
within-window token arrangement can affect that summary through positional
embeddings and self-attention, but it is not preserved as a set of separate
tokens after the [CLS] bottleneck.

Thus the surviving object is

```math
\{x_{irpc}\}
\longmapsto
\{h_{irp}\}
\longmapsto
\{h_{ir}\}
\longmapsto
z_i,
```

with a distinct learned statistic at each scale. The architecture can express
coarse morphology and long-range region interactions, but it cannot recover
cell-level distinctions that are removed by a [CLS] projection.

## 7. C/R/G/S placement

```text
C: self-attention and feed-forward context within each nested token sequence.

R: [CLS] extraction at cell, patch, and region boundaries.

G: fixed non-overlapping spatial nesting and optional within-stage positional
   embeddings; the graph is implicit rather than an explicit adjacency matrix.

S: DINO cross-view self-supervision for lower stages, then slide-level labels
   or survival outcomes for downstream fine-tuning.

HIPT is hierarchical MIL when its nested token encoder is used to convert a
variable-size WSI into a slide representation for weakly supervised prediction.
It is also a foundation-style hierarchical representation learner; those are
different axes of the same model.
```

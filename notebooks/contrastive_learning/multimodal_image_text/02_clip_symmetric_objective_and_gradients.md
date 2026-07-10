# CLIP Symmetric Objective And Gradients

Primary anchor:

- Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision." ICML 2021. https://arxiv.org/abs/2103.00020

## Similarity Matrix

For a batch of `B` paired observations, stack normalized embeddings:

```math
U
=
\begin{bmatrix}
u_1^{\top}\\
\vdots\\
u_B^{\top}
\end{bmatrix}
\in
\mathbb{R}^{B\times d},
\qquad
V
=
\begin{bmatrix}
v_1^{\top}\\
\vdots\\
v_B^{\top}
\end{bmatrix}
\in
\mathbb{R}^{B\times d}.
```

CLIP forms:

```math
S
=
\gamma UV^{\top},
\qquad
S_{ij}
=
\gamma u_i^{\top}v_j,
```

where the positive logit scale is learned. Writing `\gamma=\exp(a)` ensures:

```math
\gamma>0.
```

## Two Conditional Classifiers

Image-to-text probabilities normalize rows:

```math
p^{I\rightarrow T}_{ij}
=
\frac{\exp(S_{ij})}
{\sum_{k=1}^{B}\exp(S_{ik})}.
```

Text-to-image probabilities normalize columns:

```math
p^{T\rightarrow I}_{ij}
=
\frac{\exp(S_{ij})}
{\sum_{k=1}^{B}\exp(S_{kj})}.
```

The symmetric loss is:

```math
\mathcal{L}_{\mathrm{CLIP}}
=
-\frac{1}{2B}
\sum_{i=1}^{B}
\log p^{I\rightarrow T}_{ii}
-\frac{1}{2B}
\sum_{i=1}^{B}
\log p^{T\rightarrow I}_{ii}.
```

This is two candidate-identification tasks sharing one score matrix. It is not
equivalent to applying one softmax to all `B^2` entries.

## Exact Matrix Gradient

Let `I_B` denote the identity matrix. Differentiating with respect to `S`:

```math
\frac{\partial\mathcal{L}_{\mathrm{CLIP}}}
{\partial S}
=
\frac{1}{2B}
\left(
P^{I\rightarrow T}
+
P^{T\rightarrow I}
-
2I_B
\right),
```

where the second probability matrix is understood entrywise under column
normalization. Thus:

```math
\frac{\partial\mathcal{L}}
{\partial S_{ij}}
=
\frac{1}{2B}
\left(
p^{I\rightarrow T}_{ij}
+
p^{T\rightarrow I}_{ij}
-
2\mathbb{1}\{i=j\}
\right).
```

The embedding gradients are:

```math
\frac{\partial\mathcal{L}}
{\partial U}
=
\gamma
\frac{\partial\mathcal{L}}
{\partial S}
V,
\qquad
\frac{\partial\mathcal{L}}
{\partial V}
=
\gamma
\left(
\frac{\partial\mathcal{L}}
{\partial S}
\right)^{\top}
U.
```

For one image embedding before accounting for normalization:

```math
\frac{\partial\mathcal{L}}
{\partial u_i}
=
\frac{\gamma}{2B}
\left[
\sum_{j=1}^{B}
\left(
p^{I\rightarrow T}_{ij}
-
\mathbb{1}\{i=j\}
\right)v_j
+
\sum_{j=1}^{B}
\left(
p^{T\rightarrow I}_{ij}
-
\mathbb{1}\{i=j\}
\right)v_j
\right].
```

The row term asks whether image `i` retrieves its text. The column term asks
whether text candidates retrieve image `i`.

## Normalization Jacobian

If:

```math
u
=
\frac{z}{\|z\|_2},
```

then:

```math
\frac{\partial u}{\partial z}
=
\frac{1}{\|z\|_2}
\left(
I-uu^{\top}
\right).
```

Therefore the gradient reaching `z` is tangent to the sphere:

```math
\nabla_z\mathcal{L}
=
\frac{1}{\|z\|_2}
\left(
I-uu^{\top}
\right)
\nabla_u\mathcal{L}.
```

CLIP learns directions, while radial feature scale is discarded at the
comparison interface.

## Learned Scale Gradient

For `\gamma=\exp(a)`:

```math
\frac{\partial\mathcal{L}}
{\partial a}
=
\gamma
\sum_{i=1}^{B}
\sum_{j=1}^{B}
\frac{\partial\mathcal{L}}
{\partial S_{ij}}
u_i^{\top}v_j.
```

A larger scale sharpens both conditional distributions and concentrates
gradient on hard alternatives. It cannot repair an incorrect diagonal
relation; it makes that relation more decisive.

## Duplicate-Caption Contradiction

Suppose `t_1=t_2`, so a deterministic text encoder gives `v_1=v_2`. For image
`u_1`:

```math
S_{11}=S_{12}.
```

Hence:

```math
p^{I\rightarrow T}_{11}
\le
\frac{1}{2}.
```

The image-to-text loss cannot approach zero even if both captions are
semantically correct. The one-positive objective is structurally incompatible
with duplicate or interchangeable pathology descriptions.

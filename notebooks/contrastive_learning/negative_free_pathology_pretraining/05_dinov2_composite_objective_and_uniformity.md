# DINOv2 Composite Objective And Uniformity

Primary anchor:

- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision."
  TMLR 2024. https://arxiv.org/abs/2304.07193

## Image-Level Distillation

For class-token distributions from two views:

```math
\mathcal{L}_{\mathrm{DINO}}
=
-\sum_{k=1}^{K_g}
p_{t,k}^{\mathrm{CLS}}
\log p_{s,k}^{\mathrm{CLS}}.
```

## Patch-Level Distillation

For masked patch index set `M`:

```math
\mathcal{L}_{\mathrm{iBOT}}
=
-\sum_{i\in M}
\sum_{k=1}^{K_p}
p_{t,i,k}^{\mathrm{patch}}
\log p_{s,i,k}^{\mathrm{patch}}.
```

DINOv2 uses separate projection heads for image-level and patch-level losses:

```math
h_{\mathrm{DINO}}
\ne
h_{\mathrm{iBOT}}.
```

This prevents the global and local categorical spaces from being forced to
share one output parameterization.

## Sinkhorn-Knopp Teacher Assignment

Teacher prototype assignments are balanced across a batch through iterative
row and column normalization. Abstractly, for positive score matrix `Q`:

```math
Q^{(0)}
=
\exp
\left(
S/\varepsilon
\right),
```

then iterations enforce approximate marginals:

```math
Q\mathbf{1}
\approx
\frac{1}{B}
\mathbf{1},
\qquad
Q^{\top}\mathbf{1}
\approx
\frac{1}{K}
\mathbf{1}.
```

This is an assignment-balancing mechanism, not explicit instance repulsion.

## KoLeo Regularizer

For normalized batch features `x_1,\ldots,x_B`, define nearest-neighbor
distance:

```math
d_i
=
\min_{j\ne i}
\left\|
x_i-x_j
\right\|_2.
```

DINOv2 adds:

```math
\mathcal{L}_{\mathrm{KoLeo}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log d_i.
```

As `d_i` approaches zero, the penalty diverges. This explicitly discourages
feature collisions without assigning every other sample a softmax-negative
label.

## Local Gradient Of KoLeo

When nearest neighbor `j(i)` is unique:

```math
\nabla_{x_i}
\left(
-\log d_i
\right)
=
-
\frac{
x_i-x_{j(i)}
}{
\left\|
x_i-x_{j(i)}
\right\|_2^2
}.
```

The force is local to the nearest collision and becomes large at short range.

## Composite Objective

Abstractly:

```math
\mathcal{L}_{\mathrm{DINOv2}}
=
\lambda_g
\mathcal{L}_{\mathrm{DINO}}
+
\lambda_p
\mathcal{L}_{\mathrm{iBOT}}
+
\lambda_u
\mathcal{L}_{\mathrm{KoLeo}}.
```

Image-level consistency, masked local prediction, and batch feature spreading
are separate pressures. Calling the method only self-distillation hides two of
the three geometries.

## Pathology Sampling Dependence

KoLeo spreads whatever distribution enters a batch. If a batch contains many
tiles from one WSI, nearest neighbors are largely within-slide; if it contains
one tile per slide, the regularizer acts across-slide. Batch construction is
therefore part of the learned uniformity prior.

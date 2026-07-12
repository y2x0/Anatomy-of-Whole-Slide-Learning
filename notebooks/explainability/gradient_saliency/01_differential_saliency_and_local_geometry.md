# Differential Saliency And Local Geometry

Primary anchor:

- Simonyan, Vedaldi, Zisserman. "Deep Inside Convolutional Networks:
  Visualising Image Classification Models and Saliency Maps." 2013.

## Class Score Gradient

For class score:

```math
S_c
:
\mathbb{R}^{D}
\longrightarrow
\mathbb{R},
```

the input gradient at image `x_0` is:

```math
w_c(x_0)
=
\nabla_x S_c(x)
\big|_{x=x_0}.
```

First-order expansion gives:

```math
S_c(x_0+\delta)
=
S_c(x_0)
+
w_c(x_0)^{\top}\delta
+
o
\left(
\|\delta\|_2
\right).
```

The gradient explains local directional sensitivity of the class score.

## Pixel Saliency

For RGB pixel `(i,j)`, Simonyan et al. collapse channel derivatives using the
largest absolute component:

```math
M_{ij}^{(c)}
=
\max_{q\in\{R,G,B\}}
\left|
\frac{\partial S_c(x)}
{\partial x_{ijq}}
\right|.
```

Taking absolute value removes direction. The map shows sensitivity magnitude,
not whether increasing a channel supports or opposes class `c`.

## Steepest-Ascent Direction

Under an `L2` perturbation budget:

```math
\max_{\|\delta\|_2\le\varepsilon}
w^{\top}\delta
=
\varepsilon\|w\|_2,
```

achieved by:

```math
\delta^{\star}
=
\varepsilon
\frac{w}{\|w\|_2}.
```

Thus the gradient is a local adversarial direction as much as an explanatory
object.

## Coordinate Dependence

For invertible reparameterization:

```math
u
=
Ax,
```

the gradient transforms as:

```math
\nabla_u S
=
A^{-\top}
\nabla_x S.
```

Raw saliency depends on input coordinates and units. Pixel-space, stain-space,
and embedding-space gradients answer different questions.

## Logit Versus Probability

For softmax probability:

```math
p_c
=
\frac{\exp(S_c)}{\sum_k\exp(S_k)},
```

```math
\nabla_x p_c
=
p_c
\left[
\nabla_x S_c
-
\sum_kp_k\nabla_x S_k
\right].
```

Probability saliency is contrastive across all classes and can vanish through
saturation even when logit sensitivity remains.

## Proper Claim

Raw gradient saliency explains infinitesimal score response around the observed
input. It is not finite patch contribution, lesion probability, or causal
morphologic effect.

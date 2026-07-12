# Saturation, Shattered Gradients, And Noise

Primary anchor:

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML
  2017.

## Saturation

For sigmoid score:

```math
F(x)
=
\sigma(ax+b),
```

the derivative is:

```math
\frac{\mathrm{d}F}{\mathrm{d}x}
=
aF(x)
\left(
1-F(x)
\right).
```

When probability approaches zero or one:

```math
\frac{\mathrm{d}F}{\mathrm{d}x}
\longrightarrow
0.
```

The prediction can differ greatly from a baseline while endpoint saliency is
nearly zero.

## ReLU Region Dependence

A ReLU network is piecewise affine:

```math
F(x)
=
A_rx+b_r
\qquad
x\in\mathcal{R}_r.
```

Within one activation region:

```math
\nabla_xF(x)
=
A_r^{\top}.
```

Crossing a ReLU boundary changes the gradient discontinuously even when the
score changes continuously.

## Shattered Gradients

In deep compositions:

```math
\nabla_xF
=
J_1^{\top}
J_2^{\top}
\cdots
J_L^{\top}
\nabla_{h_L}F.
```

Products of input-dependent Jacobians can create high-frequency sign changes,
vanishing directions, and exploding directions. Visual noise can reflect local
network geometry rather than many independent pathologic features.

## Noise Averaging

For perturbation noise:

```math
\varepsilon
\sim
\mathcal{N}
\left(
0,\sigma^2I
\right),
```

smoothed gradient is:

```math
\overline g(x)
=
\mathbb{E}_{\varepsilon}
\left[
\nabla_xF(x+\varepsilon)
\right].
```

Under regularity conditions:

```math
\overline g(x)
=
\nabla_x
\mathbb{E}_{\varepsilon}
\left[
F(x+\varepsilon)
\right].
```

Smoothing explains a noise-convolved predictor, not exactly the original
pointwise function.

## WSI Patch Saturation

For attention-softmax readout, a dominant patch can saturate its weight near
one. Gradients to other patches can become tiny even if their removal changes
the normalized support after recomputation.

## Diagnostic Separation

Low gradient magnitude can mean irrelevance, saturation, cancellation,
parameterization, or local flatness. These explanations require baseline paths,
finite perturbations, or alternative score targets to distinguish.

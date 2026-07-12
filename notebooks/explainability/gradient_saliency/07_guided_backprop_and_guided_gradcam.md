# Guided Backpropagation And Guided Grad-CAM

Primary anchors:

- Springenberg et al. "Striving for Simplicity: The All Convolutional Net."
  2015.
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-Based Localization." ICCV 2017.
  https://arxiv.org/abs/1610.02391

## ReLU Forward Map

```math
y
=
\max(x,0).
```

For upstream gradient `R`, ordinary backpropagation gives:

```math
R_{\mathrm{backprop}}
=
R
\mathbb{1}
\left\{
x>0
\right\}.
```

## Deconvolution Rule

The deconvolutional visualization rule uses:

```math
R_{\mathrm{deconv}}
=
R
\mathbb{1}
\left\{
R>0
\right\}.
```

It gates on the top-down signal and ignores the forward ReLU state.

## Guided Backpropagation

Guided backpropagation combines both gates:

```math
R_{\mathrm{guided}}
=
R
\mathbb{1}
\left\{
x>0
\right\}
\mathbb{1}
\left\{
R>0
\right\}.
```

Negative backward signals are suppressed.

## Not A True Gradient

The guided rule modifies the chain rule. In general there is no scalar function
`G` whose derivative equals this backward signal everywhere:

```math
R_{\mathrm{guided}}
\ne
\nabla_xG(x).
```

It is an imputed visualization rule, not the derivative of the original class
score.

## Loss Of Negative Evidence

Because negative gradients are removed:

```math
R<0
\Longrightarrow
R_{\mathrm{guided}}=0.
```

The visualization cannot faithfully display features that locally oppose the
target.

## Guided Grad-CAM

The paper combines high-resolution guided backpropagation with upsampled
class-specific Grad-CAM by elementwise multiplication:

```math
G_{\mathrm{GuidedGradCAM}}^{c}
=
G_{\mathrm{guided}}^{c}
\odot
\mathrm{Upsample}
\left(
L_{\mathrm{GradCAM}}^{c}
\right).
```

The result inherits spatial detail from guided backpropagation and coarse class
localization from Grad-CAM.

## Interpretation Boundary

Multiplying two maps does not preserve completeness or score conservation. A
visually sharp Guided Grad-CAM image is not an additive decomposition of the
class score.

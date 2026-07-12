# Grad-CAM Exact Map And Layer Dependence

Primary anchor:

- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-Based Localization." ICCV 2017.
  https://arxiv.org/abs/1610.02391

## Chosen Activation Tensor

Let convolutional feature maps at a chosen layer be:

```math
A^k
\in
\mathbb{R}^{U\times V},
\qquad
k=1,\ldots,K.
```

For pre-softmax class score `y^c`, Grad-CAM channel weights are:

```math
\alpha_k^c
=
\frac{1}{Z}
\sum_{i=1}^{U}
\sum_{j=1}^{V}
\frac{
\partial y^c
}{
\partial A_{ij}^{k}
},
```

where:

```math
Z
=
UV.
```

The localization map is:

```math
L_{\mathrm{GradCAM}}^{c}
=
\mathrm{ReLU}
\left(
\sum_{k=1}^{K}
\alpha_k^c A^k
\right).
```

## Downstream Linearization

At the chosen activation tensor:

```math
y^c(A+\Delta A)
\approx
y^c(A)
+
\sum_{k,i,j}
\frac{
\partial y^c
}{
\partial A_{ij}^{k}
}
\Delta A_{ij}^{k}.
```

Grad-CAM replaces spatially varying gradients within channel `k` by their mean
`alpha_k^c`. It is a coarse channelwise linearization.

## ReLU Semantics

The final ReLU retains locations with positive combined influence:

```math
\sum_k
\alpha_k^cA_{ij}^k
>
0.
```

Evidence opposing class `c` is discarded from the displayed map.

## Layer Dependence

For two layers `ell` and `r`:

```math
L_{\mathrm{GradCAM},\ell}^{c}
\ne
L_{\mathrm{GradCAM},r}^{c}
```

in general. Earlier layers offer finer spatial grids and lower-level features;
later layers offer coarser grids and stronger semantic abstraction.

## Exact CAM Boundary

If class score has global-average-pooling form:

```math
y^c
=
\sum_k
w_k^c
\frac{1}{Z}
\sum_{i,j}
A_{ij}^{k},
```

then:

```math
\alpha_k^c
=
\frac{w_k^c}{Z}
```

up to convention, and Grad-CAM recovers CAM-style channel weighting.

## WSI Boundary

Applying Grad-CAM within each patch explains pixels relative to a patch-level or
slide-level target only through the selected computational path. If the frozen
patch encoder is detached from the slide model, slide gradients cannot reach
its convolutional maps.

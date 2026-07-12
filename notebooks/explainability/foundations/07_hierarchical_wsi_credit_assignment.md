# Hierarchical WSI Credit Assignment

## Composed Predictor

Write a WSI predictor as:

```math
F
=
\mathcal{H}
\circ
\mathcal{R}
\circ
\mathcal{C}
\circ
\mathcal{T},
```

where tiling `T`, context `C`, readout `R`, and head `H` are separate maps.

For slide `X`:

```math
X
\xrightarrow{\mathcal{T}}
\left\{
x_j
\right\}_{j=1}^{n}
\xrightarrow{\mathcal{C}}
\left\{
h_j
\right\}_{j=1}^{n}
\xrightarrow{\mathcal{R}}
z
\xrightarrow{\mathcal{H}}
F(X).
```

## Patch Gradient Decomposition

For patch pixels `x_j`:

```math
\frac{\partial F}
{\partial x_j}
=
\frac{\partial F}
{\partial z}
\frac{\partial z}
{\partial h_j}
\frac{\partial h_j}
{\partial x_j}
+
\sum_{k\ne j}
\frac{\partial F}
{\partial z}
\frac{\partial z}
{\partial h_k}
\frac{\partial h_k}
{\partial x_j}.
```

The cross terms vanish only when context encoding is patch-independent.

## Attention Readout

For:

```math
z
=
\sum_j
\alpha_j(H)h_j,
```

the derivative is:

```math
\frac{\partial z}
{\partial h_i}
=
\alpha_i I
+
\sum_j
h_j
\left(
\frac{\partial\alpha_j}
{\partial h_i}
\right)^{\top}.
```

Attention weight `alpha_i` captures only the direct value coefficient. It
omits how changing patch `i` redistributes every attention weight.

## Softmax Redistribution

For logits `a_j`:

```math
\frac{\partial\alpha_j}
{\partial a_i}
=
\alpha_j
\left(
\mathbb{1}\{i=j\}
-
\alpha_i
\right).
```

Removing or changing one patch affects the normalized weight of all others.
Patch contribution is therefore relational.

## Additive Readout

If:

```math
F(X)
=
\sum_j
g(h_j),
```

then exact patch credit is available:

```math
\phi_j
=
g(h_j)
```

up to baseline convention. Additivity improves credit identifiability but
forbids explicit cross-patch interactions in the final score.

## Graph And Transformer Context

When patch `j` changes neighbors through message passing or self-attention:

```math
h_k
=
h_k(X)
\qquad
\forall k,
```

an explanation assigned only to output token `j` can miss mediated effects on
other tokens.

## Scale Consistency

Pixel credits, patch credits, region credits, and slide credits should obey a
declared aggregation rule. Without additivity:

```math
\phi_{\mathrm{region}}
\ne
\sum_{j\in\mathrm{region}}
\phi_j
```

in general. Upsampling a patch score to pixels does not create pixel-level
evidence.

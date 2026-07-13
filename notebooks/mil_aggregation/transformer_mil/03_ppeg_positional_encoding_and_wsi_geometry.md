# PPEG Positional Encoding and WSI Geometry

## 1. Set-to-Grid Injection

TransMIL uses a positional encoding module to inject spatial structure into a
token representation. Abstractly, arrange tokens on a padded grid:

```math
H\in\mathbb R^{n\times d}
\longmapsto
\Pi(H)\in\mathbb R^{h\times w\times d}.
```

A depthwise convolutional operator produces positional context:

```math
P=\Pi(H)+\mathrm{DWConv}(\Pi(H)).
```

The grid is then flattened back into tokens.

## 2. Geometry Assumption

The padding and token arrangement define a coordinate relation. If patch order
or grid placement changes, the same multiset of features can produce a
different representation.

## 3. Set Versus Spatial Transformer

Without positional injection, self-attention is permutation equivariant and a
permutation-invariant readout gives a set model. With PPEG-like geometry,

```math
F(H,\Pi(H))
\ne
F(PH,\Pi(H))
```

for arbitrary permutation matrix `P` holding the grid fixed.

## 4. Failure

Padding, missing tissue, uneven tessellation, or coordinate mismatch can create
position shortcuts. Spatial ablation must preserve the patch multiset while
changing the arrangement.


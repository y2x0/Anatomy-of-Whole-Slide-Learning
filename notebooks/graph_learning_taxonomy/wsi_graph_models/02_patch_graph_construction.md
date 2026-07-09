# Patch-GCN Graph Construction

Patch-GCN constructs edges from Euclidean patch coordinates.

This is the main graph-representation choice in the paper.

## Tissue Mask And Patches

For each WSI, tissue is segmented by:

```text
1. downsampled slide image
2. HSV color transform
3. Otsu thresholding on saturation
4. tissue-background separation
```

Then the tissue region is tiled into non-overlapping patches:

```math
x_m
\in
\mathbb{R}^{256\times 256\times 3}.
```

The paper uses `20x` magnification.

## Feature Matrix

The fixed patch encoder produces:

```math
h_m^{(0)}
=
f_{\mathrm{enc}}(x_m)
\in
\mathbb{R}^{1024}.
```

Stacking patch features gives:

```math
X
=
\begin{bmatrix}
h_1^{(0)}\\
\vdots\\
h_M^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{M\times 1024}.
```

The paper reports that an average WSI contains roughly:

```math
M
\approx
13487
```

patches, with some patient graphs reaching:

```math
M
\approx
100000.
```

## Coordinate kNN Support

For patch coordinates:

```math
C
=
\{c_m\}_{m=1}^{M},
\qquad
c_m\in\mathbb{R}^{2},
```

Patch-GCN constructs adjacency with approximate k-nearest neighbors:

```math
\mathcal{A}(v)
=
\mathrm{kNN}_{8}(c_v;C).
```

Using the row-target convention in this graph-learning family:

```math
A_{vu}
=
1
\quad
\Longleftrightarrow
\quad
u\in\mathcal{A}(v).
```

The paper uses:

```math
k=8.
```

This is chosen to model a local `3 by 3` image receptive field around the
central patch.

The paper does not specify:

```text
radius cutoff
edge symmetrization
self-loops
edge direction
cross-slide edges
```

So the faithful public statement is only:

```text
fast approximate coordinate kNN with k = 8
```

## Why Coordinate Graphs Matter

A feature-similarity graph would use:

```math
\mathcal{A}_{\mathrm{feat}}(v)
=
\mathrm{kNN}_{k}(h_v;H).
```

Patch-GCN instead uses:

```math
\mathcal{A}_{\mathrm{coord}}(v)
=
\mathrm{kNN}_{8}(c_v;C).
```

These are different hypotheses.

Feature graph:

```text
similar-looking patches should communicate
```

Coordinate graph:

```text
nearby tissue regions should communicate
```

For survival, the paper argues that local tissue context matters because
prognostic meaning can depend on spatial relationships, such as whether immune
cells are near tumor cells or stroma.

## Graph Object

A single WSI graph is:

```math
G
=
(X,A,C).
```

For patient `i` with multiple WSIs:

```math
\mathcal{G}_i
=
\{G_{ij}\}_{j=1}^{K_i}.
```

## Inductive Bias

The graph construction assumes:

```text
local Euclidean tissue adjacency is the support on which prognostic morphology
should exchange information.
```

The context operator can only move information along this support:

```math
A_{vu}=0
\quad
\Longrightarrow
\quad
\text{no direct message }u\to v.
```

## Failure Mode Already Present

Before any graph convolution is applied, the model has committed to:

```text
8-neighbor local support
```

This can fail when:

```text
important interactions are long-range
the tissue mask creates missing or spurious neighbors
coordinates are distorted by preprocessing
prognostic similarity is nonlocal rather than spatially adjacent
```

# Patch-GCN Slide Object

Patch-GCN represents a patient as a collection of WSI patch graphs.

The paper's data object is a patient-level survival observation:

```math
(P_i,T_i,C_i),
```

where:

```text
P_i:
    patient with one or more WSIs

T_i:
    overall survival time

C_i:
    censoring status
```

If patient `i` has `K_i` slides:

```math
P_i
=
\{W_{ij}\}_{j=1}^{K_i}.
```

Patch-GCN builds a graph for each WSI:

```math
G_{ij}
=
(X_{ij},A_{ij}).
```

The patient-level WSI graph object is therefore:

```math
\mathcal{G}_i
=
\{G_{ij}\}_{j=1}^{K_i}.
```

This is important: the paper's construction is naturally a set of slide
subgraphs per patient, not one abstract bag of independent patches.

The paper's notation for the patient-level graph omits the upper index in one
display. The safest reading is the collection over that patient's WSIs:

```math
\{G_{ij}\}_{j=1}^{K_i}.
```

## Node Features

Each WSI is decomposed into non-overlapping patches:

```math
x_{ijm}
\in
\mathbb{R}^{256\times 256\times 3}
```

at `20x` magnification.

A truncated ResNet-50 pretrained on ImageNet maps each patch to a feature:

```math
h_{ijm}^{(0)}
=
f_{\mathrm{ResNet}}
\left(
x_{ijm}
\right)
\in
\mathbb{R}^{1024}.
```

The paper states that the feature comes from spatial average pooling after the
third residual block.

For WSI `j`:

```math
X_{ij}
=
\begin{bmatrix}
h_{ij1}^{(0)}\\
\vdots\\
h_{ijM_{ij}}^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{M_{ij}\times 1024}.
```

## Coordinates

Each patch has a coordinate from tissue segmentation:

```math
c_{ijm}
=
(x_{ijm}^{\mathrm{coord}},y_{ijm}^{\mathrm{coord}})
\in
\mathbb{R}^{2}.
```

The coordinate is not just metadata. It defines the graph support.

## Difference From MIL

An attention MIL slide object is:

```math
H_i
=
\{h_{im}\}_{m=1}^{M_i}.
```

Patch-GCN keeps:

```math
\mathcal{G}_i
=
\{(X_{ij},A_{ij},C_{ij})\}_{j=1}^{K_i}.
```

The slide object contains:

```text
patch features:
    what each local region looks like

coordinates:
    where each patch lies on tissue

adjacency:
    which local regions can exchange messages
```

## C/R/G/S Placement

```text
G:
    A_ij derived from coordinates C_ij

C:
    graph convolution over each WSI graph

R:
    global attention pooling over contextualized node states

S:
    survival time and censoring supervise the patient-level prediction
```

## What Survives At This Stage

Before graph learning, no context has been injected. The object is:

```math
\{(h_{ijm}^{(0)},c_{ijm})\}_{m=1}^{M_{ij}}
\quad
\text{plus}
\quad
A_{ij}.
```

The important design decision is already made:

```text
neighbor relations are spatial, not feature-similarity based
```

Thus two patches can communicate because they are nearby in tissue, even if
their feature vectors are very different.

The paper does not fully specify edge direction, symmetrization, self-loops, or
inter-WSI edges. Those should not be silently added to the mathematical object.

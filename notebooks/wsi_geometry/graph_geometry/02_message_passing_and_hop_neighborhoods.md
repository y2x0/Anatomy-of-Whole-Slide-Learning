# Message Passing And Hop Neighborhoods

Graph geometry becomes useful through message passing.

Let
```math
h_j^{(0)}
```
be node features. A generic graph layer is:

```math
m_j^{(\ell)}
=
\mathrm{AGG}_{k\in\mathcal{N}(j)}
\psi_\theta
\left(
h_j^{(\ell)},
h_k^{(\ell)},
e_{jk}
\right),
```

```math
h_j^{(\ell+1)}
=
\phi_\theta
\left(
h_j^{(\ell)},
m_j^{(\ell)}
\right).
```

After
```math
L
```
layers:

```math
h_j^{(L)}
=
F_\theta
\left(
\{h_k^{(0)}:d_G(j,k)\le L\}
\right).
```

The graph distance
```math
d_G
```
defines the context radius.

## GCN-Style Linear Smoothing

A simplified graph convolution is:

```math
H^{(\ell+1)}
=
\sigma
\left(
\widehat A H^{(\ell)}W^{(\ell)}
\right),
```

where:

```math
\widehat A
=
D^{-1/2}(A+I)D^{-1/2}.
```

The normalized adjacency averages information over neighbors. This creates
spatial context, but repeated layers tend to smooth node states.

## Graph Attention

Graph attention uses learned edge weights:

```math
\alpha_{jk}
=
\mathrm{softmax}_{k\in\mathcal{N}(j)}
a_\theta(h_j,h_k,e_{jk}),
```

```math
h_j^{(\ell+1)}
=
\sum_{k\in\mathcal{N}(j)}
\alpha_{jk}W h_k^{(\ell)}.
```

This keeps the graph support fixed while learning which edges matter.

## Graph-Level Readout

After context:

```math
z_i
=
\mathcal{R}
\left(
\{h_j^{(L)}\}_{j\in V_i}
\right).
```

For survival:

```math
\eta_i
=
w^\top z_i.
```

Patch-GCN-style survival models can be read as:

```math
(H_i,C_i)
\to
A_i
\to
H_i^{(L)}
\to
z_i
\to
\eta_i.
```

## C/R/G/S Placement

```text
G:
    graph neighborhoods

C:
    message passing over L-hop neighborhoods

R:
    graph-level pooling

S:
    label or survival loss shapes node context through backpropagation
```

## Dense Summary

Graph context replaces isolated patch features with neighborhood-dependent
features:

```math
h_j
\to
h_j^{(L)}
=
F_\theta(\text{L-hop tissue neighborhood of }j).
```

The model can only reason along the edges it is given or learns.

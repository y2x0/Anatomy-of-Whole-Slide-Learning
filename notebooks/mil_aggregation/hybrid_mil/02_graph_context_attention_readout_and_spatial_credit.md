# Graph Context Attention Readout And Spatial Credit

## 1. Graph contextualization

Let patch features and coordinates be

```math
H_i\in\mathbb R^{n_i\times d},
\qquad
X_i\in\mathbb R^{n_i\times 2}.
```

A coordinate graph has adjacency A_i. A one-layer graph context operator can be
written abstractly as

```math
\widetilde H_i
=
\sigma
\left(
A_iH_iW_N+H_iW_S
\right).
```

Patch-GCN uses spatially constructed neighbors before a slide-level attention
readout. The graph is therefore part of C, while attention remains part of R.

## 2. Attention after graph context

The readout is

```math
q_{ij}
=
w^{\mathsf T}
\left[
\tanh(V\widetilde h_{ij})
\odot
\sigma(U\widetilde h_{ij})
\right],
\qquad
\alpha_{ij}
=
\frac{\exp q_{ij}}{\sum_k\exp q_{ik}},
```

```math
z_i
=
\sum_{j=1}^{n_i}\alpha_{ij}\widetilde h_{ij},
\qquad
\widehat y_i=\mathcal H_{\omega}(z_i).
```

The attention is not applied to isolated patch features. It weights features
that already contain local coordinate-neighborhood information.

## 3. Effective support

If A has one-hop support, then the contextual state at node j depends on

```math
\widetilde h_j
=
f
\left(
h_j,\{h_k:k\in\mathcal N(j)\}
\right).
```

The final slide statistic therefore has the expansion

```math
z
=
\sum_j\alpha_j(H,A)
f
\left(
h_j,\{h_k:k\in\mathcal N(j)\}
\right).
```

A selected node can carry evidence from several physical patches. Heatmaps of
alpha are node-selection maps, not necessarily maps of all causal evidence.

## 4. Patch-GCN survival placement

For a Cox-style risk head,

```math
\eta_i=w^{\mathsf T}z_i+b,
\qquad
\mathcal L_{\mathrm{Cox}}
=
-\sum_{i:\delta_i=1}
\left(
\eta_i
-
\log\sum_{j\in\mathcal R(t_i)}\exp\eta_j
\right).
```

The graph changes the covariate representation entering eta_i. It does not
change the Cox likelihood into a graph-specific survival objective.

## 5. WiKG as a learned-support variant

WiKG constructs a directed top-K support from feature similarity and applies
knowledge-aware attention to feature interactions. Abstractly, let

```math
\mathcal N_K(j;H)
=
\mathrm{TopK}_{k\neq j}
\mathrm{sim}(h_j,h_k).
```

A feature-conditioned contextual state has the form

```math
\widetilde h_j
=
h_j
+
\sum_{k\in\mathcal N_K(j;H)}
\gamma_{jk}(H)\,
\Psi(h_j,h_k,e_{jk}),
```

where e_{jk} can encode an edge relation and gamma is a learned interaction
weight. A later mean or max readout gives

```math
z
=
\frac{1}{n}\sum_j\widetilde h_j
\quad\text{or}\quad
z=\max_j\widetilde h_j.
```

This is graph context followed by an ordinary invariant readout. If WiKG is
described as attention MIL, the support-learning graph step is being omitted.

## 6. Spatial credit versus attention credit

For a scalar output f, the total derivative with respect to a pre-context patch is

```math
\frac{\partial f}{\partial h_k}
=
\sum_j
\frac{\partial f}{\partial \widetilde h_j}
\frac{\partial \widetilde h_j}{\partial h_k}.
```

The term with j=k is direct credit; terms with k in a neighbor receptive field
are propagated spatial credit. Attention alpha_j only appears as one factor in
the first derivative and is not equal to the total derivative.

A deletion test should recompute graph construction and attention after removing
patch k. Holding A and alpha fixed measures a different local approximation.

## 7. C/R/G/S placement

`text
C: coordinate graph convolution for Patch-GCN; learned feature-support and
   knowledge-aware interactions for WiKG.

R: attention, mean, or max over contextualized nodes.

G: spatial kNN coordinates for Patch-GCN; dynamic directed top-K feature graph
   for WiKG.

S: slide classification or survival labels, with any self-supervised feature
   pretraining treated as an upstream objective.

The surviving statistic is a readout of relationally contextualized features,
not a readout of independent patches.
`

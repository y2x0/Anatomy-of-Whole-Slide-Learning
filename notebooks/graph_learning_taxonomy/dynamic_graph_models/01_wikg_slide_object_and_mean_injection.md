# WiKG Slide Object And Mean Injection

This note defines the patch-level WSI object, the released-code slide-mean injection, the resulting attributed directed graph, and its C/R/G/S placement.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Patch Embeddings

For slide `b`, let the foreground tissue patches be:

```math
X_b
=
\{x_{bv}\}_{v=1}^{n_b}.
```

A patch encoder and learned input projection produce:

```math
e_{bv}
=
f_{\mathrm{enc}}(x_{bv})
\in
\mathbb{R}^{d_0},
```

```math
g_{bv}
=
\phi_{\mathrm{in}}
\left(
e_{bv}W_{\mathrm{in}}+b_{\mathrm{in}}
\right)
\in
\mathbb{R}^{d}.
```

The paper reports a ViT-Small patch encoder with 384-dimensional patch
features, followed by a projection to 512 dimensions in its implementation
description.

## Released-Code Mean Injection

The released implementation adds a slide mean before graph construction:

```math
\overline g_b
=
\frac{1}{n_b}
\sum_{v=1}^{n_b}g_{bv},
```

```math
\widetilde g_{bv}
=
\frac{1}{2}
\left(
g_{bv}+\overline g_b
\right).
```

This operation appears in the released forward pass but is not part of the
paper's displayed methodology equations. It means that the implementation is
not purely local before graph construction: every node already contains the
bag first moment.

The mean injection is permutation equivariant. For a permutation matrix
`P_b` acting on node rows:

```math
\widetilde G_b(P_bG_b)
=
P_b\widetilde G_b(G_b).
```

It nevertheless changes the hypothesis class because the learned topology is
constructed from globally shifted node states rather than independent patch
states.

## The Full Graph Object

After neighbor selection and edge embedding construction, represent slide `b`
as:

```math
\mathcal{G}_{\theta,b}^{\mathrm{WiKG}}
=
\left(
V_b,
E_{\theta,b},
Q_b,
T_b,
R_{\theta,b}
\right),
```

where:

```math
Q_b
=
[q_{b1};\ldots;q_{bn_b}]
\in
\mathbb{R}^{n_b\times d},
```

```math
T_b
=
[t_{b1};\ldots;t_{bn_b}]
\in
\mathbb{R}^{n_b\times d},
```

and:

```math
R_{\theta,b}
=
\{r_{bvu}\in\mathbb{R}^{d}:
(u,v)\in E_{\theta,b}\}.
```

This is an attributed directed graph. It is not a physical tissue graph unless
the learned head-tail relation is separately shown to recover physical tissue
relations.

## C/R/G/S Placement

```text
\mathcal{G}:
    learned directed top-k support plus head, tail, and edge states

\mathcal{C}:
    knowledge-aware attention on the selected support

\mathcal{R}:
    graph-level pooling over updated head states

\mathcal{S}:
    slide label indirectly shapes W_H, W_T, and therefore the topology
```

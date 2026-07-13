# Graph-Contrast Statistical Objects

Primary anchors:

- Velickovic et al. "Deep Graph Infomax." ICLR 2019.
  https://arxiv.org/abs/1809.10341
- You et al. "Graph Contrastive Learning with Augmentations." NeurIPS 2020.
  https://arxiv.org/abs/2010.13902
- Zhu et al. "Deep Graph Contrastive Representation Learning." 2020.
  https://arxiv.org/abs/2006.04131

## Graph-Valued Input

Let:

```math
G
=
\left(
V,E,X
\right),
```

with:

```math
|V|=N,
\qquad
A\in\left\{0,1\right\}^{N\times N},
\qquad
X\in\mathbb{R}^{N\times F}.
```

A message-passing encoder produces:

```math
H
=
f_{\theta}(X,A)
\in
\mathbb{R}^{N\times d}.
```

The row `h_i` is not merely a transform of `x_i`. After `L` layers it depends
on an `L`-hop rooted neighborhood:

```math
h_i^{(L)}
=
F_{\theta}^{(L)}
\left(
G[i,L]
\right).
```

Contrast between node embeddings is therefore contrast between encoded rooted
subgraphs.

## Three Positive Relations

DGI declares a node embedding compatible with a summary of its uncorrupted
graph:

```math
R_{\mathrm{DGI}}
\left(
h_i,s
\right)
=
1.
```

GraphCL declares two graph-level readouts positive when generated from the same
source graph:

```math
R_{\mathrm{GraphCL}}
\left(
z_i^{(1)},z_j^{(2)}
\right)
=
\mathbb{1}
\left\{
i=j
\right\}.
```

GRACE declares two node embeddings positive when they correspond to the same
node under two graph corruptions:

```math
R_{\mathrm{GRACE}}
\left(
u_i,v_j
\right)
=
\mathbb{1}
\left\{
i=j
\right\}.
```

These relations differ in unit, scale, and negative distribution.

## Augmentation As A Markov Kernel

A graph view is sampled from:

```math
\widetilde G
\sim
q
\left(
\widetilde G\mid G
\right).
```

Two views satisfy conditional independence in the usual construction:

```math
\widetilde G^{(1)}
\perp
\widetilde G^{(2)}
\mid
G,
```

with different kernels allowed:

```math
\widetilde G^{(1)}
\sim
q_1(\cdot\mid G),
\qquad
\widetilde G^{(2)}
\sim
q_2(\cdot\mid G).
```

The shared information is controlled jointly by the source distribution and
the two corruption kernels.

## Invariance Is A Supervision Claim

The generative construction often implies:

```math
Y
\perp
\widetilde G
\mid
G
```

when the corruption kernel samples `\widetilde G` from `G` without directly
observing `Y`. This conditional independence is a Markov property, not a
semantic-preservation guarantee. A label-preserving view mechanism instead
requires, for `q(\widetilde G\mid G)`-almost every reachable pair:

```math
p
\left(
Y\mid \widetilde G
\right)
\equiv
p
\left(
Y\mid G
\right).
```

For deterministic labels, the corresponding requirement is:

```math
Y(G)
=
Y(\widetilde G).
```

Graph contrast cannot verify this condition without task information. It
enforces the invariance encoded by `q`.

## C/R/G/S Preview

```math
\mathcal{C}
=
\text{message passing and optional graph corruption},
```

```math
\mathcal{R}
=
\text{node, graph summary, or graph readout},
```

```math
\mathcal{G}
=
\text{bilinear BCE or temperature-scaled cosine geometry},
```

```math
\mathcal{S}
=
\text{local-global, graph identity, or node identity relation}.
```

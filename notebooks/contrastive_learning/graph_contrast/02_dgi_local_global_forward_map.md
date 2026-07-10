# DGI Local-Global Forward Map

Primary anchor:

- Velickovic et al. "Deep Graph Infomax." ICLR 2019.
  https://arxiv.org/abs/1809.10341

## Positive Graph Path

Given graph features and adjacency:

```math
H
=
\mathcal{E}_{\theta}(X,A)
=
\left[
h_1^{\top};
\ldots;
h_N^{\top}
\right].
```

DGI calls `h_i` a patch representation because message passing makes it a
function of a rooted graph neighborhood.

The transductive readout used in the paper is:

```math
s
=
\sigma
\left(
\frac{1}{N}
\sum_{i=1}^{N}
h_i
\right),
```

where `\sigma` is the logistic sigmoid applied coordinatewise.

The readout is permutation invariant:

```math
\mathcal{R}(PH)
=
\mathcal{R}(H)
```

for every permutation matrix `P`.

## Corrupted Path

A corruption function generates:

```math
\left(
\widetilde X,
\widetilde A
\right)
\sim
\mathcal{C}
\left(
X,A
\right).
```

The shared encoder gives:

```math
\widetilde H
=
\mathcal{E}_{\theta}
\left(
\widetilde X,
\widetilde A
\right).
```

The positive summary `s` is retained. DGI contrasts:

```math
\left(h_i,s\right)
\quad\text{against}\quad
\left(\widetilde h_j,s\right).
```

It does not contrast two independently pooled graph summaries.

## Bilinear Discriminator

The discriminator is:

```math
D_{\omega}(h,s)
=
\sigma
\left(
h^{\top}Ws
\right),
```

with:

```math
W
\in
\mathbb{R}^{d\times d}.
```

The unnormalized compatibility is:

```math
a(h,s)
=
h^{\top}Ws.
```

Unlike cosine contrast, this score is sensitive to embedding norms and learns
an anisotropic cross-statistic through `W`.

## Exact Objective

For `N` positive and `M` corrupted local representations, DGI maximizes:

```math
\mathcal{J}_{\mathrm{DGI}}
=
\frac{1}{N+M}
\left[
\sum_{i=1}^{N}
\log D_{\omega}(h_i,s)
+
\sum_{j=1}^{M}
\log
\left(
1-D_{\omega}(\widetilde h_j,s)
\right)
\right].
```

Equivalently, minimize:

```math
\mathcal{L}_{\mathrm{DGI}}
=
-\mathcal{J}_{\mathrm{DGI}}.
```

## Exact Score Gradients

Let:

```math
a_i
=
h_i^{\top}Ws,
\qquad
\widetilde a_j
=
\widetilde h_j^{\top}Ws.
```

Then:

```math
\frac{\partial\mathcal{L}}
{\partial a_i}
=
-\frac{1}{N+M}
\left(
1-\sigma(a_i)
\right),
```

```math
\frac{\partial\mathcal{L}}
{\partial \widetilde a_j}
=
\frac{1}{N+M}
\sigma(\widetilde a_j).
```

Therefore:

```math
\nabla_W\mathcal{L}
=
\frac{1}{N+M}
\left[
-\sum_i
\left(
1-D_i
\right)
h_i s^{\top}
+
\sum_j
\widetilde D_j
\widetilde h_j s^{\top}
\right].
```

The discriminator learns the difference between positive and corrupted
local-summary cross-moments.

## Summary-Coupled Gradient

Because `s` depends on every positive node embedding, each `h_k` receives a
direct discriminator gradient and an indirect global path:

```math
\frac{\mathrm{d}\mathcal{L}}
{\mathrm{d}h_k}
=
\frac{\partial\mathcal{L}}
{\partial h_k}
+
\left(
\frac{\partial s}{\partial h_k}
\right)^{\top}
\frac{\partial\mathcal{L}}
{\partial s}.
```

For mean-sigmoid readout:

```math
\frac{\partial s}{\partial h_k}
=
\frac{1}{N}
\mathrm{diag}
\left(
s\odot(1-s)
\right).
```

The indirect path scales as `1/N` and can weaken on very large graphs.

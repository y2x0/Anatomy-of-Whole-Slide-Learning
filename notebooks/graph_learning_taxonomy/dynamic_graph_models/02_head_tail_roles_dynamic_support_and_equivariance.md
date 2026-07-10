# Head-Tail Roles, Dynamic Support, And Equivariance

This note isolates the asymmetric head-tail parameterization, directed support, the precise meaning of dynamic graph construction, and permutation behavior.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Head And Tail Roles

WiKG gives each patch two learned roles:

```math
q_{bv}
=
\widetilde g_{bv}W_H+b_H
\in
\mathbb{R}^{d},
```

```math
t_{bu}
=
\widetilde g_{bu}W_T+b_T
\in
\mathbb{R}^{d}.
```

The head vector `q_{bv}` asks which patches should send information to target
node `v`. The tail vector `t_{bu}` describes source node `u` in its
message-sending role.

Separate projections make the pair score asymmetric:

```math
s_{bvu}
=
\frac{q_{bv}t_{bu}^{\top}}{\sqrt d}.
```

In general:

```math
s_{bvu}
\ne
s_{buv}.
```

Therefore, selecting `u` as a neighbor of `v` does not imply that `v` is a
neighbor of `u`.

## Directed Dynamic Support

For each target node `v`, define the selected source set:

```math
\mathcal{N}_{b,K}(v)
=
\underset{U\subseteq V_b,\ |U|=K}{\arg\max}
\sum_{u\in U}s_{bvu}.
```

Ignoring ties, this is the set of indices of the `K` largest head-tail logits
in row `v`.

Use the message-flow edge convention:

```math
E_{\theta,b}
=
\left\{
(u,v):
u\in\mathcal{N}_{b,K}(v)
\right\}.
```

Thus:

```text
u -> v:
    tail node u sends a message to head node v
```

The adjacency matrix is:

```math
A_{\theta,b}[v,u]
=
\mathbb{1}
\left\{
u\in\mathcal{N}_{b,K}(v)
\right\}.
```

Each row has `K` selected entries when `K` does not exceed the number of
available nodes:

```math
\sum_{u=1}^{n_b}A_{\theta,b}[v,u]
=
K.
```

The released implementation does not mask the diagonal before top-k
selection. Consequently, a self-edge can be selected:

```math
(v,v)
\in
E_{\theta,b}.
```

## What "Dynamic" Means Here

The support is a function of the slide features and learned parameters:

```math
A_{\theta,b}
=
\Gamma_K
\left(
\widetilde G_bW_H,
\widetilde G_bW_T
\right).
```

It can change:

```text
across slides:
    different patch embeddings induce different neighbors

across optimization steps:
    W_H and W_T change, so the same slide can induce different neighbors
```

But the released WiKG model constructs this support once in its graph block.
It does not repeatedly reconstruct neighborhoods after multiple graph layers.

This separates WiKG from the stronger layerwise meaning of dynamic used by
DGCNN. In DGCNN, a feature-space kNN graph is recomputed from the current node
states after successive EdgeConv layers:

```math
A_b^{(\ell)}
=
\Gamma_K
\left(
H_b^{(\ell)}
\right),
```

```math
H_b^{(\ell+1)}
=
\mathcal{C}_{\theta}^{(\ell)}
\left(
H_b^{(\ell)};
A_b^{(\ell)}
\right).
```

WiKG is input- and parameter-dynamic; DGCNN is additionally layer-dynamic.

Specific source for this distinction:

```text
Wang et al., Dynamic Graph CNN for Learning on Point Clouds,
ACM Transactions on Graphics 2019.
https://arxiv.org/abs/1801.07829
```

## Permutation Equivariance

Let `P_b` permute the node order. With no exact top-k ties:

```math
Q_b'
=
P_bQ_b,
\qquad
T_b'
=
P_bT_b,
```

```math
S_b'
=
P_bS_bP_b^{\top},
```

and:

```math
A_{\theta,b}'
=
P_bA_{\theta,b}P_b^{\top}.
```

Therefore graph construction is permutation equivariant. A permutation-
invariant graph readout can make the final slide prediction permutation
invariant.

Exact ties require a tie-breaking convention. Index-based tie breaking can
violate strict equivariance because permuting equal-score nodes can change
which tied index enters the top-k set.

## Surviving Structural Assumption

WiKG assumes that task-relevant patch interaction can be represented by a
directed graph with exactly `K` selected incoming message sources per target in
a learned bilinear feature geometry:

```math
u
\text{ can influence }
v
\quad\Longleftrightarrow\quad
q_{bv}t_{bu}^{\top}
\text{ is among the rowwise top-}K\text{ scores}.
```

That is a task-learned relational hypothesis, not an observed anatomical fact.

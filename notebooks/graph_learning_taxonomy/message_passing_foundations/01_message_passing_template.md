# Message Passing Template

The message-passing template is the common mathematical shape behind many GNNs.
The key reference for this form is Gilmer et al., who write graph learning as
message functions, update functions, and graph-level readout.

## Graph And Node States

Let:

```math
G
=
(V,E),
\qquad
H^{(0)}
=
\{h_v^{(0)}\}_{v\in V}.
```

For each target node `v`, define the allowed source set:

```math
\mathcal{A}(v)
=
\{u:(u,v)\in E\}.
```

The convention is:

```math
u\to v
```

means that source node `u` can send a message to target node `v`.

Self-loops are often added:

```math
\widetilde{\mathcal{A}}(v)
=
\mathcal{A}(v)\cup\{v\}.
```

This matters because without a self-loop a node can lose its own previous state
unless the update function explicitly retains it.

## Message Phase

At layer `ell`, the directed edge message from `u` to `v` is:

```math
m_{vu}^{(\ell)}
=
M_{\theta}^{(\ell)}
\left(
h_v^{(\ell)},
h_u^{(\ell)},
e_{vu}
\right).
```

The message may depend on:

```text
target state:
    h_v

source state:
    h_u

edge feature:
    distance, edge type, adjacency weight, relative coordinate
```

For WSI graphs, `e_vu` can encode:

```text
Euclidean distance
relative coordinate
same-region indicator
tissue compartment relation
cell-type pair relation
```

## Aggregation Phase

The incoming messages form a multiset:

```math
\mathcal{M}_v^{(\ell)}
=
\{\!\{m_{vu}^{(\ell)}:u\in\mathcal{A}(v)\}\!\}.
```

Aggregation must be permutation invariant over neighbor order:

```math
\bar m_v^{(\ell)}
=
\rho
\left(
\mathcal{M}_v^{(\ell)}
\right).
```

Common choices:

```text
sum:
    preserves count and total evidence

mean:
    normalizes degree and estimates local average state

max:
    preserves strongest local feature

attention-weighted sum:
    learns data-dependent edge weights
```

## Update Phase

The node state is updated as:

```math
h_v^{(\ell+1)}
=
U_{\theta}^{(\ell)}
\left(
h_v^{(\ell)},
\bar m_v^{(\ell)}
\right).
```

The update decides whether the model retains the old node state, overwrites it,
or mixes it with neighbor context.

A residual update is:

```math
h_v^{(\ell+1)}
=
h_v^{(\ell)}
+
F_{\theta}^{(\ell)}
\left(
\bar m_v^{(\ell)}
\right).
```

Without residual or self-information, repeated graph learning can quickly erase
node identity.

## Permutation Equivariance

Let `P` be a permutation matrix over nodes. A graph node operator should satisfy:

```math
\Phi_{\theta}
\left(
PH,
PAP^\top
\right)
=
P\Phi_{\theta}(H,A).
```

This holds when:

```text
message maps are shared across nodes and edges
aggregation is invariant to neighbor ordering
edge features are permuted consistently
```

Graph-level readout should be invariant:

```math
\mathcal{R}
\left(
P\widetilde H,
PAP^\top
\right)
=
\mathcal{R}
\left(
\widetilde H,
A
\right).
```

## What Survives

After `L` message-passing layers, each node stores an `L`-hop rooted graph
statistic:

```math
h_v^{(L)}
=
F_{\theta}^{(L)}
\left(
G[\mathcal{B}_L(v)],
\{h_u^{(0)}:u\in\mathcal{B}_L(v)\}
\right),
```

where:

```math
\mathcal{B}_L(v)
=
\{u:d_G(u,v)\le L\}.
```

The graph-level slide vector then survives readout:

```math
z
=
\mathcal{R}
\left(
\{h_v^{(L)}\}_{v\in V}
\right).
```

The full pipeline therefore preserves:

```text
local graph neighborhoods:
    inside node states

global slide statistic:
    only after graph readout
```

## Failure Mode

Message passing assumes the relevant pathology signal can move along available
paths before it is needed by the readout.

This fails when:

```text
important nodes are disconnected
the needed interaction is longer than the chosen depth
many signals must pass through a narrow graph bottleneck
the aggregator destroys rare local configurations
```

## Dense Summary

The template is:

```math
\boxed{
h_v^{(\ell+1)}
=
U_{\theta}^{(\ell)}
\left(
h_v^{(\ell)},
\rho
\left(
\{\!\{
M_{\theta}^{(\ell)}
(h_v^{(\ell)},h_u^{(\ell)},e_{vu})
:
u\in\mathcal{A}(v)
\}\!\}
\right)
\right)
}
```

GCN, GraphSAGE, and GAT are not unrelated architectures. They are different
choices for `M`, `rho`, `U`, and `A(v)`.

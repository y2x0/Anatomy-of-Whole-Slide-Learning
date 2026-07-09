# HEAT Context Operator

The Heterogeneous Edge Attribute Transformer is the context operator in Chan et
al. The paper's displayed equations are compact, so this note gives a
dimension-safe reconstruction that matches the paper-level design:

```text
node-type-specific projections
edge-attribute modulation
attention over feature-kNN neighbors
```

This note uses the row-vector convention used by the graph taxonomy folder:

```math
h_v W.
```

The exact tensor contraction for the edge attribute can vary by implementation.
The invariant claim is that edge attributes modulate source-target
compatibility before the softmax.

## Inputs

At layer `ell - 1`, node `v` has:

```math
h_{iv}^{(\ell-1)}
\in
\mathbb{R}^{d_{\ell-1}}.
```

Each node has a pseudo-label type:

```math
\tau_i(v)
\in
\mathcal{T}.
```

For the formulas below, orient the feature-kNN relation as source-neighbor to
target:

```math
E_i
=
\{(s,t):s\in\mathcal{K}_k(t)\}.
```

Thus target node `t` receives messages from:

```math
\mathcal{N}_i(t)
=
\{s:(s,t)\in E_i\}
=
\mathcal{K}_k(t).
```

If an implementation stores query-to-neighbor edges instead:

```math
(t,s)
\quad
\text{for}
\quad
s\in\mathcal{K}_k(t),
```

then the message-passing formulas use the transposed adjacency. The paper-level
claim is feature-kNN support, not a unique storage orientation.

Each edge:

```math
e=(s,t)
```

has an edge-attribute state:

```math
b_{ie}^{(\ell-1)}
\in
\mathbb{R}^{d_{e,\ell-1}}.
```

Initially, this state is derived from the Pearson edge attribute:

```math
b_{ie}^{(0)}
=
\mathrm{Embed}_{\mathrm{edge}}
\left(
\phi_i(e)
\right).
```

For scalar Pearson input, this is a map:

```math
\mathbb{R}
\to
\mathbb{R}^{d_{e,0}}.
```

A static-edge simplification sets:

```math
b_{ie}^{(\ell)}
=
b_{ie}^{(0)}
```

for all layers. The edge-state recurrence below is the learned edge-attribute
version of the HEAT idea.

## Type-Specific Q/K/V Projections

For attention head `m`, let:

```math
d_h
```

be the per-head dimension.

The source key is:

```math
k_{is}^{m}
=
h_{is}^{(\ell-1)}
W_{K,\tau_i(s)}^{m}
\in
\mathbb{R}^{d_h}.
```

The source value is:

```math
v_{is}^{m}
=
h_{is}^{(\ell-1)}
W_{V,\tau_i(s)}^{m}
\in
\mathbb{R}^{d_h}.
```

The target query is:

```math
q_{it}^{m}
=
h_{it}^{(\ell-1)}
W_{Q,\tau_i(t)}^{m}
\in
\mathbb{R}^{d_h}.
```

The important paper-level point is not that these three matrices are tied. It
is that the projection family is selected by node pseudo-label type.

## Edge Attribute Projection

For edge:

```math
e=(s,t),
```

project the edge state into the head dimension:

```math
g_{ie}^{m}
=
b_{ie}^{(\ell-1)}
W_{E}^{m}
\in
\mathbb{R}^{d_h}.
```

Thus a scalar Pearson input can become a learned feature-wise edge gate after
embedding and projection.

## Dimension-Safe Edge-Modulated Score

A dimension-safe row-vector score is:

```math
a_{ist}^{m}
=
\frac{
\left(
k_{is}^{m}
\odot
g_{ie}^{m}
\right)
\left(
q_{it}^{m}
\right)^\top
}{
\sqrt{d_h}
}.
```

Here:

```text
odot:
    feature-wise multiplication

d_h:
    per-head dimension
```

This makes the attention score scalar:

```math
a_{ist}^{m}
\in
\mathbb{R}.
```

If the projected edge attribute is represented as a matrix, the same design can
be written as a bilinear score:

```math
a_{ist}^{m}
=
\frac{
k_{is}^{m}
M_{ie}^{m}
\left(
q_{it}^{m}
\right)^\top
}{
\sqrt{d_h}
}.
```

The safe statement is therefore:

```text
edge attributes modulate source-target compatibility before softmax
```

not that the paper forces one implementation-independent scalar contraction.

## Per-Head Softmax

For each target node `t` and head `m`, normalize over the target's feature-kNN
source nodes:

```math
\alpha_{ist}^{m}
=
\frac{
\exp(a_{ist}^{m})
}{
\sum_{u\in\mathcal{N}_i(t)}
\exp(a_{iut}^{m})
}.
```

The normalization is local to:

```math
\mathcal{N}_i(t).
```

So the support bottleneck is the feature-kNN graph: nodes outside
`\mathcal{N}_i(t)` cannot contribute directly to target `t` in one HEAT layer.

## Headwise Node Update

For each head:

```math
\bar h_{it}^{m}
=
\sum_{s\in\mathcal{N}_i(t)}
\alpha_{ist}^{m}
v_{is}^{m}.
```

Concatenating heads gives:

```math
\bar h_{it}
=
\big\Vert_{m=1}^{Q}
\bar h_{it}^{m}
\in
\mathbb{R}^{Qd_h}.
```

If the next layer expects dimension `d_ell`, include an output projection:

```math
h_{it}^{(\ell)}
=
\bar h_{it}
W_{O}^{(\ell)}
\in
\mathbb{R}^{d_{\ell}}.
```

If:

```math
d_{\ell}
=
Qd_h,
```

then the output projection can be the identity.

## Edge-State Update

The edge projection can also be carried forward as a learned edge state:

```math
\bar b_{ie}
=
\big\Vert_{m=1}^{Q}
g_{ie}^{m}
\in
\mathbb{R}^{Qd_h}.
```

With an optional output projection:

```math
b_{ie}^{(\ell)}
=
\bar b_{ie}
W_{E,O}^{(\ell)}
\in
\mathbb{R}^{d_{e,\ell}}.
```

For a static-edge ablation, replace this by:

```math
b_{ie}^{(\ell)}
=
b_{ie}^{(0)}.
```

The full HEAT recurrence is therefore:

```math
\left(
H_i^{(\ell)},
B_i^{(\ell)}
\right)
=
\mathcal{C}_{\theta}^{\mathrm{HEAT},\ell}
\left(
H_i^{(\ell-1)},
B_i^{(\ell-1)};
E_i,
\tau_i
\right),
```

where:

```math
B_i^{(\ell)}
=
\{b_{ie}^{(\ell)}:e\in E_i\}.
```

## Reduction Proposition

If there is only one node type:

```math
|\mathcal{T}|=1,
```

then all node-type-specific projections collapse to shared Q/K/V projections:

```math
W_{K,\tau_i(v)}^{m}
=
W_K^{m},
\qquad
W_{V,\tau_i(v)}^{m}
=
W_V^{m},
\qquad
W_{Q,\tau_i(v)}^{m}
=
W_Q^{m}.
```

If all edge gates are constant:

```math
g_{ie}^{m}
=
\mathbf{1},
```

then:

```math
a_{ist}^{m}
=
\frac{
k_{is}^{m}
\left(
q_{it}^{m}
\right)^\top
}{
\sqrt{d_h}
}.
```

Thus HEAT reduces to graph-supported multi-head attention over the constructed
edge set. In this limit, the heterogeneous pieces no longer contribute:

```text
node type no longer changes projections
edge attribute no longer modulates attention
```

This is GAT-like graph attention, using dot-product compatibility rather than
the original additive GAT score.

## What HEAT Adds Over Homogeneous Attention

A homogeneous graph attention layer can learn:

```text
which neighboring nodes matter
```

HEAT can additionally learn:

```text
which neighboring pseudo-label types matter
how the source and target pseudo-label types change projections
how edge attributes modulate compatibility
```

The inductive bias is:

```text
patch relations depend on teacher-derived pseudo-label type and encoder-space
edge evidence
```

## C/R/G/S Placement

```text
\mathcal{G}:
    E, tau, and phi define support, node type, and edge attribute

\mathcal{C}:
    HEAT updates node and possibly edge states by type-aware edge-modulated
    attention

\mathcal{R}:
    no graph-level readout happens inside HEAT

\mathcal{S}:
    slide labels do not enter inference-time message passing
```

## Surviving Statistic At This Stage

After `L` HEAT layers:

```math
\widetilde H_i
=
H_i^{(L)}.
```

Each node state has mixed information from:

```text
feature-neighbor nodes
source and target node pseudo-types
Pearson-derived edge attributes
incoming-edge attention weights
```

But the support is still limited to the constructed feature-kNN graph.

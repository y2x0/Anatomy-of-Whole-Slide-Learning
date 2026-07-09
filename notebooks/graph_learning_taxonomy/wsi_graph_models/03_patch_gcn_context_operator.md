# Patch-GCN Context Operator

Patch-GCN injects context with DeepGCN-style message passing over the coordinate
patch graph.

The context map is:

```math
H^{(0)}
\longmapsto
H^{(L)}.
```

For Patch-GCN:

```math
L=4.
```

## Generic Layer

For node `v` and spatial neighbors:

```math
u
\in
\mathcal{A}(v),
```

Patch-GCN writes each graph convolution layer with three functions:

```math
\phi^{(\ell)}:
\text{message construction},
```

```math
\rho^{(\ell)}:
\text{permutation-invariant aggregation},
```

```math
\zeta^{(\ell)}:
\text{node update}.
```

The layer has the form:

```math
m_v^{(\ell)}
=
\rho^{(\ell)}
\left(
\{
\phi^{(\ell)}
(h_v^{(\ell)},h_u^{(\ell)},h_{e_{vu}}^{(\ell)})
:
u\in\mathcal{A}(v)
\}
\right),
```

```math
h_v^{(\ell+1)}
=
\zeta^{(\ell)}
\left(
h_v^{(\ell)},
m_v^{(\ell)}
\right).
```

## Message Construction

The paper adapts DeepGCN message functions. The message from `u` to `v` is:

```math
m_{vu}^{(\ell)}
=
\mathrm{ReLU}
\left(
h_u^{(\ell)}
+
\mathbf{1}(h_{e_{vu}}^{(\ell)})
h_{e_{vu}}^{(\ell)}
\right)
+
\epsilon.
```

Here:

```math
\epsilon
=
10^{-7}.
```

The edge feature term appears only when an edge feature exists.

In the Patch-GCN WSI construction, the main support comes from coordinates. The
paper's equation is written generally enough to allow edge features, but the
central WSI graph object is the coordinate kNN adjacency.

## Softmax Aggregation

Patch-GCN uses a softmax aggregation scheme from DeepGCN:

```math
m_v^{(\ell)}
=
\sum_{u\in\mathcal{A}(v)}
a_{vu}^{(\ell)}
\odot
m_{vu}^{(\ell)}.
```

Because the paper's message `m_vu` is vector-valued and the exponential is
written directly on the message, the exact scalar-versus-feature-wise
interpretation is not fully specified. A faithful feature-wise reading is:

```math
a_{vu,r}^{(\ell)}
=
\frac{
\exp(\beta m_{vu,r}^{(\ell)})
}{
\sum_{s\in\mathcal{A}(v)}
\exp(\beta m_{vs,r}^{(\ell)})
},
```

```math
m_{v,r}^{(\ell)}
=
\sum_{u\in\mathcal{A}(v)}
a_{vu,r}^{(\ell)}
m_{vu,r}^{(\ell)}.
```

The paper sets:

```math
\beta
=
1.
```

This is attention-like, but it is not the same as a single scalar ABMIL
attention weight per neighbor. It is a softmax aggregation over local graph
messages, applied feature coordinate by feature coordinate.

If an implementation instead collapses each message to a scalar edge weight,
the operator is still local softmax message aggregation, but the surviving
statistic differs. The paper's equation alone should not be overread as one
specific implementation beyond ReLU plus Softmax Aggregation.

## Update

The update function is:

```math
h_v^{(\ell+1)}
=
\mathrm{MLP}
\left(
h_v^{(\ell)}
+
m_v^{(\ell)}
\right).
```

The self-state is therefore additively combined with the aggregated
neighborhood message before the MLP.

## Residual Mapping

The paper follows DeepGCN and makes the graph layer residual:

```math
G^{(\ell+1)}
=
\mathcal{F}_{\mathrm{GCN}}^{(\ell)}
\left(
G^{(\ell)}
\right)
+
G^{(\ell)}.
```

At the node level:

```math
H^{(\ell+1)}
=
\mathcal{F}_{\mathrm{GCN}}^{(\ell)}
\left(
H^{(\ell)},A
\right)
+
H^{(\ell)}.
```

This reduces the risk that repeated context aggregation erases the original
patch representation.

## Dense Connections

Patch-GCN also uses dense connections from every GCN layer to the final hidden
representation:

```math
H^{(L)}
=
\left[
X^{(1)},
\ldots,
X^{(L)}
\right].
```

This means the final node state is not merely the last smoothed state. It is a
concatenation of multiple context scales.

The displayed dense-connection notation starts at `X^(1)`. The surrounding
prose says the representation combines instance-level embedding and learned
context, but the paper does not explicitly display `X^(0)` in the concatenation.

## Receptive Field

The paper uses:

```math
L=4
```

graph convolutional layers.

Therefore a node state after context depends on a 4-hop graph neighborhood:

```math
h_v^{(4)}
=
F_{\theta}
\left(
G[\mathcal{B}_4(v)],
\{h_u^{(0)}:u\in\mathcal{B}_4(v)\}
\right).
```

For 256 by 256 patches with 8-nearest-neighbor support, the paper reports an
effective image receptive field of:

```math
2302
\times
2302.
```

This note preserves the paper's stated number. Simple grid arithmetic for a
`9 by 9` patch neighborhood would suggest `2304 by 2304`, so the exact value
should be treated as paper-reported rather than rederived here.

## What Survives

The context operator preserves:

```text
instance feature:
    original patch embedding path through residual/dense connections

local tissue context:
    messages from coordinate neighbors

multi-hop context:
    deeper graph neighborhoods up to 4 hops

multi-scale node statistic:
    concatenated outputs from multiple graph layers
```

The object reaching global pooling is:

```math
H^{(L)}
\in
\mathbb{R}^{M\times d_{\mathrm{out}}}.
```

It is a field of contextualized patch states.

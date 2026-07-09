# GAT As Masked Edge Attention

Graph Attention Networks replace fixed degree-based neighbor weights with
data-dependent edge weights. The support is still the graph neighborhood.

## Edge Scores

For node states:

```math
h_v
\in
\mathbb{R}^{d},
```

project features:

```math
\tilde h_v
=
h_v W.
```

For an edge from source `u` to target `v`, compute:

```math
e_{vu}
=
\mathrm{LeakyReLU}
\left(
a^\top
[\tilde h_v\Vert \tilde h_u]
\right).
```

The score is only used when:

```math
u
\in
\widetilde{\mathcal{A}}(v),
\qquad
\widetilde{\mathcal{A}}(v)
=
\mathcal{A}(v)\cup\{v\}.
```

This self-looped neighborhood matches the usual GAT convention: a node can
attend to itself as well as to its graph neighbors.

## Neighbor Softmax

Attention weights are:

```math
\alpha_{vu}
=
\frac{\exp(e_{vu})}
{\sum_{r\in\widetilde{\mathcal{A}}(v)}\exp(e_{vr})}.
```

Thus:

```math
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\alpha_{vu}
=
1.
```

The update is:

```math
h_v'
=
\sigma
\left(
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\alpha_{vu}h_u W
\right).
```

## Masked Self-Attention View

Let:

```math
M_{vu}
=
\mathbf{1}\{u\in\widetilde{\mathcal{A}}(v)\}.
```

Then GAT can be written:

```math
\alpha_{vu}
=
\frac{M_{vu}\exp(e_{vu})}
{\sum_r M_{vr}\exp(e_{vr})}.
```

This is graph-masked attention. The graph controls support, attention controls
relative weight inside support.

## Multi-Head Form

With `K` heads:

```math
h_v'
=
\mathrm{concat}_{k=1}^{K}
\sigma
\left(
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\alpha_{vu}^{(k)}
h_u W^{(k)}
\right)
```

for intermediate layers, or an average over heads in the final layer:

```math
h_v'
=
\sigma
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\alpha_{vu}^{(k)}
h_u W^{(k)}
\right).
```

The surviving node statistic is a collection of head-specific weighted local
moments.

## Difference From GCN

GCN uses structural weights:

```math
w_{vu}^{\mathrm{GCN}}
=
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}.
```

GAT uses feature-dependent weights:

```math
w_{vu}^{\mathrm{GAT}}
=
\alpha_{vu}(h_v,h_u).
```

GCN says:

```text
trust the normalized topology
```

GAT says:

```text
trust topology for support, learn relevance inside the neighborhood
```

## WSI Interpretation

For a patch graph, GAT can learn that not every spatial neighbor matters equally:

```text
tumor neighbor:
    high weight for tumor-context task

artifact neighbor:
    low weight if recognized as irrelevant

immune neighbor:
    high weight for interface-sensitive task
```

But it cannot attend to a missing long-range patch unless the graph support
includes that patch:

```math
M_{vu}=0
\quad
\Longrightarrow
\quad
\alpha_{vu}=0.
```

## Failure Modes

GAT fails when:

```text
the graph support is wrong:
    attention cannot recover missing edges

the attention score follows artifacts:
    edge weights become shortcut weights

neighborhoods are large:
    rare relevant neighbors compete in the softmax denominator

heads are redundant:
    multiple heads preserve the same local moment
```

## Dense Summary

GAT is:

```math
\boxed{
h_v'
=
\sigma
\left(
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\frac{\exp(e_{vu})}
{\sum_{r\in\widetilde{\mathcal{A}}(v)}\exp(e_{vr})}
h_u W
\right)
}
```

It is not graph construction. It is learned weighting over an existing
neighborhood.

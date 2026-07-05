# Readout And Surviving Statistics

After message passing:

```math
\widetilde{H}
=
\{h_v^{(L)}\}_{v\in V}.
```

A graph-level slide embedding requires readout:

```math
z=\mathcal{R}(G,\widetilde{H}).
```

## Mean Readout

```math
z=\frac{1}{|V|}\sum_{v\in V}h_v^{(L)}.
```

Survives:

```text
first moment of contextualized node features
```

The graph affects $h_v^{(L)}$, but the readout is still a mean.

## Attention Readout

```math
a_v
=
\frac{\exp(s_\theta(h_v^{(L)}))}
{\sum_{u\in V}\exp(s_\theta(h_u^{(L)}))},
\qquad
z=\sum_va_vh_v^{(L)}.
```

Survives:

```text
weighted first moment of contextualized node features
```

## Hierarchical Readout

Cluster nodes:

```math
V=\bigcup_{r=1}^{R}V_r.
```

Region embeddings:

```math
z_r=\mathcal{R}_{\operatorname{local}}(\{h_v^{(L)}:v\in V_r\}).
```

Slide embedding:

```math
z=\mathcal{R}_{\operatorname{global}}(\{z_r\}_{r=1}^{R}).
```

Survives:

```text
regional summaries plus global summary
```

## Graph-Level Pooling

Differentiable pooling learns assignment:

```math
S\in\mathbb{R}^{|V|\times R}.
```

Cluster features:

```math
H_{\operatorname{cluster}}=S^\top H.
```

Cluster adjacency:

```math
A_{\operatorname{cluster}}=S^\top AS.
```

This creates a coarsened graph.

## What Survives

The final statistic depends on both message passing and readout:

```math
z
=
\mathcal{R}
\left(
\operatorname{GNN}(H,A)
\right).
```

Graph context can encode pairwise interactions, but only interactions embedded
in node states or retained by graph pooling survive.

## Dense Summary

```text
GNN:
    makes node features contextual

readout:
    decides which contextual statistics become the slide

mean:
    first moment

attention:
    weighted first moment

hierarchical pooling:
    multiscale graph summary
```

A graph representation does not automatically preserve all topology. The
readout determines the final graph statistic.

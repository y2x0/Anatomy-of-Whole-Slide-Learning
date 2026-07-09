# C/R/G/S For Message Passing Foundations

This note places the first graph-learning portion inside the repository
decomposition:

```math
\widetilde H
=
\mathcal{C}(H;G),
\qquad
z
=
\mathcal{R}(\widetilde H;G),
\qquad
\widehat y
=
\mathcal{H}(z).
```

For this first portion, `S` is the training signal, not an inference-time input
to vanilla GCN, GraphSAGE, or GAT.

## Operator Slots

For message passing:

```math
\mathcal{C}
=
\Phi_{\theta}^{(L)}
\left(
\cdot;G
\right).
```

Graph structure:

```math
G
=
(V,E,A,E^{\mathrm{feat}}).
```

Readout:

```math
z
=
\mathcal{R}
\left(
\{h_v^{(L)}\}_{v\in V}
\right).
```

Supervision:

```math
S
=
\{y_i\}
```

for classification, or:

```math
S
=
\{t_i,\delta_i\}
```

for survival.

It enters through the objective:

```math
\mathcal{L}
\left(
\mathcal{H}(z_i),
S_i
\right),
```

unless a later method explicitly conditions the graph operator on supervision.

## Anchor Placement

| Method | Message | Aggregation | Support | Surviving node statistic |
|---|---|---|---|---|
| MPNN | learned edge message | invariant aggregate | graph neighbors | learned L-hop rooted statistic |
| GCN | projected source state with degree weight | symmetric normalized sum | self-loop graph neighborhood | degree-weighted smoothed neighborhood mixture |
| GraphSAGE mean | source state | sampled mean | sampled graph neighborhood | sampled local first moment plus self state |
| GAT | projected source state | attention-weighted sum | graph-masked neighborhood | data-dependent weighted local moment |

## What Changes Across Papers

GCN changes the weighting rule:

```math
w_{vu}
=
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}.
```

GraphSAGE changes the source set:

```math
\mathcal{A}(v)
\longrightarrow
S(v)
\subseteq
\mathcal{A}(v).
```

GAT changes the edge weight from structural to learned:

```math
w_{vu}
\longrightarrow
\alpha_{vu}(h_v,h_u).
```

## Failure Mode Matrix

| Choice | Bias | Failure mode |
|---|---|---|
| GCN normalized sum | connected neighbors should smooth | rare local evidence is smoothed away |
| GraphSAGE sampling | sampled neighbors estimate local distribution | rare tissue interfaces may be missed |
| GAT edge attention | relevant neighbors can be reweighted | wrong support cannot be fixed by attention |
| fixed graph support | messages only move along existing edges | missing long-range interactions cannot appear |
| deep message passing | larger L-hop context | oversmoothing and oversquashing |

## Design Rule For WSI Graphs

After the graph representation and graph construction are specified, choose the
graph-learning operator by specifying:

```text
message:
    what source information is transmitted?

aggregation:
    sum, mean, sampled mean, attention, max

update:
    overwrite, concatenate, residual, dense, gated

depth:
    how many graph hops can influence each node?

readout:
    what graph-level statistic survives?
```

The graph model is not just a layer choice. It is a biological claim about what
tissue units should communicate before the slide-level prediction is made.

## Dense Summary

The first portion gives this map:

```math
\boxed{
(H,G)
\xrightarrow{\mathrm{message\ passing}}
H^{(L)}
\xrightarrow{\mathrm{readout}}
z
\xrightarrow{\mathrm{head}}
\widehat y
}
```

Each anchor paper changes one precise part of the context operator. That is the
level of specificity future graph notes should preserve.

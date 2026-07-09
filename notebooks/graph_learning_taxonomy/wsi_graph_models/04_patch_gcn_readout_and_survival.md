# Patch-GCN Readout And Survival

Patch-GCN does not stop at node-level graph context. It compresses the
contextualized node field into a WSI-level embedding with global attention
pooling.

The pipeline is:

```math
(X,A)
\xrightarrow{\mathcal{C}_{\mathrm{PatchGCN}}}
H^{(L)}
\xrightarrow{\mathcal{R}_{\mathrm{AttnMIL}}}
z
\xrightarrow{\mathcal{H}_{\mathrm{surv}}}
\eta.
```

## Global Attention Pooling

The paper applies an attention-based MIL pooling layer to the penultimate node
feature matrix:

```math
H^{(L)}
=
\{h_v^{(L)}\}_{v=1}^{M}.
```

Write a generic attention score:

```math
s_v
=
f_{\theta}
\left(
h_v^{(L)}
\right).
```

The global attention weights are:

```math
b_v
=
\frac{\exp(s_v)}
{\sum_{u=1}^{M}\exp(s_u)}.
```

The WSI-level embedding is:

```math
z
=
\sum_{v=1}^{M}
b_v h_v^{(L)}.
```

This is the paper's global neighborhood aggregation step. It generalizes the
local graph aggregation idea to all nodes in the graph.

## Shape

The paper describes:

```math
H^{(L)}
\in
\mathbb{R}^{M\times d_{\mathrm{out}}}.
```

Global attention pooling maps:

```math
\mathbb{R}^{M\times d_{\mathrm{out}}}
\longrightarrow
\mathbb{R}^{1\times d_{\mathrm{out}}}.
```

Thus:

```math
z
\in
\mathbb{R}^{d_{\mathrm{out}}}.
```

## Survival Head

The WSI-level embedding is supervised using a Cox-style survival objective. A
minimal scalar-risk head is:

```math
\eta_i
=
w^\top z_i.
```

Then a Cox-style loss compares patient risks under censoring:

```math
\mathcal{L}_{\mathrm{surv}}
=
\mathcal{L}
\left(
\{\eta_i,T_i,C_i\}_{i=1}^{N}
\right).
```

The paper describes this as a cross-entropy-based Cox proportional loss. The
important graph-learning point is:

```text
survival supervision acts after graph context and global attention readout
```

not directly on patch nodes.

## Patient With Multiple WSIs

The paper allows:

```math
P_i
=
\{W_{ij}\}_{j=1}^{K_i}.
```

Each WSI has a subgraph:

```math
G_{ij}
=
(X_{ij},A_{ij}).
```

A patient-level risk must therefore be a function of all WSI graph outputs:

```math
\eta_i
=
\mathcal{H}
\left(
\mathcal{R}
\left(
\{\mathcal{C}(G_{ij})\}_{j=1}^{K_i}
\right)
\right).
```

The paper states the patient-level graph as a collection of subgraphs. It does
not require cross-slide edges. That is a design distinction:

```text
within-slide:
    coordinate graph message passing

across-slide:
    patient-level aggregation/supervision
```

The exact across-WSI aggregation rule is not given as a separate equation in
the paper. The safe statement is that each WSI yields a graph object and the
patient-level survival label supervises the resulting prediction.

## What Survives Readout

Before readout:

```math
H^{(L)}
=
\{h_v^{(L)}\}_{v=1}^{M}
```

is a contextualized patch field.

After readout:

```math
z
=
\sum_v b_v h_v^{(L)}
```

is one weighted first moment of contextualized graph-node features.

Therefore Patch-GCN has two aggregation stages:

```text
local graph aggregation:
    node neighborhoods become contextualized node states

global attention aggregation:
    contextualized node states become one WSI statistic
```

## Attention Heatmaps

The paper visualizes heatmaps from the global attention weights:

```math
\{b_v\}_{v=1}^{M}.
```

These weights indicate which contextualized node states contributed strongly to
the WSI embedding. They should not be confused with the local softmax
aggregation weights inside graph message passing:

```math
a_{vu,r}^{(\ell)}.
```

There are two different attention-like objects:

```text
local graph softmax aggregation:
    neighbor-to-node message weighting

global attention pooling:
    node-to-slide readout weighting
```

## Dense Summary

Patch-GCN survival prediction can be written:

```math
\boxed{
\eta_i
=
\mathcal{H}_{\mathrm{surv}}
\left(
\sum_{v=1}^{M_i}
b_{iv}
h_{iv}^{(L)}
\right)
}
```

where `h_iv^(L)` is already graph-contextualized by coordinate-neighborhood
message passing.

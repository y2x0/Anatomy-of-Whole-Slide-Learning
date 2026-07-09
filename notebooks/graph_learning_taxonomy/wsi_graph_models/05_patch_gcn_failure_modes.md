# Patch-GCN Failure Modes

Patch-GCN is powerful because it inserts spatial graph context before survival
readout. Its failure modes follow from that same structure.

## 1. Coordinate Support Is A Hard Constraint

The coordinate kNN graph defines:

```math
A_{vu}
=
1
\quad
\Longleftrightarrow
\quad
u\in\mathrm{kNN}_{8}(c_v).
```

The paper does not specify edge symmetrization, self-loops, edge direction, or a
radius cutoff. Those details affect the exact support but are not derivable from
the text alone.

If:

```math
A_{vu}=0,
```

then node `u` cannot directly send a message to node `v`.

Even if two distant patches have clinically related morphology:

```math
\mathrm{sim}(h_u,h_v)
\text{ high},
```

they do not exchange information unless a multi-hop path connects them within
the chosen depth.

Failure mode:

```text
global repeated morphology may be missed by local coordinate support
```

## 2. Four-Hop Context Is Still Finite

Patch-GCN uses:

```math
L=4.
```

So:

```math
h_v^{(4)}
```

depends on:

```math
\mathcal{B}_4(v)
=
\{u:d_G(u,v)\le 4\}.
```

If a prognostic relation requires:

```math
d_G(u,v)>4,
```

then it cannot be represented inside a single node state before global pooling.

The global readout may still combine both regions, but the local interaction is
not explicitly contextualized.

## 3. Softmax Aggregation Can Select Or Dilute Messages

The local aggregation is:

```math
m_{v,r}^{(\ell)}
=
\sum_{u\in\mathcal{A}(v)}
a_{vu,r}^{(\ell)}
m_{vu,r}^{(\ell)}.
```

The weights are:

```math
a_{vu,r}^{(\ell)}
=
\frac{
\exp(\beta m_{vu,r}^{(\ell)})
}{
\sum_{s\in\mathcal{A}(v)}
\exp(\beta m_{vs,r}^{(\ell)})
}.
```

If one neighbor dominates a coordinate:

```math
m_{vu,r}^{(\ell)}
\gg
m_{vs,r}^{(\ell)}
\quad
s\ne u,
```

then that coordinate becomes locally winner-take-most.

If relevant neighbors have weak separation:

```math
m_{vu,r}^{(\ell)}
\approx
m_{vs,r}^{(\ell)},
```

then important local evidence can be diluted across the neighborhood.

## 4. Dense Connections Preserve More Than The Last Layer

Patch-GCN forms:

```math
H^{(L)}
=
[X^{(1)},\ldots,X^{(L)}].
```

This helps retain multi-scale information, but it also means the final node
state is a concatenated summary, not a clean single-scale graph statistic.

Failure mode:

```text
the readout may rely on shallow instance-level features even when the graph
context operator exists
```

This is not automatically bad. It means performance gains should not be
attributed only to the deepest 4-hop representation without checking what the
attention readout uses.

## 5. Global Attention Is A Bottleneck

The graph context field is:

```math
H^{(L)}
=
\{h_v^{(L)}\}_{v=1}^{M}.
```

The WSI embedding is:

```math
z
=
\sum_v b_v h_v^{(L)}.
```

Thus the final slide statistic is still one weighted first moment of
contextualized node states.

Failure mode:

```text
multi-region mechanisms may be compressed into one weighted average
```

Graph context helps create better node states, but the readout can still erase
multimodal or distributional structure.

This is also why a cancer type driven by broad global morphology can be harder
for a finite-depth local graph model. The paper reports that Patch-GCN wins on
four of five cancer types, while DeepGraphConv performs better on UCEC.

## 6. Attention Heatmaps Are Not The Whole Explanation

Patch-GCN has at least two weight systems:

```text
local graph softmax weights:
    a_vu,r^(ell)

global attention pooling weights:
    b_v
```

A high global attention patch means:

```text
this contextualized node state contributed strongly to z
```

It does not by itself identify:

```text
which local neighbor messages made the node prognostic
which tissue interaction caused the risk prediction
whether the highlighted patch is causally relevant
```

To explain a prediction, one would need to inspect the path:

```math
h_u^{(0)}
\to
m_{vu}^{(\ell)}
\to
h_v^{(L)}
\to
b_v
\to
\eta.
```

## 7. Survival Compression

With a scalar Cox-style head:

```math
\eta_i
=
w^\top z_i,
```

all graph-contextualized morphology is compressed into one risk ordering.

This is appropriate for proportional-risk ranking, but it does not directly
return:

```text
time-varying hazard
cause-specific risk
calibrated survival curve
multiple risk mechanisms
```

## Dense Summary

Patch-GCN's inductive bias is:

```text
survival-relevant morphology is partly local, spatial, and recoverable through
coordinate-neighborhood message passing before global attention readout.
```

The corresponding failure mode is:

```text
wrong support, finite context depth, softmax message competition, and final
attention pooling can each erase a different part of the survival signal.
```

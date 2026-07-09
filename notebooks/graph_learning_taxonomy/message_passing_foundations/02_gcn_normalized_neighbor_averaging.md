# GCN As Degree-Normalized Neighbor Smoothing

Kipf and Welling's GCN can be read as a particular message-passing operator:
linear value projection followed by normalized neighbor summation.

## Propagation Rule

Let:

```math
A
\in
\{0,1\}^{n\times n}
```

be the adjacency matrix. Add self-loops:

For this Kipf-Welling form, assume `A` is undirected or has been symmetrized.
With the row-target convention:

```math
A_{vu}=1
\quad
\Longleftrightarrow
\quad
u\to v.
```

```math
\widetilde A
=
A+I.
```

Let:

```math
\widetilde D_{vv}
=
\sum_u \widetilde A_{vu}.
```

The GCN layer is:

```math
H^{(\ell+1)}
=
\sigma
\left(
\widetilde D^{-1/2}
\widetilde A
\widetilde D^{-1/2}
H^{(\ell)}
W^{(\ell)}
\right).
```

Define:

```math
\widehat A
=
\widetilde D^{-1/2}
\widetilde A
\widetilde D^{-1/2}.
```

Then:

```math
H^{(\ell+1)}
=
\sigma
\left(
\widehat A H^{(\ell)}W^{(\ell)}
\right).
```

## Nodewise Form

For node `v`:

```math
h_v^{(\ell+1)}
=
\sigma
\left(
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}
h_u^{(\ell)}
W^{(\ell)}
\right).
```

Thus the message is:

```math
m_{vu}^{(\ell)}
=
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}
h_u^{(\ell)}
W^{(\ell)}.
```

The aggregation is a degree-normalized sum:

```math
\bar m_v^{(\ell)}
=
\sum_{u\in\widetilde{\mathcal{A}}(v)}
m_{vu}^{(\ell)}.
```

The update is:

```math
h_v^{(\ell+1)}
=
\sigma
\left(
\bar m_v^{(\ell)}
\right).
```

## What The Normalization Does

The coefficient:

```math
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}
```

is not learned. It is a structural weight determined by graph degree.

High-degree sources and high-degree targets receive smaller pairwise weights.
This prevents raw degree from dominating the sum, but it also imposes a specific
belief:

```text
neighbor importance is mostly degree-normalized topology
not patch-specific learned relevance
```

## Smoothing View

Ignoring nonlinearities and weights:

```math
H^{(\ell+1)}
=
\widehat A H^{(\ell)}.
```

After `L` layers:

```math
H^{(L)}
=
\widehat A^L H^{(0)}.
```

This repeatedly mixes node states over graph neighborhoods using symmetric
degree weights. It is a low-pass operation on the graph: nearby connected nodes
become more similar.

With trainable weights:

```math
H^{(L)}
\approx
\widehat A^L H^{(0)}
W_{\mathrm{eff}}
```

as a schematic linearized picture. The key is that topology-driven smoothing
remains built into the operator.

## WSI Interpretation

If a WSI patch graph connects spatial neighbors, then one GCN layer mixes patch
features with immediate tissue context:

```text
tumor patch:
    receives nearby stromal, immune, necrotic, or background signals

immune patch:
    receives nearby tumor interface signals

artifact patch:
    may contaminate nearby tissue states if connected
```

The inductive bias is useful when the label depends on local tissue
neighborhoods. It is dangerous when edges connect regions that should not share
evidence.

## WSI Graph Bridge

Generic WSI graph survival modeling can be placed as:

```math
H_i^{(0)}
=
\mathrm{PatchEncoder}(X_i),
\qquad
G_i
=
\mathrm{kNN}(C_i),
```

```math
H_i^{(L)}
=
\mathrm{GraphContext}_{\theta}
\left(
H_i^{(0)},G_i
\right),
\qquad
z_i
=
\mathcal{R}_{\theta}
\left(
H_i^{(L)}
\right),
```

```math
\eta_i
=
w^\top z_i.
```

The survival head sees graph-contextualized patch states, not isolated patch
states.

## Failure Modes

GCN fails when:

```text
edges are wrong:
    smoothing spreads evidence across false neighbors

rare signal is local:
    smoothing can dilute small but important structures

depth is too large:
    node states oversmooth

degree is confounded:
    dense tissue areas get different normalization than sparse areas
```

## Dense Summary

GCN is:

```math
\boxed{
h_v'
=
\sigma
\left(
\sum_{u\in\widetilde{\mathcal{A}}(v)}
\frac{1}
{\sqrt{\widetilde d_v\widetilde d_u}}
h_u W
\right)
}
```

The surviving statistic at each node is a degree-weighted neighborhood mixture
in a learned feature space. It is smoothing, not a row-stochastic mean.

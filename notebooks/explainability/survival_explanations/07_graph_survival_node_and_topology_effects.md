# Graph Survival Node and Topology Effects

## 1. Graph Cox Readout

For a Patch-GCN-style graph,

```math
H_i,G_i
\xrightarrow{\mathcal C}
\widetilde H_i
\xrightarrow{\mathcal R}
z_i
\xrightarrow{w^{\top}}
\eta_i.
```

The graph may encode spatial patch adjacency while the final head is a scalar
Cox risk score.

## 2. Node Deletion

The risk effect of node `v` is

```math
\Delta_{iv}^{\eta}
=\eta_i(G_i)-\eta_i(G_i\setminus v).
```

The survival effect at a fixed horizon is

```math
\Delta_{iv}^{S}(t)
=S_i(t)-S_i^{(-v)}(t).
```

These have the same intervention but different target scales.

## 3. Topology Swap

Compare graph supports while holding node features fixed:

```math
\Delta_{A\to A'}^{\eta}
=\eta_i(X_i,A)-\eta_i(X_i,A').
```

This tests whether the learned survival model relies on spatial adjacency,
feature similarity, or a dynamic topology hypothesis.

## 4. Causal Limit

Deleting a graph node can disconnect neighborhoods and alter normalization. A
large risk change proves computational dependence under the graph intervention,
not that the tissue region biologically causes the patient's outcome.


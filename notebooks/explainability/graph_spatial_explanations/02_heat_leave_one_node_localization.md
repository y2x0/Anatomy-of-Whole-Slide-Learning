# HEAT Leave-One-Node Localization

HEAT uses a node-removal loss contrast inspired by Granger-style localization.

## 1. Full and Deleted Graphs

Let `M_theta` be the trained graph model and `L` the supervised loss. For node
`v`, hold the constructed graph fixed except for removing the node and incident
edges:

```math
\Delta_{iv}
=\mathcal L(y_i,M_\theta(G_i))
-\mathcal L(y_i,M_\theta(G_i\setminus\{v\})).
```

If deleting `v` increases loss, then `Delta_iv` is negative under this paper's
sign convention. A helpful-node plotting score is

```math
I_{iv}^{\mathrm{help}}
=-\Delta_{iv}.
```

## 2. What Changes on Deletion

Removing a heterogeneous node changes:

```text
its node state;
incoming messages;
outgoing messages;
edge normalization;
pseudo-label pooling counts;
the graph-level readout.
```

Thus the score is a full model perturbation, not a local coefficient.

## 3. Label Dependence

Because the contrast uses `y_i`, the same node can receive different scores for
different tasks or class losses. It is not an input-only saliency map.

## 4. Resolution

The score localizes graph nodes. If a node is a patch containing many nuclei,
the heatmap does not identify which nucleus drove the prediction. If nodes are
cell types or regions, the resolution changes with the graph construction.

## 5. Causality Boundary

Node deletion measures causal dependence of the trained computation under the
specified graph intervention. It does not establish biological causality. The
intervention changes the observed graph and can create an out-of-distribution
input, especially when the removed node is a structural bridge.


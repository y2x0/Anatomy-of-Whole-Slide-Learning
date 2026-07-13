# Node Deletion: Fixed Graph Versus Rebuilt Graph

## 1. Two Operators

For a graph built by a constructor

```math
G=\mathcal G(X,C),
```

fixed-graph deletion is

```math
G_{-v}^{\mathrm{fixed}}
=G\setminus\{v\}.
```

Rebuilt deletion removes the node and reruns construction:

```math
G_{-v}^{\mathrm{rebuild}}
=\mathcal G(X_{-v},C_{-v}).
```

The corresponding effects are

```math
\delta_v^{\mathrm{fixed}}
=F(G)-F(G_{-v}^{\mathrm{fixed}}),
```

```math
\delta_v^{\mathrm{rebuild}}
=F(G)-F(G_{-v}^{\mathrm{rebuild}}).
```

They answer different questions.

## 2. Interpretation

Fixed deletion asks:

```text
How much does this node matter in the graph that was actually constructed?
```

Rebuilt deletion asks:

```text
How much does the entire pipeline change if this node is absent before graph
construction?
```

For feature-space kNN, removing one point can change many other neighbor sets.
For region adjacency, it can change only local boundary edges. For dynamic WiKG
topology, the learned support can change globally.

## 3. No Universal Correct Choice

The fixed operator isolates the trained graph's node dependence but can leave
neighbor relations that would not exist in the reduced bag. The rebuilt operator
is closer to a pipeline intervention but mixes node importance with topology
instability.

## 4. Reporting Requirement

Every graph deletion result should state:

```text
whether adjacency was frozen; whether edge attributes were recomputed;
whether normalization was rerun; whether pooling counts changed; and whether
the graph model was evaluated on the reduced graph without retraining.
```


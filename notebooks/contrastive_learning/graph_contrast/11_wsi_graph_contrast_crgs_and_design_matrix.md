# WSI Graph Contrast: C/R/G/S And Design Matrix

No pathology-specific objective is attributed to DGI, GraphCL, or GRACE here.
This note derives what their operators would imply when the graph nodes are WSI
patches or tissue regions.

## WSI Graph

For slide `i`:

```math
G_i
=
\left(
V_i,E_i,X_i,C_i
\right),
```

where patch features and coordinates are:

```math
X_i
\in
\mathbb{R}^{n_i\times d},
\qquad
C_i
\in
\mathbb{R}^{n_i\times 2}.
```

Adjacency may be spatial kNN, radius based, feature based, or learned. The
contrastive method inherits every assumption used to construct `E_i`.

## C/R/G/S Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|
| DGI | message passing on original and corrupted graph | sigmoid of mean node embedding | bilinear BCE; joint versus corruption law | local node belongs with positive slide summary |
| GraphCL | two graph augmentations plus shared GNN | graph-level pooling then MLP projection | temperature-scaled cosine graph discrimination | two views share source slide identity |
| GRACE | edge removal, shared-coordinate masking, shared GNN | node embeddings then MLP projection | symmetric node softmax with inter- and intra-view negatives | same patch index across views |

## WSI Surviving Statistics

DGI preserves local representations useful for distinguishing:

```math
(h_{ij},s_i)
\quad\text{from}\quad
(\widetilde h_{ik},s_i).
```

GraphCL preserves one pooled graph statistic:

```math
z_i
=
g
\left(
\mathcal{R}_{G}
\left(
\{h_{ij}\}_{j=1}^{n_i}
\right)
\right).
```

GRACE preserves patch identity under corrupted neighborhood and feature
context:

```math
h_{ij}^{(1)}
\approx
h_{ij}^{(2)}.
```

## Rare-Positive Stress Test

Let tumor nodes be:

```math
T_i
=
\left\{
j:y_{ij}^{\mathrm{tumor}}=1
\right\}.
```

Node or subgraph dropping can produce:

```math
T_i
\cap
V_i^{(a)}
=
\varnothing.
```

GraphCL still labels that view positive with a tumor-containing view. DGI mean
summary weights the tumor contribution by `|T_i|/n_i`. GRACE may preserve a
tumor node but remove the boundary context that makes it diagnostic.

## Geometry-Preserving Augmentation Criterion

For edge set generated from coordinates:

```math
E_i
=
\Gamma(C_i),
```

a view transformation should ideally commute with graph construction:

```math
\widetilde E_i
\approx
\Gamma
\left(
\widetilde C_i
\right).
```

Randomly editing adjacency while leaving coordinates fixed can violate this
consistency:

```math
\widetilde E_i
\ne
\Gamma(C_i).
```

The model is then trained on graph states impossible under its own spatial
construction rule.

## Complexity Boundary

For slide graph size `n_i`, message passing with `|E_i|` edges costs roughly:

```math
\Theta
\left(
L|E_i|d
\right).
```

Full GRACE node contrast adds:

```math
\Theta
\left(
n_i^2d
\right).
```

GraphCL graph-level batch contrast instead adds:

```math
\Theta
\left(
B^2d
\right).
```

DGI adds local-summary scores:

```math
\Theta
\left(
n_i d^2
\right)
```

for dense bilinear `W`, reducible with structured scoring. The contrast scale
therefore determines feasibility independently of the GNN itself.

## Design Matrix

| Desired invariance | Plausible relation | Principal risk |
|---|---|---|
| robustness to missing spatial edges | same node across edge-dropped views | erase boundary architecture or isolate low-degree nodes |
| robustness to missing feature channels | same node across feature-masked views | suppress a rare diagnostic feature dimension |
| slide identity under partial sampling | same slide across graph views | sparse tumor omitted from one view |
| topology-feature dependence | positive local-summary versus shuffled-feature local-summary | solve via corruption artifact or dominant tissue mode |
| morphologic equivalence across slides | supervised or mined multi-positive graph relation | noisy positives and cohort shortcuts |

## Failure Principle

For graph contrast, the learned invariance is the composition:

```math
\text{graph construction}
\longrightarrow
\text{view kernel}
\longrightarrow
\text{message passing}
\longrightarrow
\text{readout}
\longrightarrow
\text{contrast relation}.
```

An error in an earlier operator cannot be repaired by a later contrastive loss.
The loss can only make the chosen graph relation more faithfully represented.

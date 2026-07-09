# C/R/G/S: Patch-GCN Versus HACT

Patch-GCN and HACT are both graph methods, but they do not instantiate the same
mathematical object.

Patch-GCN builds one coordinate graph over patches. HACT builds a two-level
typed hierarchy over cells and tissue regions.

## Patch-GCN

Patch-GCN starts with patch embeddings:

```math
H_i^{\mathrm{patch},0}
\in
\mathbb{R}^{N_i^{\mathrm{patch}}\times d}.
```

The graph is coordinate kNN:

```math
A_i^{\mathrm{patch}}
=
\Gamma_{\mathrm{kNN}}
\left(
\left\{
c_{ia}^{\mathrm{patch}}
\right\}_{a=1}^{N_i^{\mathrm{patch}}}
\right).
```

The context operator is graph convolution:

```math
\widetilde H_i^{\mathrm{patch}}
=
\mathcal{C}_{\theta}^{\mathrm{PatchGCN}}
\left(
H_i^{\mathrm{patch},0};
A_i^{\mathrm{patch}}
\right).
```

The readout is attention MIL over contextualized graph nodes:

```math
z_i
=
\sum_{a=1}^{N_i^{\mathrm{patch}}}
\alpha_{ia}
\widetilde h_{ia}^{\mathrm{patch}}.
```

The supervision is survival:

```math
\eta_i
=
z_i w,
```

trained with a Cox-style objective in the paper.

Thus:

```text
G:
    coordinate kNN patch graph

C:
    local graph message passing over patch nodes

R:
    global attention pooling over contextualized patches

S:
    survival time and censoring
```

## HACT

HACT starts with a typed hierarchical object:

```math
G_i^{\mathrm{HACT}}
=
\left\{
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right\}.
```

The cell graph is:

```math
G_i^{\mathrm{cell}}
=
\left(
H_i^{\mathrm{cell}},
A_i^{\mathrm{cell}}
\right),
```

where nodes are nuclei and edges are thresholded spatial kNN.

The tissue graph is:

```math
G_i^{\mathrm{tissue}}
=
\left(
H_i^{\mathrm{tissue}},
A_i^{\mathrm{tissue}}
\right),
```

where nodes are tissue regions and edges come from region adjacency.

The hierarchy is:

```math
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\in
\{0,1\}^{N_i^{\mathrm{cell}}\times N_i^{\mathrm{tissue}}}.
```

The context operator is sequential:

```math
\widetilde H_i^{\mathrm{cell}}
=
\mathcal{C}_{\theta}^{\mathrm{cell}}
\left(
H_i^{\mathrm{cell}};
A_i^{\mathrm{cell}}
\right),
```

```math
H_i^{\mathrm{tissue},0}
=
\left[
H_i^{\mathrm{tissue}}
\;\middle|\;
\left(B_i^{\mathrm{cell}\to\mathrm{tissue}}\right)^\top
\widetilde H_i^{\mathrm{cell}}
\right],
```

```math
\widetilde H_i^{\mathrm{tissue}}
=
\mathcal{C}_{\theta}^{\mathrm{tissue}}
\left(
H_i^{\mathrm{tissue},0};
A_i^{\mathrm{tissue}}
\right).
```

The readout is over tissue nodes:

```math
z_i
=
\sum_{r\in V_i^{\mathrm{tissue}}}
\widetilde h_{ir}^{\mathrm{tissue}},
```

or, in the short HACT-Net formulation, a concatenation of layerwise tissue-node
sum readouts.

The supervision is region classification:

```math
\widehat p_i
=
\mathrm{softmax}
\left(
z_i W_{\mathrm{cls}}
+
b_{\mathrm{cls}}
\right).
```

Thus:

```text
G:
    cell graph, tissue graph, hard cell-to-tissue assignment

C:
    cell GNN -> assignment pooling -> tissue GNN

R:
    sum or layerwise sum-concat over tissue nodes

S:
    TRoI subtype label
```

## Same Word, Different Geometry

Both methods are graph learning methods. But the word graph hides two different
geometries:

```text
Patch-GCN:
    graph over same-type patch nodes

HACT:
    typed hierarchy over cells and tissue regions
```

Patch-GCN context is local patch-to-patch propagation:

```math
h_a^{\mathrm{patch}}
\leftarrow
\left\{
h_b^{\mathrm{patch}}
:
b\in\mathcal{N}_{\mathrm{patch}}(a)
\right\}.
```

HACT context is first local cell-to-cell propagation:

```math
h_a^{\mathrm{cell}}
\leftarrow
\left\{
h_b^{\mathrm{cell}}
:
b\in\mathcal{N}_{\mathrm{cell}}(a)
\right\},
```

then coarse tissue-to-tissue propagation:

```math
h_r^{\mathrm{tissue}}
\leftarrow
\left\{
h_s^{\mathrm{tissue}}
:
s\in\mathcal{N}_{\mathrm{tissue}}(r)
\right\},
```

with a hard transfer between them:

```math
h_r^{\mathrm{tissue},0}
\leftarrow
\sum_{a:B_{ar}=1}
\widetilde h_a^{\mathrm{cell}}.
```

## Comparison Matrix

```text
axis:
    Patch-GCN
    HACT

node ontology:
    patch nodes
    typed cell nodes and tissue-region nodes

edge support:
    coordinate kNN over patches
    thresholded cell kNN plus tissue region adjacency graph

hierarchy operator:
    none
    hard cell-to-tissue assignment B

context operator:
    local patch graph convolution
    cell GNN, assignment transfer, tissue GNN

main additive interface:
    attention-weighted patch readout
    B^T cell transfer and tissue-node readout

equivariance:
    permutation equivariant over patch-node indexing for fixed graph
    permutation equivariant separately over cell and tissue-node indexing,
    with assignment matrix transformed consistently

bottleneck:
    attention readout compresses contextual patch graph to one vector
    B^T H_cell compresses all assigned cells to tissue-region sums

label scope:
    patient-level survival in the Patch-GCN paper
    TRoI-level classification in HACT papers
```

## Surviving Statistic

Patch-GCN's final statistic is:

```math
z_i^{\mathrm{PatchGCN}}
=
\sum_a
\alpha_{ia}
\widetilde h_{ia}^{\mathrm{patch}}.
```

A weighted first moment over contextualized patch nodes survives.

HACT's final statistic is:

```math
z_i^{\mathrm{HACT}}
=
\sum_r
\widetilde h_{ir}^{\mathrm{tissue}}.
```

But each tissue state already contains:

```math
\sum_{a:B_{ar}=1}
\widetilde h_{ia}^{\mathrm{cell}}.
```

So HACT contains additive bottlenecks at two interfaces:

```math
\text{tissue-node readout}
\circ
\text{tissue context operator}
\circ
\text{cell-to-tissue sum transfer}
\circ
\text{cell context operator}.
```

The full 2022 HACT forward map is not merely additive, because PNA,
normalization, and LSTM jumping knowledge sit inside the context operators.
The additive parts are still the important compression interfaces.

## Failure Boundary

Patch-GCN fails when:

```text
patch coordinate kNN is the wrong support
local patch message passing misses larger structures
attention readout collapses graph structure into a bag statistic
```

HACT fails when:

```text
cell or tissue segmentation is wrong
hard assignment sends cell information to the wrong tissue region
sum transfer erases within-region cellular distributions
TRoI labels do not teach patient-level WSI aggregation
```

The design question is therefore:

```text
Do we believe the task is governed by spatially contextualized patches,
or by interactions among typed biological entities across scales?
```

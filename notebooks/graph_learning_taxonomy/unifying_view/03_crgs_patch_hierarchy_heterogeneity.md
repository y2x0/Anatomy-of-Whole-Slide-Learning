# C/R/G/S: Patch Graphs, Hierarchies, And Heterogeneity

The first three WSI graph anchors differ in the object they call a graph.

```text
Patch-GCN:
    one node type, coordinate graph over patches

HACT:
    two entity levels, cell graph plus tissue graph plus assignment

Chan et al. HEAT:
    feature-kNN patch graph with node pseudo-types and continuous edge
    attributes
```

## Graph Object

Patch-GCN:

```math
G_i^{\mathrm{PatchGCN}}
=
\left(
H_i^{\mathrm{patch}},
A_i^{\mathrm{coord}}
\right).
```

HACT:

```math
G_i^{\mathrm{HACT}}
=
\left(
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right).
```

HEAT:

```math
G_i^{\mathrm{HEAT}}
=
\left(
H_i^{\mathrm{patch}},
B_i^{\mathrm{edge},0},
E_i^{\mathrm{feature}},
\tau_i,
\phi_i
\right).
```

The three graphs are not interchangeable.

## Geometry

Patch-GCN uses coordinate support:

```math
(s,t)\in E_i
\quad
\Longleftrightarrow
\quad
s
\text{ is coordinate-near }t.
```

HACT uses two supports:

```text
cell graph:
    thresholded kNN over nuclei centroids

tissue graph:
    region adjacency graph
```

HEAT uses feature support:

```math
(u,v)\in E_i
\quad
\Longleftrightarrow
\quad
u\in\mathcal{K}_k(v).
```

Thus, graph edges mean different things:

```text
Patch-GCN:
    nearby patch regions

HACT:
    nearby cells or adjacent tissue regions

HEAT:
    feature-similar patch embeddings under the encoder
```

## Context Operator

Patch-GCN:

```math
\widetilde H_i
=
\mathcal{C}_{\theta}^{\mathrm{PatchGCN}}
\left(
H_i^{\mathrm{patch}};
A_i^{\mathrm{coord}}
\right).
```

HACT:

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

HEAT:

```math
\widetilde H_i
=
\mathcal{C}_{\theta}^{\mathrm{HEAT}}
\left(
H_i^{\mathrm{patch}},
B_i^{\mathrm{edge},0};
E_i^{\mathrm{feature}},
\tau_i
\right).
```

HEAT changes the attention kernel:

```text
source node type chooses key/value projection
target node type chooses query projection
edge attribute modulates compatibility
```

## Readout

Patch-GCN uses attention pooling:

```math
z_i
=
\sum_v
\alpha_{iv}
\widetilde h_{iv}.
```

HACT uses tissue-node readout:

```math
z_i
=
\sum_{r\in V_i^{\mathrm{tissue}}}
\widetilde h_{ir}^{\mathrm{tissue}}
```

or layerwise tissue-node sum concatenation, depending on source version.

HEAT uses pseudo-label pooling:

```math
z_{ia}
=
\mathcal{R}_a
\left(
\{\widetilde h_{iv}:\tau_i(v)=a\}
\right),
```

then:

```math
Z_i^{\mathrm{PL}}
=
\left[
z_{ia}
\right]_{a\in\mathcal{T}},
```

then:

```math
z_i
=
\mathcal{R}_{\mathrm{graph}}
\left(
Z_i^{\mathrm{PL}}
\right).
```

So the surviving statistics are:

```text
Patch-GCN:
    weighted first moment over patch nodes

HACT:
    tissue-level graph statistic after cell-to-tissue transfer

HEAT:
    type-wise graph statistic over pseudo-label clusters
```

## Bottlenecks

Patch-GCN bottleneck:

```math
\{\widetilde h_{iv}\}_{v\in V_i}
\longmapsto
\sum_v
\alpha_{iv}
\widetilde h_{iv}.
```

HACT bottleneck:

```math
\widetilde H_i^{\mathrm{cell}}
\longmapsto
\left(B_i^{\mathrm{cell}\to\mathrm{tissue}}\right)^\top
\widetilde H_i^{\mathrm{cell}}.
```

HEAT bottleneck:

```math
\{\widetilde h_{iv}:v\in V_i\}
\longmapsto
Z_i^{\mathrm{PL}}.
```

In HEAT, all nodes of the same pseudo-label type are compressed into one
type-level vector.

## Comparison Matrix

```text
axis:
    Patch-GCN
    HACT
    HEAT

node ontology:
    patch
    cell and tissue region
    patch with pseudo-label type

edge support:
    coordinate kNN
    cell spatial kNN and tissue RAG
    feature-space kNN

heterogeneity:
    none in node type
    entity-level hierarchy
    node pseudo-types and edge attributes

context:
    graph convolution
    cell GNN -> assignment transfer -> tissue GNN
    node-type and edge-attribute transformer

readout:
    attention over patch nodes
    tissue-node sum or layerwise sum-concat
    pseudo-label type pooling

main failure:
    coordinate support and attention collapse
    segmentation and assignment bottleneck
    pseudo-label noise and feature-kNN support

label scope:
    patient survival
    TRoI classification
    WSI classification, staging, typing
```

## Design Lesson

Adding graph learning is not one design choice. It splits into at least four:

```text
what is a node?
what creates an edge?
what heterogeneity is exposed?
what statistic survives readout?
```

These choices define the model more deeply than the word `GNN`.

# Graph Learning Taxonomy

This notebook family asks:

```text
How does information move through a graph?
```

The slide representation notes ask what it means to treat a WSI as a graph.
The geometry notes ask where the edges come from. This family asks what a graph
learning operator does after the graph exists.

The first portion covers message passing foundations. The second uses
Patch-GCN as an anchor WSI-specific graph learning model. The third uses HACT
as an anchor hierarchical entity-graph model. The fourth uses Chan et al.'s
HEAT framework as an anchor heterogeneous WSI graph model.

```text
message_passing_foundations/
    generic MPNN update
    GCN degree-normalized neighbor smoothing
    GraphSAGE sample-and-aggregate
    GAT masked edge attention

wsi_graph_models/
    Patch-GCN slide object
    coordinate kNN patch graph construction
    DeepGCN-style context operator
    global attention readout and Cox-style survival supervision
    Patch-GCN failure modes

hierarchical_graph_models/
    HACT cell graph + tissue graph object
    cell graph and tissue graph construction
    hard cell-to-tissue assignment
    GIN/PNA hierarchical context operators
    tissue-node readout and classification supervision
    HACT failure modes

heterogeneous_graph_models/
    heterogeneous WSI graph object
    HoVer-Net pseudo-label node types
    KimiaNet features and feature-space kNN edges
    Pearson edge attributes
    HEAT context operator
    pseudo-label graph pooling
    causal localization and failure modes
    full HEAT forward map

unifying_view/
    C/R/G/S placement for this first portion
    C/R/G/S comparison for Patch-GCN versus HACT
    C/R/G/S comparison for Patch-GCN, HACT, and HEAT
```

## Core Object

For slide `i`, let:

```math
G_i
=
(V_i,E_i),
\qquad
H_i^{(0)}
=
\{h_{iv}^{(0)}\}_{v\in V_i}.
```

Throughout this first portion, matrix formulas store node states as rows:

```math
H_{v:}
=
h_v.
```

Nodewise formulas use the same row-vector convention, so feature projections
are written:

```math
h_v W.
```

A graph learning layer maps:

```math
\Phi_\theta:
(G_i,H_i^{(\ell)})
\longmapsto
H_i^{(\ell+1)}.
```

After `L` layers:

```math
\widetilde H_i
=
H_i^{(L)}.
```

The graph-level slide statistic is then:

```math
z_i
=
\mathcal{R}_{\theta}
\left(
\widetilde H_i
\right).
```

## What This Family Will Separate

```text
graph representation:
    what object is the slide?

graph construction:
    which edges exist?

graph learning:
    how do node states move along edges?

graph readout:
    what graph statistic survives?
```

## Anchor Papers For This Portion

Only papers whose forward maps are explicitly derived in this family are listed
here.

- Gilmer et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
  General message passing template.
  https://arxiv.org/abs/1704.01212
- Kipf and Welling. "Semi-Supervised Classification with Graph Convolutional
  Networks." ICLR 2017. Normalized graph convolution.
  https://arxiv.org/abs/1609.02907
- Hamilton, Ying, and Leskovec. "Inductive Representation Learning on Large
  Graphs." NeurIPS 2017. GraphSAGE sample-and-aggregate.
  https://arxiv.org/abs/1706.02216
- Velickovic et al. "Graph Attention Networks." ICLR 2018. Masked
  self-attention over graph neighborhoods.
  https://arxiv.org/abs/1710.10903
- Chen et al. "Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction using Patch-based Graph Convolutional Networks." MICCAI 2021.
  Coordinate WSI patch graph, DeepGCN-style local context, global attention
  pooling, and Cox-style survival supervision.
  https://arxiv.org/abs/2107.13048
- Pati et al. "HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for
  Histopathological Image Classification." arXiv 2020. Cell graph, tissue
  graph, hard cell-to-tissue assignment, GIN-style hierarchical graph learning.
  https://arxiv.org/abs/2007.00584
- Pati et al. "Hierarchical graph representations in digital pathology."
  Medical Image Analysis 2022. Extended HACT representation, PNA-based graph
  learning, jumping knowledge, BRACS dataset, and entity-graph evaluation.
  https://doi.org/10.1016/j.media.2021.102264
  https://arxiv.org/abs/2102.11057
- Chan et al. "Histopathology Whole Slide Image Analysis With Heterogeneous
  Graph Representation Learning." CVPR 2023. Heterogeneous patch graph,
  HoVer-Net pseudo-label node types, Pearson edge attributes, HEAT message
  passing, pseudo-label graph pooling, and causal localization.
  https://arxiv.org/abs/2307.04189

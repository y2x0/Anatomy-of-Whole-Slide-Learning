# C/R/G/S Across Four WSI Graph Families

This note places Patch-GCN, HACT, HEAT, and WiKG inside one operator decomposition while retaining each method's actual graph object, context map, readout, and supervision.

The four anchor WSI graph families differ most sharply in what determines an
edge and whether slide supervision can change that edge.

Specific anchors:

```text
Patch-GCN:
    Chen et al., Whole Slide Images are 2D Point Clouds, MICCAI 2021.
    https://arxiv.org/abs/2107.13048

HACT:
    Pati et al., HACT-Net, 2020.
    https://arxiv.org/abs/2007.00584
    Pati et al., Hierarchical graph representations in digital pathology,
    Medical Image Analysis 2022.
    https://arxiv.org/abs/2102.11057

HEAT:
    Chan et al., Histopathology Whole Slide Image Analysis With Heterogeneous
    Graph Representation Learning, CVPR 2023.
    https://arxiv.org/abs/2307.04189

WiKG:
    Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
    Histopathology Whole Slide Image Analysis, CVPR 2024.
    https://arxiv.org/abs/2403.07719
```

## Common Predictor

For slide `b`, write the whole graph predictor as:

```math
\widehat y_b
=
\mathcal{H}_{\theta}
\left(
\mathcal{R}_{\theta}
\left(
\mathcal{C}_{\theta}
\left(
H_b;
\mathcal{G}_b
\right)
\right)
\right).
```

The decomposition is:

```text
\mathcal{G}:
    graph object and relation support

\mathcal{C}:
    context operator moving information along that support

\mathcal{R}:
    graph-to-slide statistic

\mathcal{S}:
    target and objective constraining the predictor
```

The central comparison is whether `G_b` is observed, constructed, or learned.

## Patch-GCN: Coordinate Patch Graph

Patch-GCN represents a slide or patient WSI subgraph as:

```math
\mathcal{G}_b^{\mathrm{PatchGCN}}
=
\left(
H_b^{\mathrm{patch}},
A_b^{\mathrm{coord}}
\right).
```

Coordinate kNN support is constructed before graph learning:

```math
A_b^{\mathrm{coord}}[v,u]
=
\mathbb{1}
\left\{
u\in\mathrm{kNN}(c_{bv})
\right\}.
```

Context is a DeepGCN-style patch graph operator:

```math
\widetilde H_b^{\mathrm{patch}}
=
\mathcal{C}_{\theta}^{\mathrm{PatchGCN}}
\left(
H_b^{\mathrm{patch}};
A_b^{\mathrm{coord}}
\right).
```

Global attention readout gives:

```math
z_b^{\mathrm{PatchGCN}}
=
\sum_{v}
\alpha_{bv}
\widetilde h_{bv}^{\mathrm{patch}}.
```

Survival supervision acts on the patient or slide statistic:

```math
\eta_b
=
\mathcal{H}_{\mathrm{surv}}
\left(
z_b^{\mathrm{PatchGCN}}
\right).
```

C/R/G/S placement:

```text
\mathcal{G}:
    coordinate kNN patch graph

\mathcal{C}:
    DeepGCN-style local graph context

\mathcal{R}:
    global attention weighted first moment

\mathcal{S}:
    survival target and censoring-aware objective
```

The survival loss changes context and readout parameters but does not redefine
coordinate kNN support in the presented construction.

## HACT: Cell-To-Tissue Hierarchy

HACT represents a tissue region with two entity graphs and an assignment map:

```math
\mathcal{G}_b^{\mathrm{HACT}}
=
\left(
G_b^{\mathrm{cell}},
G_b^{\mathrm{tissue}},
B_b^{\mathrm{cell}\to\mathrm{tissue}}
\right).
```

Cell context is:

```math
\widetilde H_b^{\mathrm{cell}}
=
\mathcal{C}_{\theta}^{\mathrm{cell}}
\left(
H_b^{\mathrm{cell}};
A_b^{\mathrm{cell}}
\right).
```

Cell states transfer into tissue regions:

```math
U_b^{\mathrm{cell}\to\mathrm{tissue}}
=
\left(
B_b^{\mathrm{cell}\to\mathrm{tissue}}
\right)^{\top}
\widetilde H_b^{\mathrm{cell}}.
```

Tissue input combines region features and transferred cell context:

```math
H_b^{\mathrm{tissue},0}
=
\left[
H_b^{\mathrm{tissue}}
\middle\Vert
U_b^{\mathrm{cell}\to\mathrm{tissue}}
\right].
```

Tissue context is:

```math
\widetilde H_b^{\mathrm{tissue}}
=
\mathcal{C}_{\theta}^{\mathrm{tissue}}
\left(
H_b^{\mathrm{tissue},0};
A_b^{\mathrm{tissue}}
\right).
```

The graph statistic is a tissue-level sum or layerwise sum-concatenation,
depending on the HACT source version:

```math
z_b^{\mathrm{HACT}}
=
\mathcal{R}_{\mathrm{tissue}}
\left(
\widetilde H_b^{\mathrm{tissue}}
\right).
```

C/R/G/S placement:

```text
\mathcal{G}:
    cell graph, tissue graph, and hard cell-to-tissue assignment

\mathcal{C}:
    cell GNN -> assignment transfer -> tissue GNN

\mathcal{R}:
    tissue-node sum or layerwise sum-concatenation

\mathcal{S}:
    histopathology region classification
```

The decisive bottleneck is the assignment matrix. Cell information reaches a
tissue node only through the region containing that cell.

## HEAT: Typed Patch Graph With Edge Attributes

Chan et al. construct:

```math
\mathcal{G}_b^{\mathrm{HEAT}}
=
\left(
H_b^{\mathrm{patch}},
B_b^{\mathrm{edge},0},
E_b^{\mathrm{feature}},
\tau_b,
\phi_b
\right).
```

Here:

```text
tau_b(v):
    HoVer-Net-derived patch pseudo-type

phi_b(u,v):
    continuous edge attribute

E_b^feature:
    feature-kNN support
```

HEAT context uses node-type-specific projections and edge-attribute modulation:

```math
\left(
\widetilde H_b,
\widetilde B_b^{\mathrm{edge}}
\right)
=
\mathcal{C}_{\theta}^{\mathrm{HEAT}}
\left(
H_b^{\mathrm{patch}},
B_b^{\mathrm{edge},0};
E_b^{\mathrm{feature}},
\tau_b
\right).
```

Pseudo-label pooling first compresses nodes within each type:

```math
z_{ba}
=
\mathcal{R}_a
\left(
\{\widetilde h_{bv}:\tau_b(v)=a\}
\right).
```

The type-indexed matrix is:

```math
Z_b^{\mathrm{PL}}
=
\left[
z_{ba}
\right]_{a\in\mathcal{T}}.
```

Graph-level pooling then gives:

```math
z_b^{\mathrm{HEAT}}
=
\mathcal{R}_{\mathrm{graph}}
\left(
Z_b^{\mathrm{PL}}
\right).
```

C/R/G/S placement:

```text
\mathcal{G}:
    feature-kNN patch support, node pseudo-types, continuous edge attributes

\mathcal{C}:
    heterogeneous edge-attribute transformer

\mathcal{R}:
    pseudo-type pooling followed by graph pooling

\mathcal{S}:
    WSI classification, staging, or typing label
```

Heterogeneity changes the message kernel and the readout vocabulary, but the
feature-kNN support is constructed before task learning in the presented
pipeline.

## WiKG: Task-Learned Directed Support

WiKG constructs:

```math
\mathcal{G}_{\theta,b}^{\mathrm{WiKG}}
=
\left(
Q_b,
T_b,
A_{\theta,b},
R_{\theta,b}
\right).
```

The support is:

```math
A_{\theta,b}[v,u]
=
\mathbb{1}
\left\{
u
\in
\mathrm{TopK}
\left(
\left\{
q_{bv}t_{ba}^{\top}
/
\sqrt d
\right\}_{a=1}^{n_b}
\right)
\right\}.
```

Relation states are:

```math
r_{bvu}
=
\omega_{bvu}t_{bu}
+
\left(
1-\omega_{bvu}
\right)q_{bv}.
```

Knowledge-aware context gives:

```math
m_{bv}
=
\sum_{u\in\mathcal{N}_{b,K}(v)}
\pi_{bvu}t_{bu},
```

```math
h_{bv}^{+}
=
\phi_1
\left(
(q_{bv}+m_{bv})W_1+b_1
\right)
+
\phi_2
\left(
(q_{bv}\odot m_{bv})W_2+b_2
\right).
```

The released training script uses mean readout:

```math
z_b^{\mathrm{WiKG}}
=
\frac{1}{n_b}
\sum_{v=1}^{n_b}h_{bv}^{+}.
```

After LayerNorm and a linear classifier:

```math
\ell_b
=
\mathrm{LN}
\left(
z_b^{\mathrm{WiKG}}
\right)
W_C+b_C.
```

C/R/G/S placement:

```text
\mathcal{G}:
    task-learned directed top-k support and endpoint-derived relation states

\mathcal{C}:
    knowledge-aware selected-neighbor attention and dual interaction

\mathcal{R}:
    mean in the released training script; max or attention are model options

\mathcal{S}:
    WSI classification or staging cross-entropy
```

Unlike the other three anchor constructions, slide supervision updates the
head-tail projections that define the support itself.

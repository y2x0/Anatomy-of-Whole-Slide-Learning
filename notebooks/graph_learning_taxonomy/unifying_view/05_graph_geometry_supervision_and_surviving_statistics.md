# Graph Geometry, Supervision, And Surviving Statistics

This note compares what an edge means, whether task supervision can reshape support, and which graph statistic survives each anchor pipeline.

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

## Geometry Comparison

Patch-GCN relation:

```math
u\sim v
\quad\Longleftrightarrow\quad
c_u
\text{ and }
c_v
\text{ are coordinate-near}.
```

HACT relation:

```math
u\sim v
\quad\Longleftrightarrow\quad
u,v
\text{ are nearby entities at the same level},
```

plus:

```math
\text{cell}
\xrightarrow{B^{\mathrm{cell}\to\mathrm{tissue}}}
\text{containing tissue region}.
```

HEAT relation:

```math
u\sim v
\quad\Longleftrightarrow\quad
h_u
\text{ and }
h_v
\text{ are feature-near}.
```

WiKG relation:

```math
u\to v
\quad\Longleftrightarrow\quad
q_vt_u^{\top}
\text{ is among target }v\text{'s top-}K\text{ scores}.
```

These relations answer different questions. Coordinate proximity, entity
containment, encoder similarity, and task-learned directed compatibility are
not interchangeable notions of tissue geometry.

## Where Supervision Enters

Let `theta_task` denote parameters updated by the downstream task objective.
For Patch-GCN, HACT, and HEAT as presented in their construction pipelines,
adjacency is independent of those parameters:

```math
\frac{\partial A_b}
{\partial\theta_{\mathrm{task}}}
=
0
```

because adjacency is produced by coordinate, segmentation, region, or
precomputed feature rules rather than a differentiable task-trained graph
constructor.

For WiKG, score geometry depends directly on task-trained head and tail
projections:

```math
\frac{\partial S_b}
{\partial W_H}
\ne
0,
\qquad
\frac{\partial S_b}
{\partial W_T}
\ne
0.
```

because:

```math
S_b
=
\frac{1}{\sqrt d}
Q_bT_b^{\top}
```

depends on trainable head and tail projections. The discrete support identities
still have zero derivative almost everywhere under hard top-k and change only
when logits cross a rank boundary. Gradient updates to `W_H` and `W_T` can move
the model across those boundaries between optimization steps.

## Surviving Statistics

Patch-GCN:

```math
z_b
=
\sum_v
\alpha_{bv}
\widetilde h_{bv}.
```

Survives:

```text
one weighted first moment of coordinate-contextualized patch nodes
```

HACT:

```math
z_b
=
\mathcal{R}_{\mathrm{tissue}}
\left(
\widetilde H_b^{\mathrm{tissue}}
\right).
```

Survives:

```text
tissue-level statistic after hard cell-to-region transfer
```

HEAT:

```math
Z_b^{\mathrm{PL}}
=
\left[
\mathcal{R}_a
\left(
\{\widetilde h_{bv}:\tau_b(v)=a\}
\right)
\right]_{a\in\mathcal{T}}.
```

Survives:

```text
type-wise statistics indexed by the pseudo-label vocabulary
```

WiKG:

```math
m_{bv}
=
\sum_u\pi_{bvu}t_{bu},
```

then, under released mean readout:

```math
z_b
=
\frac{1}{n_b}
\sum_v h_{bv}^{+}.
```

Survives:

```text
one neighborhood weighted first moment per target, followed by one global
first moment of dynamically contextualized targets
```

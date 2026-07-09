# HACT Slide Object

HACT-Net is best read as a hierarchical graph model for annotated histology
regions of interest. The Medical Image Analysis paper works with tumor regions
of interest, not arbitrary patient-level gigapixel WSIs.

So the input object is:

```math
I_i
```

where `I_i` is an H&E tumor region of interest extracted from a WSI.

The paper calls the region a TRoI. The dataset-level object is still derived
from WSIs, but the model's forward map classifies the region graph:

```math
I_i
\longmapsto
G_i^{\mathrm{HACT}}
\longmapsto
\widehat y_i.
```

This caveat matters. HACT is not a MIL survival model, and it should not be
described as if it directly aggregates all WSI patches into a patient risk
score.

## The HACT Object

The hierarchical graph object is:

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
V_i^{\mathrm{cell}},
E_i^{\mathrm{cell}},
H_i^{\mathrm{cell}}
\right),
```

with:

```math
H_i^{\mathrm{cell}}
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d_{\mathrm{cell}}}.
```

The tissue graph is:

```math
G_i^{\mathrm{tissue}}
=
\left(
V_i^{\mathrm{tissue}},
E_i^{\mathrm{tissue}},
H_i^{\mathrm{tissue}}
\right),
```

with:

```math
H_i^{\mathrm{tissue}}
\in
\mathbb{R}^{N_i^{\mathrm{tissue}}\times d_{\mathrm{tissue}}}.
```

The assignment matrix is:

```math
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\in
\{0,1\}^{N_i^{\mathrm{cell}}\times N_i^{\mathrm{tissue}}}.
```

The paper's hierarchy is hard and spatial:

```math
B_{iab}^{\mathrm{cell}\to\mathrm{tissue}}
=
1
```

when cell `a` belongs to tissue region `b`.

The row constraint is:

```math
\sum_{b=1}^{N_i^{\mathrm{tissue}}}
B_{iab}^{\mathrm{cell}\to\mathrm{tissue}}
=
1
\qquad
\text{for every cell }a.
```

Thus each detected cell is assigned to exactly one tissue region.

The usual size relation is:

```math
N_i^{\mathrm{cell}}
\gg
N_i^{\mathrm{tissue}}.
```

## Equivalent Typed-Graph View

If one wanted to flatten HACT into a single graph, the typed node set would be:

```math
V_i^{\mathrm{HACT}}
=
V_i^{\mathrm{cell}}
\sqcup
V_i^{\mathrm{tissue}}.
```

The block adjacency would be:

```math
A_i^{\mathrm{HACT}}
=
\begin{bmatrix}
A_i^{\mathrm{cell}} &
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\\
\left(B_i^{\mathrm{cell}\to\mathrm{tissue}}\right)^\top &
A_i^{\mathrm{tissue}}
\end{bmatrix}.
```

But this is not the HACT-Net forward map.

HACT-Net does not simply run one homogeneous GNN on this block graph. It uses a
sequential fine-to-coarse computation:

```text
cell graph message passing
    then
cell-to-tissue pooling
    then
tissue graph message passing
    then
graph readout
```

That ordering is a strong inductive bias.

## Comparison To Patch-GCN

Patch-GCN constructs:

```math
G_i^{\mathrm{patch}}
=
\left(
V_i^{\mathrm{patch}},
E_i^{\mathrm{patch}},
H_i^{\mathrm{patch}}
\right),
```

where all nodes are patches.

HACT constructs:

```math
\left\{
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right\}.
```

The difference is not just node scale. The difference is typed structure.

Patch-GCN says:

```text
nearby patches exchange messages
```

HACT says:

```text
cells form neighborhoods,
tissue regions form neighborhoods,
and cells are nested inside tissue regions
```

## The Factorization Assumption

The papers define models whose predictions factor through the constructed HACT
object. The safer mathematical statement is therefore:

```math
\Gamma_{\mathrm{HACT}}(I_i)
=
G_i^{\mathrm{HACT}},
```

```math
\widehat y_i
=
f_\theta
\left(
\Gamma_{\mathrm{HACT}}(I_i)
\right).
```

Equivalently:

```math
F_\theta(I_i)
=
f_\theta
\left(
G_i^{\mathrm{HACT}}
\right).
```

This is a model factorization assumption, not a statistical sufficiency theorem.
The HACT papers do not prove that the label is conditionally independent of the
raw image given the extracted HACT graph. They build a predictor that only sees
the extracted entity graph.

More explicitly:

```math
\widehat y_i
=
\mathcal{H}_\theta
\left(
\mathcal{R}_\theta
\left(
\mathcal{C}_\theta
\left(
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right)
\right)
\right).
```

The object passed into the graph learner is already a lossy entity extraction.
Nuclei segmentation, tissue region segmentation, graph construction, and hard
assignment are all part of the representation.

## C/R/G/S Placement

```text
G:
    hierarchical entity graph
    {cell graph, tissue graph, cell-to-tissue assignment}

C:
    sequential cell GNN -> assignment pooling -> tissue GNN

R:
    graph-level readout over contextualized tissue nodes

S:
    TRoI cancer subtype label used in cross-entropy training
```

## What Survives Before Learning

Before message passing, the representation contains:

```math
\left(
H_i^{\mathrm{cell}},
A_i^{\mathrm{cell}},
H_i^{\mathrm{tissue}},
A_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right).
```

It does not contain:

```text
the full pixel field
all pixel-level texture
soft uncertainty over tissue membership
top-down tissue-to-cell refinement
patient-level survival time
whole-slide MIL structure
```

That is not a criticism. It is the price of making the tissue object
entity-based and mathematically explicit.

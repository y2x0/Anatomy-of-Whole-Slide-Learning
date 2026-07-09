# HACT Failure Modes

HACT is powerful because it makes the histology object explicit:

```math
G_i^{\mathrm{HACT}}
=
\left\{
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\right\}.
```

That same explicitness creates precise failure modes.

## 1. Segmentation Error Becomes Graph Error

The model does not merely receive noisy features. It receives a graph whose
nodes depend on segmentation:

```math
I_i
\xrightarrow{\mathrm{segmentation}}
V_i^{\mathrm{cell}},
V_i^{\mathrm{tissue}}.
```

If nuclei are missed:

```math
V_i^{\mathrm{cell}}
\text{ is wrong}.
```

If tissue regions are overmerged:

```math
V_i^{\mathrm{tissue}}
\text{ is wrong}.
```

If boundaries are wrong:

```math
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\text{ is wrong}.
```

The GNN cannot recover entities that never enter the graph.

## 2. Hard Assignment Removes Boundary Uncertainty

HACT uses:

```math
B_{iab}
\in
\{0,1\}.
```

Each cell belongs to exactly one tissue region:

```math
\sum_b B_{iab}=1.
```

A soft assignment would preserve uncertainty:

```math
S_{iab}
\in
[0,1],
\qquad
\sum_b S_{iab}=1.
```

HACT does not use this. A border cell assigned to the wrong region sends all of
its downstream signal to that region:

```math
p_{ib}^{\mathrm{cell}}
=
\sum_a B_{iab}\widetilde h_{ia}^{\mathrm{cell}}.
```

There is no partial membership unless one modifies the representation.

## 3. Sum Pooling Couples Phenotype And Count

Cell transfer uses:

```math
B_i^\top
\widetilde H_i^{\mathrm{cell}}.
```

For tissue region `b`:

```math
p_{ib}^{\mathrm{cell}}
=
\sum_{a\in\mathcal{M}_i(b)}
\widetilde h_{ia}^{\mathrm{cell}}.
```

If all assigned cell embeddings have similar mean:

```math
\widetilde h_{ia}^{\mathrm{cell}}
\approx
\mu_b,
```

then:

```math
p_{ib}^{\mathrm{cell}}
\approx
n_{ib}^{\mathrm{cell}}\mu_b.
```

Thus the magnitude contains cell count.

This is not automatically bad. Cellularity can be diagnostic. But it means the
model can rely on:

```text
how many cells were segmented in the region
```

as much as:

```text
what those cells look like
```

If segmentation density shifts across scanners or stain conditions, this
becomes a shortcut.

The 2022 implementation uses GraphNorm and BatchNorm after PNA layers, which
can reduce scale instability across graphs of different sizes. But
normalization does not remove the fact that the transfer operator itself is a
sum, and therefore exposes cell count before later normalization.

## 4. Within-Region Distributions Collapse To A Sum

After transfer, the tissue GNN sees:

```math
B_i^\top
\widetilde H_i^{\mathrm{cell}}.
```

For each tissue node, many cell configurations can map to the same sum:

```math
\sum_{a\in\mathcal{M}_i(b)}
u_a
=
\sum_{a\in\mathcal{M}_i(b)}
v_a.
```

If this happens, the upper graph cannot distinguish:

```math
\{u_a\}
\quad
\text{from}
\quad
\{v_a\}
```

unless the lower cell GNN already encoded the relevant higher moments into
individual embeddings before summation.

## 5. One-Way Hierarchy Limits Feedback

HACT is fine-to-coarse:

```math
G_i^{\mathrm{cell}}
\to
G_i^{\mathrm{tissue}}.
```

The canonical forward map does not do:

```math
G_i^{\mathrm{tissue}}
\to
G_i^{\mathrm{cell}}
\to
G_i^{\mathrm{tissue}}.
```

So a tissue-level hypothesis cannot revise the cell-level representation before
cell embeddings are pooled.

This matters when cell phenotype is ambiguous without larger tissue context.

## 6. RAG Edges Can Oversimplify Tissue Geometry

The tissue graph uses region adjacency:

```math
A_{iab}^{\mathrm{tissue}}
=
1
\quad
\Longleftrightarrow
\quad
R_{ia}\sim R_{ib}.
```

This captures direct contact between tissue regions. It does not automatically
encode:

```text
metric distance between non-adjacent regions
shape of the shared boundary
orientation
long-range arrangement
global gland architecture
```

Those signals must either appear in tissue node features, emerge through
multiple message-passing layers, or be lost.

## 7. PNA Helps, But Does Not Make Geometry Exact

PNA uses multiple neighborhood statistics:

```math
\left[
\mu,
\sigma,
\mathrm{max},
\mathrm{min}
\right]
```

with degree scalers. This is richer than mean aggregation.

But PNA still operates on the constructed graph support:

```math
\mathcal{N}(v)
=
\{u:A_{vu}=1\}.
```

If the construction misses an important relation, PNA cannot aggregate over it.

The expressive message passing block cannot fix a wrong graph ontology.

## 8. ROI-Level Scope Is Not WSI-Level MIL

The Medical Image Analysis paper uses BRACS tumor regions of interest from
WSIs. The model predicts a region label:

```math
G_i^{\mathrm{HACT}}
\longmapsto
\widehat y_i^{\mathrm{TRoI}}.
```

It is not:

```math
\{G_{ij}^{\mathrm{HACT}}\}_{j=1}^{K_i}
\longmapsto
\widehat y_i^{\mathrm{patient}}.
```

So if one wants full WSI or patient-level learning, an additional aggregation
layer is required:

```math
z_{ij}
=
F_\theta(G_{ij}^{\mathrm{HACT}}),
\qquad
z_i
=
\mathcal{R}
\left(
\{z_{ij}\}_{j=1}^{K_i}
\right).
```

HACT supplies the entity graph encoder. It does not by itself solve
patient-level MIL.

## 9. Short Paper And Journal Version Should Not Be Collapsed

The short HACT-Net paper presents a GIN-style model:

```text
neighbor sum
MLP update
layerwise sum readout
```

The journal paper presents a stronger implementation:

```text
PNA layers
multiple aggregators
degree scalers
LSTM jumping knowledge
GraphNorm and BatchNorm
```

They share the HACT representation, but the message-passing operator differs.

When citing a claim, keep the source specific:

```text
HACT object:
    both papers

GIN equations:
    2020 HACT-Net paper

PNA plus jumping knowledge:
    2022 Medical Image Analysis paper

BRACS 4391 TRoI dataset:
    2022 Medical Image Analysis paper
```

## Failure Table

```text
representation failure:
    wrong nuclei, wrong tissue regions, wrong hard assignment

context failure:
    graph support omits biologically relevant interactions

readout failure:
    sums collapse within-region cell distributions

supervision failure:
    ROI labels do not teach patient-level WSI aggregation
```

## Concrete Stress Tests

These are natural checks for the failure modes above.

```text
segmentation jitter:
    perturb nuclei centroids or tissue boundaries and measure prediction drift

boundary reassignment:
    move border nuclei between adjacent tissue regions and measure sensitivity

sum-versus-mean transfer:
    replace B^T H_cell with degree-normalized transfer and test whether
    performance depends on cell count shortcuts

cell-count ablation:
    append or remove explicit n_cell per tissue node and test whether the model
    was already using count through the sum magnitude

RAG rewiring:
    randomly rewire tissue adjacency while preserving degree distribution to
    test whether tissue topology matters beyond tissue-node features

k and d_min sweep:
    vary cell-graph kNN support and distance threshold to test whether local
    cellular neighborhoods are stable

WSI-level split audit:
    confirm that regions from the same WSI are not split across train,
    validation, and test
```

## C/R/G/S Diagnostic Questions

```text
G:
    Are the cell and tissue graphs biologically correct?
    Are the assignment boundaries reliable?

C:
    Does the cell GNN encode enough before pooling?
    Does the tissue GNN need top-down feedback?

R:
    Is sum pooling preserving the right statistic,
    or mostly encoding region size and cell count?

S:
    Is the label a TRoI label, WSI label, patient label, or survival endpoint?
```

The core lesson is simple:

```text
hierarchical graph learning is only as good as the hierarchy that the
preprocessing pipeline makes available
```

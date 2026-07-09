# HACT-Net Context And Readout

HACT-Net applies graph learning in a fixed order:

```text
cell-level context
cell-to-tissue transfer
tissue-level context
tissue graph readout
classification head
```

The short HACT-Net paper presents this with GIN layers. The later Medical Image
Analysis paper uses PNA layers with jumping knowledge. The hierarchy is the
same; the local message-passing block changes.

## Source Boundary

The two HACT sources should not be collapsed into one architecture.

| Axis | 2020 HACT-Net arXiv | 2022 Medical Image Analysis |
| --- | --- | --- |
| Input scope | BRACS regions of interest | BRACS tumor regions of interest and BACH images |
| Labels | 5 breast categories | 7 BRACS categories, plus other evaluation settings |
| Cell features | hand-crafted morphology, texture, and spatial features | nucleus-centered CNN features plus normalized spatial features |
| Tissue features | color, texture, random-forest selected features, and spatial centroids | CNN features over superpixel-centered patches, merged-region averages, and spatial centroids |
| Cell graph | thresholded kNN over nuclei centroids | same graph-support idea |
| Tissue graph | region adjacency graph | same graph-support idea |
| Graph layer | GIN-style neighbor sum plus MLP | PNA layers |
| Depth aggregation | layerwise tissue-node sum concatenation | LSTM jumping knowledge |
| Normalization | not the emphasized operator | GraphNorm followed by BatchNorm after PNA layers |

The notes below derive both forward maps, but any paper-specific claim should
use the correct row of this table.

## GIN Version

Let the initial cell embedding be:

```math
h_{ia}^{\mathrm{cell},0}
=
f_i^{\mathrm{cell}}(a).
```

The GIN-style cell update in the short paper is:

```math
h_{ia}^{\mathrm{cell},t+1}
=
\mathrm{MLP}_{\mathrm{cell}}^{t}
\left(
h_{ia}^{\mathrm{cell},t}
+
\sum_{b\in\mathcal{N}_i^{\mathrm{cell}}(a)}
h_{ib}^{\mathrm{cell},t}
\right).
```

After `T_cell` layers:

```math
\widetilde h_{ia}^{\mathrm{cell}}
=
h_{ia}^{\mathrm{cell},T_{\mathrm{cell}}}.
```

For tissue node `r`, HACT initializes the tissue state by concatenating tissue
features with pooled cell features:

```math
h_{ir}^{\mathrm{tissue},0}
=
\mathrm{Concat}
\left(
f_i^{\mathrm{tissue}}(r),
\sum_{a\in\mathcal{M}_i(r)}
\widetilde h_{ia}^{\mathrm{cell}}
\right).
```

Then the tissue graph is processed analogously:

```math
h_{ir}^{\mathrm{tissue},t+1}
=
\mathrm{MLP}_{\mathrm{tissue}}^{t}
\left(
h_{ir}^{\mathrm{tissue},t}
+
\sum_{s\in\mathcal{N}_i^{\mathrm{tissue}}(r)}
h_{is}^{\mathrm{tissue},t}
\right).
```

The short paper reads out graph-level information by summing tissue-node
embeddings across layers and concatenating the results:

```math
h_i^{\mathrm{HACT}}
=
\mathrm{Concat}
\left(
\left\{
\sum_{r\in V_i^{\mathrm{tissue}}}
h_{ir}^{\mathrm{tissue},t}
:
t=0,\dots,T_{\mathrm{tissue}}
\right\}
\right).
```

This is a layerwise graph statistic:

```text
for each depth, sum all tissue-region states
then concatenate depth summaries
```

## PNA Version

The Medical Image Analysis paper uses Principal Neighbourhood Aggregation.

For a cell node `a`, define messages:

```math
m_{iab}^{\mathrm{cell},t}
=
M_{\mathrm{cell}}^t
\left(
h_{ia}^{\mathrm{cell},t},
h_{ib}^{\mathrm{cell},t}
\right),
\qquad
b\in\mathcal{N}_i^{\mathrm{cell}}(a).
```

PNA aggregates multiple statistics:

```math
\mathcal{A}
=
\left[
\mu,
\sigma,
\mathrm{max},
\mathrm{min}
\right].
```

It also applies degree scalers:

```math
\mathcal{D}
=
\left[
I,
\mathcal{S}(D,\alpha=1),
\mathcal{S}(D,\alpha=-1)
\right].
```

Interpreting the journal paper through the standard PNA form, the degree scaler
can be written as:

```math
\mathcal{S}(D,\alpha)
=
\left(
\frac{\log(D+1)}{\delta}
\right)^\alpha,
```

with:

```math
\delta
=
\frac{1}{|\mathrm{train}|}
\sum_{j\in \mathrm{train}}
\log(d_j+1).
```

The combined PNA neighborhood statistic can be written abstractly as:

```math
a_{ia}^{\mathrm{cell},t+1}
=
\left(
\mathcal{D}
\otimes
\mathcal{A}
\right)
\left(
\left\{
m_{iab}^{\mathrm{cell},t}
:
b\in\mathcal{N}_i^{\mathrm{cell}}(a)
\right\}
\right).
```

Then:

```math
h_{ia}^{\mathrm{cell},t+1}
=
U_{\mathrm{cell}}^t
\left(
h_{ia}^{\mathrm{cell},t},
a_{ia}^{\mathrm{cell},t+1}
\right).
```

The important difference from a GCN-style average is that the neighborhood is
not summarized by a single first moment. PNA keeps several statistics:

```text
mean
standard deviation
maximum
minimum
degree-scaled variants
```

So its surviving local statistic is closer to a small distribution summary.

## Jumping Knowledge

The journal paper applies an LSTM-based jumping knowledge operator to the cell
states:

```math
\widetilde h_{ia}^{\mathrm{cell}}
=
\mathrm{LSTM}_{\mathrm{JK}}
\left(
\left\{
h_{ia}^{\mathrm{cell},t}
:
t=1,\dots,T_{\mathrm{cell}}
\right\}
\right).
```

This makes the final cell state depend on multiple message-passing depths:

```text
one-hop cellular context
two-hop cellular context
...
T-hop cellular context
```

Then HACT transfers the cell states to tissue nodes:

```math
h_{ir}^{\mathrm{tissue},0}
=
\mathrm{Concat}
\left(
h_{ir}^{\mathrm{tissue}},
\sum_{a\in\mathcal{M}_i(r)}
\widetilde h_{ia}^{\mathrm{cell}}
\right).
```

The tissue graph then receives analogous PNA updates:

```math
a_{ir}^{\mathrm{tissue},t+1}
=
\left(
\mathcal{D}
\otimes
\mathcal{A}
\right)
\left(
\left\{
M_{\mathrm{tissue}}^t
\left(
h_{ir}^{\mathrm{tissue},t},
h_{is}^{\mathrm{tissue},t}
\right)
:
s\in\mathcal{N}_i^{\mathrm{tissue}}(r)
\right\}
\right),
```

```math
h_{ir}^{\mathrm{tissue},t+1}
=
U_{\mathrm{tissue}}^t
\left(
h_{ir}^{\mathrm{tissue},t},
a_{ir}^{\mathrm{tissue},t+1}
\right).
```

The 2022 implementation also applies GraphNorm followed by BatchNorm after PNA
layers. Those normalizations are not the hierarchy itself, but they matter
because HACT graphs vary strongly in node count and because assignment sums can
carry size and cellularity information.

Tissue-level jumping knowledge gives:

```math
\widetilde h_{ir}^{\mathrm{tissue}}
=
\mathrm{LSTM}_{\mathrm{JK}}
\left(
\left\{
h_{ir}^{\mathrm{tissue},t}
:
t=1,\dots,T_{\mathrm{tissue}}
\right\}
\right).
```

The graph-level representation is obtained by summing tissue-node
representations:

```math
h_i^{\mathrm{HACT}}
=
\sum_{r\in V_i^{\mathrm{tissue}}}
\widetilde h_{ir}^{\mathrm{tissue}}.
```

## Classification Head

For `K_cls` cancer subtype classes:

```math
\ell_i
=
h_i^{\mathrm{HACT}}W_{\mathrm{cls}}
+
b_{\mathrm{cls}}
\in
\mathbb{R}^{K_{\mathrm{cls}}}.
```

The predicted class probabilities are:

```math
\widehat p_i
=
\mathrm{softmax}(\ell_i).
```

For one-hot label:

```math
y_i
\in
\{0,1\}^{K_{\mathrm{cls}}},
```

the cross-entropy loss is:

```math
\mathcal{L}_i
=
-
\sum_{c=1}^{K_{\mathrm{cls}}}
y_{ic}
\log
\widehat p_{ic}.
```

This is ordinary supervised classification. The task signal does not define the
graph support during inference.

## Whole Forward Map

Putting the PNA version together:

```math
G_i^{\mathrm{cell}}
\xrightarrow{\mathrm{CellGNN+Norm+JK}}
\widetilde H_i^{\mathrm{cell}}
\xrightarrow{B_i^\top}
B_i^\top \widetilde H_i^{\mathrm{cell}}
\xrightarrow{\mathrm{Concat}\;H_i^{\mathrm{tissue}}}
H_i^{\mathrm{tissue},0}
\xrightarrow{\mathrm{TissueGNN+Norm+JK}}
\widetilde H_i^{\mathrm{tissue}}
\xrightarrow{\mathrm{sum}}
h_i^{\mathrm{HACT}}
\xrightarrow{\mathrm{MLP}}
\widehat p_i.
```

The critical bottleneck is:

```math
B_i^\top\widetilde H_i^{\mathrm{cell}}.
```

That is the only channel through which lower-level cell information enters the
upper-level tissue graph.

## C/R/G/S Placement

```text
G:
    G_cell, G_tissue, and hard assignment B

C:
    CellGNN, assignment pooling, TissueGNN

R:
    sum or layerwise concat readout over tissue nodes

S:
    cross-entropy label for cancer subtype
```

## What Survives Aggregation

At the cell-neighborhood level:

```text
GIN:
    neighbor sum

PNA:
    neighborhood distribution summary
```

At the cell-to-tissue level:

```text
sum of contextualized cell states per tissue region
```

At the graph-readout level:

```text
sum of contextualized tissue states
or layerwise concatenation of tissue-state sums
```

Thus HACT has additive interfaces at the cell-to-tissue transfer and final
tissue-node readout. In the 2022 model, those additive interfaces are separated
by nonlinear PNA, normalization, and LSTM jumping-knowledge operators, so the
whole forward map is not merely a nested sum.

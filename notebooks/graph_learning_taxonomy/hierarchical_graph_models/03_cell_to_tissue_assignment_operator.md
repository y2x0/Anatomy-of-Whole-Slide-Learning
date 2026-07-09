# Cell-To-Tissue Assignment Operator

The distinctive HACT operator is not just a GNN layer. It is the hard
assignment from cell nodes to tissue-region nodes.

For TRoI `i`, define:

```math
B_i
=
B_i^{\mathrm{cell}\to\mathrm{tissue}}
\in
\{0,1\}^{N_i^{\mathrm{cell}}\times N_i^{\mathrm{tissue}}}.
```

The entry:

```math
B_{iab}=1
```

means:

```text
cell a belongs to tissue region b
```

The row-stochastic hard-assignment constraint is:

```math
\sum_{b=1}^{N_i^{\mathrm{tissue}}}
B_{iab}
=
1.
```

The column count is:

```math
n_{ib}^{\mathrm{cell}}
=
\sum_{a=1}^{N_i^{\mathrm{cell}}}
B_{iab}.
```

This is the number of cells assigned to tissue region `b`.

## Assignment Set

For tissue region `b`, define:

```math
\mathcal{M}_i(b)
=
\left\{
a\in V_i^{\mathrm{cell}}
:
B_{iab}=1
\right\}.
```

The paper uses this set to inject cell information into tissue nodes.

If a nucleus lies on a tissue-region boundary, the journal paper assigns it to
the tissue region with maximum overlap. Thus boundary uncertainty is resolved
before learning.

## Cell-To-Tissue Pooling

Let the final contextualized cell embeddings be:

```math
\widetilde H_i^{\mathrm{cell}}
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d_c}.
```

HACT pools cell information into tissue nodes with a sum:

```math
p_{ib}^{\mathrm{cell}}
=
\sum_{a\in \mathcal{M}_i(b)}
\widetilde h_{ia}^{\mathrm{cell}}.
```

In matrix form:

```math
P_i^{\mathrm{cell}\to\mathrm{tissue}}
=
B_i^\top
\widetilde H_i^{\mathrm{cell}}
\in
\mathbb{R}^{N_i^{\mathrm{tissue}}\times d_c}.
```

The initialized tissue features are:

```math
\bar H_i^{\mathrm{tissue},0}
=
\left[
H_i^{\mathrm{tissue}}
\;\middle|\;
B_i^\top
\widetilde H_i^{\mathrm{cell}}
\right].
```

Equivalently, for tissue node `b`:

```math
\bar h_{ib}^{\mathrm{tissue},0}
=
\mathrm{Concat}
\left(
h_{ib}^{\mathrm{tissue}},
\sum_{a\in\mathcal{M}_i(b)}
\widetilde h_{ia}^{\mathrm{cell}}
\right).
```

This is exactly the coarse-scale feature injection:

```text
cells are summarized inside the tissue node they belong to
```

## Sum Is Not Mean

The HACT equations use a sum, not a mean.

Mean pooling would be:

for:

```math
n_{ib}^{\mathrm{cell}}
>
0,
```

```math
\bar p_{ib}^{\mathrm{cell}}
=
\frac{1}{n_{ib}^{\mathrm{cell}}}
\sum_{a\in \mathcal{M}_i(b)}
\widetilde h_{ia}^{\mathrm{cell}}.
```

HACT instead uses:

```math
p_{ib}^{\mathrm{cell}}
=
\sum_{a\in \mathcal{M}_i(b)}
\widetilde h_{ia}^{\mathrm{cell}}.
```

So the tissue node receives both:

```text
cell phenotype signal
cell abundance signal
```

because the norm of the sum can scale with:

```math
n_{ib}^{\mathrm{cell}}.
```

This can be desirable when cellularity is diagnostic. It can also leak region
size or segmentation density into the classifier.

## What Information Survives

The assignment operator preserves:

```text
which tissue region each cell belongs to
summed contextual cell embeddings per tissue region
cell abundance through sum magnitude
```

It discards:

```text
ordering of cells inside the tissue region
individual cell identities after pooling
soft boundary uncertainty
cross-tissue cell membership
fine cell topology after it has been compressed into each tissue node
```

Formally, after assignment pooling, the upper graph only sees:

```math
\left(
H_i^{\mathrm{tissue}},
B_i^\top\widetilde H_i^{\mathrm{cell}},
A_i^{\mathrm{tissue}}
\right).
```

It does not see the individual multiset:

```math
\left\{
\widetilde h_{ia}^{\mathrm{cell}}
:
a\in\mathcal{M}_i(b)
\right\}
```

except through its sum.

## Assignment Bottleneck Proposition

Define the assignment pooling map:

```math
\Pi_{B_i}
\left(
H
\right)
=
B_i^\top H,
```

where:

```math
H
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d}.
```

For any perturbation:

```math
\Delta
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d}
```

such that:

```math
B_i^\top \Delta
=
0,
```

assignment pooling is invariant:

```math
\Pi_{B_i}
\left(
H+\Delta
\right)
=
\Pi_{B_i}(H).
```

Proof:

```math
\Pi_{B_i}
\left(
H+\Delta
\right)
=
B_i^\top
\left(
H+\Delta
\right)
=
B_i^\top H
+
B_i^\top \Delta
=
B_i^\top H.
```

So every perturbation in:

```math
\mathrm{Null}
\left(
B_i^\top
\right)
```

is invisible after the cell-to-tissue transfer.

## Lost Linear Degrees Of Freedom

Let:

```math
r_i
=
\mathrm{rank}(B_i).
```

The nullity of:

```math
B_i^\top:
\mathbb{R}^{N_i^{\mathrm{cell}}}
\to
\mathbb{R}^{N_i^{\mathrm{tissue}}}
```

is:

```math
N_i^{\mathrm{cell}}
-
r_i.
```

For `d`-dimensional cell embeddings, the invisible linear subspace has
dimension:

```math
d
\left(
N_i^{\mathrm{cell}}
-
r_i
\right).
```

Because `B_i` is a hard assignment matrix, `r_i` equals the number of nonempty
tissue regions. If every tissue region receives at least one cell, then:

```math
r_i
=
N_i^{\mathrm{tissue}},
```

and the lost linear dimension is:

```math
d
\left(
N_i^{\mathrm{cell}}
-
N_i^{\mathrm{tissue}}
\right)
=
d
\sum_{b=1}^{N_i^{\mathrm{tissue}}}
\left(
n_{ib}^{\mathrm{cell}}-1
\right).
```

Equivalently, within a tissue region containing `n_ib_cell` cells, all
zero-sum perturbations:

```math
\sum_{a\in\mathcal{M}_i(b)}
\Delta_a
=
0
```

are removed by assignment pooling.

This proposition is about the transfer after the lower cell GNN has produced
cell embeddings. The lower GNN can try to encode useful within-region
differences before the sum. But once the map:

```math
B_i^\top
\widetilde H_i^{\mathrm{cell}}
```

has been applied, the tissue GNN only receives the column sums.

This is the key mathematical bottleneck:

```text
within-region cellular structure must survive a sum before tissue-level
message passing begins
```

## Difference From Learned Pooling

The assignment matrix is not a learned DiffPool-style matrix.

A learned soft pooling would be:

```math
S_i
\in
[0,1]^{N_i^{\mathrm{cell}}\times K_i},
\qquad
\sum_{b=1}^{K_i}S_{iab}=1,
```

with:

```math
S_i
=
g_\theta
\left(
H_i^{\mathrm{cell}},
A_i^{\mathrm{cell}}
\right).
```

HACT instead uses:

```math
B_i
=
\Gamma_{\mathrm{geometry}}
\left(
\text{cell centroids},
\text{tissue regions}
\right).
```

So the hierarchy is a pathological and geometric prior, not a latent clustering
learned from labels.

## One-Way Hierarchy

The canonical HACT-Net computation is fine-to-coarse:

```math
G_i^{\mathrm{cell}}
\to
B_i^\top\widetilde H_i^{\mathrm{cell}}
\to
G_i^{\mathrm{tissue}}.
```

There is no simultaneous top-down update:

```math
G_i^{\mathrm{tissue}}
\to
G_i^{\mathrm{cell}}.
```

So tissue context can influence the final graph embedding after pooling, but it
does not revise the cell embeddings before they are pooled.

## C/R/G/S Placement

```text
G:
    B_cell_to_tissue is part of the hierarchical graph geometry

C:
    B^T H_cell is a context transfer operator from cell scale to tissue scale

R:
    no final graph readout happens here; the final readout is over tissue
    nodes after tissue-level context

S:
    task labels do not build B; B is constructed from segmentation geometry
```

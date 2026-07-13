# HACT Hierarchy As Graph MIL

Sources:

```text
Pati et al., HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for
Histopathological Image Classification, 2020.
https://arxiv.org/abs/2007.00584

Pati et al., Hierarchical graph representations in digital pathology,
Medical Image Analysis 2022.
https://arxiv.org/abs/2102.11057
```

HACT is not flat patch MIL. It is a hierarchical graph representation in which
fine entities are summarized into coarse entities before the final graph-level
readout.

## 1. Hierarchical Object

For slide or region `i`, define:

```math
\mathcal{G}_i^{\mathrm{HACT}}
=
\left(
G_i^{\mathrm{cell}},
G_i^{\mathrm{tissue}},
B_i
\right),
```

where `B_i` is the cell-to-tissue assignment matrix:

```math
B_i
\in
\{0,1\}^{N_i^{\mathrm{cell}}\times N_i^{\mathrm{tissue}}}.
```

Hard assignment means:

```math
\sum_{b=1}^{N_i^{\mathrm{tissue}}}B_{iab}=1
\qquad
\text{for every cell }a.
```

The cell graph and tissue graph have separate supports:

```math
A_i^{\mathrm{cell}}
\in
\{0,1\}^{N_i^{\mathrm{cell}}\times N_i^{\mathrm{cell}}},
\qquad
A_i^{\mathrm{tissue}}
\in
\{0,1\}^{N_i^{\mathrm{tissue}}\times N_i^{\mathrm{tissue}}}.
```

The hierarchy is part of the input object. It is not recoverable from the
tissue-node feature matrix after the assignment map has been discarded.

## 2. Fine-Scale Context

Let cell states after the cell graph operator be:

```math
\widetilde H_i^{\mathrm{cell}}
=
\mathcal{C}_{\theta}^{\mathrm{cell}}
\left(
H_i^{\mathrm{cell}},A_i^{\mathrm{cell}}
\right)
\in
\mathbb{R}^{N_i^{\mathrm{cell}}\times d_c}.
```

The original HACT-Net paper uses the following GIN-style update in its
dimension-safe reconstruction:

```math
h_{ia}^{\mathrm{cell},\ell+1}
=
\mathrm{MLP}_{\ell}
\left(
h_{ia}^{\mathrm{cell},\ell}
+
\sum_{b\in\mathcal{N}_i^{\mathrm{cell}}(a)}
h_{ib}^{\mathrm{cell},\ell}
\right).
```

This is the paper-level self-plus-neighbor form. A standard GIN
implementation may expose a learnable self coefficient; that parameterization
should not be attributed to the HACT paper unless the implementation being
reproduced uses it.

The later Medical Image Analysis version uses PNA-style multiple neighborhood
statistics and jumping knowledge. These are different context operators over
the same hierarchical object and should not be silently merged.

## 3. Assignment Readout

Cell-to-tissue pooling is a matrix multiplication:

```math
U_i^{\mathrm{cell}\to\mathrm{tissue}}
=
B_i^{\top}\widetilde H_i^{\mathrm{cell}}
\in
\mathbb{R}^{N_i^{\mathrm{tissue}}\times d_c}.
```

For tissue region `b`:

```math
u_{ib}
=
\sum_{a:B_{iab}=1}
\widetilde h_{ia}^{\mathrm{cell}}.
```

The initialized tissue state is:

```math
H_i^{\mathrm{tissue},0}
=
\left[
H_i^{\mathrm{tissue}}
\middle\Vert
U_i^{\mathrm{cell}\to\mathrm{tissue}}
\right].
```

The sum preserves a cellularity-sensitive statistic. A mean would instead be:

```math
\overline u_{ib}
=
\frac{1}{n_{ib}^{\mathrm{cell}}}
\sum_{a:B_{iab}=1}
\widetilde h_{ia}^{\mathrm{cell}},
```

but that is not the HACT sum used in the derivation above.

## 4. Coarse-Scale Context And Readout

The tissue graph operator is:

```math
\widetilde H_i^{\mathrm{tissue}}
=
\mathcal{C}_{\theta}^{\mathrm{tissue}}
\left(
H_i^{\mathrm{tissue},0},A_i^{\mathrm{tissue}}
\right).
```

The graph statistic is a tissue-node sum in the extended model:

```math
z_i^{\mathrm{HACT}}
=
\sum_{b=1}^{N_i^{\mathrm{tissue}}}
\widetilde h_{ib}^{\mathrm{tissue}}.
```

The short HACT-Net source also describes layerwise sum-concatenation:

```math
z_i^{\mathrm{HACT\text{-}Net}}
=
\mathrm{Concat}_{\ell}
\left(
\sum_{b=1}^{N_i^{\mathrm{tissue}}}
h_{ib}^{\mathrm{tissue},\ell}
\right).
```

The classifier then maps:

```math
\widehat p_i
=
\mathrm{softmax}
\left(
z_iW_{\mathrm{cls}}+b_{\mathrm{cls}}
\right).
```

## 5. Assignment Information Bottleneck

The transfer map is:

```math
\Pi_{B_i}(H)
=
B_i^{\top}H.
```

For any perturbation `Delta` satisfying:

```math
B_i^{\top}\Delta=0,
```

the coarse representation cannot change:

```math
\Pi_{B_i}(H+\Delta)
=
\Pi_{B_i}(H).
```

Thus every within-region redistribution with zero sum lies in the nullspace of
the assignment readout. If all tissue regions are nonempty and `B_i` has rank
`N_i^(tissue)`, the linear nullity is:

```math
N_i^{\mathrm{cell}}-N_i^{\mathrm{tissue}}.
```

Per feature coordinate, that many degrees of freedom are discarded before the
tissue graph context begins.

## 6. C/R/G/S Placement

```text
C:
    cell graph context, assignment transfer, then tissue graph context

R:
    tissue-node sum or layerwise sum-concatenation

G:
    cell graph, tissue graph, and explicit cell-to-tissue hierarchy

S:
    region or slide classification labels; cell and tissue structures are
    upstream representation choices
```

HACT's defining surviving statistic is not a patch first moment. It is a
coarse-scale statistic of fine-scale sums after a fixed assignment operator.

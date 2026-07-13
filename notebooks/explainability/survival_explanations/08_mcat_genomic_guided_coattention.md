# MCAT Genomic-Guided Co-Attention

MCAT makes multimodal survival attention conditional on genomic pathway or gene
embeddings.

## 1. Cross-Modal Map

Let pathology instances and genomic tokens be

```math
H\in\mathbb R^{N\times d},
\qquad
G\in\mathbb R^{M\times d}.
```

Genomic-guided co-attention produces

```math
A_{\mathrm{coattn}}\in\mathbb R^{N\times M},
```

where column `m` maps genomic token `m` to pathology patches.

## 2. Gene-Guided Visual Concept

For genomic token `m`, the corresponding patch map is

```math
z_m^{\mathrm{path}}
=\sum_{n=1}^N A_{nm}h_n.
```

The heatmap answers which visual instances were selected conditional on the
genomic token. It does not prove a molecular mechanism.

## 3. Survival Head

The fused representation is read by a survival model:

```math
\eta_i=\mathcal H
\left(\mathcal R
\left(\{z_m^{\mathrm{path}}\}_m,G\right)\right).
```

Attention weights are not automatically contributions to `eta_i`. A signed
co-attention credit requires the fused head and target score.

## 4. Overlapping Gene Sets

If gene embeddings are grouped into overlapping biological sets, one patch can
appear in several gene-guided maps. Summing these maps double-counts shared
pathway membership unless a normalization or attribution rule is specified.


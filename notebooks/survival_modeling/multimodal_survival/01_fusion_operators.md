# Fusion Operators

Let:

```math
z_i^{p}=\text{pathology representation},
\qquad
z_i^{g}=\text{genomic representation},
\qquad
z_i^{c}=\text{clinical representation}.
```

A multimodal survival model defines:

```math
z_i=\Phi(z_i^{p},z_i^{g},z_i^{c}).
```

## Concatenation

```math
z_i=[z_i^{p};z_i^{g};z_i^{c}].
```

Then:

```math
\eta_i=w^\top z_i.
```

This is simple but only models cross-modal interactions through downstream
layers.

## Additive Fusion

```math
z_i=W_pz_i^{p}+W_gz_i^{g}+W_cz_i^{c}.
```

Risk:

```math
\eta_i=w^\top z_i.
```

This assumes modalities contribute additively after projection.

## Gated Fusion

For modality
```math
m
```
:

```math
\alpha_i^{m}
=
\sigma(a_m^\top z_i^m+b_m).
```

Fused representation:

```math
z_i=\sum_m\alpha_i^mW_mz_i^m.
```

The gates learn modality reliability or relevance.

## Bilinear Fusion

For two modalities:

```math
z_i^{pg}
=
(U_pz_i^p)\odot(U_gz_i^g).
```

This models pairwise multiplicative interactions.

Tensor fusion augments with constants:

```math
\widetilde{z}_i^p=[z_i^p;1],
\qquad
\widetilde{z}_i^g=[z_i^g;1],
```

then:

```math
T_i=\widetilde{z}_i^p\otimes\widetilde{z}_i^g.
```

The tensor includes unimodal and interaction terms.

## Cross-Attention Fusion

Let pathology tokens be:

```math
P_i=\{p_{ij}\}_{j=1}^{n_i},
```

and omic/pathway tokens:

```math
G_i=\{g_{ir}\}_{r=1}^{R}.
```

Cross-attention from omics to pathology:

```math
\mathrm{Attn}(G_i,P_i,P_i)
=
\mathrm{softmax}
\left(
\frac{G_iW_Q(P_iW_K)^\top}{\sqrt{d}}
\right)
P_iW_V.
```

This creates token-level interactions.

## Product-Of-Experts Fusion

If each modality predicts survival:

```math
S_i^m(t),
```

a product-of-experts style combination is:

```math
S_i(t)
\propto
\prod_m[S_i^m(t)]^{\alpha_m}.
```

In cumulative hazard space:

```math
\Lambda_i(t)
=
\sum_m\alpha_m\Lambda_i^m(t).
```

This is natural because hazards add under independent multiplicative survival
experts.

## Dense Summary

```text
concatenation:
    weak interaction prior

gating:
    modality relevance

bilinear/tensor:
    multiplicative cross-modal interactions

cross-attention:
    token-level alignment

product of experts:
    additive cumulative hazards
```

Fusion is an inductive bias about how modalities interact to determine risk.

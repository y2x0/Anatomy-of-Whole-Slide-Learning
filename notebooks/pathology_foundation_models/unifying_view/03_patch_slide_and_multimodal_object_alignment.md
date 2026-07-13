# Patch, Slide, and Multimodal Object Alignment

## 1. Commuting Diagram Question

Patch pretraining followed by aggregation gives

```math
z_i^{\mathrm{patch}}
=\mathcal R\left(\{f_\phi(x_{ij})\}_j\right).
```

Slide pretraining gives

```math
z_i^{\mathrm{slide}}
=F_\psi(\{x_{ij},c_{ij}\}_j).
```

These are equivalent only if the slide operator `F_psi` factors through the
patch statistic retained by `R`.

## 2. Multimodal Alignment

For modalities `m=1,...,M`,

```math
z_i^{(m)}=F_m(X_i^{(m)}),
\qquad
z_i=\mathcal F(z_i^{(1)},\ldots,z_i^{(M)}).
```

Alignment can preserve cross-modal correspondences while losing modality-
specific information. Missing modality robustness is therefore part of the
representation claim.

## 3. Retrieval Alignment

A slide-text model supports a query function

```math
q\mapsto\arg\max_r\mathrm{sim}(z_q,z_r).
```

That readout differs from a survival or classification head even when it uses
the same embedding.


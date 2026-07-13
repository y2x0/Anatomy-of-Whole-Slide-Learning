# CONCH and PLIP: Image-Text Alignment

## 1. Paired Representation

For pathology image `x` and text `t`,

```math
u=\frac{f_{\mathrm{img}}(x)}{\|f_{\mathrm{img}}(x)\|_2},
\qquad
v=\frac{f_{\mathrm{text}}(t)}{\|f_{\mathrm{text}}(t)\|_2}.
```

The pair similarity is `u^T v`.

## 2. Symmetric Objective

For batch size `B`,

```math
\mathcal L_{I\to T}
=-\frac1B\sum_i
\log\frac{\exp(u_i^{\top}v_i/\tau)}
{\sum_j\exp(u_i^{\top}v_j/\tau)},
```

and the text-to-image term reverses the roles. CONCH and PLIP differ in data,
scale, architecture, and training details, but share this cross-modal geometry
as a central interpretive object.

## 3. Prompt Readout

For candidate pathology concepts `t_1,...,t_C`, zero-shot scores are

```math
s_c(x)=f_{\mathrm{img}}(x)^{\top}f_{\mathrm{text}}(t_c).
```

Prompt choice changes the classifier direction. A score is semantic alignment
under the VLM, not a calibrated pathology probability.

## 4. WSI Extension

For patch scores `s_jc`, a slide concept statistic might be

```math
z_{ic}=\sum_j a_{ij}s_{ijc}.
```

Its meaning combines image-text similarity, patch attention, and prevalence.


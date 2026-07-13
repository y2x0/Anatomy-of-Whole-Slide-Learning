# Phikon Histology Masked Modeling

Primary anchor:

- Filiot et al. "Scaling Self-Supervised Learning for Histopathology with
  Masked Image Modeling." medRxiv 2023.
  https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2
- Official Phikon implementation and release:
  https://github.com/owkin/HistoSSLscaling

The original Phikon study applies iBOT to histology and studies domain, data,
and model scaling. Its objective is masked self-distillation, not pixel-level
reconstruction.

## Histology View Distribution

For each tile, the reported setup samples two global and ten local crops. Crop
area fractions lie in:

```math
\alpha_g
\in
\left[
0.32,1
\right]
```

for global crops and:

```math
\alpha_{\ell}
\in
\left[
0.05,0.32
\right]
```

for local crops.

Random masked image modeling is applied to global crops. The masked proportion
is sampled in the reported range:

```math
\rho
\sim
\mathrm{Uniform}
\left(
0.10,0.50
\right)
```

when masking is active.

## Objective Map

For global crop `x^g`, mask `M`, visible teacher, and masked student:

```math
\mathcal{L}_{\mathrm{patch}}
=
-
\sum_{i\in M}
p_{t,i}(x^g)^{\top}
\log
p_{s,i}
\left(
\widehat x^g
\right).
```

The class-token objective also aligns views:

```math
\mathcal{L}_{\mathrm{global}}
=
\sum_{v_t}
\sum_{v_s\ne v_t}
H
\left(
p_t^{\mathrm{CLS}}(v_t),
p_s^{\mathrm{CLS}}(v_s)
\right).
```

## Dataset Scaling As Distribution Expansion

The study compares colorectal-only and pan-cancer datasets, culminating in more
than 40 million histology images across 16 cancer types. Formally, scaling
changes:

```math
p_{\mathrm{COAD}}(x)
\longrightarrow
p_{\mathrm{pan\text{-}cancer}}(x)
=
\sum_{c=1}^{16}
\pi_c p(x\mid c).
```

This expands support but also changes mixture weights. A model can improve
average transfer while losing fidelity for a low-weight tissue component.

## Masked Conditional Entropy

The irreducible difficulty at position `i` is:

```math
H
\left(
Z_i^t
\mid
X_O
\right).
```

Histology patterns predictable from neighborhood context have low conditional
entropy. Isolated mitoses, microorganisms, or rare atypical cells may have high
conditional entropy and be averaged by the optimal predictor.

## Patch-To-WSI Boundary

Even when masked modeling improves patch features on slide tasks, the WSI map
remains composite:

```math
X_i
\xrightarrow{\mathrm{tiling}}
\left\{
x_{ij}
\right\}
\xrightarrow{f_{\mathrm{Phikon}}}
\left\{
h_{ij}
\right\}
\xrightarrow{\mathcal{R}}
\widehat y_i.
```

The patch objective cannot be credited with spatial or aggregation behavior
introduced only by the downstream slide model.

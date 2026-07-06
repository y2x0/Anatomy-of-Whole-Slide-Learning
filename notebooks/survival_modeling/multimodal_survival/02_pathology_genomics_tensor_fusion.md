# Pathology-Genomics Tensor Fusion

Pathomic Fusion-style models combine histology and genomic representations using
gating and tensor interactions.

Let:

```math
z_i^{p}=\mathcal{R}_{p}(\mathcal{C}_{p}(H_i)),
\qquad
z_i^{g}=E_g(g_i).
```

## Gated Unimodal Representations

For pathology:

```math
\widetilde{z}_i^{p}
=
z_i^{p}\odot\sigma(W_pz_i^{p}+b_p).
```

For genomics:

```math
\widetilde{z}_i^{g}
=
z_i^{g}\odot\sigma(W_gz_i^{g}+b_g).
```

Gating controls which features enter cross-modal fusion.

## Tensor Fusion

Augment with constants:

```math
\bar{z}_i^{p}=[\widetilde{z}_i^{p};1],
\qquad
\bar{z}_i^{g}=[\widetilde{z}_i^{g};1].
```

Tensor product:

```math
T_i=\bar{z}_i^{p}\otimes\bar{z}_i^{g}.
```

This contains:

```text
pathology-only terms
genomics-only terms
pathology-genomics interaction terms
constant bias term
```

Flatten and project:

```math
z_i=\phi(W\mathrm{vec}(T_i)+b).
```

Survival:

```math
\eta_i=f_{\text{surv}}(z_i).
```

## Bilinear Interaction View

A linear Cox head after tensor fusion gives:

```math
\eta_i
=
\sum_{a,b}
\Theta_{ab}
\bar{z}_{ia}^{p}
\bar{z}_{ib}^{g}.
```

This is a bilinear risk surface:

```math
\eta_i
=
(\bar{z}_i^{p})^\top
\Theta
\bar{z}_i^{g}.
```

The model can represent:

```text
a morphology feature is risky only under a molecular subtype
a genomic alteration changes the meaning of histologic grade
```

## PORPOISE-Style Pan-Cancer View

PORPOISE extends the multimodal idea across many cancer types. Mathematically,
the challenge is that:

```math
P(T\mid z^p,z^g,\text{cancer type})
```

may have type-specific baselines and interactions.

A generic model can include cancer-type embedding $a_i$:

```math
z_i=\Phi(z_i^p,z_i^g,a_i),
\qquad
\eta_i=f(z_i).
```

or type-specific baselines:

```math
\lambda(t\mid z_i,a_i)
=
\lambda_{0,a_i}(t)\exp(\eta_i).
```

## Dense Summary

```math
\begin{aligned}
\widetilde{z}^{p}&=z^{p}\odot\sigma(W_pz^{p}+b_p),\\
\widetilde{z}^{g}&=z^{g}\odot\sigma(W_gz^{g}+b_g),\\
T&=[\widetilde{z}^{p};1]\otimes[\widetilde{z}^{g};1],\\
\eta&=f_{\text{surv}}(\mathrm{vec}(T)).
\end{aligned}
```

Tensor fusion makes cross-modal interactions explicit rather than hoping a late
MLP learns them from concatenation.

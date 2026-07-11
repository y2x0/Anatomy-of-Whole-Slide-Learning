# Cross-Slide Augmentation Is Not Contrast

Primary anchor:

- Chen et al. "Cross-Slide Augmentation for Whole Slide Image Classification
  Based on Class Activation Map." Scientific Reports 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12657889/

CAMCSA uses WSI class-activation scores to select significant instances and
mix them across slides. This is cross-slide interaction, not a contrastive
objective.

## Significant-Instance Selection

```math
I_a^{(k)}
=
\mathrm{TopK}
\left(
\left\{
s_{aj}
\right\}_{j=1}^{n_a},
k
\right),
```

```math
\mathcal{S}_a
=
\left\{
h_{aj}:j\in I_a^{(k)}
\right\}.
```

## Cross-Slide Mixing

```math
\widetilde{\mathcal{S}}_{ab}
=
\mathcal{M}_{\lambda}
\left(
\mathcal{S}_a,
\mathcal{S}_b
\right),
```

```math
\widetilde y_{ab}
=
\lambda y_a
+
\left(
1-\lambda
\right)y_b.
```

The paper operates on selected instance features and mixed targets for
classifier training.

## Contrastive Boundary

A contrastive objective normalizes a positive against alternatives:

```math
-\log
\frac{
\exp(s_{+}/\tau)
}{
\exp(s_{+}/\tau)
+
\sum_j\exp(s_j^{-}/\tau)
}.
```

Mixing instead changes the training law:

```math
p_{\mathrm{train}}
\longrightarrow
\left(
1-\rho
\right)p_{\mathrm{real}}
+
\rho p_{\mathrm{mixed}}.
```

## MIL Semantic Problem

Binary MIL labels are existential:

```math
Y
=
\max_j y_j.
```

Linear target interpolation is not the exact label law of an existential bag.
It is a regularizer rather than a literal MIL probability.

## Error Propagation

If selected instances contain nuisance evidence, mixing transports that
nuisance into synthetic bags from other slides. Cross-slide augmentation can
diversify lesions or disseminate shortcuts.

## C/R/G/S Placement

```math
\mathcal{R}
=
\text{CAM-selected subbag},
\qquad
\mathcal{S}
=
\text{mixed-label supervision}.
```

It does not introduce a contrastive candidate geometry.

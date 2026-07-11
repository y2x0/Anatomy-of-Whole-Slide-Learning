# Cross-Slide Augmentation Is Not Contrast

Primary anchor:

- Chen et al. "Cross-Slide Augmentation for Whole Slide Image Classification
  Based on Class Activation Map." Scientific Reports 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12657889/

CAMCSA uses WSI class-activation scores to select significant instances and
mix them across slides. This is cross-slide interaction, but it is not a
contrastive objective.

## Significant-Instance Selection

For slide `a`, let class-activation contribution scores be `s_{aj}` and select:

```math
I_a^{(k)}
=
\mathrm{TopK}
\left(
\left\{
s_{aj}
\right\}_{j=1}^{n_a},
k
\right).
```

The selected bag is:

```math
\mathcal{S}_a
=
\left\{
h_{aj}:j\in I_a^{(k)}
\right\}.
```

## Cross-Slide Mixing

For slides `a,b` and a mixing coefficient:

```math
\widetilde{\mathcal{S}}_{ab}
=
\mathcal{M}_{\lambda}
\left(
\mathcal{S}_a,
\mathcal{S}_b
\right),
```

with mixed target conceptually of the form:

```math
\widetilde y_{ab}
=
\lambda y_a
+
\left(
1-\lambda
\right)y_b.
```

The exact paper construction operates on selected instance features and mixed
labels for classifier training.

## Why It Is Not Contrastive

A contrastive loss requires a candidate relation and normalized comparison,
for example:

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

Cross-slide mixing instead enlarges the empirical training distribution:

```math
p_{\mathrm{train}}
\longrightarrow
\left(
1-\rho
\right)p_{\mathrm{real}}
+
\rho p_{\mathrm{mixed}}.
```

No positive-negative candidate identification is necessary.

## MIL Semantic Problem

For binary MIL, a bag label is existential:

```math
Y
=
\max_j y_j.
```

Linear label interpolation is not the exact label law of an existential bag.
The mixed target is a regularization device, not a literal probability derived
from the standard MIL assumption.

## Selection Error Propagation

If the selected set contains nuisance fraction `epsilon_a`, mixing transports
that nuisance into synthetic bags from other slides. Cross-slide augmentation
can diversify true evidence or disseminate selection shortcuts.

## Taxonomic Placement

```math
\mathcal{S}
=
\text{mixed-label supervision},
\qquad
\mathcal{R}
=
\text{CAM-selected subbag},
```

not:

```math
\mathcal{G}
=
\text{contrastive candidate geometry}.
```

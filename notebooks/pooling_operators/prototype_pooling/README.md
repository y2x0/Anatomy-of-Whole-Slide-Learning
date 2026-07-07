# Prototype Pooling

Prototype pooling represents a slide by how its patches assign to recurring
morphology prototypes.

The core question:

```text
Which morphologic types are present, and how much of each survives?
```

The generic form is:

```math
q_{ijm}
=
q_m(h_{ij}),
\qquad
p_{im}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}q_{ijm}.
```

The surviving statistic is a prevalence vector, residual vector, or mixture
summary over prototypes.

## C/R/G/S Placement

```text
G:
    prototype geometry, mixture covariance, or transport cost between prototypes

C:
    assignment to morphology prototypes or GMM components

R:
    prevalence, burden, residual, or Fisher-like prototype statistic

S:
    unsupervised prototype learning plus downstream slide-level supervision
```

Surviving statistic:

```math
z_i
=
\left\{
\frac{1}{n_i}\sum_j q_{ijm},
\frac{1}{n_i}\sum_j q_{ijm}(h_{ij}-c_m)
\right\}_{m=1}^{M}.
```

## Files

- `01_prototypes_as_quantized_measure.md`: assignments as a pushforward of the
  patch empirical measure.
- `02_counts_residuals_and_transport.md`: prevalence, residuals, and prototype
  geometry.
- `03_panther_gmm_statistics.md`: PANTHER-style GMM responsibilities and slide
  statistics.
- `04_failure_modes.md`: drift, collapse, finite-sample noise, and lost layout.

## Anchor Papers

- Song et al. "Morphological Prototyping for Unsupervised Slide Representation
  Learning in Computational Pathology" / PANTHER. CVPR 2024.
  https://arxiv.org/abs/2405.11643
- ProtoMIL / PAMIL prototype MIL are later related prototype-based pathology
  directions tracked in the private paper map.

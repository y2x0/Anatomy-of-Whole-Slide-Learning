# Additive Evidence Pooling

Additive evidence pooling represents the slide as a sum of patch-level
contributions.

The core question:

```text
How much signed evidence does each patch add to the bag?
```

The generic readout is:

```math
r_i
=
\sum_{j=1}^{n_i} e_\theta(u_{ij}).
```

## C/R/G/S Placement

```text
G:
    ignored unless context changes the evidence function

C:
    patch evidence map e_theta, optionally after graph or transformer context

R:
    sum or normalized sum of signed evidence

S:
    slide-level labels or survival outcomes shaping the evidence scale
```

Surviving statistic:

```math
r_i
=
\sum_j e_{ij}.
```

## Files

- `01_sum_of_patch_evidence.md`: additive readout and normalization choices.
- `02_signed_evidence_and_credit_assignment.md`: positive/negative evidence and
  exact contribution maps.
- `03_additive_survival_and_burden.md`: count-like risk and tissue burden.
- `04_failure_modes.md`: double counting, calibration, and missing interactions.

## Anchor Paper

- Javed et al. "Additive MIL: Intrinsically Interpretable Multiple Instance
  Learning for Pathology." NeurIPS 2022. https://arxiv.org/abs/2206.01794


# Partial Labels

Partial labels observe some latent variables but not all of them.

The supervision has the form:

```math
S_i^{\mathrm{obs}}
=
\{(a,y_a):a\in\mathcal{O}_i\},
```

where
```math
\mathcal{O}_i
```
is the observed subset of patches, regions, pairs, or
constraints.

## Files

- `01_observed_subset_constraints.md`: masking, partial likelihood, and missingness.
- `02_region_labels_and_sparse_annotations.md`: region labels as aggregate constraints.
- `03_ranking_and_order_constraints.md`: ranking-aware partial annotations.
- `04_pu_and_semi_supervised_risks.md`: labeled positives, unlabeled instances, and consistency.
- `05_failure_modes.md`: missing-not-at-random labels and annotation bias.

## C/R/G/S Placement

```text
G:
    annotation geometry, regions, or selected coordinates

C:
    features shaped by labeled and unlabeled subsets

R:
    readout constrained by partial observations

S:
    subset of latent truth, not complete truth
```

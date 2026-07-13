# Unifying View

This folder places weak supervision in the repository-wide framework.

The key object is the supervision channel:

```math
U
\to
S^{\mathrm{obs}}.
```

The model never sees complete latent truth
```math
U
```
. It sees
```math
S^{\mathrm{obs}}
```
,
may generate extra targets
```math
\widehat U_t
```
, and optimizes a surrogate objective.

## Files

- `01_supervision_channel_decomposition.md`: latent truth, observed signal, and loss.
- `02_crgs_supervision_matrix.md`: where supervision enters C/R/G/S.
- `03_paper_placement_matrix.md`: anchor papers by supervision type.
- `04_noise_and_identifiability_matrix.md`: what is identifiable under each signal.
- `05_design_checklist.md`: questions every weak-supervision method should answer.
- `06_information_and_elbo_view.md`: data processing, missing information, and variational bounds.
- `07_assumption_theorem_template.md`: theorem-style reporting block for method notes.
- `08_toy_counterexamples.md`: minimal worlds where weak supervision cannot identify the desired latent object.

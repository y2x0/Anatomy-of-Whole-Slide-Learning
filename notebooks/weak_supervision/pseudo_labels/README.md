# Pseudo-Labels

Pseudo-label methods create supervision from a model, teacher, attention rule,
clusterer, or heuristic.

The supervision is:

```math
\widehat S_i
=
\Psi_{\theta,\phi}(H_i,G_i,S_i),
```

where $\Psi$ may depend on the current model, a frozen teacher, an external
clusterer, or a previous training stage.

## Files

- `01_self_training_as_latent_variable_optimization.md`: pseudo-labels as hard latent assignments.
- `02_clam_top_bottom_k_constraints.md`: CLAM-style attention extremes.
- `03_teacher_student_and_consistency.md`: mean teacher, noisy student, and consistency.
- `04_pseudo_bags_and_distillation.md`: DTFD-MIL and coarse-to-fine pseudo-bags.
- `05_failure_modes.md`: confirmation bias, threshold bias, and drifting targets.

## C/R/G/S Placement

```text
G:
    pseudo-labels may be region-, graph-, or coordinate-dependent

C:
    pseudo-labels shape embeddings and instance classifiers

R:
    pseudo-labels may select top-k, pseudo-bags, prototypes, or attention regions

S:
    observed labels plus generated labels
```

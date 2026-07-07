# Pseudo-Labels

Pseudo-label methods create generated targets from a model, teacher, attention
rule, clusterer, or heuristic. They are not usually new observations from the
world; they are algorithmic consequences of the observed supervision.

The observed supervision is:

```math
S_i^{\mathrm{obs}}
\sim
Q_\alpha(S\mid U_i,H_i,G_i).
```

The generated target is:

```math
\widehat U_{i,t}
=
\Psi_t(H_i,G_i,S_i^{\mathrm{obs}},\theta_t,\mathcal{D}),
```

where $\Psi$ may depend on the current model, a frozen teacher, an external
clusterer, or a previous training stage.

## Files

- `01_self_training_as_latent_variable_optimization.md`: pseudo-labels as hard latent assignments.
- `02_clam_top_bottom_k_constraints.md`: CLAM-style attention extremes.
- `03_teacher_student_and_consistency.md`: mean teacher, noisy student, and consistency.
- `04_pseudo_bags_and_distillation.md`: DTFD-MIL and coarse-to-fine pseudo-bags.
- `05_failure_modes.md`: confirmation bias, threshold bias, and drifting targets.
- `06_cluster_to_conquer.md`: cluster sampling, weak patch CE, slide CE, and KL attention regularization.

## C/R/G/S Placement

```text
G:
    pseudo-labels may be region-, graph-, or coordinate-dependent

C:
    pseudo-labels shape embeddings and instance classifiers

R:
    pseudo-labels may select top-k, pseudo-bags, prototypes, or attention regions

S:
    observed labels plus generated latent targets
```

# Full Fine-Tuning

Full fine-tuning allows downstream gradients to update the foundation encoder.

```math
\phi
\leftarrow
\phi_0
-
\alpha
\nabla_\phi
\mathcal{L}_{\mathrm{task}}.
```

## Files

- `01_end_to_end_gradient_paths.md`: how task loss reaches patches and encoder layers.
- `02_regularized_representation_drift.md`: weight-space and function-space drift.
- `03_catastrophic_forgetting_and_small_cohorts.md`: overfitting and loss of pretrained invariances.
- `04_failure_modes.md`: compute limits, shortcut adaptation, and unstable evidence.

## C/R/G/S Placement

```text
G:
    can be updated if geometry-aware layers are trainable

C:
    encoder and context operator both receive task gradients

R:
    readout may also be learned end to end

S:
    downstream labels shape the whole pipeline
```

## Dense Summary

Full fine-tuning has maximum adaptation capacity and maximum risk of
task-specific overfitting.

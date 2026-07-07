# Parameter-Efficient Tuning

Parameter-efficient fine-tuning adapts a foundation model through a restricted
parameter subspace.

The generic form is:

```math
\phi
=
\phi_0+\Delta_\eta,
\qquad
\Delta_\eta\in\mathcal{A}_{\mathrm{small}}.
```

The constraint set $\mathcal{A}_{\mathrm{small}}$ is the method.

## Files

- `01_lora_low_rank_updates.md`: low-rank matrix perturbations.
- `02_adapters_as_residual_bottlenecks.md`: residual bottleneck modules.
- `03_bias_norm_and_head_tuning.md`: tiny trainable subsets and normalization.
- `04_rank_capacity_and_interference.md`: capacity, rank, and layer placement.
- `05_failure_modes.md`: under-adaptation, overfitting, and shortcut drift.

## C/R/G/S Placement

```text
G:
    inherited from the pretrained architecture, sometimes coordinate-aware

C:
    adapted through low-rank or residual modules

R:
    may be frozen or adapted depending on layer placement

S:
    downstream labels shape a small parameter subspace
```

## Dense Summary

PEFT asks whether the downstream task can be solved by a small perturbation of a
large pretrained computation.

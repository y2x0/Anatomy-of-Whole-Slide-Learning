# Mean Pooling

Mean pooling is the simplest aggregation operator:

```math
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}u_{ij}.
```

It answers:

```text
What is the average instance state?
```

This folder treats mean pooling as a first-moment estimator, then separates raw
mean pooling from mean pooling after a context operator.

## C/R/G/S Placement

```text
G:
    ignored by the readout unless geometry has already been encoded into u_ij

C:
    identity for raw mean; graph, attention, transformer, or sequence context
    for mean-after-context

R:
    normalized first moment

S:
    slide-level task loss shaping the encoder and context operator
```

Surviving statistic:

```math
z_i
=
\mathbb{E}_{u\sim\mu_i}[u].
```

## Files

- `01_mean_as_first_moment.md`: empirical measure, first moment, and invariance.
- `02_mean_after_context.md`: mean readout after graph, attention, or sequence
  context.
- `03_mean_failure_modes.md`: dilution, collisions, nuisance mass, and gradient
  spreading.

## Core Claim

Mean pooling is not weak because it is simple. It is weak when the task-relevant
signal is not identifiable from a first moment.

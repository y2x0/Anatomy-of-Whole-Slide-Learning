# Max And Extreme Pooling

Max pooling is the canonical sparse-positive MIL operator.

For instance scores:

```math
s_{ij}
=
g_\theta(u_{ij}),
```

hard max pooling computes:

```math
z_i
=
\max_j s_{ij}.
```

It answers:

```text
Is there at least one strongly positive instance?
```

## Files

- `01_max_as_extreme_statistic.md`: max as an extreme-value functional.
- `02_logsumexp_softmax_limits.md`: smooth max, temperature, and gradients.
- `03_sparse_positive_mil.md`: MIL OR assumptions and failure modes.

## Core Claim

Max pooling is not a prevalence estimator. It is an extreme-evidence detector.
That is exactly why it works for sparse positives and fails for diffuse or
multi-region evidence.

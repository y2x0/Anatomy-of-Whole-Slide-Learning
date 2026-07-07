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

## C/R/G/S Placement

```text
G:
    ignored unless geometry changes the instance score before max

C:
    instance scoring map g_theta, optionally after a context operator

R:
    maximum, smooth maximum, top-k, or noisy-or extreme readout

S:
    bag label under a positive-instance or sparse-evidence assumption
```

Surviving statistic:

```math
z_i
=
\sup_{u\in\mathrm{supp}(\mu_i)}g_\theta(u).
```

## Files

- `01_max_as_extreme_statistic.md`: max as an extreme-value functional.
- `02_logsumexp_softmax_limits.md`: smooth max, temperature, and gradients.
- `03_sparse_positive_mil.md`: MIL OR assumptions and failure modes.
- `04_topk_quantile_generalized_means.md`: upper-tail statistics between mean
  and max.

## Core Claim

Max pooling is not a prevalence estimator. It is an extreme-evidence detector.
That is exactly why it works for sparse positives and fails for diffuse or
multi-region evidence.

# Noisy-Or Pooling

Noisy-or pooling is the probabilistic version of the MIL positive-instance
assumption.

The core question:

```text
What is the probability that at least one instance triggers the bag label?
```

For instance probabilities
```math
p_{ij}
```
:

```math
P(y_i=1\mid H_i)
=
1-\prod_{j=1}^{n_i}(1-p_{ij}).
```

## C/R/G/S Placement

```text
G:
    ignored unless instance probabilities contain context

C:
    instance probability model p_ij = sigmoid(g_theta(u_ij))

R:
    probabilistic OR over independent instance failures

S:
    bag label under an at-least-one-positive assumption
```

Surviving statistic:

```math
r_i
=
1-\prod_j(1-p_{ij}).
```

## Files

- `01_noisy_or_bag_probability.md`: OR probability and independence assumption.
- `02_log_space_gradients_and_limits.md`: stable computation and gradients.
- `03_noisy_or_vs_max_vs_sum.md`: relation to extreme and additive evidence.
- `04_failure_modes.md`: identifiability, saturation, and correlated instances.

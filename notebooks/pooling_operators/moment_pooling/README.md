# Moment Pooling

Moment pooling represents a slide by expectations of chosen feature functions.

The core question:

```text
Which expectations of the instance distribution should survive?
```

Given contextualized instance states $u_{ij}$ and empirical measure:

```math
\mu_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{u_{ij}},
```

moment pooling computes:

```math
z_i
=
\left(
\int \phi_1(u)\,d\mu_i(u),
\ldots,
\int \phi_M(u)\,d\mu_i(u)
\right).
```

## C/R/G/S Placement

```text
G:
    ignored unless geometry is encoded into u_ij or the moment functions

C:
    identity, learned feature map, or upstream context operator

R:
    empirical expectations of fixed or learned moment functions

S:
    task loss shaping learned moment maps, or unsupervised/statistical choice
```

Surviving statistic:

```math
z_i
=
\mathbb{E}_{u\sim\mu_i}[\Phi(u)].
```

## Files

- `01_moments_as_statistics.md`: first and higher moments as readouts.
- `02_second_order_and_covariance.md`: covariance, bilinear features, and
  interaction surrogates.
- `03_learned_moment_maps_and_kernel_means.md`: Deep Sets, learned moments, and
  kernel mean embeddings.
- `04_failure_modes.md`: moment collisions, sample noise, and lost layout.

## Anchor Ideas

- Deep Sets: transform and sum as a universal set-learning template.
- Distribution representations: moments and kernel means as finite statistics
  of the empirical patch measure.


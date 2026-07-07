# Moment Pooling Failure Modes

Moment pooling fails when the selected expectations are insufficient for the
task.

## 1. Moment Collision

Two distributions can share the same chosen moments:

```math
\mathbb{E}_{\mu_i}[\Phi(u)]
=
\mathbb{E}_{\mu_k}[\Phi(u)]
```

while:

```math
\mu_i\ne\mu_k.
```

The downstream head cannot separate them because it only sees $z_i$.

## 2. Rare Event Dilution

If a rare positive morphology occupies fraction $\epsilon_i$ of patches, then a
bounded moment coordinate contributes at most:

```math
O(\epsilon_i).
```

Unless $\Phi$ amplifies rare evidence, moment pooling behaves like prevalence
estimation and can miss sparse positives.

## 3. High-Dimensional Sample Noise

With $M$ moment coordinates:

```math
z_i
\in
\mathbb{R}^{M},
```

finite-patch estimation noise grows as the statistic becomes richer. A large
$M$ can represent more distribution shape but may be noisy for small tissue
regions or aggressive subsampling.

## 4. Lost Layout

Moment pooling is invariant to patch permutation:

```math
\frac{1}{n_i}\sum_j\Phi(u_{ij})
=
\frac{1}{n_i}\sum_j\Phi(u_{i\pi(j)}).
```

If layout matters, moments of individual states are insufficient unless context
encoded spatial relationships into $u_{ij}$ before pooling.

## 5. Learned Shortcut Moments

If $\Phi=\phi_\theta$ is trained by slide labels, it may learn nuisance
statistics:

```text
scanner style
stain intensity
tissue amount
cohort artifact
```

The pooling formula remains mathematically clean, but the learned moments may
not be biological.

## Dense Summary

Moment pooling asks:

```text
which expectations are enough?
```

Its failure mode is always an insufficiency statement:

```math
T(\mu_i)=T(\mu_k)
\quad
\text{but}
\quad
y_i\ne y_k.
```


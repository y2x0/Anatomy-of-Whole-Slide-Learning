# Distribution Pooling Failure Modes

Distribution pooling is broad, but every finite statistic still discards
information.

## 1. Statistic Mismatch

If the readout is:

```math
z_i=T(\mu_i),
```

then any two slides with:

```math
T(\mu_i)=T(\mu_k)
```

are indistinguishable to the head. Choosing the wrong
```math
T
```
creates collisions.

## 2. Sample Complexity

High-resolution statistics require many patches. If a statistic has dimension
```math
M
```
, empirical estimation error can dominate when:

```math
M
\gg
n_i
```

or when rare bins have very small mass.

## 3. Lost Spatial Layout

Distribution pooling is invariant to permutation:

```math
T\left(
\frac{1}{n_i}\sum_j\delta_{u_{ij}}
\right)
=
T\left(
\frac{1}{n_i}\sum_j\delta_{u_{i\pi(j)}}
\right).
```

Two slides with identical patch distributions but different tissue arrangement
collide.

## 4. Metric Mismatch

For transport or kernel pooling, the geometry matters:

```math
k(u,v)
\quad
\text{or}
\quad
C_{mn}.
```

If the metric reflects stain or scanner variation more than morphology, the
distribution comparison will be wrong.

## 5. Tail Instability

Quantiles near the tail:

```math
Q_i(0.99)
```

depend on very few patches and can be unstable under artifacts or sampling.

## Dense Summary

Distribution pooling fails when:

```text
the statistic is too weak,
the statistic is too noisy,
or the metric sees the wrong geometry.
```

It should always be paired with the question: which equivalence relation does
```math
T(\mu)
```
impose on slides?

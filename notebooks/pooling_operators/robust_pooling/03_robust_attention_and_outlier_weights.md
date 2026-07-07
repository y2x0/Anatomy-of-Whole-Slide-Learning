# Robust Attention And Outlier Weights

Attention can be made more robust by bounding the influence of any one instance.

Standard attention uses:

```math
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_\ell\exp(s_{i\ell})}.
```

The maximum weight can approach $1$:

```math
\max_j a_{ij}
\to
1.
```

## Weight Clipping

A clipped attention readout should enforce the cap on the normalized weights:

```math
a_{ij}
\le
\tau.
```

with $\tau\in[1/n_i,1]$. The constrained attention vector can be written as a
capped-simplex problem:

```math
a_i
=
\arg\min_{a\in\Delta^{n_i-1}}
\left[
-\sum_{j=1}^{n_i}a_j s_{ij}
+\lambda\sum_{j=1}^{n_i}a_j\log a_j
\right]
\quad
\mathrm{subject\ to}
\quad
0\le a_j\le\tau.
```

Here:

```math
\Delta^{n_i-1}
=
\{a:a_j\ge 0,\ \sum_j a_j=1\}.
```

This prevents a single patch from taking all mass.

## Entropy Regularization

Add an entropy term:

```math
\mathcal{L}
=
\mathcal{L}_{\text{task}}
-
\lambda H(a_i),
```

where:

```math
H(a_i)
=
-\sum_j a_{ij}\log a_{ij}.
```

Encouraging entropy prevents collapse. Penalizing entropy encourages sparse
selection. The sign must match the desired inductive bias.

## Bounded Influence

For attention pooling:

```math
z_i
=
\sum_j a_{ij}v_{ij}.
```

If:

```math
a_{ij}\le\tau
```

and values are bounded:

```math
\|v_{ij}\|\le B,
```

then one instance can contribute at most:

```math
\tau B
```

to the pooled norm.

## C/R/G/S Placement

```text
G:
    none unless outlier detection uses geometry

C:
    attention score and outlier score

R:
    bounded-weight weighted average

S:
    task loss plus robustness regularizer or clipping rule
```

## Dense Summary

Robust attention modifies the learned measure:

```math
\nu_{i,\theta}
=
\sum_j a_{ij}\delta_{v_{ij}}
```

so no single atom can dominate unless the design explicitly allows it.

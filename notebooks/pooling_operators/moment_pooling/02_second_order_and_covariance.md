# Second Order And Covariance

First moments cannot see dispersion or co-activation. Second-order pooling
keeps quadratic statistics.

Let:

```math
m_i
=
\frac{1}{n_i}\sum_j u_{ij}.
```

The raw second moment is:

```math
M_i
=
\mathbb{E}_{\mu_i}[uu^\top]
=
\frac{1}{n_i}\sum_j u_{ij}u_{ij}^\top.
```

The covariance is:

```math
\Sigma_i
=
\mathbb{E}_{\mu_i}[(u-m_i)(u-m_i)^\top]
=
M_i-m_i m_i^\top.
```

## What Second Order Adds

Two slides can share the same mean:

```math
m_i=m_k
```

but have different covariance:

```math
\Sigma_i\ne\Sigma_k.
```

Mean pooling cannot distinguish them. Second-order pooling can.

## Bilinear Readout

A task head can score covariance through a matrix $A$:

```math
r_i
=
\langle A,M_i\rangle
=
\mathrm{tr}(A^\top M_i).
```

Expanding:

```math
r_i
=
\frac{1}{n_i}\sum_j u_{ij}^\top A u_{ij}.
```

So a bilinear moment head is still additive over patches after applying a
quadratic patch evidence function.

## Cross-Feature Co-Occurrence

If $u_a$ measures one morphology coordinate and $u_b$ measures another, then:

```math
M_{iab}
=
\frac{1}{n_i}\sum_j u_{ija}u_{ijb}
```

captures within-patch co-activation. It does not capture two different patches
co-occurring spatially unless context has already encoded neighborhood
information into $u_{ij}$.

## Compression

The full covariance has:

```math
O(d^2)
```

coordinates. Practical readouts often use:

```text
diagonal covariance:
    O(d)

low-rank bilinear map:
    A = BB^top

random projections:
    phi(u) = (r_1^top u)^2, ..., (r_M^top u)^2
```

## C/R/G/S Placement

```text
G:
    absent unless context puts geometry into u_ij

C:
    optional context or feature map before quadratic statistics

R:
    first and second empirical moments

S:
    downstream loss choosing which covariance directions matter
```

## Dense Summary

Second-order pooling preserves:

```math
z_i
=
(m_i,\Sigma_i).
```

It repairs some mean collisions but still summarizes the slide as a distribution
of individual states. It does not automatically preserve spatial arrangements
or cross-patch motifs.


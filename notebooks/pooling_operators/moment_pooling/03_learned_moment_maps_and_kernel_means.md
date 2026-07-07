# Learned Moment Maps And Kernel Means

Deep Sets-style pooling learns the moment functions:

```math
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\phi_\theta(u_{ij}).
```

This is moment pooling where:

```math
\Phi=\phi_\theta.
```

The task head then computes:

```math
\widehat y_i
=
\rho_\theta(z_i).
```

## Learned Moment Interpretation

Each coordinate of $\phi_\theta(u)$ is a learned statistic:

```math
z_{ir}
=
\frac{1}{n_i}\sum_j \phi_{\theta,r}(u_{ij}).
```

If $\phi_{\theta,r}$ behaves like an indicator of a morphology, $z_{ir}$
estimates prevalence. If it behaves like a risk score, $z_{ir}$ estimates
average risk evidence.

## Kernel Mean Embedding

A kernel mean embedding represents a distribution by:

```math
m_{\mu_i}
=
\int k(u,\cdot)\,d\mu_i(u).
```

Empirically:

```math
m_{\mu_i}
=
\frac{1}{n_i}
\sum_j k(u_{ij},\cdot).
```

If the kernel is characteristic, equality of population embeddings implies
equality of distributions:

```math
m_{\mu_i}=m_{\mu_k}
\quad
\Longleftrightarrow
\quad
\mu_i=\mu_k.
```

Finite neural maps approximate this idea with:

```math
\phi_\theta(u)
\approx
(k(u,r_1),\ldots,k(u,r_M))
```

or learned feature maps.

## Random Feature Sketch

A random feature map:

```math
\varphi(u)
\in
\mathbb{R}^{M}
```

approximates a kernel:

```math
k(u,v)
\approx
\varphi(u)^\top \varphi(v).
```

The pooled sketch is:

```math
z_i
=
\frac{1}{n_i}\sum_j\varphi(u_{ij}).
```

Then:

```math
z_i^\top z_k
\approx
\mathbb{E}_{u\sim\mu_i,v\sim\mu_k}[k(u,v)].
```

## C/R/G/S Placement

```text
G:
    kernel geometry or learned feature geometry

C:
    feature map phi_theta or random/kernel map

R:
    empirical expectation of mapped features

S:
    task labels if phi is learned; unsupervised/statistical objective otherwise
```

## Dense Summary

Learned moment pooling is the bridge between mean pooling and distribution
embedding:

```math
\boxed{
z_i
=
\mathbb{E}_{\mu_i}[\phi_\theta(u)]
}
```

The better $\phi_\theta$ separates task-relevant morphology, the less damaging
the final average becomes.


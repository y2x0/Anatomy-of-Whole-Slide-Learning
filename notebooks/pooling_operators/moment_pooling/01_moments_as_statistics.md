# Moments As Statistics

Moment pooling begins with the empirical measure:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{u_{ij}}.
```

A moment readout chooses functions:

```math
\phi_m:\mathbb{R}^{d}\to\mathbb{R},
\qquad
m=1,\ldots,M,
```

and computes:

```math
z_{im}
=
\int \phi_m(u)\,d\mu_i(u)
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\phi_m(u_{ij}).
```

Thus:

```math
z_i
=
\mathbb{E}_{u\sim\mu_i}[\Phi(u)],
\qquad
\Phi(u)
=
(\phi_1(u),\ldots,\phi_M(u)).
```

## First Moment

Mean pooling is the special case:

```math
\Phi(u)=u.
```

Then:

```math
z_i
=
\mathbb{E}_{\mu_i}[u].
```

This preserves only average feature mass.

## Scalar Evidence Moments

If an instance evidence function is:

```math
e_\theta(u)
\in
\mathbb{R},
```

then an average evidence statistic is:

```math
r_i
=
\mathbb{E}_{\mu_i}[e_\theta(u)]
=
\frac{1}{n_i}\sum_j e_\theta(u_{ij}).
```

This is not max MIL. It says every instance contributes to the slide score
through an average.

## Higher Raw Moments

Coordinate moments use monomials:

```math
\phi_{\alpha}(u)
=
u_1^{\alpha_1}\cdots u_d^{\alpha_d},
```

where:

```math
|\alpha|
=
\sum_{r=1}^{d}\alpha_r.
```

First moments have total degree one. Second moments have total degree two. Higher
moments capture more distribution shape but grow quickly in dimension.

## Central Moments

Let:

```math
m_i
=
\mathbb{E}_{\mu_i}[u].
```

Central moment functions are:

```math
\phi_{\alpha}^{\mathrm{central}}(u)
=
(u_1-m_{i1})^{\alpha_1}\cdots(u_d-m_{id})^{\alpha_d}.
```

These depend on the slide mean, so the readout has two stages:

```math
\mu_i
\xrightarrow{\text{mean}}
m_i
\xrightarrow{\text{central moments}}
z_i.
```

## C/R/G/S Placement

```text
G:
    not used unless u_ij already contains geometric context

C:
    optional transform u_ij = C(h_ij)

R:
    empirical expectation of Phi(u)

S:
    task loss if Phi is learned, none if Phi is fixed
```

## Dense Summary

Moment pooling is:

```math
\boxed{
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\Phi(u_{ij})
}
```

The operator is simple, but its power depends entirely on
```math
\Phi
```
. A weak
```math
\Phi
```
 gives mean-pooling collisions. A rich
```math
\Phi
```
approaches a learned
distribution embedding.

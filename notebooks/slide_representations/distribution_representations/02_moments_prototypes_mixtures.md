# Moments, Prototypes, And Mixtures

Distribution representations differ by how they compress $\mu_i$.

## Moment Family

Given basis functions:

```math
\phi_1,\ldots,\phi_M:\mathbb{R}^{d}\to\mathbb{R},
```

the slide statistic is:

```math
z_i
=
\left(
\int\phi_1(h)\,d\mu_i(h),
\ldots,
\int\phi_M(h)\,d\mu_i(h)
\right).
```

In empirical form:

```math
z_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\Phi(h_{ij}),
```

where:

```math
\Phi(h)=(\phi_1(h),\ldots,\phi_M(h)).
```

This is exactly the Deep Sets statistic with a fixed interpretation:

```text
slide = estimated expectations of morphology features
```

## Kernel Mean Embedding

A kernel mean embedding maps a distribution to a reproducing-kernel Hilbert
space:

```math
m_{\mu_i}
=
\int k(h,\cdot)\,d\mu_i(h).
```

Empirically:

```math
m_{\mu_i}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}k(h_{ij},\cdot).
```

If the kernel is characteristic, then:

```math
m_{\mu_i}=m_{\mu_k}
\quad
\Longleftrightarrow
\quad
\mu_i=\mu_k.
```

Finite neural summaries approximate this idea with learned feature maps:

```math
z_i
=
\frac{1}{n_i}\sum_j\phi_\theta(h_{ij}).
```

## Prototype Prevalence

Prototype methods choose morphology exemplars:

```math
c_1,\ldots,c_M\in\mathbb{R}^{d}.
```

Assignments:

```math
q_{jm}
=
q_m(h_{ij}).
```

Prevalence:

```math
p_{im}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}q_{jm}.
```

The slide statistic is:

```math
p_i\in\Delta^{M-1}.
```

This is interpretable when prototypes correspond to recurring morphology.

## Mixture Model Representation

Assume patch embeddings are generated from a mixture:

```math
p(h)
=
\sum_{m=1}^{M}\pi_m
\mathcal{N}(h;\mu_m,\Sigma_m).
```

For slide $i$, infer responsibilities:

```math
\gamma_{ijm}
=
\frac{
\pi_m\mathcal{N}(h_{ij};\mu_m,\Sigma_m)
}{
\sum_{r=1}^{M}\pi_r\mathcal{N}(h_{ij};\mu_r,\Sigma_r)
}.
```

Then estimate slide-specific mixture statistics:

```math
\widehat{\pi}_{im}
=
\frac{1}{n_i}
\sum_j\gamma_{ijm}.
```

One can also store component-specific residuals:

```math
r_{im}
=
\frac{1}{n_i}
\sum_j
\gamma_{ijm}
\Sigma_m^{-1/2}(h_{ij}-\mu_m).
```

The slide representation can be:

```math
z_i
=
[\widehat{\pi}_{i1},r_{i1},\ldots,\widehat{\pi}_{iM},r_{iM}].
```

PANTHER-style morphological prototyping fits naturally in this family: recurring
patch morphologies are modeled as mixture components, and the slide is
summarized by component usage and deviations.

## Fisher Vector View

For a generative model $p_\theta(h)$, the Fisher score is:

```math
G_{\theta}(\mu_i)
=
\int
\nabla_\theta\log p_\theta(h)
d\mu_i(h).
```

Empirically:

```math
G_{\theta}(\mu_i)
=
\frac{1}{n_i}
\sum_j
\nabla_\theta\log p_\theta(h_{ij}).
```

This represents a slide by how its patch distribution would change the
parameters of a global morphology model.

## Dense Summary

```text
moments:
    expectations of chosen functions

kernel mean:
    distribution embedding

prototype prevalence:
    soft histogram over morphology exemplars

mixture model:
    component weights and residuals

Fisher vector:
    gradient of log likelihood under a global distribution model
```

All are answers to the same question:

```text
Which finite statistic of the patch distribution should represent the slide?
```

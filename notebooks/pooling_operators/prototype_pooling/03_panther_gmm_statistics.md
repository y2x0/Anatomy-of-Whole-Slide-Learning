# PANTHER GMM Statistics

PANTHER-style morphological prototyping treats recurring patch morphologies as
mixture components. The slide is represented by how its patch distribution uses
those components.

## Global Mixture Model

Let a global patch distribution be modeled by:

```math
p(h)
=
\sum_{m=1}^{M}
\pi_m
\mathcal{N}(h;\mu_m,\Sigma_m).
```

The prototypes are the mixture components:

```math
\{(\pi_m,\mu_m,\Sigma_m)\}_{m=1}^{M}.
```

For patch
```math
h_{ij}
```
, the responsibility of component
```math
m
```
is:

```math
\gamma_{ijm}
=
\frac{
\pi_m\mathcal{N}(h_{ij};\mu_m,\Sigma_m)
}{
\sum_{r=1}^{M}
\pi_r\mathcal{N}(h_{ij};\mu_r,\Sigma_r)
}.
```

Responsibilities are soft prototype assignments:

```math
\sum_{m=1}^{M}\gamma_{ijm}=1.
```

## Slide-Level Mixture Weights

The slide-specific component prevalence is:

```math
\widehat\pi_{im}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\gamma_{ijm}.
```

This estimates how much of slide
```math
i
```
 belongs to morphology component
```math
m
```
.

The simplest PANTHER-style slide vector is:

```math
z_i
=
(\widehat\pi_{i1},\ldots,\widehat\pi_{iM}).
```

## Residual Deviations

A component-specific residual can be whitened by covariance:

```math
r_{ijm}
=
\Sigma_m^{-1/2}(h_{ij}-\mu_m).
```

The slide-level residual statistic is:

```math
\bar r_{im}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\gamma_{ijm}
\Sigma_m^{-1/2}(h_{ij}-\mu_m).
```

A richer slide representation is:

```math
z_i
=
[
\widehat\pi_{i1},
\bar r_{i1},
\ldots,
\widehat\pi_{iM},
\bar r_{iM}
].
```

This is close in spirit to Fisher-vector-style summaries: the slide is described
by how its patch distribution uses and deviates from global morphology
components.

## What Makes This Pooling?

The readout is:

```math
\mathcal{R}(H_i)
=
\left\{
\frac{1}{n_i}\sum_j \gamma_{ijm},
\frac{1}{n_i}\sum_j \gamma_{ijm}\Sigma_m^{-1/2}(h_{ij}-\mu_m)
\right\}_{m=1}^{M}.
```

It is not attention pooling because the weights are not trained per downstream
class. They are assignment weights to global morphology components.

It is not mean pooling because the output is not one average embedding. It is a
structured distribution summary over prototypes.

## Downstream Head

A downstream task head sees:

```math
\widehat y_i
=
\mathcal{H}_\theta(z_i).
```

For survival with a Cox head:

```math
\eta_i
=
w^\top z_i.
```

Then each prototype prevalence or residual coordinate receives a direct risk
coefficient.

## Dense Summary

PANTHER-style pooling can be written:

```math
H_i
\xrightarrow{\text{GMM responsibilities}}
\gamma_{ijm}
\xrightarrow{\text{slide averages}}
z_i.
```

The surviving statistic is:

```text
component prevalence plus component-specific deviation
```

This is a distribution readout, not a patch-selection readout.

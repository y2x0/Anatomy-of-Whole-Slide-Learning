# Neural Hazards And Mixtures

A continuous-time neural hazard model parameterizes:

```math
\lambda_\theta(t\mid z)
=
\rho(g_\theta(t,z)),
```

where $\rho$ enforces positivity.

The most direct choice is:

```math
\rho(a)=\mathrm{softplus}(a).
```

## Time As An Input

The network receives both time and covariates:

```math
g_\theta(t,z)
=
\mathrm{MLP}_{\theta}([e(t),z]).
```

Here $e(t)$ may be:

```text
raw time
log time
Fourier features
spline basis
learned time embedding
```

The model can learn time-varying effects:

```math
\frac{\partial}{\partial t}
\left[
\log \frac{\lambda(t\mid z_i)}
{\lambda(t\mid z_j)}
\right]
\ne 0.
```

This escapes the Cox proportional hazards assumption.

## Additive Hazard Decomposition

One useful parameterization is:

```math
\lambda_\theta(t\mid z)
=
\rho\left(
b_\theta(t)
+
r_\theta(t,z)
\right).
```

Here:

```text
b_theta(t) = population baseline time trend
r_theta(t,z) = patient-specific deviation
```

This mirrors the Cox decomposition while allowing non-proportional effects.

## Mixture Models

Deep Cox Mixtures and related models represent heterogeneous survival behavior
with latent groups.

A generic mixture density is:

```math
f(t\mid z)
=
\sum_{m=1}^{M}
\pi_m(z)f_m(t\mid z),
```

with:

```math
\pi_m(z)\ge0,
\qquad
\sum_m \pi_m(z)=1.
```

The survival function is:

```math
S(t\mid z)
=
\sum_{m=1}^{M}
\pi_m(z)S_m(t\mid z).
```

For censored data:

```math
\log p_i
=
\delta_i\log
\left[
\sum_m \pi_m(z_i)f_m(X_i\mid z_i)
\right]
+
(1-\delta_i)\log
\left[
\sum_m \pi_m(z_i)S_m(X_i\mid z_i)
\right].
```

## Why Mixtures Matter In Pathology

WSI cohorts often combine subtypes:

```text
histologic subtypes
molecular subtypes
treatment groups
institutions
stages
immune phenotypes
```

A mixture model can represent survival as:

```text
which latent disease regime is this slide in?
plus
what is the time distribution inside that regime?
```

This matches prototype and distribution-pooling intuitions.

## Density Parameterization

Instead of hazards, one can model the event-time density directly:

```math
f_\theta(t\mid z)\ge0,
\qquad
\int_0^\infty f_\theta(t\mid z)\,dt=1.
```

Then:

```math
S_\theta(t\mid z)
=
\int_t^\infty f_\theta(u\mid z)\,du.
```

The likelihood remains:

```math
f_\theta(X_i\mid z_i)^{\delta_i}
S_\theta(X_i\mid z_i)^{1-\delta_i}.
```

Density modeling makes the output object explicit but requires normalization.

## Key Takeaway

Continuous-time neural models replace the Cox scalar:

```math
\eta_i
```

with a time-aware function:

```math
(t,z_i)\mapsto \lambda_i(t), S_i(t), \text{ or } f_i(t).
```

The extra expressivity is real, but numerical integration and calibration become
central.

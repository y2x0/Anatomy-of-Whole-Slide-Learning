# Hazard, Survival, And Density

Continuous-time survival models start from:

```math
\lambda(t\mid z)
=
\lim_{\Delta t\to0}
\frac{\Pr(t\le T<t+\Delta t\mid T\ge t,z)}{\Delta t}.
```

The cumulative hazard is:

```math
\Lambda(t\mid z)
=
\int_0^t \lambda(u\mid z)\,du.
```

The survival function is:

```math
S(t\mid z)
=
\exp[-\Lambda(t\mid z)].
```

The density is:

```math
f(t\mid z)
=
\lambda(t\mid z)S(t\mid z).
```

## Likelihood With Censoring

For observed pair $(X_i,\delta_i)$:

```math
p(X_i,\delta_i\mid z_i)
=
f(X_i\mid z_i)^{\delta_i}
S(X_i\mid z_i)^{1-\delta_i}.
```

Using $f=\lambda S$:

```math
\log p_i
=
\delta_i\log\lambda(X_i\mid z_i)
+
\log S(X_i\mid z_i).
```

Since:

```math
\log S(X_i\mid z_i)
=
-\int_0^{X_i}\lambda(u\mid z_i)\,du,
```

the negative log likelihood is:

```math
\mathcal{L}_i
=
-
\delta_i\log\lambda(X_i\mid z_i)
+
\int_0^{X_i}\lambda(u\mid z_i)\,du.
```

This is the continuous-time survival likelihood.

## Interpretation

The event term:

```math
-\delta_i\log\lambda(X_i\mid z_i)
```

rewards high hazard at the observed event time.

The integral term:

```math
\int_0^{X_i}\lambda(u\mid z_i)\,du
```

penalizes accumulated hazard before the observed time.

For censored patients $\delta_i=0$, only the integral appears. The model is
penalized for assigning too much hazard before censoring.

## Positivity Constraint

A hazard must be nonnegative:

```math
\lambda(t\mid z)\ge0.
```

Neural models usually enforce this with:

```math
\lambda_\theta(t,z)=\mathrm{softplus}(g_\theta(t,z))
```

or:

```math
\lambda_\theta(t,z)=\exp(g_\theta(t,z)).
```

Softplus is numerically gentler. Exponential can produce unstable hazards.

## Numerical Integration

The loss requires:

```math
\int_0^{X_i}\lambda_\theta(u,z_i)\,du.
```

This may be computed by:

```text
closed form if parameterization allows it
quadrature
Monte Carlo integration
piecewise-constant approximation
neural ODE style integration
```

The integration method becomes part of the model.

## Key Takeaway

Continuous-time hazard models answer:

```text
what is the instantaneous event rate at time t?
```

and train by balancing:

```text
high hazard at observed event times
low accumulated hazard before observed times.
```

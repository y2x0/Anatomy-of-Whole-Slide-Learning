# Mixture Survival Algebra

Mixture survival models introduce a latent regime:

```math
M_i\in\{1,\ldots,M\}.
```

The slide representation $z_i$ determines mixture weights:

```math
\pi_m(z_i)
=
\Pr(M_i=m\mid z_i),
\qquad
\sum_m\pi_m(z_i)=1.
```

Each component has its own survival distribution:

```math
f_m(t\mid z_i),
\qquad
S_m(t\mid z_i),
\qquad
\lambda_m(t\mid z_i).
```

## Mixture Density And Survival

The event-time density is:

```math
f(t\mid z)
=
\sum_{m=1}^{M}\pi_m(z)f_m(t\mid z).
```

The survival function is:

```math
S(t\mid z)
=
\sum_{m=1}^{M}\pi_m(z)S_m(t\mid z).
```

The mixture hazard is not the weighted average of component hazards with fixed
weights. It is:

```math
\lambda(t\mid z)
=
\frac{f(t\mid z)}{S(t\mid z)}
=
\frac{
\sum_m\pi_m(z)f_m(t\mid z)
}{
\sum_m\pi_m(z)S_m(t\mid z)
}.
```

Since:

```math
f_m(t\mid z)=\lambda_m(t\mid z)S_m(t\mid z),
```

we get:

```math
\lambda(t\mid z)
=
\sum_m
\omega_m(t,z)\lambda_m(t\mid z),
```

where the time-dependent posterior survival weight is:

```math
\omega_m(t,z)
=
\frac{\pi_m(z)S_m(t\mid z)}
{\sum_{\ell}\pi_\ell(z)S_\ell(t\mid z)}.
```

Thus mixture hazards naturally create non-proportional hazards because the
effective component weights change with time.

## Censored Likelihood

For observed $(X_i,\delta_i)$:

```math
p_i
=
f(X_i\mid z_i)^{\delta_i}
S(X_i\mid z_i)^{1-\delta_i}.
```

Therefore:

```math
\log p_i
=
\delta_i
\log
\sum_m\pi_m(z_i)f_m(X_i\mid z_i)
+
(1-\delta_i)
\log
\sum_m\pi_m(z_i)S_m(X_i\mid z_i).
```

## Posterior Component Assignment

If the event is observed:

```math
\Pr(M_i=m\mid X_i,\delta_i=1,z_i)
=
\frac{\pi_m(z_i)f_m(X_i\mid z_i)}
{\sum_{\ell}\pi_\ell(z_i)f_\ell(X_i\mid z_i)}.
```

If censored:

```math
\Pr(M_i=m\mid X_i,\delta_i=0,z_i)
=
\frac{\pi_m(z_i)S_m(X_i\mid z_i)}
{\sum_{\ell}\pi_\ell(z_i)S_\ell(X_i\mid z_i)}.
```

Censored patients update component responsibility through survival probability,
not event density.

## Mixture Of Cox Components

A component may be Cox-like:

```math
\lambda_m(t\mid z)
=
\lambda_{0m}(t)\exp(\eta_m(z)).
```

Then:

```math
S_m(t\mid z)
=
\exp[-\Lambda_{0m}(t)\exp(\eta_m(z))].
```

The full model is:

```math
S(t\mid z)
=
\sum_m
\pi_m(z)
\exp[-\Lambda_{0m}(t)\exp(\eta_m(z))].
```

Even if each component is proportional hazards, the mixture need not be
proportional hazards.

## Prototype Interpretation For WSI

Suppose a WSI is represented by prototype proportions:

```math
p_i\in\Delta^{M-1}.
```

One can set:

```math
\pi(z_i)=p_i
```

or:

```math
\pi(z_i)=\mathrm{softmax}(Az_i+b).
```

Then each mixture component can correspond to a latent prognosis regime:

```text
immune-rich tumor
necrotic aggressive tumor
stromal invasion
low-grade glandular morphology
treatment-response phenotype
```

This is not guaranteed semantically, but the math makes the interpretation
available if prototypes are constrained or inspected.

## Gradient For Mixture Weights

Let:

```math
a_m(z)=\text{mixture logit},
\qquad
\pi_m(z)=\mathrm{softmax}_{m}(a(z)).
```

For event observations, define:

```math
\gamma_m^{(e)}
=
\frac{\pi_mf_m(X)}
{\sum_\ell\pi_\ell f_\ell(X)}.
```

Then:

```math
\frac{\partial[-\log p]}{\partial a_m}
=
\pi_m-\gamma_m^{(e)}.
```

For censored observations:

```math
\gamma_m^{(c)}
=
\frac{\pi_mS_m(X)}
{\sum_\ell\pi_\ell S_\ell(X)},
```

and:

```math
\frac{\partial[-\log p]}{\partial a_m}
=
\pi_m-\gamma_m^{(c)}.
```

The mixture gate learns from both events and censoring.

## Failure Modes

Mixtures can collapse:

```math
\pi_1(z_i)\approx1
\quad\forall i.
```

They can also swap labels across runs because component labels are not
identifiable:

```math
(1,2,\ldots,M)
```

can be permuted without changing likelihood.

Finally, a mixture may fit cohort heterogeneity rather than biological
heterogeneity if site, scanner, stage, or treatment dominate the survival signal.

## Dense Summary

```math
\begin{aligned}
f(t\mid z)&=\sum_m\pi_m(z)f_m(t\mid z),\\
S(t\mid z)&=\sum_m\pi_m(z)S_m(t\mid z),\\
\lambda(t\mid z)
&=
\sum_m
\frac{\pi_m(z)S_m(t\mid z)}
{\sum_\ell\pi_\ell(z)S_\ell(t\mid z)}
\lambda_m(t\mid z).
\end{aligned}
```

Mixture survival is a mathematically natural way to make risk regime-specific
without forcing one global proportional-hazards ordering.

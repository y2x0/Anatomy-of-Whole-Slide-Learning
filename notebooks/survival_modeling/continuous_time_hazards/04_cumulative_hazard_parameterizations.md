# Cumulative Hazard Parameterizations

Continuous-time survival models must enforce:

```math
\lambda(t\mid z)\ge0,
\qquad
\Lambda(t\mid z)=\int_0^t\lambda(u\mid z)\,du,
\qquad
S(t\mid z)=\exp[-\Lambda(t\mid z)].
```

Equivalently:

```math
\Lambda(0\mid z)=0,
\qquad
\Lambda(t\mid z)\ \text{nondecreasing in }t.
```

The parameterization decides whether these constraints are automatic.

## Direct Hazard Parameterization

Let:

```math
\lambda_\theta(t,z)=\rho(g_\theta(t,z)),
```

with:

```math
\rho(a)=\operatorname{softplus}(a)
```

or:

```math
\rho(a)=\exp(a).
```

The likelihood is:

```math
\mathcal{L}_i
=
-
\delta_i\log \rho(g_\theta(X_i,z_i))
+
\int_0^{X_i}\rho(g_\theta(u,z_i))\,du.
```

The hard part is the integral.

## Piecewise-Constant Hazard

Choose knots:

```math
0=\tau_0<\tau_1<\cdots<\tau_K.
```

Let:

```math
\lambda(t\mid z)=\lambda_k(z)
\quad
t\in(\tau_{k-1},\tau_k].
```

Then:

```math
\Lambda(t\mid z)
=
\sum_{\ell<k}\lambda_\ell(z)(\tau_\ell-\tau_{\ell-1})
+
\lambda_k(z)(t-\tau_{k-1})
```

for \(t\in(\tau_{k-1},\tau_k]\).

This is continuous-time but close to discrete hazards. The interval hazard is:

```math
h_k(z)
=
1-\exp[-\lambda_k(z)\Delta_k],
\qquad
\Delta_k=\tau_k-\tau_{k-1}.
```

Thus:

```math
\lambda_k(z)
=
-\frac{1}{\Delta_k}\log(1-h_k(z)).
```

Discrete hazards are piecewise-constant continuous hazards viewed at interval
boundaries.

## Direct Cumulative Hazard

Instead of modeling \(\lambda\), model:

```math
\Lambda_\theta(t,z)
```

with monotonicity in \(t\). Then:

```math
S_\theta(t,z)=\exp[-\Lambda_\theta(t,z)].
```

The event density requires:

```math
f_\theta(t,z)
=
\frac{\partial \Lambda_\theta(t,z)}{\partial t}
\exp[-\Lambda_\theta(t,z)].
```

The event log likelihood is:

```math
\log f_\theta(X_i,z_i)
=
\log
\frac{\partial\Lambda_\theta(X_i,z_i)}{\partial t}
-
\Lambda_\theta(X_i,z_i).
```

The censored log likelihood is:

```math
\log S_\theta(X_i,z_i)
=
-\Lambda_\theta(X_i,z_i).
```

## Positive Basis Expansion

Let:

```math
\lambda(t\mid z)
=
\sum_{m=1}^{M}
\alpha_m(z)b_m(t),
```

where:

```math
\alpha_m(z)\ge0,
\qquad
b_m(t)\ge0.
```

Then \(\lambda\ge0\), and:

```math
\Lambda(t\mid z)
=
\sum_{m=1}^{M}
\alpha_m(z)B_m(t),
```

where:

```math
B_m(t)=\int_0^tb_m(u)\,du.
```

If \(B_m\) is known analytically, the likelihood is cheap and exact.

For WSI:

```math
\alpha(z_i)=\operatorname{softplus}(Az_i+b).
```

The slide predicts nonnegative weights over temporal hazard basis functions.

## Monotone Neural Cumulative Hazard

A neural cumulative hazard can be constructed as:

```math
\Lambda_\theta(t,z)
=
\int_0^t
\operatorname{softplus}(g_\theta(u,z))\,du.
```

This is always monotone but still requires integration.

Alternatively:

```math
\Lambda_\theta(t,z)
=
\operatorname{softplus}(a_\theta(t,z))
```

is nonnegative but not necessarily monotone in \(t\). Nonnegativity alone is not
enough.

## Likelihood Gradients

For direct hazard:

```math
\mathcal{L}_i
=
-\delta_i\log\lambda_\theta(X_i,z_i)
+
\int_0^{X_i}\lambda_\theta(u,z_i)\,du.
```

The gradient is:

```math
\nabla_\theta\mathcal{L}_i
=
-
\delta_i
\frac{\nabla_\theta\lambda_\theta(X_i,z_i)}
{\lambda_\theta(X_i,z_i)}
+
\int_0^{X_i}
\nabla_\theta\lambda_\theta(u,z_i)\,du.
```

For WSI representation \(z_i\):

```math
\nabla_{z_i}\mathcal{L}_i
=
-
\delta_i
\nabla_{z_i}\log\lambda_\theta(X_i,z_i)
+
\int_0^{X_i}
\nabla_{z_i}\lambda_\theta(u,z_i)\,du.
```

The slide embedding is pulled to increase hazard at event time and decrease
integrated hazard before observed time.

## Dense Summary

```math
\begin{aligned}
\mathcal{L}_i
&=
-\delta_i\log\lambda_\theta(X_i,z_i)
+\Lambda_\theta(X_i,z_i),\\
\Lambda_\theta(t,z)
&=
\int_0^t\lambda_\theta(u,z)\,du,\\
S_\theta(t,z)
&=
\exp[-\Lambda_\theta(t,z)],\\
h_k(z)
&=
1-\exp[-\lambda_k(z)\Delta_k].
\end{aligned}
```

Continuous-time survival is mostly a constrained function-learning problem:
hazards must be positive, cumulative hazards must be monotone, and likelihoods
must be numerically faithful.

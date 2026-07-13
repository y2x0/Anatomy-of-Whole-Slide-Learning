# Counting-Process Derivation

The Cox partial likelihood can be derived from a multiplicative intensity model.

For subject
```math
i
```
, define:

```math
N_i(t)=\mathbf{1}[X_i\le t,\delta_i=1],
\qquad
Y_i(t)=\mathbf{1}[X_i\ge t].
```

Here
```math
N_i(t)
```
 counts whether the event has occurred by time
```math
t
```
, and
```math
Y_i(t)
```
indicates whether the subject is still at risk immediately before
time
```math
t
```
.

The Cox intensity model is:

```math
\lambda_i(t)
=
Y_i(t)\lambda_0(t)\exp(\eta_i),
\qquad
\eta_i=f_\theta(z_i).
```

The process
```math
N_i(t)
```
has compensator:

```math
A_i(t)
=
\int_0^t
Y_i(u)\lambda_0(u)\exp(\eta_i)\,du.
```

The martingale residual is:

```math
M_i(t)=N_i(t)-A_i(t).
```

## Full Likelihood Before Profiling

For observed event/censoring data, the likelihood contribution under independent
censoring is proportional to:

```math
\prod_i
\left[
\lambda_0(X_i)\exp(\eta_i)
\right]^{\delta_i}
\exp\left(
-\int_0^{X_i}\lambda_0(u)\exp(\eta_i)\,du
\right).
```

The log likelihood is:

```math
\ell(\theta,\lambda_0)
=
\sum_i \delta_i[\log\lambda_0(X_i)+\eta_i]
-
\sum_i
\exp(\eta_i)
\int_0^{X_i}\lambda_0(u)\,du.
```

Using
```math
Y_i(u)=\mathbf{1}[X_i\ge u]
```
:

```math
\ell(\theta,\lambda_0)
=
\sum_i \delta_i[\log\lambda_0(X_i)+\eta_i]
-
\int_0^\infty
\lambda_0(u)
\sum_i Y_i(u)\exp(\eta_i)
\,du.
```

Let:

```math
S^{(0)}(u;\theta)
=
\sum_iY_i(u)\exp(\eta_i).
```

Then:

```math
\ell(\theta,\lambda_0)
=
\sum_i \delta_i[\log\lambda_0(X_i)+\eta_i]
-
\int_0^\infty
\lambda_0(u)S^{(0)}(u;\theta)\,du.
```

## Profiling The Baseline Hazard

Assume distinct event times:

```math
t_1<t_2<\cdots<t_m.
```

Represent the baseline cumulative hazard as jumps:

```math
\Lambda_0(t)=\sum_{\ell:t_\ell\le t}\Delta\Lambda_{0\ell}.
```

The log likelihood terms involving jumps are:

```math
\ell
=
\sum_{\ell=1}^m
\left[
\log\Delta\Lambda_{0\ell}
+\eta_{i_\ell}
-
\Delta\Lambda_{0\ell}S^{(0)}(t_\ell;\theta)
\right],
```

where
```math
i_\ell
```
 is the subject failing at
```math
t_\ell
```
.

Maximize over each
```math
\Delta\Lambda_{0\ell}
```
:

```math
\frac{\partial \ell}{\partial \Delta\Lambda_{0\ell}}
=
\frac{1}{\Delta\Lambda_{0\ell}}
-
S^{(0)}(t_\ell;\theta)
=0.
```

Therefore:

```math
\widehat{\Delta\Lambda}_{0\ell}
=
\frac{1}{S^{(0)}(t_\ell;\theta)}.
```

Plugging this back into the likelihood leaves:

```math
\ell_p(\theta)
=
\sum_{\ell=1}^m
\left[
\eta_{i_\ell}
-
\log S^{(0)}(t_\ell;\theta)
\right]
+\text{constant}.
```

Since:

```math
S^{(0)}(t_\ell;\theta)
=
\sum_{j\in R(t_\ell)}\exp(\eta_j),
```

we recover the partial log likelihood:

```math
\ell_p(\theta)
=
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log
\sum_{j\in R_i}
\exp(\eta_j)
\right].
```

## Tied Events

If
```math
d_t
```
 subjects fail at time
```math
t
```
, the Breslow approximation is:

```math
\ell_t^{\mathrm{Breslow}}
=
\sum_{i\in D_t}\eta_i
-
d_t\log
\sum_{j\in R_t}\exp(\eta_j).
```

The baseline jump becomes:

```math
\widehat{\Delta\Lambda}_0(t)
=
\frac{d_t}{\sum_{j\in R_t}\exp(\eta_j)}.
```

The Efron approximation corrects the denominator as tied events are removed
fractionally:

```math
\ell_t^{\mathrm{Efron}}
=
\sum_{i\in D_t}\eta_i
-
\sum_{r=0}^{d_t-1}
\log
\left[
\sum_{j\in R_t}\exp(\eta_j)
-
\frac{r}{d_t}
\sum_{i\in D_t}\exp(\eta_i)
\right].
```

## What Was Assumed

The derivation uses:

```math
T_i \perp C_i \mid z_i
```

and assumes the event intensity factorizes as:

```math
\lambda_i(t)=Y_i(t)\lambda_0(t)\exp(\eta_i).
```

The second assumption is stronger than censoring independence. It says all
subject-specific information acts through one time-invariant log-risk score.

## WSI Translation

For WSI survival:

```math
\eta_i=f_\theta(\mathcal{R}(\mathcal{C}(H_i;G_i))).
```

The counting-process derivation does not care how
```math
z_i
```
was built. Therefore
all WSI complexity enters only through
```math
\eta_i
```
.

The Cox loss cannot directly distinguish:

```math
\eta_i
=
f_\theta(\text{tumor grade})
```

from:

```math
\eta_i
=
f_\theta(\text{stroma plus immune context})
```

unless those choices change risk-set ordering.

## Dense Summary

```math
\begin{aligned}
N_i(t)&=\mathbf{1}[X_i\le t,\delta_i=1],\\
Y_i(t)&=\mathbf{1}[X_i\ge t],\\
\lambda_i(t)&=Y_i(t)\lambda_0(t)\exp(\eta_i),\\
S^{(0)}(t)&=\sum_jY_j(t)\exp(\eta_j),\\
\ell_p(\theta)
&=
\sum_{i:\delta_i=1}
\left[
\eta_i-\log S^{(0)}(X_i)
\right].
\end{aligned}
```

The partial likelihood is the profile likelihood after removing an unspecified
baseline hazard from a multiplicative intensity model.

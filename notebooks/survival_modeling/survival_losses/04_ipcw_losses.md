# IPCW Losses

Inverse probability of censoring weighting constructs supervised losses under
right censoring.

Let:

```math
G(t\mid z)=\Pr(C>t\mid z)
```

be the censoring survival function. In many applications, a marginal
```math
G(t)
```
is estimated by Kaplan-Meier on censoring times.

## Horizon Binary Target

At horizon
```math
t
```
:

```math
Y_i(t)=\mathbf{1}[T_i\le t].
```

This target is observed if:

```text
event occurs by t
or subject is still observed after t
```

It is unobserved if censored before
```math
t
```
.

## IPCW Risk

For predicted event probability:

```math
\widehat{F}_i(t)=\Pr_\theta(T_i\le t\mid z_i),
```

a censoring-weighted squared loss is:

```math
\mathcal{L}_{\text{IPCW}}(t)
=
\sum_i
\left[
\frac{\mathbf{1}[X_i\le t,\delta_i=1]}{\widehat{G}(X_i)}
\ell(1,\widehat{F}_i(t))
+
\frac{\mathbf{1}[X_i>t]}{\widehat{G}(t)}
\ell(0,\widehat{F}_i(t))
\right].
```

For Brier loss:

```math
\ell(y,p)=(y-p)^2.
```

For log loss:

```math
\ell(y,p)=-y\log p-(1-y)\log(1-p).
```

## Why The Weights Work

Under independent censoring:

```math
T\perp C\mid z,
```

the observed event-before-
```math
t
```
indicator is:

```math
\mathbf{1}[X\le t,\delta=1]
=
\mathbf{1}[T\le t,T\le C].
```

Weighting by
```math
1/G(T)
```
corrects the probability of being observed at the event
time.

The observed event-free-after-
```math
t
```
indicator is:

```math
\mathbf{1}[X>t]
=
\mathbf{1}[T>t,C>t].
```

Weighting by
```math
1/G(t)
```
corrects the probability of being uncensored through
the horizon.

## Instability Near Tail

If:

```math
\widehat{G}(t)\approx0,
```

then IPCW weights explode. Evaluation and training horizons should usually be
restricted to time ranges where censoring survival remains sufficiently large.

## WSI Training Use

For WSI, an IPCW horizon loss can train:

```math
\widehat{F}_i(t^\star)
=
1-\widehat{S}_i(t^\star)
```

from slide representation:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i)).
```

The gradient is a supervised classification gradient at
```math
t^\star
```
, weighted
by censoring.

This can be useful when the clinical question is a fixed horizon:

```text
2-year recurrence
5-year cancer-specific death
10-year overall survival
```

It is not a full event-time likelihood unless integrated across horizons with a
proper scoring rule.

## Dense Summary

```math
\mathcal{L}_{\text{IPCW}}(t)
=
\sum_i
\frac{\mathbf{1}[X_i\le t,\delta_i=1]}{\widehat{G}(X_i)}
\ell(1,\widehat{F}_i(t))
+
\frac{\mathbf{1}[X_i>t]}{\widehat{G}(t)}
\ell(0,\widehat{F}_i(t)).
```

IPCW turns censored survival evaluation into weighted supervised learning at a
specified time horizon.

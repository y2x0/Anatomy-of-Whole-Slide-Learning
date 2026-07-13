# Problem Setup

Survival learning differs from ordinary supervised learning because the target
is only partially observed.

For each individual
```math
i
```
, there is an event time
```math
T_i
```
and a censoring time
```math
C_i
```
. We observe:

```math
X_i = \min(T_i,C_i),
\qquad
\delta_i = \mathbf{1}[T_i \le C_i].
```

If
```math
\delta_i=1
```
, the event occurred at
```math
X_i
```
. If
```math
\delta_i=0
```
, the only
known fact is:

```math
T_i > X_i.
```

That inequality is the whole reason survival losses look different from
classification or regression losses.

## Survival Objects

The central functions are:

```math
S(t\mid x) = \Pr(T>t\mid x)
```

```math
F(t\mid x) = \Pr(T\le t\mid x)=1-S(t\mid x)
```

```math
f(t\mid x) = \frac{\partial}{\partial t}F(t\mid x)
```

```math
\lambda(t\mid x)
= \lim_{\Delta t\to 0}
\frac{\Pr(t\le T<t+\Delta t\mid T\ge t,x)}{\Delta t}.
```

The hazard
```math
\lambda
```
is not a probability. It is an instantaneous event rate.
It connects to survival by:

```math
S(t\mid x)
=
\exp\left(
-\int_0^t \lambda(u\mid x)\,du
\right).
```

The density is:

```math
f(t\mid x)
=
\lambda(t\mid x)S(t\mid x).
```

## Whole-Slide Inputs

For computational pathology, the covariate
```math
x_i
```
is not usually a small vector.
It is a slide:

```math
S_i = \text{whole slide image}.
```

After tiling and feature extraction:

```math
H_i = \{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^d.
```

A WSI survival model first builds a slide representation:

```math
z_i
=
\mathcal{R}(\mathcal{C}(H_i;G_i,S_i)).
```

Here:

```text
C = context operator over patches
R = readout operator
G = geometry, graph, order, or hierarchy
S = supervision and censoring structure
```

Then the survival head maps
```math
z_i
```
into a risk object:

```math
\mathcal{H}(z_i)
\in
\{\eta_i,\ h_i,\ \lambda_i(t),\ S_i(t),\ f_i(t)\}.
```

## The Censoring Assumption

Most standard survival losses assume independent, non-informative censoring:

```math
T_i \perp C_i \mid x_i.
```

This says that, after conditioning on observed covariates, censoring does not
carry extra information about the event time.

In pathology this can be fragile. Patients are censored because of follow-up
systems, treatment access, institution, trial design, and cohort construction.
If those variables correlate with morphology or cancer subtype, the censoring
mechanism can leak structure into the task.

## Three Output Families

### Cox Family

The model outputs a scalar:

```math
\eta_i = f_\theta(z_i).
```

The scalar is an ordering of risk, not a calibrated time distribution.

### Discrete-Time Hazards

The model outputs interval hazards:

```math
h_{ik}
=
\Pr(T_i\in I_k \mid T_i\ge \tau_{k-1}, z_i).
```

This gives a stepwise survival curve.

### Continuous-Time Hazards

The model outputs a function:

```math
\lambda_i(t)=\lambda_\theta(t,z_i)
```

or an equivalent density/survival parameterization.

This is the most expressive family, but it carries the most burden: positivity,
integrability, calibration, and numerical integration.

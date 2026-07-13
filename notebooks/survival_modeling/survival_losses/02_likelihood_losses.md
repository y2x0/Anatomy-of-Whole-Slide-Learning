# Likelihood Losses

Likelihood losses are the cleanest survival objectives because they correspond
to a probabilistic model for observed
```math
(X,\delta)
```
.

## Continuous-Time Likelihood

For a hazard model:

```math
\lambda_i(t)=\lambda_\theta(t,z_i),
\qquad
S_i(t)=\exp\left[-\int_0^t\lambda_i(u)\,du\right].
```

The observed likelihood is:

```math
p(X_i,\delta_i\mid z_i)
=
[\lambda_i(X_i)S_i(X_i)]^{\delta_i}
[S_i(X_i)]^{1-\delta_i}.
```

Therefore:

```math
\mathcal{L}_i
=
-\delta_i\log\lambda_i(X_i)
+\int_0^{X_i}\lambda_i(u)\,du.
```

The likelihood uses both events and censoring:

```text
events add a log hazard term
censored observations add only survival-through-censoring
```

## Piecewise-Constant Hazard Likelihood

For intervals
```math
I_k=(\tau_{k-1},\tau_k]
```
, let:

```math
\lambda_i(t)=\lambda_{ik}
\quad
t\in I_k.
```

If
```math
X_i\in I_r
```
 and
```math
\delta_i=1
```
:

```math
\mathcal{L}_i
=
-\log\lambda_{ir}
+
\sum_{k<r}\lambda_{ik}\Delta_k
+
\lambda_{ir}(X_i-\tau_{r-1}).
```

If censored in
```math
I_r
```
:

```math
\mathcal{L}_i
=
\sum_{k<r}\lambda_{ik}\Delta_k
+
\lambda_{ir}(X_i-\tau_{r-1}).
```

This preserves within-interval censoring time, unlike coarse discrete hazards.

## Discrete Hazard Likelihood

Discrete hazard:

```math
h_{ik}
=
\Pr(T_i\in I_k\mid T_i>\tau_{k-1},z_i).
```

Event in interval
```math
r
```
:

```math
p_i
=
h_{ir}\prod_{k<r}(1-h_{ik}).
```

Censored after interval
```math
r
```
:

```math
p_i
=
\prod_{k\le r}(1-h_{ik}).
```

Loss:

```math
\mathcal{L}_i
=
-
\sum_km_{ik}
[y_{ik}\log h_{ik}+(1-y_{ik})\log(1-h_{ik})].
```

The exact mask depends on whether censoring inside an interval is included or
only fully observed intervals are used.

## PMF Likelihood

Let:

```math
p_{ik}=\Pr(T_i\in I_k\mid z_i),
\qquad
p_{i,>K}=\Pr(T_i>\tau_K\mid z_i).
```

Event in interval
```math
r
```
:

```math
\mathcal{L}_i=-\log p_{ir}.
```

Censored after interval
```math
r
```
:

```math
\mathcal{L}_i
=
-\log
\left[
\sum_{k>r}p_{ik}+p_{i,>K}
\right].
```

The PMF likelihood directly models
```math
T
```
. It is not conditional on surviving
earlier intervals.

## Cox Partial Likelihood As Profile Likelihood

Cox starts from:

```math
\lambda_i(t)=\lambda_0(t)\exp(\eta_i).
```

Profiling out
```math
\lambda_0
```
gives:

```math
\mathcal{L}_{\text{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
\eta_i-\log\sum_{j\in R_i}\exp(\eta_j)
\right].
```

It is a likelihood for ordering under proportional hazards, not a full
likelihood for calibrated survival until a baseline hazard is recovered.

## Dense Comparison

```text
continuous hazard:
    needs integration, preserves continuous time

piecewise hazard:
    exact within-bin exposure, positive rates

discrete hazard:
    simple masked BCE, bin-dependent

PMF:
    direct distribution over event bins

Cox:
    avoids baseline hazard, learns relative ordering
```

For WSI, the statistical difference matters because cohorts are small,
censoring is heavy, and event times are often coarsely recorded.

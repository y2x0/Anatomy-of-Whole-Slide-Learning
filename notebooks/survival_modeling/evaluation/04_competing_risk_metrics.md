# Competing-Risk Metrics

Competing-risk models output cause-specific cumulative incidence:

```math
\widehat{F}_{ic}(t)
=
\Pr_\theta(T_i\le t,J_i=c\mid z_i).
```

Evaluation should respect event type.

## Cause-Specific Horizon Brier Score

At horizon
```math
t
```
, define:

```math
Y_{ic}(t)=\mathbf{1}[T_i\le t,J_i=c].
```

The predicted probability is:

```math
\widehat{F}_{ic}(t).
```

An IPCW Brier score has the same censoring logic as survival Brier, but the
target is cause-specific event occurrence by
```math
t
```
:

```math
\mathrm{BS}_{c}(t)
=
\frac{1}{n}\sum_i
w_i(t)
\left[
Y_{ic}(t)-\widehat{F}_{ic}(t)
\right]^2.
```

Weights account for whether
```math
Y_{ic}(t)
```
is observed under censoring.

## Cause-Specific AUC

At horizon
```math
t
```
:

```math
\mathrm{AUC}_{c}(t)
=
\Pr(
\widehat{F}_{ic}(t)>\widehat{F}_{jc}(t)
\mid
T_i\le t,J_i=c,\ T_j>t
).
```

Controls may be defined as event-free at
```math
t
```
, or sometimes as not having cause
```math
c
```
 by
```math
t
```
. The definition must be stated.

## Probability Constraint

Predicted CIFs must satisfy:

```math
\sum_c\widehat{F}_{ic}(t)\le1
```

for all
```math
t
```
, and each
```math
\widehat{F}_{ic}(t)
```
 must be nondecreasing in
```math
t
```
.

Metric reporting should check validity:

```math
\widehat{F}_{ic}(t_{k+1})\ge\widehat{F}_{ic}(t_k),
\qquad
\sum_c\widehat{F}_{ic}(t_k)\le1.
```

## Cause-Specific Concordance

A cause-specific comparable pair can be:

```math
A_{ij}^{(c)}
=
\mathbf{1}[X_i<X_j,\Delta_i=c].
```

Then:

```math
\widehat{C}_{c}
=
\frac{
\sum_{i,j}A_{ij}^{(c)}
\mathbf{1}[r_{ic}>r_{jc}]
}{
\sum_{i,j}A_{ij}^{(c)}
}.
```

The risk score
```math
r_{ic}
```
may be:

```math
r_{ic}=\widehat{F}_{ic}(t^\star)
```

or a cause-specific hazard score.

## WSI Reporting Rule

If a WSI model predicts multiple event types, report:

```text
per-cause discrimination
per-cause calibration
CIF validity
event counts per cause
clinically meaningful horizons
```

Do not collapse all causes into one all-cause C-index unless the task is
explicitly all-cause survival.

## Dense Summary

Competing-risk evaluation should be built around:

```math
\widehat{F}_{ic}(t),
\qquad
Y_{ic}(t)=\mathbf{1}[T_i\le t,J_i=c].
```

The metric must specify the event type, horizon, control definition, and
censoring adjustment.

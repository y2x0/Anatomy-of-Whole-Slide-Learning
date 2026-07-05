# Metric And Output Matching

A survival metric is only meaningful relative to the model output.

## Output: Scalar Risk

Examples:

```math
\eta_i=f_\theta(z_i).
```

Appropriate metrics:

```text
Harrell C-index
Uno C-index
time-dependent AUC using eta as marker
```

Not directly appropriate without calibration:

```text
Brier score
calibration plots
horizon probability error
```

Those require $\widehat{S}_i(t)$ or $\widehat{F}_i(t)$.

## Output: Survival Curve

Examples:

```math
\widehat{S}_i(t).
```

Appropriate metrics:

```text
C-index after choosing risk functional
time-dependent AUC
IPCW Brier score
integrated Brier score
calibration at horizons
```

Need to state:

```math
r_i=\rho[\widehat{S}_i].
```

## Output: Discrete Hazards

Examples:

```math
\widehat{h}_{ik}.
```

Convert to survival:

```math
\widehat{S}_i(\tau_k)
=
\prod_{\ell\le k}(1-\widehat{h}_{i\ell}).
```

Then evaluate as a survival curve.

## Output: PMF

Examples:

```math
\widehat{p}_{ik}.
```

Convert to:

```math
\widehat{F}_i(\tau_k)=\sum_{\ell\le k}\widehat{p}_{i\ell},
\qquad
\widehat{S}_i(\tau_k)=1-\widehat{F}_i(\tau_k).
```

Then evaluate distribution, horizon risk, or curve metrics.

## Output: Competing-Risk CIFs

Examples:

```math
\widehat{F}_{ic}(t).
```

Appropriate metrics:

```text
cause-specific AUC at horizon
cause-specific Brier score
CIF calibration
validity of CIF sums
```

All-cause C-index is not enough.

## Dense Table

```text
eta:
    ranking metrics

S(t):
    ranking plus calibration plus Brier

h_k:
    convert to S_k, then evaluate

p_k:
    convert to F_k and S_k, then evaluate

F_c(t):
    cause-specific metrics

lambda(t):
    convert to S(t) through cumulative hazard
```

## Key Rule

Never report a metric without specifying:

```text
model output object
risk functional used for scalar ranking
censoring adjustment
evaluation time horizon
event type, if competing risks exist
```

# Discrete-Time Hazard Failure Modes

Discrete-time models are flexible, but the flexibility moves burden onto time
binning, censoring masks, and calibration.

## 1. Binning Defines The Task

The model predicts:

```math
h_{ik}
=
\Pr(T_i\in I_k\mid T_i>\tau_{k-1},z_i).
```

Change the intervals
```math
I_k
```
, and the target changes.

Coarse bins hide temporal structure. Fine bins create sparse event targets.

## 2. Interval Censoring Ambiguity

If a patient is censored inside interval
```math
I_k
```
, we know:

```math
T_i>X_i
```

but we may not know whether the patient would survive through
```math
\tau_k
```
. Treating
the entire interval as event-free can introduce bias.

## 3. Too Many Empty Event Bins

For small pathology cohorts:

```math
K \text{ large}
\quad\Rightarrow\quad
\text{few events per bin}.
```

The model can learn a noisy time profile, especially with high-dimensional WSI
features.

## 4. Shared Representation Bottleneck

If:

```math
h_i=\sigma(Wz_i+b),
```

all hazards depend on the same slide embedding
```math
z_i
```
.

This can still miss time-specific morphology if the aggregator discards it.

## 5. Non-Monotone Risk Summaries

Hazards can vary by interval:

```math
h_{i1},h_{i2},\ldots,h_{iK}.
```

If evaluation reduces the curve to one risk score, the choice of summary matters:

```text
cumulative incidence at 2 years
expected event time
sum of hazards
negative survival at final time
```

Different summaries can rank patients differently.

## 6. Calibration Drift

A model can rank well but produce poorly calibrated hazards:

```math
\widehat{h}_{ik}
\ne
\Pr(T_i\in I_k\mid T_i>\tau_{k-1},z_i).
```

This is especially likely when ranking losses are mixed with likelihood losses.

## 7. WSI Attention Reuse

If the same attention weights define
```math
z_i
```
for all time bins, then all hazards
see the same morphological summary.

This can silently turn a time-dependent survival head into:

```text
one slide summary plus many linear probes.
```

## Diagnostic Questions

1. Are the time bins clinically meaningful?
2. How many events are in each bin?
3. Are censored intervals handled conservatively?
4. Is attention shared or time-dependent?
5. Are metrics computed at clinically meaningful horizons?
6. Is calibration checked separately from concordance?

## Design Escape Routes

```text
quantile event-time bins
time-dependent attention
multi-head early/late risk decoders
smoothness penalties across adjacent hazards
calibration losses
continuous-time models when binning is unstable
```

## Key Takeaway

Discrete hazards make risk time-aware, but time awareness is only as good as the
binning and the slide statistics that survive aggregation.

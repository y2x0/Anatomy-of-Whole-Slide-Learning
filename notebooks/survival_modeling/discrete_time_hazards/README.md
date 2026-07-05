# Discrete-Time Hazards

Discrete-time survival models divide time into intervals and predict the event
probability in each interval, conditional on surviving to that interval.

Let:

```math
0=\tau_0 < \tau_1 < \cdots < \tau_K.
```

Define interval:

```math
I_k = (\tau_{k-1},\tau_k].
```

The model predicts:

```math
h_{ik}
=
\Pr(T_i\in I_k \mid T_i>\tau_{k-1}, z_i).
```

The risk object is a vector:

```math
h_i=(h_{i1},\ldots,h_{iK}).
```

This is more expressive than Cox because risk can vary with time.

## Why This Matters For WSI

A WSI model can map one slide representation to many time-specific hazards:

```math
z_i
\longmapsto
(h_{i1},\ldots,h_{iK}).
```

Early and late risk can depend on different coordinates of \(z_i\):

```math
h_{ik}=\sigma(w_k^\top z_i+b_k).
```

This makes discrete hazards a natural fit when morphology has different
prognostic meaning across time horizons.

## Files

- `01_time_binning_and_targets.md`: intervals, labels, censoring masks.
- `02_hazard_likelihood.md`: likelihood derivation.
- `03_nnet_survival_deephit_and_wsi.md`: neural heads and WSI readouts.
- `04_failure_modes.md`: binning artifacts, calibration, and censoring issues.
- `05_hazard_pmf_survival_algebra.md`: invertible maps between hazards, PMFs,
  survival curves, cumulative incidence, and logits.
- `06_masked_likelihood_gradients.md`: censoring masks, exact gradients, and
  batch-level WSI consequences.
- `07_ranking_calibration_and_time_summaries.md`: ranking surrogates, horizon
  risk, calibration, and proper scoring tension.

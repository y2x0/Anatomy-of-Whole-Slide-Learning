# Cox Family

The Cox family represents risk as a scalar score:

```math
\eta_i = f_\theta(z_i).
```

The score is not a probability, not an event time, and not a survival curve. It
is a relative-risk coordinate.

The proportional hazards model writes:

```math
\lambda(t\mid z_i)
=
\lambda_0(t)\exp(\eta_i).
```

The baseline hazard \(\lambda_0(t)\) is unspecified. Training depends on event
ordering through the partial likelihood.

## Why This Matters For WSI

If the WSI encoder produces:

```math
z_i = \mathcal R(\mathcal C(H_i)),
```

then the Cox head keeps only:

```math
\eta_i = w^\top z_i
```

or a nonlinear scalar \(f_\theta(z_i)\).

That means the entire slide is compressed into one coordinate before the
survival loss sees it.

For pathology, the Cox family is attractive because it works with censored data
and small cohorts. Its main weakness is also its elegance: it only learns an
ordering unless the baseline hazard is estimated afterward.

## Files

- `01_cox_model.md`: proportional hazards and the risk score.
- `02_partial_likelihood.md`: risk sets, likelihood, gradients, and censoring.
- `03_deep_cox_and_wsi_readouts.md`: DeepSurv, Cox-nnet, and WSI encoders.
- `04_failure_modes.md`: proportional hazards failures, ranking blind spots,
  and WSI-specific issues.
- `05_counting_process_derivation.md`: Cox as a counting-process likelihood.
- `06_score_hessian_and_baseline.md`: score, Hessian, Breslow baseline, and
  calibrated survival recovery.
- `07_wsi_cox_statistic.md`: the exact WSI statistic optimized by Cox heads.

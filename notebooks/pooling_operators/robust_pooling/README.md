# Robust Pooling

Robust pooling modifies aggregation so artifacts, outliers, and mislabeled
patches have limited influence.

The core question:

```text
How much can one bad instance move the slide statistic?
```

## C/R/G/S Placement

```text
G:
    ignored unless robustness is region-aware or geometry-aware

C:
    optional score, residual, or context map used to identify outliers

R:
    trimmed mean, winsorized mean, median, geometric median, or M-estimator

S:
    task loss, artifact assumptions, or robust-statistical objective
```

Surviving statistic:

```math
z_i
=
\mathrm{RobustAgg}(\{u_{ij}\}_{j=1}^{n_i}).
```

## Files

- `01_trimmed_and_winsorized_means.md`: clipping and removing extremes.
- `02_median_geometric_median_and_m_estimators.md`: robust location estimates.
- `03_robust_attention_and_outlier_weights.md`: bounded influence attention.
- `04_failure_modes.md`: discarding rare positives and hiding artifacts.


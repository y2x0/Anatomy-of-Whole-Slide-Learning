# Evaluation

Survival evaluation must match the model output.

Risk-score models, survival-curve models, hazard models, and competing-risk
models require different metrics.

The central rule:

```text
Do not evaluate every survival model only by C-index.
```

C-index measures ranking. It does not measure calibration, horizon probability,
or quality of the full event-time distribution.

## Files

- `01_concordance.md`: Harrell/Uno C-index and comparable pairs.
- `02_time_dependent_auc.md`: cumulative/dynamic and incident/dynamic AUC.
- `03_brier_score_calibration.md`: IPCW Brier score and calibration.
- `04_competing_risk_metrics.md`: CIF metrics and cause-specific evaluation.
- `05_external_validation_and_wsi_leakage.md`: pathology-specific evaluation
  failure modes.
- `06_metric_output_matching.md`: which metrics match which output objects.

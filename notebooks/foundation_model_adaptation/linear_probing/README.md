# Linear Probing

Linear probing freezes the representation and trains a linear task map.

For a frozen slide embedding:

```math
z_i
=
F_{\phi_0}(X_i),
```

the probe is:

```math
o_i
=
Wz_i+b.
```

## Files

- `01_linear_probe_geometry.md`: hyperplanes, margins, and frozen separability.
- `02_logistic_and_cox_probes.md`: classification and survival probes.
- `03_calibration_and_domain_shift.md`: probability calibration under frozen features.
- `04_failure_modes.md`: nonlinear targets, nuisance axes, and false confidence.

## C/R/G/S Placement

```text
G:
    inherited from the pretrained representation

C:
    frozen

R:
    frozen if probing slide embeddings; fixed or simple if probing pooled patch features

S:
    downstream labels shape only W and b
```

## Dense Summary

Linear probing tests whether the task is linearly readable from the pretrained
space. It does not test whether the foundation model could solve the task after
adaptation.

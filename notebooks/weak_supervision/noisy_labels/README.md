# Noisy Labels

Noisy-label supervision observes a corrupted target:

```math
\widetilde Y_i
\ne
Y_i.
```

In WSI, noise can come from reports, cohort construction, uncertain diagnoses,
label thresholding, missing clinical context, scanner artifacts, or weak slide
selection.

## Files

- `01_label_noise_channel.md`: class-conditional and instance-dependent noise.
- `02_forward_backward_risk_correction.md`: transition matrices and unbiased losses.
- `03_robust_losses_and_small_loss_selection.md`: robust objectives and co-teaching logic.
- `04_instance_noise_under_bag_labels.md`: how slide noise corrupts latent patch supervision.
- `05_failure_modes.md`: unidentifiable noise and shortcut amplification.
- `06_report_derived_labels_and_campanella.md`: report labels, case/slide mapping, and clinical weak supervision.

## C/R/G/S Placement

```text
G:
    geometry may correlate with noise through scanner, lab, or tissue handling

C:
    representation can fit noise if high-capacity

R:
    aggregation can amplify mislabeled witnesses

S:
    corrupted label channel
```

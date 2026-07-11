# Negative-Free Pathology Pretraining

This portion asks:

```text
How can pathology representations be learned without explicit negative pairs,
and what prevents a constant solution?
```

Only primary papers already listed in the private research map are used. The
portion distinguishes self-distillation from masked token prediction rather
than treating every modern pathology foundation model as one objective.

## Notes

```text
01_negative_free_statistical_object.md
02_dino_teacher_student_objective.md
03_dino_centering_sharpening_and_collapse.md
04_ibot_masked_patch_token_distillation.md
05_dinov2_composite_objective_and_uniformity.md
06_hipt_hierarchical_self_distillation.md
07_uni_and_virchow_scaling_maps.md
08_phikon_histology_masked_modeling.md
09_wsi_interface_crgs_and_failure_matrix.md
```

The central distinction is:

```math
\text{absence of explicit negatives}
\ne
\text{absence of distributional constraints}.
```

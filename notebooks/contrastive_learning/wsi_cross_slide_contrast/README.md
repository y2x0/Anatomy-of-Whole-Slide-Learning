# WSI And Cross-Slide Contrast

This final contrastive portion asks:

```text
What exactly is compared across slides: whole-bag embeddings, selected patch
subbags, or mixed bags?
```

Primary anchors are SCL-WC, SC-MIL, and CAMCSA, all already listed in the
private research map. SimMIL is excluded because its classification pretraining
propagates bag labels to instances rather than defining a cross-slide
contrastive classification objective.

## Notes

```text
01_cross_slide_statistical_objects.md
02_sclwc_class_specific_attention_and_pseudo_instances.md
03_sclwc_positive_negative_subspaces.md
04_sclwc_memory_banks_and_selected_subbags.md
05_sclwc_exact_wscl_objective_and_gradients.md
06_sclwc_support_feedback_and_failure_modes.md
07_scmil_bag_level_supervised_contrast.md
08_scmil_curriculum_imbalance_and_geometry.md
09_cross_slide_augmentation_is_not_contrast.md
10_patient_leakage_false_positives_and_identifiability.md
11_crgs_cross_slide_design_matrix.md
```

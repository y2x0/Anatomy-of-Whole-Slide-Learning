# Counterfactual Explanations

This portion asks:

```text
What is the smallest meaningful change that would alter the model's decision?
```

A decision-changing perturbation is not automatically a counterfactual
explanation. The edit must be defined in a particular state space, measured by
a particular cost, constrained by a feasibility set, and evaluated against a
particular model. WSI counterfactuals can remove patches, alter interpretable
slide features, or synthesize changed morphology. These operations answer
different questions.

## Notes

1. `01_counterfactual_objects_and_optimization.md`
2. `02_wachter_objective_and_distance_geometry.md`
3. `03_validity_feasibility_actionability_and_causality.md`
4. `04_dice_diversity_and_determinantal_geometry.md`
5. `05_wsi_set_counterfactuals_and_hippo.md`
6. `06_hif_slide_object_and_gmm_ceflow.md`
7. `07_quadratic_boundary_projection_and_surrogate_gap.md`
8. `08_diffusion_counterfactual_foundations.md`
9. `09_mopadi_linear_feature_guidance.md`
10. `10_mopadi_mil_gradient_guidance.md`
11. `11_validation_crgs_and_failure_matrix.md`

## Primary Sources

- Wachter, Mittelstadt, Russell, [Counterfactual Explanations without Opening
  the Black Box](https://arxiv.org/abs/1711.00399).
- Mothilal, Sharma, Tan, [Explaining Machine Learning Classifiers through
  Diverse Counterfactual Explanations](https://arxiv.org/abs/1905.07697).
- Kaczmarzyk et al., [Explainable AI for Computational Pathology Identifies
  Model Limitations and Tissue Biomarkers](https://pmc.ncbi.nlm.nih.gov/articles/PMC11398542/).
- Benkirane et al., [Counterfactual Analysis for Digital Histopathology Slides
  Using Human Interpretable Features](https://inserm.hal.science/inserm-05034913v1).
- Dreyer et al., [Counterfactual Diffusion Models for Interpretable
  Morphology-based Explanations in Pathology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11565818/).


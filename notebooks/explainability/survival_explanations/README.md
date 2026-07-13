# Survival Explanations

This portion asks:

```text
What exactly is being explained when a survival model highlights a patch?
```

Survival outputs are not one scalar object. A Cox model outputs a relative-risk
score, a discrete model outputs hazards over bins, a continuous model defines a
hazard function, and a competing-risk model may define cause-specific hazards
or cumulative incidence functions. The same WSI attention map can therefore
be reused by several heads while supporting different claims.

## Notes

1. `01_survival_explanation_targets.md`
2. `02_cox_risk_score_and_patch_credit.md`
3. `03_discrete_hazard_and_horizon_explanations.md`
4. `04_continuous_hazard_and_survival_curve_explanations.md`
5. `05_competing_risk_and_cause_specific_explanations.md`
6. `06_wsi_attention_risk_heatmaps.md`
7. `07_graph_survival_node_and_topology_effects.md`
8. `08_mcat_genomic_guided_coattention.md`
9. `09_c2mil_semantic_causal_intervention.md`
10. `10_c2mil_topological_causal_subgraphs.md`
11. `11_survival_explanation_validation_and_crgs.md`

## Primary Sources

- Cox, [Regression Models and Life-Tables](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x).
- Yao et al., [Deep Multi-instance Learning for Survival Prediction from Whole
  Slide Images](https://doi.org/10.1007/978-3-030-32239-7_51).
- Chen et al., [Multimodal Co-Attention Transformer for Survival Prediction in
  Gigapixel Whole Slide Images](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html).
- Chan et al., [Histopathology Whole Slide Image Analysis with Heterogeneous
  Graph Representation Learning](https://arxiv.org/abs/2307.04189).
- Cen et al., [C2MIL: Synchronizing Semantic and Topological Causalities in
  Multiple Instance Learning for Robust and Interpretable Survival
  Analysis](https://arxiv.org/abs/2509.20152).


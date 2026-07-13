# Graph and Spatial Explanations

This portion asks:

```text
What does a graph explanation attribute: a node, an edge, a path, a subgraph,
or the spatial hypothesis used to construct the graph?
```

Whole-slide graph papers expose several distinct quantities: graph attention,
node-removal loss changes, stain-level pooling weights, edge interaction
scores, integrated gradients over interpretable region features, and topology
perturbation effects. They should not be collapsed into one heatmap.

## Notes

1. `01_graph_explanation_objects_and_paths.md`
2. `02_heat_leave_one_node_localization.md`
3. `03_node_deletion_fixed_graph_vs_rebuilt_graph.md`
4. `04_message_path_attribution_and_chain_rules.md`
5. `05_edge_interaction_and_heterogeneous_type_credit.md`
6. `06_pseudo_label_pooling_and_readout_localization.md`
7. `07_biox_stain_aware_node_attention.md`
8. `08_biox_stain_entropy_and_interaction_scores.md`
9. `09_region_graph_integrated_gradients.md`
10. `10_hierarchical_graph_attribution.md`
11. `11_spatial_rendering_and_topology_counterfactuals.md`
12. `12_graph_spatial_crgs_and_failure_matrix.md`

## Primary Sources

- Chan et al., [Histopathology Whole Slide Image Analysis with Heterogeneous
  Graph Representation Learning](https://arxiv.org/abs/2307.04189).
- Gallagher-Syed et al., [BioX-CPath: Biologically-driven Explainable
  Diagnostics for Multistain IHC Computational Pathology](https://openaccess.thecvf.com/content/CVPR2025/html/Gallagher-Syed_BioX-CPath_Biologically-driven_Explainable_Diagnostics_for_Multistain_IHC_Computational_Pathology_CVPR_2025_paper.html).
- [A Graph-Based Framework for Interpretable Whole Slide Image
  Analysis](https://arxiv.org/html/2503.11846v1).
- Patch-GCN, [Whole Slide Images are 2D Point Clouds: Context-Aware Survival
  Prediction](https://arxiv.org/abs/2107.13048).
- WiKG, [Dynamic Graph Representation with Knowledge-aware Attention for
  Whole-Slide Image Classification](https://arxiv.org/html/2403.07719).
- HACT-Net, [A Hierarchical Cell-to-Tissue Graph Neural Network for
  Histopathology Image Classification](https://arxiv.org/abs/2103.09658).


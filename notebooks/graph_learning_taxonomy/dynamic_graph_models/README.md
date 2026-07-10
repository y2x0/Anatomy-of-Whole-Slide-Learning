# Dynamic Graph Models

This folder studies graph learners for which the edge set is inferred from the
current slide representation rather than fixed entirely by physical
coordinates or preprocessing.

The pathology anchor is:

```text
Li et al.
Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis
CVPR 2024
https://arxiv.org/abs/2403.07719
```

The general machine-learning anchor used only to distinguish two meanings of
dynamic graph construction is:

```text
Wang et al.
Dynamic Graph CNN for Learning on Point Clouds
ACM Transactions on Graphics 2019
https://arxiv.org/abs/1801.07829
```

WiKG learns one directed graph from patch embeddings during each forward pass.
DGCNN recomputes feature-space neighborhoods after successive EdgeConv layers.
Both are input-dependent constructions, but only the second performs layerwise
graph rewiring.

## Object And Construction

- `01_wikg_slide_object_and_mean_injection.md`: patch embeddings, the
  released-code slide-mean injection, and the attributed directed graph object.
- `02_head_tail_roles_dynamic_support_and_equivariance.md`: asymmetric node
  roles, directed support, meanings of dynamic, and permutation equivariance.
- `03_directed_score_geometry.md`: score matrix, asymmetry, and learned
  bilinear relation geometry.
- `04_topk_normalization_and_stability.md`: printed normalization versus
  released computation, sparse support, top-k margins, and gradients.
- `05_relation_embeddings_and_two_stage_normalization.md`: endpoint
  interpolation and the distinct roles of construction and message weights.
- `06_dense_discovery_scaling_and_batching.md`: quadratic all-pairs discovery,
  sparse selected tensors, and one-slide batching.

## Context Operator

- `07_triplet_compatibility_paper_vs_code.md`: paper dot product versus the
  released rank-one coordinate-sum einsum.
- `08_knowledge_attention_and_dual_interaction.md`: local attention, GAT
  comparison, additive fusion, multiplicative fusion, and matrix form.
- `09_attention_limits_gradients_and_surviving_statistic.md`: limiting cases,
  nested weight dependence, gradient paths, and the local moment bottleneck.

## Readout And Supervision

- `10_dropout_mean_and_max_readout.md`: contextualized nodes, dropout, mean,
  and coordinatewise max.
- `11_attention_readout_invariance_and_statistics.md`: optional global
  attention, permutation invariance, and surviving graph statistics.
- `12_layernorm_classification_and_cross_entropy.md`: LayerNorm, class logits,
  probabilities, cross-entropy, and supervision paths.

## Failure Analysis

- `13_topology_identifiability_geometry_and_shortcuts.md`: weakly identified
  edges, rank instability, non-spatial relations, encoder shortcuts, and
  bag-conditioned topology.
- `14_aggregation_bottlenecks_and_graph_collapse.md`: fixed degree,
  self-edges, relation-state restriction, attention collapse, and local/global
  readout collisions.
- `15_reproducibility_and_implementation_boundaries.md`: one-shot graph
  construction, paper-code differences, and the final failure audit.

## Full Tensor Maps

- `16_full_construction_tensor_map.md`: dimensions from patch features through
  selected relation tensors.
- `17_full_context_readout_and_prediction_map.md`: dimensions from relation
  gates through slide logits and loss.
- `18_complete_wikg_composition.md`: end-to-end composition, C/R/G/S,
  complexity, and surviving statistics.

Cross-family synthesis is split across:

```text
../unifying_view/04_crgs_four_wsi_graph_families.md
../unifying_view/05_graph_geometry_supervision_and_surviving_statistics.md
../unifying_view/06_graph_design_matrix_and_failure_principle.md
```

## C/R/G/S Placement

```text
\mathcal{G}:
    directed top-k graph inferred from learned head-tail compatibility

\mathcal{C}:
    knowledge-aware attention followed by dual-interaction fusion

\mathcal{R}:
    global mean, max, or attention readout over updated head nodes

\mathcal{S}:
    slide-level cross-entropy; topology receives indirect task supervision
```

## Source Boundary

The WiKG paper and released implementation are not identical in every detail.
These notes distinguish:

```text
paper equation:
    the expression printed in the CVPR paper

paper algorithm:
    the PyTorch-like pseudocode printed in the paper

released implementation:
    https://github.com/WonderLandxD/WiKG

reconstruction:
    dimensionally explicit formulas with paper and executed code kept separate
```

The distinction matters for similarity normalization, edge embeddings, the
pre-graph slide-mean injection, the triplet-score einsum, readout choice, and
batch structure.

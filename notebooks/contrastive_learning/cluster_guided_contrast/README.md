# Cluster-Guided Contrast In Computational Pathology

This folder studies contrastive and self-supervised methods in which learned
groups, prototypes, or cluster centroids become supervision objects.

Primary anchors:

- Caron et al. "Deep Clustering for Unsupervised Learning of Visual Features."
  ECCV 2018. https://arxiv.org/abs/1807.05520
- Caron et al. "Unsupervised Learning of Visual Features by Contrasting Cluster
  Assignments." NeurIPS 2020. https://arxiv.org/abs/2006.09882
- Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
  NeurIPS 2013. https://arxiv.org/abs/1306.0895
- Wang et al. "RetCCL: Clustering-guided Contrastive Learning for Whole-slide
  Image Retrieval." Medical Image Analysis 2023.
  https://pubmed.ncbi.nlm.nih.gov/36270093/
- RetCCL implementation and pretrained encoder:
  https://github.com/Xiyue-Wang/RetCCL

## Mathematical Route

- `01_cluster_variables_and_three_regimes.md`: hard labels, soft codes,
  and clusters used as side information.
- `02_deepcluster_alternating_map.md`: global k-means assignments,
  pseudo-label prediction, feature preprocessing, and the non-EM boundary.
- `03_deepcluster_degeneracy_balance_and_stability.md`: empty clusters,
  inverse-size weighting, prior shift, and assignment margins.
- `04_swav_swapped_prediction_and_gradients.md`: prototype
  probabilities, cross-view codes, stop-gradient targets, and exact gradients.
- `05_sinkhorn_balanced_codes.md`: entropy-regularized transport,
  scaling form, Sinkhorn iterations, and transport-plan versus code
  normalization.
- `06_swav_multicrop_partial_global_consistency.md`: directed
  global-to-local code supervision and its pathology assumptions.
- `07_retccl_subqueues_and_weighted_infonce.md`: online k-means
  subqueues, the exact piecewise negative weight, and weighted-logit behavior.
- `08_retccl_group_level_infonce.md`: auxiliary heads, cross-branch
  centroids, group-level candidate classification, and combined objective.
- `09_retccl_wsi_mosaic_and_retrieval_map.md`: dual-cluster mosaics,
  patch retrieval bags, diagnosis-adjusted probabilities, entropy ranking,
  filtering, and slide vote.
- `10_crgs_cluster_geometry_and_failure_matrix.md`: unified C/R/G/S,
  surviving objects, complexity, and failure analysis.

## Three Non-Equivalent Uses Of Clusters

```text
DeepCluster:
    cluster the entire feature collection, then predict hard assignments

SwAV:
    solve a balanced soft assignment problem online, then swap codes across
    views

RetCCL:
    use k-means groups to modify negative logits, define group targets, and
    construct a retrieval mosaic
```

Calling all three "clustering-based contrastive learning" hides different
optimization variables and different meanings of a cluster.

## Family Boundary

A cluster index is latent and permutation invariant. It is not automatically a
tissue label:

```math
C_n
\in
\left\{
1,\ldots,K
\right\}
```

does not imply:

```math
C_n
=
Y_n.
```

The notes preserve that distinction throughout.

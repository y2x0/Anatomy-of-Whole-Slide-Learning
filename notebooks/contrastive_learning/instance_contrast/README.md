# Instance Contrast In Computational Pathology

This folder studies the case in which an image patch is treated as an
individual identity and stochastic views of that patch define the positive
relation.

The pathology anchors are:

- Li, Li, and Eliceiri. "Dual-stream Multiple Instance Learning Network for
  Whole Slide Image Classification with Self-supervised Contrastive Learning."
  CVPR 2021. https://arxiv.org/abs/2011.08939
- Wang et al. "Transformer-based Unsupervised Contrastive Learning for
  Histopathological Image Classification." Medical Image Analysis 2022.
  https://pubmed.ncbi.nlm.nih.gov/35952419/

The objective anchor used by DSMIL is:

- Chen et al. "A Simple Framework for Contrastive Learning of Visual
  Representations." ICML 2020. https://arxiv.org/abs/2002.05709

## Mathematical Route

- `01_patch_identity_and_positive_relation.md`: what is declared to be the
  same instance, which conditional distribution is learned, and which
  pathology relations are absent.
- `02_dsmil_simclr_pretraining_map.md`: the exact separation between
  patch-level SimCLR pretraining, multiscale feature construction, and the
  DSMIL slide head.
- `03_augmentation_orbits_and_pathology_invariance.md`: augmentation
  kernels, quotient geometry, nuisance removal, and destructive invariance.
- `04_ctranspath_local_global_backbone.md`: the CNN stem, windowed and
  shifted-window attention, tensor shapes, and within-patch context.
- `05_srcl_positive_mining_and_objective.md`: CTransPath's online,
  target, and shared-target paths, memory-bank top-S mining, and exact SRCL
  loss.
- `06_srcl_gradients_support_switching_and_bias.md`: positive-mass
  gradients, top-S discontinuities, stale mining, and self-reinforcing
  neighborhoods.
- `07_instance_geometry_and_identifiability.md`: what instance
  discrimination identifies and what it cannot identify.
- `08_patch_encoder_to_wsi_predictor_interface.md`: the mathematical
  boundary between a pretrained patch map and a weakly supervised slide
  learner.
- `09_crgs_and_failure_matrix.md`: C/R/G/S placement, surviving
  statistics, complexity, and failure modes.

## Family Boundary

Instance contrast uses:

```math
x
\longrightarrow
\left(
t_1(x),t_2(x)
\right)
```

as its primitive positive relation. CTransPath extends this relation by mining
other patch identities from a memory bank, but it does not form explicit
cluster assignments. Explicit hard or soft cluster variables are treated in:

```text
../cluster_guided_contrast/
```

## Central Warning

The contrastive learner does not see a whole slide merely because its patches
were cropped from slides. Unless slide identity, coordinates, region
membership, or a slide objective enters the construction, pretraining learns a
patch relation. Slide semantics enter only through the downstream aggregation
stage.

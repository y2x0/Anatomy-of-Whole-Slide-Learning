# Contrastive Learning

This notebook family asks:

```text
What defines similarity?
```

Weak-supervision notes ask where positive and negative relations come from.
This family asks what objective those relations define, which statistical
quantity is estimated, what embedding geometry survives, and how sampling
changes the learned notion of similarity.

The generic pipeline is:

```math
X
\xrightarrow{\mathcal{A}}
(V^{(1)},V^{(2)})
\xrightarrow{\mathcal{C}_{\theta}}
(H^{(1)},H^{(2)})
\xrightarrow{\mathcal{R}_{\theta}}
(Z^{(1)},Z^{(2)})
\xrightarrow{\mathcal{S}}
\mathcal{L}_{\mathrm{contrast}}.
```

Here `A` constructs views or paired objects, `C` encodes them, `R` selects and
projects the representation being compared, and `S` specifies positive,
negative, teacher, or batch-moment supervision.

## Current Portions

```text
00_problem_setup.md
    views, relation distributions, encoder/projector separation, and C/R/G/S

objective_foundations/
    candidate identification and density ratios
    InfoNCE mutual-information bound
    temperature and hyperspherical geometry
    exact gradients, Hessian, and normalization Jacobian
    multi-positive supervised contrastive learning
    false negatives and batch bias
    MoCo dictionary size, queue age, and key consistency
    BYOL and SimSiam stop-gradient dynamics
    Barlow Twins and VICReg moment constraints
    C/R/G/S synthesis and failure matrix

instance_contrast/
    patch identity and same-patch positive statistics
    DSMIL's SimCLR-to-MIL composition
    augmentation orbits and destructive invariance
    CTransPath's CNN-Swin within-patch context
    SRCL top-S memory positives, exact gradients, and support switching
    patch-encoder-to-WSI information boundary
    C/R/G/S synthesis and failure matrix

cluster_guided_contrast/
    hard, soft, and side-information cluster regimes
    DeepCluster alternation, balancing, degeneracy, and assignment stability
    SwAV swapped prediction, Sinkhorn transport, and multi-crop consistency
    RetCCL subqueue-weighted and group-level InfoNCE
    RetCCL mosaic construction and full WSI retrieval map
    C/R/G/S synthesis and failure matrix

multimodal_image_text/
    observed pair semantics and latent compatibility
    CLIP's symmetric matrix objective, exact gradients, and density ratios
    zero-shot text prototypes and prompt-defined decision geometry
    PLIP's OpenPath sampling law, false negatives, and retrieval selection
    CONCH's dual visual readouts and contrastive-captioning objective
    CONCH's prompt-conditioned top-k WSI statistic
    CPLIP's generated bags and exact MIL-NCE gradient hierarchy
    PathCLIP corruption geometry
    C/R/G/S synthesis and failure matrix

graph_contrast/
    local-global, graph-identity, and node-identity contrastive objects
    DGI's exact forward map, discriminator gradients, and corruption law
    GraphCL's graph augmentation kernels and graph-level objective
    augmentation quotients and destructive graph invariance
    GRACE's node-level objective and dual corruption geometry
    graph automorphisms, false negatives, and WSI C/R/G/S synthesis

negative_free_pathology_pretraining/
    negative-free statistical objects and collapse solutions
    DINO teacher-student distributions, centering, sharpening, and multi-crop
    iBOT masked patch-token distillation
    DINOv2 image, patch, assignment-balance, and KoLeo objectives
    HIPT hierarchical self-distillation and frozen scale boundaries
    UNI, Virchow, and Phikon pathology scaling maps
    WSI interface, C/R/G/S synthesis, and failure matrix
```

Later portions will treat the pathology families already listed in the private
research map:

```text
WSI and cross-slide contrast
```

## Boundary With Weak Supervision

```text
weak_supervision/contrastive_labels/:
    why a pair or candidate relation is observed or generated

contrastive_learning/:
    what mathematical objective that relation induces
```

The folders are complementary. A pair can be a valid supervision object while
the resulting estimator is biased, saturated, or geometrically unsuitable.

## Source Policy

Every theory note uses only primary papers already present in:

```text
private/12_topic_private_research_map.md
```

Pathology-specific papers will enter only when their exact construction or
forward map is derived.

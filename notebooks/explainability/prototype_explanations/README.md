# Prototype Explanations

This portion asks:

```text
When does resemblance to a learned representative explain a WSI prediction?
```

The word *prototype* names several different mathematical objects. ProtoPNet
learns latent vectors and projects them onto training patches. PANTHER uses
shared mixture-component identities and estimates slide-specific distributional
parameters. PAMIL couples prototype similarity to two predictive branches.
ProtoMIL learns sparse concept coordinates and displays highly activating
patches as visual representatives. Conflating these objects also conflates
similarity, assignment, prevalence, attention, and score contribution.

## Notes

1. `01_prototype_explanation_objects.md`
2. `02_protopnet_similarity_geometry.md`
3. `03_protopnet_training_projection_and_score_credit.md`
4. `04_similarity_presence_and_contribution.md`
5. `05_prototype_purity_coverage_redundancy_stability.md`
6. `06_panther_gmm_prototype_object.md`
7. `07_panther_em_slide_representation_and_maps.md`
8. `08_pamil_dual_branch_prototype_attention.md`
9. `09_protomil_sparse_concepts_and_visual_prototypes.md`
10. `10_protomil_exact_concept_credit_and_intervention.md`
11. `11_wsi_prototype_crgs_and_failure_matrix.md`

## Primary Sources

- Chen et al., [This Looks Like That: Deep Learning for Interpretable Image
  Recognition](https://arxiv.org/abs/1806.10574).
- Song et al., [PANTHER: A Generalized Deep Learning Framework for Interpretable
  Whole-Slide Image Classification](https://arxiv.org/abs/2405.11643).
- Liu et al., [Prototype Attention-based Multiple Instance Learning for Whole
  Slide Image Classification](https://papers.miccai.org/miccai-2024/587-Paper1022.html).
- Sun et al., [Prototype-Based Multiple Instance Learning for Gigapixel Whole
  Slide Image Classification](https://papers.miccai.org/miccai-2025/0731-Paper0542.html).
